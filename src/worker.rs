//! Background daemon & worker coordinator shared by both Headless mode and GUI mode.
//! Runs two non-blocking background pipelines:
//! 1. Async Tokio HTTP ingestion loop downloading public climate feeds into SQLite.
//! 2. Multi-threaded Rayon physics simulation loop computing all 34 predictive layers
//!    and publishing atomic snapshots to GUI / CLI consumers.

use crate::db::{ClimateDatabase, DbStatsSummary, GlobalLiveReadouts, StoredGridRecord};
use crate::ingest::{IngestionConfig, TelemetryIngestor};
use crate::physics::{ClimateSimulationEngine, SimulationConfig};
use crate::schema::HubSchema;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex, RwLock,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{error, info};

#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub config: SimulationConfig,
    pub readouts: GlobalLiveReadouts,
    pub grids: HashMap<String, StoredGridRecord>,
    pub db_stats: DbStatsSummary,
    pub last_step_ms: u64,
    pub last_ingest_status: String,
}

pub struct BackgroundCoordinator {
    schema: HubSchema,
    db: ClimateDatabase,
    desired_config: Arc<(Mutex<(SimulationConfig, u64)>, Condvar)>,
    latest_snapshot: Arc<RwLock<Option<TelemetrySnapshot>>>,
    ingest_status: Arc<RwLock<String>>,
    shutdown: Arc<AtomicBool>,
}

pub struct RunningWorkers {
    pub coordinator: Arc<BackgroundCoordinator>,
    sim_thread: Option<JoinHandle<()>>,
    ingest_thread: Option<JoinHandle<()>>,
}

impl BackgroundCoordinator {
    pub fn new(schema: HubSchema, db: ClimateDatabase, initial_cfg: SimulationConfig) -> Arc<Self> {
        Arc::new(Self {
            schema,
            db,
            desired_config: Arc::new((Mutex::new((initial_cfg, 1)), Condvar::new())),
            latest_snapshot: Arc::new(RwLock::new(None)),
            ingest_status: Arc::new(RwLock::new("Initializing telemetry sync...".to_string())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn schema(&self) -> &HubSchema {
        &self.schema
    }

    pub fn db(&self) -> &ClimateDatabase {
        &self.db
    }

    /// Update desired simulation parameters (scenario, year, pressure slice, depth slice)
    /// and wake the background Rayon simulation worker immediately.
    pub fn request_simulation_update(&self, new_cfg: SimulationConfig) {
        let (lock, cvar) = &*self.desired_config;
        if let Ok(mut guard) = lock.lock() {
            guard.0 = new_cfg;
            guard.1 = guard.1.wrapping_add(1);
            cvar.notify_one();
        }
    }

    pub fn latest_snapshot(&self) -> Option<TelemetrySnapshot> {
        self.latest_snapshot
            .read()
            .ok()
            .and_then(|guard| guard.clone())
    }

    pub fn current_ingest_status(&self) -> String {
        self.ingest_status
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let (_, cvar) = &*self.desired_config;
        cvar.notify_all();
    }

    /// Spawn both the background async public data downloader thread and the Rayon physics worker thread.
    pub fn spawn_background_workers(
        self: &Arc<Self>,
        ingest_interval: Duration,
        sim_refresh_interval: Duration,
    ) -> RunningWorkers {
        let coord_ingest = Arc::clone(self);
        let ingest_thread = std::thread::Builder::new()
            .name("terratwin-ingest".to_string())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(e) => {
                        error!("Failed starting Tokio runtime for ingestion: {}", e);
                        return;
                    }
                };

                rt.block_on(async move {
                    let ingestor = match TelemetryIngestor::new(IngestionConfig::default()) {
                        Ok(i) => i,
                        Err(e) => {
                            error!("Failed creating HTTP TelemetryIngestor: {}", e);
                            return;
                        }
                    };

                    while !coord_ingest.shutdown.load(Ordering::SeqCst) {
                        if let Ok(mut st) = coord_ingest.ingest_status.write() {
                            *st = "Downloading public climate swaths & station feeds...".to_string();
                        }
                        match ingestor.run_sync_cycle(&coord_ingest.db).await {
                            Ok(rep) => {
                                let msg = if rep.live_records > 0 {
                                    format!(
                                        "Live synced ({} obs | CO2: {} | OISST: {} | FIRMS: {} | GloFAS: {})",
                                        rep.live_records,
                                        rep.mauna_loa_co2_ppm
                                            .map(|v| format!("{:.1}ppm", v))
                                            .unwrap_or_else(|| "ok".to_string()),
                                        rep.erddap_oisst_cells,
                                        rep.firms_fire_clusters,
                                        rep.glofas_river_nodes
                                    )
                                } else {
                                    format!(
                                        "Offline climatology boundary active ({} seed nodes)",
                                        rep.fallback_records
                                    )
                                };
                                info!("{}", msg);
                                if let Ok(mut st) = coord_ingest.ingest_status.write() {
                                    *st = msg;
                                }
                                let (lock, cvar) = &*coord_ingest.desired_config;
                                if let Ok(mut guard) = lock.lock() {
                                    guard.1 = guard.1.wrapping_add(1);
                                    cvar.notify_one();
                                }
                            }
                            Err(err) => {
                                error!("Telemetry ingestion cycle error: {}", err);
                                if let Ok(mut st) = coord_ingest.ingest_status.write() {
                                    *st = format!("Sync warning: {}", err);
                                }
                            }
                        }

                        let mut slept = Duration::ZERO;
                        while slept < ingest_interval
                            && !coord_ingest.shutdown.load(Ordering::SeqCst)
                        {
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            slept += Duration::from_millis(500);
                        }
                    }
                });
            })
            .expect("Failed spawning ingestion thread");

        let coord_sim = Arc::clone(self);
        let sim_thread = std::thread::Builder::new()
            .name("terratwin-rayon-sim".to_string())
            .spawn(move || {
                let engine =
                    ClimateSimulationEngine::new(coord_sim.schema.clone(), coord_sim.db.clone());
                let mut last_seen_gen = 0u64;

                while !coord_sim.shutdown.load(Ordering::SeqCst) {
                    let cfg = {
                        let (lock, cvar) = &*coord_sim.desired_config;
                        let guard = lock.lock().expect("Config mutex poisoned");
                        let guard = if guard.1 == last_seen_gen {
                            cvar.wait_timeout(guard, sim_refresh_interval)
                                .expect("Condvar wait poisoned")
                                .0
                        } else {
                            guard
                        };
                        last_seen_gen = guard.1;
                        guard.0.clone()
                    };

                    if coord_sim.shutdown.load(Ordering::SeqCst) {
                        break;
                    }

                    let t0 = Instant::now();
                    match engine.step_all_layers(&cfg) {
                        Ok(readouts) => {
                            let elapsed_ms = t0.elapsed().as_millis() as u64;
                            let grids = engine
                                .load_layer_bundle(
                                    cfg.scenario,
                                    cfg.atmospheric_level_hpa,
                                    cfg.ocean_depth_m,
                                )
                                .unwrap_or_default();
                            let db_stats = coord_sim.db.summary_stats().unwrap_or(DbStatsSummary {
                                db_path: coord_sim.db.path().display().to_string(),
                                registered_layers: 34,
                                raw_observation_count: 0,
                                simulated_grid_count: grids.len() as i64,
                                readout_snapshots: 1,
                                sync_events: 0,
                                latest_simulation_utc: Some(readouts.timestamp_utc.clone()),
                            });
                            let snap = TelemetrySnapshot {
                                config: cfg,
                                readouts,
                                grids,
                                db_stats,
                                last_step_ms: elapsed_ms,
                                last_ingest_status: coord_sim.current_ingest_status(),
                            };
                            if let Ok(mut w) = coord_sim.latest_snapshot.write() {
                                *w = Some(snap);
                            }
                        }
                        Err(e) => {
                            error!("Multi-threaded climate simulation step failed: {}", e);
                        }
                    }
                }
            })
            .expect("Failed spawning Rayon simulation worker thread");

        RunningWorkers {
            coordinator: Arc::clone(self),
            sim_thread: Some(sim_thread),
            ingest_thread: Some(ingest_thread),
        }
    }
}

impl Drop for RunningWorkers {
    fn drop(&mut self) {
        self.coordinator.signal_shutdown();
        if let Some(h) = self.sim_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.ingest_thread.take() {
            let _ = h.join();
        }
    }
}

/// Run one deterministic synchronous ingest + simulation step (ideal for CLI `step` and integration tests).
pub async fn run_single_headless_cycle(
    schema: &HubSchema,
    db: &ClimateDatabase,
    cfg: &SimulationConfig,
    attempt_network: bool,
) -> Result<GlobalLiveReadouts> {
    if attempt_network {
        let ingestor = TelemetryIngestor::new(IngestionConfig::default())?;
        let _ = ingestor.run_sync_cycle(db).await?;
    } else {
        let coords = crate::interp::fibonacci_sphere_points(36);
        let now = chrono::Utc::now().to_rfc3339();
        let seeded = crate::ingest::seed_climatological_boundary_samples(&coords, &now);
        db.insert_raw_samples(&seeded)?;
    }
    let engine = ClimateSimulationEngine::new(schema.clone(), db.clone());
    engine.step_all_layers(cfg)
}
