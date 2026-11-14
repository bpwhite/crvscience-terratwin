//! Unified binary entrypoint (`terratwin`) supporting both:
//! 1. **Native Desktop Global Map GUI + Background Worker Threads** (`terratwin` or `terratwin gui`)
//! 2. **Native Headless Background Daemon & CLI Utilities** (`terratwin --headless`, `terratwin daemon`,
//!    `terratwin step`, `terratwin status`, `terratwin export`)

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;
use terratwin_climate_hub::{
    db::{ClimateDatabase, RetentionPolicy, ZarrCompressionCodec},
    physics::SimulationConfig,
    schema::{HubSchema, SspScenario},
    worker::{run_single_headless_cycle, BackgroundCoordinator},
};
use tracing::info;

#[derive(Debug, Parser)]
#[command(
    name = "terratwin",
    version = "2.5.0",
    about = "TerraTwin HPC Climate Modeling & Telemetry Hub — Headless Daemon + Native Global Map GUI"
)]
struct Cli {
    /// SQLite WAL database file path for raw telemetry & simulated layer grids
    #[arg(long, default_value = "terratwin_climate.db", global = true)]
    db: PathBuf,

    /// Optional external schema JSON override (defaults to embedded climate_modeling_hub_schema.json)
    #[arg(long, global = true)]
    schema: Option<PathBuf>,

    /// Force headless background daemon mode without opening a GUI window
    #[arg(long, default_value_t = false)]
    headless: bool,

    /// Default SSP climate forcing scenario (SSP1-1.9, SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5)
    #[arg(long, default_value = "SSP2-4.5", global = true)]
    scenario: String,

    /// Number of Rayon worker threads (0 = use all available CPU cores)
    #[arg(long, default_value_t = 0, global = true)]
    threads: usize,

    /// Maximum historical simulation ticks retained per SSP scenario in SQLite (0 = unlimited)
    #[arg(long, default_value_t = 96, global = true)]
    max_history_ticks: usize,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Launch the interactive global map GUI with background telemetry & Rayon simulation workers
    Gui,

    /// Run continuously in headless background daemon mode, downloading raw data & updating SQLite
    Daemon {
        /// Telemetry HTTP download interval in seconds
        #[arg(long, default_value_t = 300)]
        ingest_interval_secs: u64,
        /// Multi-threaded climate physics refresh interval in seconds
        #[arg(long, default_value_t = 30)]
        sim_interval_secs: u64,
        /// Optional maximum daemon cycles before exiting (0 = run indefinitely)
        #[arg(long, default_value_t = 0)]
        max_cycles: usize,
    },

    /// Execute a single synchronous ingestion + 34-layer multi-threaded simulation pass and exit
    Step {
        /// Target simulation year (1950.0 .. 2100.0)
        #[arg(long, default_value_t = 2026.86)]
        year: f64,
        /// Atmospheric vertical slice in hPa (1000 .. 10)
        #[arg(long, default_value_t = 850.0)]
        hpa: f64,
        /// Ocean vertical slice depth in meters (0 .. 6000)
        #[arg(long, default_value_t = 0.0)]
        depth_m: f64,
        /// Skip network calls and seed from deterministic physical climatology
        #[arg(long, default_value_t = false)]
        offline: bool,
    },

    /// Print SQLite database catalog statistics and latest simulated global climate readouts
    Status,

    /// Export a chunked Zarr v3 binary directory store (`.zarr`) or legacy `.json` subset from SQLite
    Export {
        /// Layer identifier from climate_modeling_hub_schema.json (e.g. sst_marine_heatwaves)
        #[arg(long)]
        layer: String,
        /// Output Zarr v3 directory (e.g. `layer_subset.zarr`) or `.json` subset path
        #[arg(long, default_value = "layer_subset.zarr")]
        output: PathBuf,
        #[arg(long, default_value_t = -90.0)]
        min_lat: f64,
        #[arg(long, default_value_t = 90.0)]
        max_lat: f64,
        #[arg(long, default_value_t = -180.0)]
        min_lon: f64,
        #[arg(long, default_value_t = 180.0)]
        max_lon: f64,
        #[arg(long, default_value_t = 850.0)]
        level: f64,
        #[arg(long, default_value_t = 36)]
        chunk_rows: usize,
        #[arg(long, default_value_t = 72)]
        chunk_cols: usize,
        /// Zarr v3 compression codec (`zstd`, `gzip`, or `none`)
        #[arg(long, default_value = "zstd")]
        codec: String,
        /// Compression level (`1..=22` for `zstd`, `1..=9` for `gzip`)
        #[arg(long, default_value_t = 3)]
        compression_level: i32,
        #[arg(long, default_value_t = false)]
        json_subset: bool,
    },

    /// Prune expired SQLite simulation ticks, raw observations, and sync logs (with optional VACUUM)
    Prune {
        /// Number of most recent 4D ticks to keep per SSP scenario (0 = keep all)
        #[arg(long, default_value_t = 96)]
        keep_ticks: usize,
        /// Number of most recent raw telemetry observations to keep per layer (0 = keep all)
        #[arg(long, default_value_t = 4096)]
        keep_raw_per_layer: usize,
        /// Run `PRAGMA wal_checkpoint(TRUNCATE)` and `VACUUM` to reclaim disk pages immediately
        #[arg(long, default_value_t = false)]
        vacuum: bool,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    let schema = HubSchema::load(cli.schema.as_deref())?;
    let db = ClimateDatabase::open(&cli.db)?;
    db.set_retention_policy(RetentionPolicy {
        max_ticks_per_scenario: cli.max_history_ticks,
        ..RetentionPolicy::default()
    });
    let synced_layers = db.sync_schema_catalog(&schema)?;
    info!(
        "Initialized SQLite WAL store ({}) with {} predictive layers across {} groups",
        cli.db.display(),
        synced_layers,
        schema.layer_groups.len()
    );

    let ssp = SspScenario::from_code(&cli.scenario);
    let mut base_cfg = SimulationConfig::default();
    base_cfg.scenario = ssp;

    if cli.headless && cli.command.is_none() {
        return run_daemon_loop(schema, db, base_cfg, 300, 30, 0);
    }

    match cli.command {
        Some(Commands::Daemon {
            ingest_interval_secs,
            sim_interval_secs,
            max_cycles,
        }) => run_daemon_loop(
            schema,
            db,
            base_cfg,
            ingest_interval_secs,
            sim_interval_secs,
            max_cycles,
        ),

        Some(Commands::Step {
            year,
            hpa,
            depth_m,
            offline,
        }) => {
            base_cfg.sim_year = year;
            base_cfg.atmospheric_level_hpa = hpa;
            base_cfg.ocean_depth_m = depth_m;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let readouts =
                rt.block_on(run_single_headless_cycle(&schema, &db, &base_cfg, !offline))?;
            let stats = db.summary_stats()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "mode": "headless_single_step",
                    "readouts": readouts,
                    "sqlite_stats": stats,
                }))?
            );
            Ok(())
        }

        Some(Commands::Status) => {
            let stats = db.summary_stats()?;
            let latest_readouts = db.latest_global_readouts(ssp)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "application": schema.application.name,
                    "version": schema.application.version,
                    "sqlite_stats": stats,
                    "latest_readouts": latest_readouts,
                }))?
            );
            Ok(())
        }

        Some(Commands::Export {
            layer,
            output,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            level,
            chunk_rows,
            chunk_cols,
            codec,
            compression_level,
            json_subset,
        }) => {
            let is_json = json_subset
                || output
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("json"))
                    .unwrap_or(false);
            if is_json {
                let cells = db.export_layer_subset_json(
                    &layer, ssp, level, min_lat, max_lat, min_lon, max_lon, &output,
                )?;
                info!(
                    "Exported {} grid cells for layer '{}' to JSON {}",
                    cells,
                    layer,
                    output.display()
                );
            } else {
                let zarr_codec = ZarrCompressionCodec::from_cli(&codec, compression_level)?;
                let summary = db.export_zarr_v3_store_with_codec(
                    &layer,
                    ssp,
                    level,
                    min_lat,
                    max_lat,
                    min_lon,
                    max_lon,
                    (chunk_rows, chunk_cols),
                    zarr_codec,
                    &output,
                )?;
                println!("{}", serde_json::to_string_pretty(&summary)?);
            }
            Ok(())
        }

        Some(Commands::Prune {
            keep_ticks,
            keep_raw_per_layer,
            vacuum,
        }) => {
            db.set_retention_policy(RetentionPolicy {
                max_ticks_per_scenario: keep_ticks,
                max_raw_samples_per_layer: keep_raw_per_layer,
                max_sync_log_rows: 1_000,
            });
            let report = db.prune_expired_history(vacuum)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }

        Some(Commands::Gui) | None => {
            #[cfg(feature = "gui")]
            {
                let coord = BackgroundCoordinator::new(schema, db, base_cfg);
                let workers = coord.spawn_background_workers(
                    Duration::from_secs(300),
                    Duration::from_secs(45),
                );
                terratwin_climate_hub::gui::launch_native_gui(workers)
            }
            #[cfg(not(feature = "gui"))]
            {
                info!("Binary compiled without `gui` feature; starting headless daemon mode");
                run_daemon_loop(schema, db, base_cfg, 300, 30, 0)
            }
        }
    }
}

fn run_daemon_loop(
    schema: HubSchema,
    db: ClimateDatabase,
    mut cfg: SimulationConfig,
    ingest_secs: u64,
    sim_secs: u64,
    max_cycles: usize,
) -> Result<()> {
    info!(
        "Starting TerraTwin headless daemon (ingest={}s, sim={}s, scenario={})",
        ingest_secs,
        sim_secs,
        cfg.scenario.code()
    );
    let coord = BackgroundCoordinator::new(schema, db, cfg.clone());
    let workers = coord.spawn_background_workers(
        Duration::from_secs(ingest_secs.max(5)),
        Duration::from_secs(sim_secs.max(2)),
    );

    let mut cycles = 0usize;
    loop {
        std::thread::sleep(Duration::from_secs(sim_secs.max(2)));
        if let Some(snap) = workers.coordinator.latest_snapshot() {
            info!(
                "[Headless Tick] Scenario={} | dT={:+.2}C | CO2={:.1}ppm | ENSO={:+.2}C | SST_Anom={:+.2}C | Grids={} ({} ms)",
                snap.readouts.scenario,
                snap.readouts.global_2m_temp_anomaly_c,
                snap.readouts.mauna_loa_co2_ppm,
                snap.readouts.enso_nino34_c,
                snap.readouts.global_sst_anomaly_c,
                snap.db_stats.simulated_grid_count,
                snap.last_step_ms
            );
            cycles += 1;
            if max_cycles > 0 && cycles >= max_cycles {
                info!("Reached max_cycles={}, shutting down headless daemon cleanly", max_cycles);
                break;
            }
            cfg.advance_tick();
            workers.coordinator.request_simulation_update(cfg.clone());
        }
    }
    Ok(())
}
