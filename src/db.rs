//! SQLite WAL data management layer (`rusqlite`).
//! Stores layer catalog metadata, raw public telemetry swaths/observations,
//! multi-threaded simulated spatial grids (little-endian `f32` raster blobs),
//! vertical sounding/ocean profiles, top-bar global climate indices, and sync logs.

use crate::schema::{HubSchema, SspScenario};
use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use flate2::{read::GzDecoder, write::GzEncoder, Compression};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTelemetrySample {
    pub layer_id: String,
    pub source_name: String,
    pub source_url: String,
    pub timestamp_utc: String,
    pub latitude: f64,
    pub longitude: f64,
    pub vertical_level: f64,
    pub primary_value: f64,
    pub secondary_value: Option<f64>,
    pub quality_flag: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalLiveReadouts {
    pub timestamp_utc: String,
    pub scenario: String,
    pub global_2m_temp_anomaly_c: f64,
    pub mauna_loa_co2_ppm: f64,
    pub enso_nino34_c: f64,
    pub global_sst_anomaly_c: f64,
}

#[derive(Debug, Clone)]
pub struct StoredGridRecord {
    pub layer_id: String,
    pub scenario: String,
    pub timestamp_utc: String,
    pub vertical_level: f64,
    pub width: usize,
    pub height: usize,
    pub min_val: f32,
    pub max_val: f32,
    pub mean_val: f32,
    pub values: Vec<f32>,
    /// Optional U/V vector components (for wind streamlines, ocean currents, sea ice drift, IVT).
    pub u_component: Option<Vec<f32>>,
    pub v_component: Option<Vec<f32>>,
}

/// Automatic SQLite ring-buffer retention policy bounding disk growth in long-running daemons.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Maximum distinct simulation `timestamp_utc` ticks kept per SSP scenario (`0` = unlimited).
    pub max_ticks_per_scenario: usize,
    /// Maximum raw telemetry rows kept per `layer_id` (`0` = unlimited).
    pub max_raw_samples_per_layer: usize,
    /// Maximum sync audit log rows kept (`0` = unlimited).
    pub max_sync_log_rows: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_ticks_per_scenario: 96,
            max_raw_samples_per_layer: 4_096,
            max_sync_log_rows: 1_000,
        }
    }
}

/// Summary returned by `ClimateDatabase::prune_expired_history`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PruneReport {
    pub grids_deleted: usize,
    pub readouts_deleted: usize,
    pub raw_samples_deleted: usize,
    pub sync_logs_deleted: usize,
    pub grid_blob_bytes_remaining: i64,
}

/// Standards-compliant Zarr v3 `bytes -> bytes` compression codec selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZarrCompressionCodec {
    None,
    Zstd { level: i32, checksum: bool },
    Gzip { level: u32 },
}

impl Default for ZarrCompressionCodec {
    fn default() -> Self {
        Self::Zstd {
            level: 3,
            checksum: true,
        }
    }
}

impl ZarrCompressionCodec {
    pub fn from_cli(name: &str, level: i32) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "zstd" | "zstandard" => Ok(Self::Zstd {
                level: level.clamp(1, 22),
                checksum: true,
            }),
            "gzip" | "gz" => Ok(Self::Gzip {
                level: level.clamp(1, 9) as u32,
            }),
            "none" | "raw" | "bytes" => Ok(Self::None),
            other => anyhow::bail!(
                "Unsupported Zarr v3 codec '{}' (expected 'zstd', 'gzip', or 'none')",
                other
            ),
        }
    }

    pub fn label(&self) -> String {
        match *self {
            Self::None => "bytes".to_string(),
            Self::Zstd { level, .. } => format!("zstd(level={})", level),
            Self::Gzip { level } => format!("gzip(level={})", level),
        }
    }

    pub fn zarr_v3_codecs_json(&self) -> serde_json::Value {
        let bytes_codec = serde_json::json!({
            "name": "bytes",
            "configuration": { "endian": "little" }
        });
        match *self {
            Self::None => serde_json::json!([bytes_codec]),
            Self::Zstd { level, checksum } => serde_json::json!([
                bytes_codec,
                {
                    "name": "zstd",
                    "configuration": { "level": level, "checksum": checksum }
                }
            ]),
            Self::Gzip { level } => serde_json::json!([
                bytes_codec,
                {
                    "name": "gzip",
                    "configuration": { "level": level }
                }
            ]),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZarrV3ExportSummary {
    pub store_dir: String,
    pub layer_id: String,
    pub codec: String,
    pub shape: [usize; 2],
    pub chunk_shape: [usize; 2],
    pub chunks_written: usize,
    pub cells_exported: usize,
    pub uncompressed_bytes: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbStatsSummary {
    pub db_path: String,
    pub registered_layers: i64,
    pub raw_observation_count: i64,
    pub simulated_grid_count: i64,
    pub readout_snapshots: i64,
    pub sync_events: i64,
    pub latest_simulation_utc: Option<String>,
}

#[derive(Clone)]
pub struct ClimateDatabase {
    path: PathBuf,
    conn: Arc<Mutex<Connection>>,
    retention: Arc<Mutex<RetentionPolicy>>,
}

impl ClimateDatabase {
    /// Open or create the SQLite database file, enable WAL concurrency, and run migrations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        if let Some(parent) = path_buf.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let conn = Connection::open(&path_buf)
            .with_context(|| format!("Failed opening SQLite DB at {}", path_buf.display()))?;

        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA temp_store = MEMORY;
            PRAGMA cache_size = -64000;

            CREATE TABLE IF NOT EXISTS app_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS layer_catalog (
                layer_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                group_name TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                public_sources_json TEXT NOT NULL,
                render_type TEXT NOT NULL,
                data_format TEXT NOT NULL,
                spatial_resolution TEXT NOT NULL,
                temporal_resolution TEXT NOT NULL,
                vertical_dimension TEXT NOT NULL,
                units TEXT NOT NULL,
                default_visible INTEGER NOT NULL,
                default_opacity REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS raw_telemetry_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_url TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                vertical_level REAL NOT NULL DEFAULT 0.0,
                primary_value REAL NOT NULL,
                secondary_value REAL,
                quality_flag INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_raw_telemetry_layer_time
                ON raw_telemetry_samples(layer_id, timestamp_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_raw_telemetry_spatial
                ON raw_telemetry_samples(layer_id, latitude, longitude);

            CREATE TABLE IF NOT EXISTS simulated_layer_grids (
                layer_id TEXT NOT NULL,
                scenario TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                vertical_level REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                min_val REAL NOT NULL,
                max_val REAL NOT NULL,
                mean_val REAL NOT NULL,
                grid_f32_le BLOB NOT NULL,
                u_f32_le BLOB,
                v_f32_le BLOB,
                computed_at_utc TEXT NOT NULL,
                PRIMARY KEY (layer_id, scenario, timestamp_utc, vertical_level)
            );

            CREATE INDEX IF NOT EXISTS idx_sim_grids_lookup
                ON simulated_layer_grids(layer_id, scenario, vertical_level);

            CREATE TABLE IF NOT EXISTS global_readouts_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                scenario TEXT NOT NULL,
                global_2m_temp_anomaly_c REAL NOT NULL,
                mauna_loa_co2_ppm REAL NOT NULL,
                enso_nino34_c REAL NOT NULL,
                global_sst_anomaly_c REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_readouts_time
                ON global_readouts_history(scenario, timestamp_utc DESC);

            CREATE TABLE IF NOT EXISTS ingestion_sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_name TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                fetched_at_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                records_ingested INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                notes TEXT
            );
            ",
        )
        .context("Failed initializing SQLite schema")?;

        Ok(Self {
            path: path_buf,
            conn: Arc::new(Mutex::new(conn)),
            retention: Arc::new(Mutex::new(RetentionPolicy::default())),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn set_retention_policy(&self, policy: RetentionPolicy) {
        if let Ok(mut guard) = self.retention.lock() {
            *guard = policy;
        }
    }

    pub fn retention_policy(&self) -> RetentionPolicy {
        self.retention
            .lock()
            .map(|g| *g)
            .unwrap_or_default()
    }

    /// Synchronize the JSON schema's 6 layer groups and 34 predictive layers into `layer_catalog`.
    pub fn sync_schema_catalog(&self, schema: &HubSchema) -> Result<usize> {
        let mut conn = self.conn.lock().expect("SQLite mutex poisoned");
        let tx = conn.transaction()?;
        let now = chrono::Utc::now().to_rfc3339();

        tx.execute(
            "INSERT INTO app_metadata (key, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            params!["schema_version", &schema.application.version, &now],
        )?;
        tx.execute(
            "INSERT INTO app_metadata (key, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            params!["app_name", &schema.application.name, &now],
        )?;

        let mut count = 0usize;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO layer_catalog (
                    layer_id, group_id, group_name, name, description, public_sources_json,
                    render_type, data_format, spatial_resolution, temporal_resolution,
                    vertical_dimension, units, default_visible, default_opacity
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
                ON CONFLICT(layer_id) DO UPDATE SET
                    group_id = excluded.group_id,
                    group_name = excluded.group_name,
                    name = excluded.name,
                    description = excluded.description,
                    public_sources_json = excluded.public_sources_json,
                    render_type = excluded.render_type,
                    data_format = excluded.data_format,
                    spatial_resolution = excluded.spatial_resolution,
                    temporal_resolution = excluded.temporal_resolution,
                    vertical_dimension = excluded.vertical_dimension,
                    units = excluded.units,
                    default_visible = excluded.default_visible,
                    default_opacity = excluded.default_opacity",
            )?;

            for (group, layer) in schema.all_layers() {
                let sources_json = serde_json::to_string(&layer.public_sources)?;
                stmt.execute(params![
                    &layer.id,
                    &group.group_id,
                    &group.group_name,
                    &layer.name,
                    &layer.description,
                    sources_json,
                    &layer.render_type,
                    &layer.data_format,
                    &layer.spatial_resolution,
                    &layer.temporal_resolution,
                    &layer.vertical_dimension,
                    &layer.units,
                    i32::from(layer.default_visible),
                    layer.default_opacity as f64,
                ])?;
                count += 1;
            }
        }
        tx.commit()?;
        Ok(count)
    }

    /// Batch-insert raw telemetry observations inside a single WAL transaction.
    pub fn insert_raw_samples(&self, samples: &[RawTelemetrySample]) -> Result<usize> {
        if samples.is_empty() {
            return Ok(0);
        }
        let mut conn = self.conn.lock().expect("SQLite mutex poisoned");
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO raw_telemetry_samples (
                    layer_id, source_name, source_url, timestamp_utc,
                    latitude, longitude, vertical_level, primary_value, secondary_value, quality_flag
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for s in samples {
                stmt.execute(params![
                    &s.layer_id,
                    &s.source_name,
                    &s.source_url,
                    &s.timestamp_utc,
                    s.latitude,
                    s.longitude,
                    s.vertical_level,
                    s.primary_value,
                    s.secondary_value,
                    s.quality_flag,
                ])?;
            }
        }
        let max_raw = self.retention_policy().max_raw_samples_per_layer;
        if max_raw > 0 {
            let mut touched_layers = BTreeSet::new();
            for s in samples {
                touched_layers.insert(s.layer_id.as_str());
            }
            for lid in touched_layers {
                tx.execute(
                    "DELETE FROM raw_telemetry_samples
                     WHERE layer_id = ?1
                       AND id NOT IN (
                           SELECT id FROM raw_telemetry_samples
                           WHERE layer_id = ?1
                           ORDER BY id DESC
                           LIMIT ?2
                       )",
                    params![lid, max_raw as i64],
                )?;
            }
        }
        tx.commit()?;
        Ok(samples.len())
    }

    /// Retrieve the most recent raw observations for a given `layer_id` (up to `limit` points)
    /// to assimilate into Spherical IDW / Barnes objective analysis.
    pub fn fetch_recent_samples(
        &self,
        layer_id: &str,
        limit: usize,
    ) -> Result<Vec<RawTelemetrySample>> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT layer_id, source_name, source_url, timestamp_utc,
                    latitude, longitude, vertical_level, primary_value, secondary_value, quality_flag
             FROM raw_telemetry_samples
             WHERE layer_id = ?1 AND quality_flag >= 0
             ORDER BY id DESC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![layer_id, limit as i64], |row| {
            Ok(RawTelemetrySample {
                layer_id: row.get(0)?,
                source_name: row.get(1)?,
                source_url: row.get(2)?,
                timestamp_utc: row.get(3)?,
                latitude: row.get(4)?,
                longitude: row.get(5)?,
                vertical_level: row.get(6)?,
                primary_value: row.get(7)?,
                secondary_value: row.get(8)?,
                quality_flag: row.get(9)?,
            })
        })?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Persist a batch of simulated 2D/vertical-slice raster grids in a single SQLite transaction.
    pub fn upsert_simulated_grids(&self, grids: &[StoredGridRecord]) -> Result<usize> {
        if grids.is_empty() {
            return Ok(0);
        }
        let mut conn = self.conn.lock().expect("SQLite mutex poisoned");
        let tx = conn.transaction()?;
        let computed_at = chrono::Utc::now().to_rfc3339();
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO simulated_layer_grids (
                    layer_id, scenario, timestamp_utc, vertical_level,
                    width, height, min_val, max_val, mean_val,
                    grid_f32_le, u_f32_le, v_f32_le, computed_at_utc
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
                ON CONFLICT(layer_id, scenario, timestamp_utc, vertical_level) DO UPDATE SET
                    width = excluded.width,
                    height = excluded.height,
                    min_val = excluded.min_val,
                    max_val = excluded.max_val,
                    mean_val = excluded.mean_val,
                    grid_f32_le = excluded.grid_f32_le,
                    u_f32_le = excluded.u_f32_le,
                    v_f32_le = excluded.v_f32_le,
                    computed_at_utc = excluded.computed_at_utc",
            )?;

            for g in grids {
                let grid_bytes = pack_f32_compressed(&g.values);
                let u_bytes = g.u_component.as_ref().map(|v| pack_f32_compressed(v));
                let v_bytes = g.v_component.as_ref().map(|v| pack_f32_compressed(v));
                stmt.execute(params![
                    &g.layer_id,
                    &g.scenario,
                    &g.timestamp_utc,
                    g.vertical_level,
                    g.width as i64,
                    g.height as i64,
                    g.min_val as f64,
                    g.max_val as f64,
                    g.mean_val as f64,
                    grid_bytes,
                    u_bytes,
                    v_bytes,
                    &computed_at,
                ])?;
            }
        }
        let max_ticks = self.retention_policy().max_ticks_per_scenario;
        if max_ticks > 0 {
            let mut scenarios = BTreeSet::new();
            for g in grids {
                scenarios.insert(g.scenario.as_str());
            }
            for scen in scenarios {
                Self::prune_scenario_ticks_tx(&tx, scen, max_ticks)?;
            }
        }
        tx.commit()?;
        Ok(grids.len())
    }

    fn prune_scenario_ticks_tx(
        tx: &rusqlite::Transaction<'_>,
        scenario_code: &str,
        max_ticks: usize,
    ) -> Result<(usize, usize)> {
        if max_ticks == 0 {
            return Ok((0, 0));
        }
        let g_del = tx.execute(
            "DELETE FROM simulated_layer_grids
             WHERE scenario = ?1
               AND timestamp_utc NOT IN (
                   SELECT DISTINCT timestamp_utc
                   FROM simulated_layer_grids
                   WHERE scenario = ?1
                   ORDER BY timestamp_utc DESC
                   LIMIT ?2
               )",
            params![scenario_code, max_ticks as i64],
        )?;
        let r_del = tx.execute(
            "DELETE FROM global_readouts_history
             WHERE scenario = ?1
               AND timestamp_utc NOT IN (
                   SELECT DISTINCT timestamp_utc
                   FROM simulated_layer_grids
                   WHERE scenario = ?1
               )
               AND id NOT IN (
                   SELECT id FROM global_readouts_history
                   WHERE scenario = ?1
                   ORDER BY id DESC
                   LIMIT ?2
               )",
            params![scenario_code, max_ticks as i64],
        )?;
        Ok((g_del, r_del))
    }

    /// Execute full retention pruning across all SSP scenarios, raw telemetry tables, and sync logs.
    pub fn prune_expired_history(&self, vacuum: bool) -> Result<PruneReport> {
        let policy = self.retention_policy();
        let mut report = PruneReport::default();

        {
            let mut conn = self.conn.lock().expect("SQLite mutex poisoned");
            let tx = conn.transaction()?;

            if policy.max_ticks_per_scenario > 0 {
                for ssp in SspScenario::ALL {
                    let (gd, rd) =
                        Self::prune_scenario_ticks_tx(&tx, ssp.code(), policy.max_ticks_per_scenario)?;
                    report.grids_deleted += gd;
                    report.readouts_deleted += rd;
                }
            }

            if policy.max_raw_samples_per_layer > 0 {
                let layer_ids: Vec<String> = {
                    let mut st =
                        tx.prepare("SELECT DISTINCT layer_id FROM raw_telemetry_samples")?;
                    let rows = st.query_map([], |r| r.get::<_, String>(0))?;
                    let mut v = Vec::new();
                    for r in rows {
                        v.push(r?);
                    }
                    v
                };
                for lid in layer_ids {
                    report.raw_samples_deleted += tx.execute(
                        "DELETE FROM raw_telemetry_samples
                         WHERE layer_id = ?1
                           AND id NOT IN (
                               SELECT id FROM raw_telemetry_samples
                               WHERE layer_id = ?1
                               ORDER BY id DESC
                               LIMIT ?2
                           )",
                        params![&lid, policy.max_raw_samples_per_layer as i64],
                    )?;
                }
            }

            if policy.max_sync_log_rows > 0 {
                report.sync_logs_deleted += tx.execute(
                    "DELETE FROM ingestion_sync_log
                     WHERE id NOT IN (
                         SELECT id FROM ingestion_sync_log
                         ORDER BY id DESC
                         LIMIT ?1
                     )",
                    params![policy.max_sync_log_rows as i64],
                )?;
            }

            report.grid_blob_bytes_remaining = tx.query_row(
                "SELECT COALESCE(SUM(LENGTH(grid_f32_le) + COALESCE(LENGTH(u_f32_le), 0) + COALESCE(LENGTH(v_f32_le), 0)), 0)
                 FROM simulated_layer_grids",
                [],
                |r| r.get(0),
            )?;
            tx.commit()?;
        }

        if vacuum {
            let conn = self.conn.lock().expect("SQLite mutex poisoned");
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE); VACUUM;")?;
        }
        Ok(report)
    }

    /// Load the latest simulated grid for `(layer_id, scenario)` closest to `vertical_level`.
    pub fn load_latest_grid(
        &self,
        layer_id: &str,
        scenario: SspScenario,
        vertical_level: f64,
    ) -> Result<Option<StoredGridRecord>> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT layer_id, scenario, timestamp_utc, vertical_level,
                    width, height, min_val, max_val, mean_val,
                    grid_f32_le, u_f32_le, v_f32_le
             FROM simulated_layer_grids
             WHERE layer_id = ?1 AND scenario = ?2
             ORDER BY ABS(vertical_level - ?3) ASC, timestamp_utc DESC
             LIMIT 1",
        )?;

        stmt.query_row(params![layer_id, scenario.code(), vertical_level], |row| {
            let grid_blob: Vec<u8> = row.get(9)?;
            let u_blob: Option<Vec<u8>> = row.get(10)?;
            let v_blob: Option<Vec<u8>> = row.get(11)?;
            let width_i64: i64 = row.get(4)?;
            let height_i64: i64 = row.get(5)?;
            let min_f64: f64 = row.get(6)?;
            let max_f64: f64 = row.get(7)?;
            let mean_f64: f64 = row.get(8)?;

            Ok(StoredGridRecord {
                layer_id: row.get(0)?,
                scenario: row.get(1)?,
                timestamp_utc: row.get(2)?,
                vertical_level: row.get(3)?,
                width: width_i64 as usize,
                height: height_i64 as usize,
                min_val: min_f64 as f32,
                max_val: max_f64 as f32,
                mean_val: mean_f64 as f32,
                values: unpack_f32_auto(&grid_blob),
                u_component: u_blob.as_deref().map(unpack_f32_auto),
                v_component: v_blob.as_deref().map(unpack_f32_auto),
            })
        })
        .optional()
        .context("Failed reading simulated grid from SQLite")
    }

    /// List chronologically ordered `timestamp_utc` ticks stored in SQLite for `scenario`.
    pub fn list_available_timestamps(
        &self,
        scenario: SspScenario,
        limit: usize,
    ) -> Result<Vec<String>> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT timestamp_utc FROM (
                SELECT DISTINCT timestamp_utc
                FROM simulated_layer_grids
                WHERE scenario = ?1
                ORDER BY timestamp_utc DESC
                LIMIT ?2
             ) sub
             ORDER BY timestamp_utc ASC",
        )?;
        let rows = stmt.query_map(params![scenario.code(), limit.max(1) as i64], |r| {
            r.get::<_, String>(0)
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Load a simulated grid at an exact SQLite `timestamp_utc` closest to `vertical_level`.
    pub fn load_grid_at_timestamp(
        &self,
        layer_id: &str,
        scenario: SspScenario,
        vertical_level: f64,
        timestamp_utc: &str,
    ) -> Result<Option<StoredGridRecord>> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT layer_id, scenario, timestamp_utc, vertical_level,
                    width, height, min_val, max_val, mean_val,
                    grid_f32_le, u_f32_le, v_f32_le
             FROM simulated_layer_grids
             WHERE layer_id = ?1 AND scenario = ?2 AND timestamp_utc = ?3
             ORDER BY ABS(vertical_level - ?4) ASC
             LIMIT 1",
        )?;

        stmt.query_row(
            params![layer_id, scenario.code(), timestamp_utc, vertical_level],
            |row| {
                let grid_blob: Vec<u8> = row.get(9)?;
                let u_blob: Option<Vec<u8>> = row.get(10)?;
                let v_blob: Option<Vec<u8>> = row.get(11)?;
                let width_i64: i64 = row.get(4)?;
                let height_i64: i64 = row.get(5)?;
                let min_f64: f64 = row.get(6)?;
                let max_f64: f64 = row.get(7)?;
                let mean_f64: f64 = row.get(8)?;

                Ok(StoredGridRecord {
                    layer_id: row.get(0)?,
                    scenario: row.get(1)?,
                    timestamp_utc: row.get(2)?,
                    vertical_level: row.get(3)?,
                    width: width_i64 as usize,
                    height: height_i64 as usize,
                    min_val: min_f64 as f32,
                    max_val: max_f64 as f32,
                    mean_val: mean_f64 as f32,
                    values: unpack_f32_auto(&grid_blob),
                    u_component: u_blob.as_deref().map(unpack_f32_auto),
                    v_component: v_blob.as_deref().map(unpack_f32_auto),
                })
            },
        )
        .optional()
        .context("Failed reading historical simulated grid from SQLite")
    }

    /// Load all 34 simulated layer grids for a specific historical SQLite `timestamp_utc`.
    pub fn load_layer_bundle_at_timestamp(
        &self,
        schema: &HubSchema,
        scenario: SspScenario,
        atm_hpa: f64,
        ocean_m: f64,
        timestamp_utc: &str,
    ) -> Result<std::collections::HashMap<String, StoredGridRecord>> {
        let mut map = std::collections::HashMap::new();
        for (group, layer) in schema.all_layers() {
            let target_lvl = if group.group_id == "physical_biogeochemical_oceanography" {
                ocean_m
            } else {
                atm_hpa
            };
            if let Some(grid) =
                self.load_grid_at_timestamp(&layer.id, scenario, target_lvl, timestamp_utc)?
            {
                map.insert(layer.id.clone(), grid);
            }
        }
        Ok(map)
    }

    /// Load `GlobalLiveReadouts` recorded at or immediately prior to `timestamp_utc`.
    pub fn load_readouts_at_timestamp(
        &self,
        scenario: SspScenario,
        timestamp_utc: &str,
    ) -> Result<Option<GlobalLiveReadouts>> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT timestamp_utc, scenario, global_2m_temp_anomaly_c,
                    mauna_loa_co2_ppm, enso_nino34_c, global_sst_anomaly_c
             FROM global_readouts_history
             WHERE scenario = ?1 AND timestamp_utc <= ?2
             ORDER BY timestamp_utc DESC, id DESC
             LIMIT 1",
        )?;
        stmt.query_row(params![scenario.code(), timestamp_utc], |row| {
            Ok(GlobalLiveReadouts {
                timestamp_utc: row.get(0)?,
                scenario: row.get(1)?,
                global_2m_temp_anomaly_c: row.get(2)?,
                mauna_loa_co2_ppm: row.get(3)?,
                enso_nino34_c: row.get(4)?,
                global_sst_anomaly_c: row.get(5)?,
            })
        })
        .optional()
        .context("Failed querying historical global_readouts_history")
    }

    pub fn record_global_readouts(&self, readout: &GlobalLiveReadouts) -> Result<()> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        conn.execute(
            "INSERT INTO global_readouts_history (
                timestamp_utc, scenario, global_2m_temp_anomaly_c,
                mauna_loa_co2_ppm, enso_nino34_c, global_sst_anomaly_c
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &readout.timestamp_utc,
                &readout.scenario,
                readout.global_2m_temp_anomaly_c,
                readout.mauna_loa_co2_ppm,
                readout.enso_nino34_c,
                readout.global_sst_anomaly_c,
            ],
        )?;
        Ok(())
    }

    pub fn latest_global_readouts(
        &self,
        scenario: SspScenario,
    ) -> Result<Option<GlobalLiveReadouts>> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let mut stmt = conn.prepare_cached(
            "SELECT timestamp_utc, scenario, global_2m_temp_anomaly_c,
                    mauna_loa_co2_ppm, enso_nino34_c, global_sst_anomaly_c
             FROM global_readouts_history
             WHERE scenario = ?1
             ORDER BY id DESC
             LIMIT 1",
        )?;

        stmt.query_row(params![scenario.code()], |row| {
            Ok(GlobalLiveReadouts {
                timestamp_utc: row.get(0)?,
                scenario: row.get(1)?,
                global_2m_temp_anomaly_c: row.get(2)?,
                mauna_loa_co2_ppm: row.get(3)?,
                enso_nino34_c: row.get(4)?,
                global_sst_anomaly_c: row.get(5)?,
            })
        })
        .optional()
        .context("Failed querying global_readouts_history")
    }

    pub fn log_sync_event(
        &self,
        feed_name: &str,
        endpoint_url: &str,
        status: &str,
        records_ingested: usize,
        latency_ms: u64,
        notes: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO ingestion_sync_log (
                feed_name, endpoint_url, fetched_at_utc, status, records_ingested, latency_ms, notes
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                feed_name,
                endpoint_url,
                now,
                status,
                records_ingested as i64,
                latency_ms as i64,
                notes,
            ],
        )?;
        let max_logs = self.retention_policy().max_sync_log_rows;
        if max_logs > 0 {
            conn.execute(
                "DELETE FROM ingestion_sync_log
                 WHERE id NOT IN (
                     SELECT id FROM ingestion_sync_log ORDER BY id DESC LIMIT ?1
                 )",
                params![max_logs as i64],
            )?;
        }
        Ok(())
    }

    pub fn summary_stats(&self) -> Result<DbStatsSummary> {
        let conn = self.conn.lock().expect("SQLite mutex poisoned");
        let registered_layers: i64 =
            conn.query_row("SELECT COUNT(*) FROM layer_catalog", [], |r| r.get(0))?;
        let raw_observation_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM raw_telemetry_samples", [], |r| r.get(0))?;
        let simulated_grid_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM simulated_layer_grids", [], |r| r.get(0))?;
        let readout_snapshots: i64 =
            conn.query_row("SELECT COUNT(*) FROM global_readouts_history", [], |r| r.get(0))?;
        let sync_events: i64 =
            conn.query_row("SELECT COUNT(*) FROM ingestion_sync_log", [], |r| r.get(0))?;
        let latest_simulation_utc: Option<String> = conn
            .query_row(
                "SELECT computed_at_utc FROM simulated_layer_grids ORDER BY computed_at_utc DESC LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()?;

        Ok(DbStatsSummary {
            db_path: self.path.display().to_string(),
            registered_layers,
            raw_observation_count,
            simulated_grid_count,
            readout_snapshots,
            sync_events,
            latest_simulation_utc,
        })
    }

    /// Export a Zarr-compatible JSON/CSV spatial subset for external NetCDF/Zarr analysis pipelines.
    pub fn export_layer_subset_json(
        &self,
        layer_id: &str,
        scenario: SspScenario,
        vertical_level: f64,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        output_path: &Path,
    ) -> Result<usize> {
        let Some(grid) = self.load_latest_grid(layer_id, scenario, vertical_level)? else {
            anyhow::bail!(
                "No simulated grid stored yet for layer '{}' ({})",
                layer_id,
                scenario.code()
            );
        };

        let mut cells = Vec::new();
        for row in 0..grid.height {
            let lat = 90.0 - (row as f64 + 0.5) * (180.0 / grid.height as f64);
            if lat < min_lat || lat > max_lat {
                continue;
            }
            for col in 0..grid.width {
                let lon = -180.0 + (col as f64 + 0.5) * (360.0 / grid.width as f64);
                if lon < min_lon || lon > max_lon {
                    continue;
                }
                let idx = row * grid.width + col;
                let u = grid.u_component.as_ref().map(|v| v[idx]);
                let v = grid.v_component.as_ref().map(|v| v[idx]);
                cells.push(serde_json::json!({
                    "lat": (lat * 1000.0).round() / 1000.0,
                    "lon": (lon * 1000.0).round() / 1000.0,
                    "val": grid.values[idx],
                    "u": u,
                    "v": v,
                }));
            }
        }

        let export_doc = serde_json::json!({
            "zarr_format": 3,
            "layer_id": grid.layer_id,
            "scenario": grid.scenario,
            "timestamp_utc": grid.timestamp_utc,
            "vertical_level": grid.vertical_level,
            "bbox": [min_lon, min_lat, max_lon, max_lat],
            "grid_shape": [grid.height, grid.width],
            "cell_count": cells.len(),
            "cells": cells,
        });

        std::fs::write(output_path, serde_json::to_string_pretty(&export_doc)?)?;
        Ok(cells.len())
    }

    /// Export a standards-compliant chunked **Zarr v3** binary directory store using default `zstd(level=3)` compression.
    #[allow(clippy::too_many_arguments)]
    pub fn export_zarr_v3_store(
        &self,
        layer_id: &str,
        scenario: SspScenario,
        vertical_level: f64,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        chunk_shape: (usize, usize),
        store_dir: &Path,
    ) -> Result<ZarrV3ExportSummary> {
        self.export_zarr_v3_store_with_codec(
            layer_id,
            scenario,
            vertical_level,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            chunk_shape,
            ZarrCompressionCodec::default(),
            store_dir,
        )
    }

    /// Export a standards-compliant chunked **Zarr v3** binary directory store (`zarr.json` + `c/<row>/<col>`)
    /// with configurable `bytes -> bytes` compression codec (`zstd`, `gzip`, or `none`).
    #[allow(clippy::too_many_arguments)]
    pub fn export_zarr_v3_store_with_codec(
        &self,
        layer_id: &str,
        scenario: SspScenario,
        vertical_level: f64,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        chunk_shape: (usize, usize),
        codec: ZarrCompressionCodec,
        store_dir: &Path,
    ) -> Result<ZarrV3ExportSummary> {
        let Some(grid) = self.load_latest_grid(layer_id, scenario, vertical_level)? else {
            anyhow::bail!(
                "No simulated grid stored yet for layer '{}' ({})",
                layer_id,
                scenario.code()
            );
        };

        let mut rows = Vec::new();
        let mut lat_coords = Vec::new();
        for r in 0..grid.height {
            let lat = 90.0 - (r as f64 + 0.5) * (180.0 / grid.height as f64);
            if lat >= min_lat && lat <= max_lat {
                rows.push(r);
                lat_coords.push(lat as f32);
            }
        }

        let mut cols = Vec::new();
        let mut lon_coords = Vec::new();
        for c in 0..grid.width {
            let lon = -180.0 + (c as f64 + 0.5) * (360.0 / grid.width as f64);
            if lon >= min_lon && lon <= max_lon {
                cols.push(c);
                lon_coords.push(lon as f32);
            }
        }

        let sub_h = rows.len();
        let sub_w = cols.len();
        if sub_h == 0 || sub_w == 0 {
            anyhow::bail!(
                "Bounding box [{}, {}] x [{}, {}] matched 0 grid cells",
                min_lat,
                max_lat,
                min_lon,
                max_lon
            );
        }

        let ch = chunk_shape.0.max(1).min(sub_h);
        let cw = chunk_shape.1.max(1).min(sub_w);
        std::fs::create_dir_all(store_dir)?;

        let root_meta = serde_json::json!({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "title": "TerraTwin HPC Climate Modeling & Telemetry Hub",
                "layer_id": grid.layer_id,
                "scenario": grid.scenario,
                "timestamp_utc": grid.timestamp_utc,
                "vertical_level": grid.vertical_level,
                "codec": codec.label(),
                "bbox": [min_lon, min_lat, max_lon, max_lat],
            }
        });
        std::fs::write(
            store_dir.join("zarr.json"),
            serde_json::to_string_pretty(&root_meta)?,
        )?;

        let codecs_json = codec.zarr_v3_codecs_json();

        let write_1d_coord = |name: &str, coords: &[f32], units: &str| -> Result<()> {
            let dir = store_dir.join(name);
            let chunk_dir = dir.join("c");
            std::fs::create_dir_all(&chunk_dir)?;
            let meta = serde_json::json!({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [coords.len()],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": { "chunk_shape": [coords.len()] }
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": { "separator": "/" }
                },
                "fill_value": "NaN",
                "codecs": codecs_json,
                "dimension_names": [name],
                "attributes": { "units": units }
            });
            std::fs::write(dir.join("zarr.json"), serde_json::to_string_pretty(&meta)?)?;
            let encoded = encode_zarr_chunk_bytes(coords, codec)?;
            std::fs::write(chunk_dir.join("0"), encoded)?;
            Ok(())
        };

        write_1d_coord("latitude", &lat_coords, "degrees_north")?;
        write_1d_coord("longitude", &lon_coords, "degrees_east")?;

        let write_2d_array = |arr_name: &str, full_vals: &[f32]| -> Result<(usize, usize, usize)> {
            let arr_dir = store_dir.join(arr_name);
            std::fs::create_dir_all(&arr_dir)?;
            let meta = serde_json::json!({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [sub_h, sub_w],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": { "chunk_shape": [ch, cw] }
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": { "separator": "/" }
                },
                "fill_value": "NaN",
                "codecs": codecs_json,
                "dimension_names": ["latitude", "longitude"],
                "attributes": {
                    "layer_id": grid.layer_id,
                    "scenario": grid.scenario,
                    "timestamp_utc": grid.timestamp_utc,
                    "vertical_level": grid.vertical_level,
                    "min_val": grid.min_val,
                    "max_val": grid.max_val,
                    "mean_val": grid.mean_val,
                }
            });
            std::fs::write(
                arr_dir.join("zarr.json"),
                serde_json::to_string_pretty(&meta)?,
            )?;

            let n_row_chunks = (sub_h + ch - 1) / ch;
            let n_col_chunks = (sub_w + cw - 1) / cw;
            let mut written = 0usize;
            let mut raw_bytes = 0usize;
            let mut comp_bytes = 0usize;

            for rc in 0..n_row_chunks {
                let row_chunk_dir = arr_dir.join("c").join(rc.to_string());
                std::fs::create_dir_all(&row_chunk_dir)?;
                for cc in 0..n_col_chunks {
                    let mut buf = vec![f32::NAN; ch * cw];
                    for lr in 0..ch {
                        let sr = rc * ch + lr;
                        if sr >= sub_h {
                            break;
                        }
                        let src_row = rows[sr];
                        for lc in 0..cw {
                            let sc = cc * cw + lc;
                            if sc >= sub_w {
                                break;
                            }
                            let src_col = cols[sc];
                            buf[lr * cw + lc] = full_vals[src_row * grid.width + src_col];
                        }
                    }
                    raw_bytes += buf.len() * 4;
                    let encoded = encode_zarr_chunk_bytes(&buf, codec)?;
                    comp_bytes += encoded.len();
                    std::fs::write(row_chunk_dir.join(cc.to_string()), encoded)?;
                    written += 1;
                }
            }
            Ok((written, raw_bytes, comp_bytes))
        };

        let (mut chunks_written, mut uncompressed_bytes, mut compressed_bytes) =
            write_2d_array(&grid.layer_id, &grid.values)?;
        if let Some(ref u_vals) = grid.u_component {
            let (cw_u, rb_u, cb_u) = write_2d_array(&format!("{}_u", grid.layer_id), u_vals)?;
            chunks_written += cw_u;
            uncompressed_bytes += rb_u;
            compressed_bytes += cb_u;
        }
        if let Some(ref v_vals) = grid.v_component {
            let (cw_v, rb_v, cb_v) = write_2d_array(&format!("{}_v", grid.layer_id), v_vals)?;
            chunks_written += cw_v;
            uncompressed_bytes += rb_v;
            compressed_bytes += cb_v;
        }

        let compression_ratio = if compressed_bytes > 0 {
            uncompressed_bytes as f64 / compressed_bytes as f64
        } else {
            1.0
        };

        Ok(ZarrV3ExportSummary {
            store_dir: store_dir.display().to_string(),
            layer_id: grid.layer_id,
            codec: codec.label(),
            shape: [sub_h, sub_w],
            chunk_shape: [ch, cw],
            chunks_written,
            cells_exported: sub_h * sub_w,
            uncompressed_bytes,
            compressed_bytes,
            compression_ratio,
        })
    }
}

/// Encode a `f32` slice into a Zarr v3 chunk (`bytes` little-endian + optional `zstd` or `gzip` codec).
pub fn encode_zarr_chunk_bytes(values: &[f32], codec: ZarrCompressionCodec) -> Result<Vec<u8>> {
    let raw = pack_f32_le(values);
    match codec {
        ZarrCompressionCodec::None => Ok(raw),
        ZarrCompressionCodec::Zstd { level, checksum } => {
            let mut enc = zstd::Encoder::new(Vec::with_capacity(raw.len() / 2), level)?;
            enc.include_checksum(checksum)?;
            enc.write_all(&raw)?;
            Ok(enc.finish()?)
        }
        ZarrCompressionCodec::Gzip { level } => {
            let mut enc = GzEncoder::new(Vec::with_capacity(raw.len() / 2), Compression::new(level));
            enc.write_all(&raw)?;
            Ok(enc.finish()?)
        }
    }
}

/// Read and decode a 2D Zarr v3 chunk `c/<row_chunk>/<col_chunk>` by inspecting `<array_dir>/zarr.json`.
pub fn read_zarr_v3_chunk_f32(
    array_dir: &Path,
    row_chunk: usize,
    col_chunk: usize,
) -> Result<Vec<f32>> {
    let meta_str = std::fs::read_to_string(array_dir.join("zarr.json"))
        .with_context(|| format!("Missing zarr.json in {}", array_dir.display()))?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
    let chunk_path = array_dir
        .join("c")
        .join(row_chunk.to_string())
        .join(col_chunk.to_string());
    let raw_file = std::fs::read(&chunk_path)
        .with_context(|| format!("Missing Zarr v3 chunk {}", chunk_path.display()))?;

    let mut has_zstd = false;
    let mut has_gzip = false;
    if let Some(codecs) = meta.get("codecs").and_then(|c| c.as_array()) {
        for c in codecs {
            if let Some(name) = c.get("name").and_then(|n| n.as_str()) {
                if name == "zstd" {
                    has_zstd = true;
                } else if name == "gzip" {
                    has_gzip = true;
                }
            }
        }
    }

    let decoded = if has_zstd {
        zstd::stream::decode_all(raw_file.as_slice())?
    } else if has_gzip {
        let mut dec = GzDecoder::new(raw_file.as_slice());
        let mut out = Vec::new();
        dec.read_to_end(&mut out)?;
        out
    } else {
        raw_file
    };

    Ok(unpack_f32_le(&decoded))
}

/// Magic header for lossless byte-shuffled LZ4 SQLite raster blobs (`"TTZ1"` + codec `1u8`).
const SQLITE_COMPRESSED_MAGIC: &[u8; 4] = b"TTZ1";
const SQLITE_CODEC_SHUFFLE_LZ4: u8 = 1;

/// Group byte lanes `0, 1, 2, 3` across `values` so IEEE-754 exponents & sign bits compress tightly.
pub fn byte_shuffle_f32(values: &[f32]) -> Vec<u8> {
    let n = values.len();
    let mut out = vec![0u8; n * 4];
    for (i, &v) in values.iter().enumerate() {
        let b = v.to_le_bytes();
        out[i] = b[0];
        out[n + i] = b[1];
        out[2 * n + i] = b[2];
        out[3 * n + i] = b[3];
    }
    out
}

/// Inverse of `byte_shuffle_f32`, reconstructing exact IEEE-754 `f32` values (including `NaN`s).
pub fn byte_unshuffle_f32(shuffled: &[u8], count: usize) -> Vec<f32> {
    assert_eq!(shuffled.len(), count * 4);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(f32::from_le_bytes([
            shuffled[i],
            shuffled[count + i],
            shuffled[2 * count + i],
            shuffled[3 * count + i],
        ]));
    }
    out
}

/// Compress `values` losslessly using IEEE-754 byte-shuffling + pure-Rust `lz4_flex`.
pub fn pack_f32_compressed(values: &[f32]) -> Vec<u8> {
    let count = values.len() as u32;
    let shuffled = byte_shuffle_f32(values);
    let lz4_bytes = lz4_flex::block::compress_prepend_size(&shuffled);
    let mut out = Vec::with_capacity(9 + lz4_bytes.len());
    out.extend_from_slice(SQLITE_COMPRESSED_MAGIC);
    out.push(SQLITE_CODEC_SHUFFLE_LZ4);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&lz4_bytes);
    out
}

/// Transparently decode either a `"TTZ1"` byte-shuffled LZ4 blob OR a legacy uncompressed `f32` blob.
pub fn unpack_f32_auto(bytes: &[u8]) -> Vec<f32> {
    if bytes.len() >= 9
        && &bytes[0..4] == SQLITE_COMPRESSED_MAGIC
        && bytes[4] == SQLITE_CODEC_SHUFFLE_LZ4
    {
        let count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
        if let Ok(shuffled) = lz4_flex::block::decompress_size_prepended(&bytes[9..]) {
            if shuffled.len() == count * 4 {
                return byte_unshuffle_f32(&shuffled, count);
            }
        }
    }
    unpack_f32_le(bytes)
}

pub fn pack_f32_le(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub fn unpack_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
