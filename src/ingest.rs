// Copyright (C) 2026 CRV Science Network (https://crvscience.com)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://gnu.org>.

//! Asynchronous background telemetry ingestion engine (`reqwest` + `tokio`).
//! Downloads open public climate, oceanographic, hydrological, and wildfire telemetry from:
//! 1. **NOAA Global Monitoring Laboratory (GML) Mauna Loa CO2**:
//!    `https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv`
//! 2. **NOAA CPC / PSL Teleconnection Indices (ENSO Niño 3.4 & NAO)**:
//!    - `https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data`
//!    - `https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii`
//!    - `https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table`
//! 3. **NOAA CoastWatch ERDDAP OISST v2.1 Global Gridded SST (`Lon+/-180`)**:
//!    `https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv`
//! 4. **NASA FIRMS VIIRS Active Fires & Fire Radiative Power (FRP)**:
//!    - Authenticated Area API (`FIRMS_MAP_KEY` env var):
//!      `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/world/1`
//!    - Open 24h Global NOAA-21 VIIRS C2 CSV fallback (no API key required):
//!      `https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_24h.csv`
//! 5. **Open-Meteo Global Forecast / ERA5 / CAMS Air Quality / Copernicus Marine / GloFAS Flood**:
//!    - `https://api.open-meteo.com/v1/forecast`
//!    - `https://air-quality-api.open-meteo.com/v1/air-quality`
//!    - `https://marine-api.open-meteo.com/v1/marine`
//!    - `https://flood-api.open-meteo.com/v1/flood` (Copernicus GloFAS v4/v5 river discharge)
//!
//! Includes deterministic physical climatology seeding if run on an air-gapped node
//! or if all upstream endpoints are unreachable.

use crate::db::{ClimateDatabase, RawTelemetrySample};
use crate::interp::fibonacci_sphere_points;
use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, warn};

pub const NOAA_MLO_CO2_URL: &str =
    "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.csv";
pub const NOAA_PSL_NINO34_URL: &str =
    "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data";
pub const NOAA_CPC_NINO34_FALLBACK_URL: &str =
    "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii";
pub const NOAA_CPC_NAO_TABLE_URL: &str =
    "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table";
pub const COASTWATCH_ERDDAP_OISST_CSV_URL: &str =
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv?sst%5B(last)%5D%5B(0.0)%5D%5B(-78.125):24:(78.125)%5D%5B(-179.875):24:(179.875)%5D";
pub const NASA_FIRMS_OPEN_GLOBAL_24H_CSV_URL: &str =
    "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_24h.csv";
pub const NASA_FIRMS_AREA_API_BASE: &str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv";

pub const OPEN_METEO_FORECAST_URL: &str = "https://api.open-meteo.com/v1/forecast";
pub const OPEN_METEO_AIR_QUALITY_URL: &str =
    "https://air-quality-api.open-meteo.com/v1/air-quality";
pub const OPEN_METEO_MARINE_URL: &str = "https://marine-api.open-meteo.com/v1/marine";
pub const OPEN_METEO_FLOOD_URL: &str = "https://flood-api.open-meteo.com/v1/flood";

/// Major global river gauging corridors queried via Open-Meteo Copernicus GloFAS `(lat, lon)`.
pub const MAJOR_RIVER_GAUGE_NODES: &[(f64, f64)] = &[
    (45.5017, -73.5673),  // St. Lawrence River Basin
    (-1.9470, -55.5110),  // Amazon River at Óbidos
    (-4.3220, 15.3080),   // Congo River at Kinshasa
    (30.7680, 117.6250),  // Yangtze River at Datong
    (32.3526, -90.8779),  // Mississippi River at Vicksburg
    (23.8900, 90.0200),   // Ganges-Brahmaputra Confluence
    (11.5620, 104.9310),  // Mekong River at Phnom Penh
    (45.2200, 28.7300),   // Danube Delta
    (30.1800, 31.1400),   // Nile River at Cairo
    (-32.9500, -60.6400), // Paraná River at Rosario
    (66.5300, 66.6000),   // Ob River at Salekhard
    (67.4500, -133.7400), // Mackenzie River at Arctic Red
];

#[derive(Debug, Clone)]
pub struct IngestionConfig {
    /// Number of Fibonacci global geodesic stations queried per sync cycle.
    pub geodesic_sample_nodes: usize,
    /// Maximum aggregated active wildfire complexes stored per sync cycle.
    pub max_firms_hotspots: usize,
    /// Optional NASA FIRMS `MAP_KEY` (automatically read from `FIRMS_MAP_KEY` env var if `None`).
    pub firms_map_key: Option<String>,
    /// HTTP timeout per upstream request.
    pub request_timeout: Duration,
    /// Allow deterministic baseline climatology seeding when offline.
    pub enable_offline_fallback: bool,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            geodesic_sample_nodes: 36,
            max_firms_hotspots: 500,
            firms_map_key: std::env::var("FIRMS_MAP_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty()),
            request_timeout: Duration::from_secs(14),
            enable_offline_fallback: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct IngestionReport {
    pub live_records: usize,
    pub fallback_records: usize,
    pub mauna_loa_co2_ppm: Option<f64>,
    pub enso_nino34_c: Option<f64>,
    pub nao_index: Option<f64>,
    pub firms_fire_clusters: usize,
    pub erddap_oisst_cells: usize,
    pub glofas_river_nodes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OpenMeteoBatch<T> {
    Many(Vec<T>),
    Single(T),
}

impl<T> OpenMeteoBatch<T> {
    fn into_vec(self) -> Vec<T> {
        match self {
            OpenMeteoBatch::Many(v) => v,
            OpenMeteoBatch::Single(item) => vec![item],
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenMeteoMultiResponse {
    latitude: f64,
    longitude: f64,
    current: Option<OpenMeteoCurrentBlock>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoCurrentBlock {
    time: Option<String>,
    temperature_2m: Option<f64>,
    relative_humidity_2m: Option<f64>,
    pressure_msl: Option<f64>,
    precipitation: Option<f64>,
    wind_speed_10m: Option<f64>,
    wind_direction_10m: Option<f64>,
    cape: Option<f64>,
    soil_moisture_0_to_7cm: Option<f64>,
    shortwave_radiation: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoMarineItem {
    latitude: f64,
    longitude: f64,
    current: Option<OpenMeteoMarineCurrent>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoMarineCurrent {
    time: Option<String>,
    sea_surface_temperature: Option<f64>,
    ocean_current_velocity: Option<f64>,
    ocean_current_direction: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoAirItem {
    latitude: f64,
    longitude: f64,
    current: Option<OpenMeteoAirCurrent>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoAirCurrent {
    time: Option<String>,
    pm2_5: Option<f64>,
    aerosol_optical_depth: Option<f64>,
    #[allow(dead_code)]
    dust: Option<f64>,
    nitrogen_dioxide: Option<f64>,
    sulphur_dioxide: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoFloodItem {
    latitude: f64,
    longitude: f64,
    daily: Option<OpenMeteoFloodDaily>,
}

#[derive(Debug, Deserialize)]
struct OpenMeteoFloodDaily {
    time: Option<Vec<String>>,
    river_discharge: Option<Vec<Option<f64>>>,
}

pub struct TelemetryIngestor {
    client: reqwest::Client,
    config: IngestionConfig,
}

impl TelemetryIngestor {
    pub fn new(config: IngestionConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent("TerraTwin-Climate-Hub/2.4.0 (Rust Native Telemetry Assimilator)")
            .timeout(config.request_timeout)
            .build()?;
        Ok(Self { client, config })
    }

    /// Execute a full multi-source telemetry download cycle across all 8 public feeds and persist to SQLite.
    pub async fn run_sync_cycle(&self, db: &ClimateDatabase) -> Result<IngestionReport> {
        let mut report = IngestionReport::default();
        let now_utc = chrono::Utc::now().to_rfc3339();

        let mut coords = fibonacci_sphere_points(self.config.geodesic_sample_nodes);
        coords.push((40.7128, -74.0061)); // New York City continental anchor
        coords.push((19.5362, -155.5763)); // Mauna Loa Observatory anchor

        // 1. NOAA GML Mauna Loa CO2 CSV
        let t0 = Instant::now();
        match self.fetch_mauna_loa_co2().await {
            Ok(co2_ppm) => {
                report.mauna_loa_co2_ppm = Some(co2_ppm);
                let sample = RawTelemetrySample {
                    layer_id: "column_co2_flux".to_string(),
                    source_name: "NOAA GML Mauna Loa Observatory".to_string(),
                    source_url: NOAA_MLO_CO2_URL.to_string(),
                    timestamp_utc: now_utc.clone(),
                    latitude: 19.5362,
                    longitude: -155.5763,
                    vertical_level: 1000.0,
                    primary_value: co2_ppm,
                    secondary_value: None,
                    quality_flag: 1,
                };
                db.insert_raw_samples(&[sample])?;
                report.live_records += 1;
                db.log_sync_event(
                    "NOAA_GML_MLO_CO2",
                    NOAA_MLO_CO2_URL,
                    "LIVE_OK",
                    1,
                    t0.elapsed().as_millis() as u64,
                    Some(&format!("Latest MLO CO2 = {:.2} ppm", co2_ppm)),
                )?;
            }
            Err(err) => {
                warn!("NOAA GML CO2 fetch unavailable ({}), using baseline", err);
                db.log_sync_event(
                    "NOAA_GML_MLO_CO2",
                    NOAA_MLO_CO2_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t0.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
        }

        // 2. NOAA CPC / PSL ENSO Niño 3.4 & North Atlantic Oscillation (NAO) Teleconnection Indices
        let t_osc = Instant::now();
        let enso_res = self.fetch_noaa_enso_nino34().await;
        let nao_res = self.fetch_noaa_cpc_nao().await;
        if enso_res.is_ok() || nao_res.is_ok() {
            let enso_val = enso_res.ok();
            let nao_val = nao_res.ok();
            report.enso_nino34_c = enso_val;
            report.nao_index = nao_val;

            let mut osc_samples = Vec::new();
            if let Some(nino34) = enso_val {
                osc_samples.push(RawTelemetrySample {
                    layer_id: "teleconnection_oscillation_masks".to_string(),
                    source_name: "NOAA CPC / PSL ENSO Nino 3.4".to_string(),
                    source_url: NOAA_PSL_NINO34_URL.to_string(),
                    timestamp_utc: now_utc.clone(),
                    latitude: 0.0,
                    longitude: -145.0, // Center of Niño 3.4 box (5S-5N, 170W-120W)
                    vertical_level: 0.0,
                    primary_value: nino34,
                    secondary_value: nao_val,
                    quality_flag: 1,
                });
            }
            if let Some(nao) = nao_val {
                osc_samples.push(RawTelemetrySample {
                    layer_id: "teleconnection_oscillation_masks".to_string(),
                    source_name: "NOAA CPC Monthly NAO Index".to_string(),
                    source_url: NOAA_CPC_NAO_TABLE_URL.to_string(),
                    timestamp_utc: now_utc.clone(),
                    latitude: 50.0,
                    longitude: -30.0, // Center of NAO North Atlantic dipole
                    vertical_level: 0.0,
                    primary_value: nao,
                    secondary_value: enso_val,
                    quality_flag: 1,
                });
            }
            let n = db.insert_raw_samples(&osc_samples)?;
            report.live_records += n;
            db.log_sync_event(
                "NOAA_CPC_ENSO_NAO",
                NOAA_PSL_NINO34_URL,
                "LIVE_OK",
                n,
                t_osc.elapsed().as_millis() as u64,
                Some(&format!("ENSO={:?} C, NAO={:?}", enso_val, nao_val)),
            )?;
        } else {
            db.log_sync_event(
                "NOAA_CPC_ENSO_NAO",
                NOAA_PSL_NINO34_URL,
                "OFFLINE_FALLBACK",
                0,
                t_osc.elapsed().as_millis() as u64,
                Some("NOAA CPC/PSL teleconnection ASCII feeds unreachable"),
            )?;
        }

        // 3. NOAA CoastWatch ERDDAP OISST v2.1 Gridded Global SST
        let t_oisst = Instant::now();
        match self.fetch_coastwatch_erddap_oisst(&now_utc).await {
            Ok(samples) if !samples.is_empty() => {
                let n = db.insert_raw_samples(&samples)?;
                report.live_records += n;
                report.erddap_oisst_cells = n;
                db.log_sync_event(
                    "NOAA_COASTWATCH_ERDDAP_OISST_V21",
                    COASTWATCH_ERDDAP_OISST_CSV_URL,
                    "LIVE_OK",
                    n,
                    t_oisst.elapsed().as_millis() as u64,
                    Some("Assimilated NOAA NCEI OISST v2.1 global griddap SST slice"),
                )?;
            }
            Err(err) => {
                warn!("CoastWatch ERDDAP OISST v2.1 fetch unavailable: {}", err);
                db.log_sync_event(
                    "NOAA_COASTWATCH_ERDDAP_OISST_V21",
                    COASTWATCH_ERDDAP_OISST_CSV_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t_oisst.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
            _ => {}
        }

        // 4. NASA FIRMS Active Wildfires & Fire Radiative Power (FRP)
        let t_firms = Instant::now();
        match self.fetch_nasa_firms_active_fires(&now_utc).await {
            Ok((samples, used_url)) if !samples.is_empty() => {
                let n = db.insert_raw_samples(&samples)?;
                report.live_records += n;
                report.firms_fire_clusters = n;
                db.log_sync_event(
                    "NASA_FIRMS_VIIRS_ACTIVE_FIRES",
                    &used_url,
                    "LIVE_OK",
                    n,
                    t_firms.elapsed().as_millis() as u64,
                    Some("Assimilated VIIRS active fire hotspots & peak FRP (MW)"),
                )?;
            }
            Err(err) => {
                warn!("NASA FIRMS active fire feed unavailable: {}", err);
                db.log_sync_event(
                    "NASA_FIRMS_VIIRS_ACTIVE_FIRES",
                    NASA_FIRMS_OPEN_GLOBAL_24H_CSV_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t_firms.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
            _ => {}
        }

        // 5. Open-Meteo Copernicus GloFAS River Discharge
        let t_flood = Instant::now();
        let mut river_coords: Vec<(f64, f64)> = MAJOR_RIVER_GAUGE_NODES.to_vec();
        for &(la, lo) in &coords {
            if is_approx_continental_interior(la, lo) && river_coords.len() < 28 {
                river_coords.push((la, lo));
            }
        }
        match self.fetch_open_meteo_glofas_flood(&river_coords, &now_utc).await {
            Ok(samples) if !samples.is_empty() => {
                let n = db.insert_raw_samples(&samples)?;
                report.live_records += n;
                report.glofas_river_nodes = n;
                db.log_sync_event(
                    "OPEN_METEO_GLOFAS_FLOOD",
                    OPEN_METEO_FLOOD_URL,
                    "LIVE_OK",
                    n,
                    t_flood.elapsed().as_millis() as u64,
                    Some("Assimilated Copernicus GloFAS daily river discharge (m3/s)"),
                )?;
            }
            Err(err) => {
                warn!("Open-Meteo GloFAS Flood API unavailable: {}", err);
                db.log_sync_event(
                    "OPEN_METEO_GLOFAS_FLOOD",
                    OPEN_METEO_FLOOD_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t_flood.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
            _ => {}
        }

        // 6. Open-Meteo Global Meteorology & Surface Hydrology Batch
        let t_met = Instant::now();
        match self.fetch_open_meteo_atmosphere(&coords, &now_utc).await {
            Ok(samples) if !samples.is_empty() => {
                let n = db.insert_raw_samples(&samples)?;
                report.live_records += n;
                db.log_sync_event(
                    "OPEN_METEO_GLOBAL_MET",
                    OPEN_METEO_FORECAST_URL,
                    "LIVE_OK",
                    n,
                    t_met.elapsed().as_millis() as u64,
                    Some("Assimilated 2m Temp, RH, MSLP, 10m Wind, CAPE, Soil Moisture, Shortwave"),
                )?;
            }
            Err(err) => {
                warn!("Open-Meteo Atmosphere fetch unavailable: {}", err);
                db.log_sync_event(
                    "OPEN_METEO_GLOBAL_MET",
                    OPEN_METEO_FORECAST_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t_met.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
            _ => {}
        }

        // 7. Open-Meteo Copernicus Marine SST & Surface Currents Batch
        let t_mar = Instant::now();
        let ocean_coords: Vec<(f64, f64)> = coords
            .iter()
            .copied()
            .filter(|&(lat, lon)| !is_approx_continental_interior(lat, lon))
            .take(24)
            .collect();
        match self.fetch_open_meteo_marine(&ocean_coords, &now_utc).await {
            Ok(samples) if !samples.is_empty() => {
                let n = db.insert_raw_samples(&samples)?;
                report.live_records += n;
                db.log_sync_event(
                    "OPEN_METEO_MARINE_CMEMS",
                    OPEN_METEO_MARINE_URL,
                    "LIVE_OK",
                    n,
                    t_mar.elapsed().as_millis() as u64,
                    Some("Assimilated marine SST and surface current velocity vectors"),
                )?;
            }
            Err(err) => {
                db.log_sync_event(
                    "OPEN_METEO_MARINE_CMEMS",
                    OPEN_METEO_MARINE_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t_mar.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
            _ => {}
        }

        // 8. Open-Meteo CAMS Speciated Aerosols & Trace Gases Batch
        let t_air = Instant::now();
        let air_subset: Vec<(f64, f64)> = coords.iter().copied().take(24).collect();
        match self.fetch_open_meteo_air_quality(&air_subset, &now_utc).await {
            Ok(samples) if !samples.is_empty() => {
                let n = db.insert_raw_samples(&samples)?;
                report.live_records += n;
                db.log_sync_event(
                    "OPEN_METEO_CAMS_AEROSOLS",
                    OPEN_METEO_AIR_QUALITY_URL,
                    "LIVE_OK",
                    n,
                    t_air.elapsed().as_millis() as u64,
                    Some("Assimilated AOD 550nm, PM2.5, NO2, SO2"),
                )?;
            }
            Err(err) => {
                db.log_sync_event(
                    "OPEN_METEO_CAMS_AEROSOLS",
                    OPEN_METEO_AIR_QUALITY_URL,
                    "OFFLINE_FALLBACK",
                    0,
                    t_air.elapsed().as_millis() as u64,
                    Some(&err.to_string()),
                )?;
            }
            _ => {}
        }

        // 9. Air-gapped deterministic physical climatology fallback if completely offline
        if report.live_records == 0 && self.config.enable_offline_fallback {
            let seeded = seed_climatological_boundary_samples(&coords, &now_utc);
            let n = db.insert_raw_samples(&seeded)?;
            report.fallback_records = n;
            info!(
                "Seeded {} baseline climatological boundary observations into SQLite (offline mode)",
                n
            );
        }

        Ok(report)
    }

    async fn fetch_mauna_loa_co2(&self) -> Result<f64> {
        let text = self
            .client
            .get(NOAA_MLO_CO2_URL)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;

        for line in text.lines().rev() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("year") {
                continue;
            }
            let cols: Vec<&str> = trimmed.split(',').collect();
            if cols.len() >= 5 {
                if let Ok(ppm) = cols[4].trim().parse::<f64>() {
                    if ppm > 300.0 && ppm < 600.0 {
                        return Ok(ppm);
                    }
                }
            }
        }
        anyhow::bail!("No valid CO2 ppm measurement found in NOAA GML CSV")
    }

    async fn fetch_noaa_enso_nino34(&self) -> Result<f64> {
        if let Ok(resp) = self.client.get(NOAA_PSL_NINO34_URL).send().await {
            if let Ok(ok_resp) = resp.error_for_status() {
                if let Ok(text) = ok_resp.text().await {
                    if let Ok(val) = parse_noaa_enso_nino34(&text) {
                        return Ok(val);
                    }
                }
            }
        }
        let fallback_txt = self
            .client
            .get(NOAA_CPC_NINO34_FALLBACK_URL)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        parse_noaa_enso_nino34(&fallback_txt)
    }

    async fn fetch_noaa_cpc_nao(&self) -> Result<f64> {
        let text = self
            .client
            .get(NOAA_CPC_NAO_TABLE_URL)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        parse_noaa_cpc_nao(&text)
    }

    async fn fetch_coastwatch_erddap_oisst(
        &self,
        fallback_time: &str,
    ) -> Result<Vec<RawTelemetrySample>> {
        let csv_text = self
            .client
            .get(COASTWATCH_ERDDAP_OISST_CSV_URL)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        parse_erddap_oisst_csv(&csv_text, fallback_time)
    }

    async fn fetch_nasa_firms_active_fires(
        &self,
        fallback_time: &str,
    ) -> Result<(Vec<RawTelemetrySample>, String)> {
        if let Some(ref map_key) = self.config.firms_map_key {
            let auth_url = format!(
                "{}/{}/VIIRS_SNPP_NRT/world/1",
                NASA_FIRMS_AREA_API_BASE,
                map_key.trim()
            );
            let redact_url = format!("{}/<FIRMS_MAP_KEY>/VIIRS_SNPP_NRT/world/1", NASA_FIRMS_AREA_API_BASE);
            if let Ok(resp) = self.client.get(&auth_url).send().await {
                if let Ok(ok_resp) = resp.error_for_status() {
                    if let Ok(csv_text) = ok_resp.text().await {
                        let parsed = parse_firms_csv_hotspots(
                            &csv_text,
                            &redact_url,
                            fallback_time,
                            self.config.max_firms_hotspots,
                        )?;
                        if !parsed.is_empty() {
                            return Ok((parsed, redact_url));
                        }
                    }
                }
            }
        }

        // Open 24h global NOAA-21 VIIRS C2 active fire CSV (no API key required)
        let csv_text = self
            .client
            .get(NASA_FIRMS_OPEN_GLOBAL_24H_CSV_URL)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        let parsed = parse_firms_csv_hotspots(
            &csv_text,
            NASA_FIRMS_OPEN_GLOBAL_24H_CSV_URL,
            fallback_time,
            self.config.max_firms_hotspots,
        )?;
        Ok((parsed, NASA_FIRMS_OPEN_GLOBAL_24H_CSV_URL.to_string()))
    }

    async fn fetch_open_meteo_glofas_flood(
        &self,
        coords: &[(f64, f64)],
        fallback_time: &str,
    ) -> Result<Vec<RawTelemetrySample>> {
        if coords.is_empty() {
            return Ok(Vec::new());
        }
        let lats = coords
            .iter()
            .map(|(la, _)| format!("{:.4}", la))
            .collect::<Vec<_>>()
            .join(",");
        let lons = coords
            .iter()
            .map(|(_, lo)| format!("{:.4}", lo))
            .collect::<Vec<_>>()
            .join(",");

        let payload: OpenMeteoBatch<OpenMeteoFloodItem> = self
            .client
            .get(OPEN_METEO_FLOOD_URL)
            .query(&[
                ("latitude", lats.as_str()),
                ("longitude", lons.as_str()),
                ("daily", "river_discharge"),
                ("past_days", "1"),
                ("forecast_days", "1"),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let mut out = Vec::new();
        for item in payload.into_vec() {
            let Some(daily) = item.daily else { continue };
            let Some(discharges) = daily.river_discharge else {
                continue;
            };
            let times = daily.time.unwrap_or_default();

            // Select most recent non-null daily river discharge (m^3 s^-1)
            let mut latest_q: Option<(f64, String)> = None;
            for (idx, q_opt) in discharges.into_iter().enumerate().rev() {
                if let Some(q) = q_opt {
                    if q.is_finite() && q >= 0.0 {
                        let ts = times
                            .get(idx)
                            .cloned()
                            .unwrap_or_else(|| fallback_time.to_string());
                        latest_q = Some((q, ts));
                        break;
                    }
                }
            }

            if let Some((discharge_m3s, ts)) = latest_q {
                out.push(RawTelemetrySample {
                    layer_id: "swot_surface_water_river_discharge".to_string(),
                    source_name: "Open-Meteo / Copernicus GloFAS".to_string(),
                    source_url: OPEN_METEO_FLOOD_URL.to_string(),
                    timestamp_utc: ts,
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 0.0,
                    primary_value: discharge_m3s,
                    secondary_value: None,
                    quality_flag: 1,
                });
            }
        }
        Ok(out)
    }

    async fn fetch_open_meteo_atmosphere(
        &self,
        coords: &[(f64, f64)],
        fallback_time: &str,
    ) -> Result<Vec<RawTelemetrySample>> {
        let lats = coords
            .iter()
            .map(|(la, _)| format!("{:.3}", la))
            .collect::<Vec<_>>()
            .join(",");
        let lons = coords
            .iter()
            .map(|(_, lo)| format!("{:.3}", lo))
            .collect::<Vec<_>>()
            .join(",");

        let batch: OpenMeteoBatch<OpenMeteoMultiResponse> = self
            .client
            .get(OPEN_METEO_FORECAST_URL)
            .query(&[
                ("latitude", lats.as_str()),
                ("longitude", lons.as_str()),
                (
                    "current",
                    "temperature_2m,relative_humidity_2m,pressure_msl,precipitation,wind_speed_10m,wind_direction_10m,cape,soil_moisture_0_to_7cm,shortwave_radiation",
                ),
                ("wind_speed_unit", "ms"),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let mut out = Vec::new();
        for item in batch.into_vec() {
            let Some(cur) = item.current else { continue };
            let ts = cur.time.unwrap_or_else(|| fallback_time.to_string());

            if let Some(mslp) = cur.pressure_msl {
                out.push(RawTelemetrySample {
                    layer_id: "synoptic_fronts_isobars".to_string(),
                    source_name: "Open-Meteo GFS/IFS".to_string(),
                    source_url: OPEN_METEO_FORECAST_URL.to_string(),
                    timestamp_utc: ts.clone(),
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 1000.0,
                    primary_value: mslp,
                    secondary_value: cur.temperature_2m,
                    quality_flag: 1,
                });
            }

            if let (Some(t2m), Some(rh)) = (cur.temperature_2m, cur.relative_humidity_2m) {
                out.push(RawTelemetrySample {
                    layer_id: "wet_bulb_globe_temp_heat_stress".to_string(),
                    source_name: "Open-Meteo ERA5/IFS".to_string(),
                    source_url: OPEN_METEO_FORECAST_URL.to_string(),
                    timestamp_utc: ts.clone(),
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 1000.0,
                    primary_value: t2m,
                    secondary_value: Some(rh),
                    quality_flag: 1,
                });
            }

            if let (Some(spd), Some(dir)) = (cur.wind_speed_10m, cur.wind_direction_10m) {
                let rad = dir.to_radians();
                let u = -spd * rad.sin();
                let v = -spd * rad.cos();
                out.push(RawTelemetrySample {
                    layer_id: "wind_vector_jet_streams".to_string(),
                    source_name: "Open-Meteo GFS/IFS".to_string(),
                    source_url: OPEN_METEO_FORECAST_URL.to_string(),
                    timestamp_utc: ts.clone(),
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 1000.0,
                    primary_value: u,
                    secondary_value: Some(v),
                    quality_flag: 1,
                });
            }

            if let Some(precip) = cur.precipitation {
                let dbz = if precip > 0.02 {
                    10.0 * (200.0 * precip.powf(1.6)).log10()
                } else {
                    0.0
                };
                out.push(RawTelemetrySample {
                    layer_id: "volumetric_doppler_radar".to_string(),
                    source_name: "Open-Meteo / GPM Proxy".to_string(),
                    source_url: OPEN_METEO_FORECAST_URL.to_string(),
                    timestamp_utc: ts.clone(),
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 1000.0,
                    primary_value: dbz.clamp(0.0, 70.0),
                    secondary_value: cur.cape,
                    quality_flag: 1,
                });
            }

            if let Some(sm) = cur.soil_moisture_0_to_7cm {
                out.push(RawTelemetrySample {
                    layer_id: "soil_moisture_groundwater_grace".to_string(),
                    source_name: "Open-Meteo Land/SMAP Assimilated".to_string(),
                    source_url: OPEN_METEO_FORECAST_URL.to_string(),
                    timestamp_utc: ts,
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 0.0,
                    primary_value: sm,
                    secondary_value: cur.shortwave_radiation,
                    quality_flag: 1,
                });
            }
        }
        Ok(out)
    }

    async fn fetch_open_meteo_marine(
        &self,
        coords: &[(f64, f64)],
        fallback_time: &str,
    ) -> Result<Vec<RawTelemetrySample>> {
        if coords.is_empty() {
            return Ok(Vec::new());
        }
        let lats = coords
            .iter()
            .map(|(la, _)| format!("{:.3}", la))
            .collect::<Vec<_>>()
            .join(",");
        let lons = coords
            .iter()
            .map(|(_, lo)| format!("{:.3}", lo))
            .collect::<Vec<_>>()
            .join(",");

        let batch: OpenMeteoBatch<OpenMeteoMarineItem> = self
            .client
            .get(OPEN_METEO_MARINE_URL)
            .query(&[
                ("latitude", lats.as_str()),
                ("longitude", lons.as_str()),
                (
                    "current",
                    "sea_surface_temperature,ocean_current_velocity,ocean_current_direction",
                ),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let mut out = Vec::new();
        for item in batch.into_vec() {
            let Some(cur) = item.current else { continue };
            let ts = cur.time.unwrap_or_else(|| fallback_time.to_string());
            if let Some(sst) = cur.sea_surface_temperature {
                out.push(RawTelemetrySample {
                    layer_id: "sst_marine_heatwaves".to_string(),
                    source_name: "Open-Meteo Marine / Copernicus".to_string(),
                    source_url: OPEN_METEO_MARINE_URL.to_string(),
                    timestamp_utc: ts.clone(),
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 0.0,
                    primary_value: sst,
                    secondary_value: None,
                    quality_flag: 1,
                });
            }
            if let (Some(vel), Some(dir)) =
                (cur.ocean_current_velocity, cur.ocean_current_direction)
            {
                let rad = dir.to_radians();
                out.push(RawTelemetrySample {
                    layer_id: "thermohaline_currents_amoc".to_string(),
                    source_name: "Open-Meteo CMEMS Ocean Currents".to_string(),
                    source_url: OPEN_METEO_MARINE_URL.to_string(),
                    timestamp_utc: ts,
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 0.0,
                    primary_value: vel * rad.sin(),
                    secondary_value: Some(vel * rad.cos()),
                    quality_flag: 1,
                });
            }
        }
        Ok(out)
    }

    async fn fetch_open_meteo_air_quality(
        &self,
        coords: &[(f64, f64)],
        fallback_time: &str,
    ) -> Result<Vec<RawTelemetrySample>> {
        if coords.is_empty() {
            return Ok(Vec::new());
        }
        let lats = coords
            .iter()
            .map(|(la, _)| format!("{:.3}", la))
            .collect::<Vec<_>>()
            .join(",");
        let lons = coords
            .iter()
            .map(|(_, lo)| format!("{:.3}", lo))
            .collect::<Vec<_>>()
            .join(",");

        let batch: OpenMeteoBatch<OpenMeteoAirItem> = self
            .client
            .get(OPEN_METEO_AIR_QUALITY_URL)
            .query(&[
                ("latitude", lats.as_str()),
                ("longitude", lons.as_str()),
                (
                    "current",
                    "pm2_5,aerosol_optical_depth,dust,nitrogen_dioxide,sulphur_dioxide",
                ),
            ])
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let mut out = Vec::new();
        for item in batch.into_vec() {
            let Some(cur) = item.current else { continue };
            let ts = cur.time.unwrap_or_else(|| fallback_time.to_string());
            if let Some(aod) = cur.aerosol_optical_depth {
                out.push(RawTelemetrySample {
                    layer_id: "aerosol_optical_depth_speciated".to_string(),
                    source_name: "Copernicus CAMS via Open-Meteo".to_string(),
                    source_url: OPEN_METEO_AIR_QUALITY_URL.to_string(),
                    timestamp_utc: ts.clone(),
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 1000.0,
                    primary_value: cur.pm2_5.unwrap_or(aod * 35.0),
                    secondary_value: Some(aod),
                    quality_flag: 1,
                });
            }
            if let Some(no2) = cur.nitrogen_dioxide {
                out.push(RawTelemetrySample {
                    layer_id: "tropospheric_no2_so2".to_string(),
                    source_name: "Copernicus CAMS / TROPOMI Assimilated".to_string(),
                    source_url: OPEN_METEO_AIR_QUALITY_URL.to_string(),
                    timestamp_utc: ts,
                    latitude: item.latitude,
                    longitude: item.longitude,
                    vertical_level: 1000.0,
                    primary_value: no2 * 1.0e14,
                    secondary_value: cur.sulphur_dioxide,
                    quality_flag: 1,
                });
            }
        }
        Ok(out)
    }
}

/// Parse NOAA CoastWatch ERDDAP `ncdcOisst21Agg_LonPM180.csv` griddap output.
/// Skips header + units row (`UTC,m,degrees_north,degrees_east,Celsius`) and `NaN` land masks.
pub fn parse_erddap_oisst_csv(
    csv_text: &str,
    fallback_time: &str,
) -> Result<Vec<RawTelemetrySample>> {
    let mut lines = csv_text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        anyhow::bail!("Empty ERDDAP OISST CSV response");
    };
    let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
    let idx_time = cols.iter().position(|&c| c == "time");
    let idx_lat = cols
        .iter()
        .position(|&c| c == "latitude")
        .ok_or_else(|| anyhow::anyhow!("ERDDAP CSV missing latitude column"))?;
    let idx_lon = cols
        .iter()
        .position(|&c| c == "longitude")
        .ok_or_else(|| anyhow::anyhow!("ERDDAP CSV missing longitude column"))?;
    let idx_sst = cols
        .iter()
        .position(|&c| c == "sst")
        .ok_or_else(|| anyhow::anyhow!("ERDDAP CSV missing sst column"))?;

    let mut out = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() <= idx_sst || parts.len() <= idx_lon || parts.len() <= idx_lat {
            continue;
        }
        // Skip ERDDAP row-2 units line (e.g. "degrees_north") or NaN land cells
        let Ok(lat) = parts[idx_lat].parse::<f64>() else {
            continue;
        };
        let Ok(lon) = parts[idx_lon].parse::<f64>() else {
            continue;
        };
        let Ok(sst) = parts[idx_sst].parse::<f64>() else {
            continue;
        };
        if !sst.is_finite() || !(-5.0..=45.0).contains(&sst) {
            continue;
        }
        let ts = idx_time
            .and_then(|i| parts.get(i))
            .filter(|s| !s.is_empty() && **s != "UTC")
            .map(|s| (*s).to_string())
            .unwrap_or_else(|| fallback_time.to_string());

        out.push(RawTelemetrySample {
            layer_id: "sst_marine_heatwaves".to_string(),
            source_name: "NOAA CoastWatch ERDDAP OISST v2.1".to_string(),
            source_url: COASTWATCH_ERDDAP_OISST_CSV_URL.to_string(),
            timestamp_utc: ts,
            latitude: lat,
            longitude: ((lon + 180.0).rem_euclid(360.0)) - 180.0,
            vertical_level: 0.0,
            primary_value: sst,
            secondary_value: None,
            quality_flag: 1,
        });
    }

    if out.is_empty() {
        anyhow::bail!("No valid ocean SST cells parsed from ERDDAP OISST CSV");
    }
    Ok(out)
}

/// Parse NASA FIRMS VIIRS/MODIS active fire CSV (`latitude,longitude,...,frp,...`),
/// spatially binning detections onto `1° x 1°` fire complexes and returning the top
/// `max_clusters` complexes ranked by aggregated Fire Radiative Power (`MW`).
pub fn parse_firms_csv_hotspots(
    csv_text: &str,
    source_url: &str,
    fallback_time: &str,
    max_clusters: usize,
) -> Result<Vec<RawTelemetrySample>> {
    let mut lines = csv_text.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next() else {
        anyhow::bail!("Empty NASA FIRMS CSV response");
    };
    let cols: Vec<String> = header
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .collect();

    let idx_lat = cols
        .iter()
        .position(|c| c == "latitude")
        .ok_or_else(|| anyhow::anyhow!("NASA FIRMS CSV missing 'latitude' header"))?;
    let idx_lon = cols
        .iter()
        .position(|c| c == "longitude")
        .ok_or_else(|| anyhow::anyhow!("NASA FIRMS CSV missing 'longitude' header"))?;
    let idx_frp = cols
        .iter()
        .position(|c| c == "frp")
        .ok_or_else(|| anyhow::anyhow!("NASA FIRMS CSV missing 'frp' header"))?;
    let idx_bright = cols
        .iter()
        .position(|c| c == "bright_ti4" || c == "brightness");
    let idx_date = cols.iter().position(|c| c == "acq_date");
    let idx_conf = cols.iter().position(|c| c == "confidence");

    // Aggregate sub-pixel VIIRS detections into 1-degree regional fire complexes:
    // (lat_sum, lon_sum, count, frp_sum_mw, max_bright_k, latest_date)
    let mut bins: HashMap<(i32, i32), (f64, f64, usize, f64, Option<f64>, String)> = HashMap::new();

    for line in lines {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() <= idx_frp || parts.len() <= idx_lat || parts.len() <= idx_lon {
            continue;
        }
        // Skip low-confidence false alarms ("l" / "low" or numeric < 25)
        if let Some(ic) = idx_conf {
            if let Some(&conf_str) = parts.get(ic) {
                let lc = conf_str.to_lowercase();
                if lc == "l" || lc == "low" {
                    continue;
                }
                if let Ok(num_c) = lc.parse::<f64>() {
                    if num_c < 25.0 {
                        continue;
                    }
                }
            }
        }

        let (Ok(lat), Ok(lon), Ok(frp)) = (
            parts[idx_lat].parse::<f64>(),
            parts[idx_lon].parse::<f64>(),
            parts[idx_frp].parse::<f64>(),
        ) else {
            continue;
        };
        if !(-90.0..=90.0).contains(&lat)
            || !(-180.0..=180.0).contains(&lon)
            || !frp.is_finite()
            || frp <= 0.1
        {
            continue;
        }

        let bright = idx_bright
            .and_then(|ib| parts.get(ib))
            .and_then(|s| s.parse::<f64>().ok());
        let ts = idx_date
            .and_then(|id| parts.get(id))
            .filter(|s| !s.is_empty())
            .map(|d| format!("{}T00:00:00Z", d))
            .unwrap_or_else(|| fallback_time.to_string());

        let key = (lat.floor() as i32, lon.floor() as i32);
        let entry = bins
            .entry(key)
            .or_insert_with(|| (0.0, 0.0, 0, 0.0, None, ts.clone()));
        entry.0 += lat * frp;
        entry.1 += lon * frp;
        entry.2 += 1;
        entry.3 += frp;
        if let Some(b) = bright {
            entry.4 = Some(entry.4.map_or(b, |prev| prev.max(b)));
        }
    }

    let mut ranked: Vec<_> = bins.into_values().collect();
    ranked.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(max_clusters.max(1));

    let samples = ranked
        .into_iter()
        .map(|(w_lat, w_lon, _cnt, frp_sum, max_bright, ts)| RawTelemetrySample {
            layer_id: "active_wildfires_frp_burn_severity".to_string(),
            source_name: "NASA FIRMS VIIRS 375m".to_string(),
            source_url: source_url.to_string(),
            timestamp_utc: ts,
            latitude: w_lat / frp_sum,
            longitude: w_lon / frp_sum,
            vertical_level: 0.0,
            primary_value: frp_sum.min(2500.0),
            secondary_value: max_bright,
            quality_flag: 1,
        })
        .collect();

    Ok(samples)
}

/// Parse NOAA PSL (`YYYY m1..m12`) or NOAA CPC (`YR MON TOTAL ClimAdjust ANOM`) ENSO Niño 3.4 ASCII feeds,
/// returning the most recent valid monthly SST anomaly (`°C`).
pub fn parse_noaa_enso_nino34(text: &str) -> Result<f64> {
    let mut latest: Option<f64> = None;
    for line in text.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 3 {
            continue;
        }
        let Ok(yr) = toks[0].parse::<i32>() else {
            continue;
        };
        if !(1870..=2100).contains(&yr) {
            continue;
        }

        if toks.len() >= 13 {
            // PSL 13-column format: YEAR JAN FEB ... DEC (missing = -99.99 / -999.0)
            for &tok in &toks[1..=12] {
                if let Ok(v) = tok.parse::<f64>() {
                    if (-6.0..=6.0).contains(&v) {
                        latest = Some(v);
                    }
                }
            }
        } else if toks.len() == 5 {
            // CPC ERSSTv5 5-column format: YR MON TOTAL ClimAdjust ANOM
            if let Ok(anom) = toks[4].parse::<f64>() {
                if (-6.0..=6.0).contains(&anom) {
                    latest = Some(anom);
                }
            }
        }
    }
    latest.ok_or_else(|| anyhow::anyhow!("No valid ENSO Nino 3.4 anomaly found in NOAA ASCII feed"))
}

/// Parse NOAA CPC Monthly North Atlantic Oscillation (`NAO`) ASCII table (`YEAR JAN..DEC` or `YEAR MON INDEX`).
pub fn parse_noaa_cpc_nao(text: &str) -> Result<f64> {
    let mut latest: Option<f64> = None;
    for line in text.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 3 {
            continue;
        }
        let Ok(yr) = toks[0].parse::<i32>() else {
            continue;
        };
        if !(1948..=2100).contains(&yr) {
            continue;
        }

        if toks.len() >= 13 {
            for &tok in &toks[1..=12] {
                if let Ok(v) = tok.parse::<f64>() {
                    if (-6.0..=6.0).contains(&v) {
                        latest = Some(v);
                    }
                }
            }
        } else if toks.len() == 3 {
            if let Ok(v) = toks[2].parse::<f64>() {
                if (-6.0..=6.0).contains(&v) {
                    latest = Some(v);
                }
            }
        }
    }
    latest.ok_or_else(|| anyhow::anyhow!("No valid NAO index found in NOAA CPC ASCII table"))
}

/// Coarse continental interior check so marine SST queries target ocean/coastal waters.
pub fn is_approx_continental_interior(lat: f64, lon: f64) -> bool {
    let n_america = (28.0..=62.0).contains(&lat) && (-118.0..=-78.0).contains(&lon);
    let eurasia = (32.0..=68.0).contains(&lat) && (15.0..=125.0).contains(&lon);
    let sahara_africa = (-18.0..=30.0).contains(&lat) && (12.0..=36.0).contains(&lon);
    let s_america = (-25.0..=5.0).contains(&lat) && (-70.0..=-48.0).contains(&lon);
    let australia = (-32.0..=-18.0).contains(&lat) && (120.0..=146.0).contains(&lon);
    let antarctica = lat < -72.0;
    n_america || eurasia || sahara_africa || s_america || australia || antarctica
}

/// Deterministic physical climatological boundary observations used when operating air-gapped.
pub fn seed_climatological_boundary_samples(
    coords: &[(f64, f64)],
    timestamp_utc: &str,
) -> Vec<RawTelemetrySample> {
    let mut out = Vec::with_capacity(coords.len() * 4);
    for &(lat, lon) in coords {
        let phi = lat.to_radians();
        let lam = lon.to_radians();
        let t2m = 27.5 * phi.cos() - 12.0 * phi.sin().powi(2) + 3.2 * (2.0 * lam).cos();
        let mslp = 1013.25 - 8.5 * (3.0 * phi).cos() + 4.0 * (2.0 * lam + phi).sin();
        let rh = (78.0 - 22.0 * (2.0 * phi).sin().powi(2)).clamp(20.0, 96.0);

        out.push(RawTelemetrySample {
            layer_id: "synoptic_fronts_isobars".to_string(),
            source_name: "ERA5 Climatology Baseline".to_string(),
            source_url: "https://cds.climate.copernicus.eu/".to_string(),
            timestamp_utc: timestamp_utc.to_string(),
            latitude: lat,
            longitude: lon,
            vertical_level: 1000.0,
            primary_value: mslp,
            secondary_value: Some(t2m),
            quality_flag: 0,
        });
        out.push(RawTelemetrySample {
            layer_id: "wet_bulb_globe_temp_heat_stress".to_string(),
            source_name: "ERA5-HEAT Climatology Baseline".to_string(),
            source_url: "https://cds.climate.copernicus.eu/".to_string(),
            timestamp_utc: timestamp_utc.to_string(),
            latitude: lat,
            longitude: lon,
            vertical_level: 1000.0,
            primary_value: t2m,
            secondary_value: Some(rh),
            quality_flag: 0,
        });
        if !is_approx_continental_interior(lat, lon) {
            let sst = (28.5 * phi.cos() - 2.0).max(-1.8);
            out.push(RawTelemetrySample {
                layer_id: "sst_marine_heatwaves".to_string(),
                source_name: "NOAA OISST v2.1 Climatology Baseline".to_string(),
                source_url: "https://www.ncei.noaa.gov/products/optimum-interpolation-sst"
                    .to_string(),
                timestamp_utc: timestamp_utc.to_string(),
                latitude: lat,
                longitude: lon,
                vertical_level: 0.0,
                primary_value: sst,
                secondary_value: None,
                quality_flag: 0,
            });
        }
    }
    out
}
