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

//! Native desktop GUI (`eframe` / `egui`) implementing `climate_modeling_hub_schema.json`:
//! - Top Bar: Global Anomaly Ticker, Projection Switcher (`EPSG:4978`, `EPSG:4326`, `EPSG:3995`, `EPSG:3031`),
//!   SSP Scenario Selector, Coordinate Geocoder/Jump bar, Crosshair & Transect toggles, and 4 Live Readouts.
//! - Left Collapsible Sidebar (`layer_tree_controller`): Basemap selector, Atmospheric Pressure (`1000..=10 hPa`)
//!   & Ocean Depth (`0..=6000 m`) sliders, Anomaly vs. Absolute mode, and all 6 Layer Groups / 34 Layers
//!   with opacity blending, palette picker, and public source citation popovers.
//! - Main Global Map Canvas (`global_map_canvas`): Rayon-parallelized inverse-projected multi-layer
//!   raster compositor + Natural Earth coastlines, graticules, animated Lagrangian wind/current particles,
//!   teleconnection masks, observatory pins, and A->B cross-section transect line.
//! - Right Collapsible Analytics Dock (`telemetry_analytics_dock`): Point Skew-T / Log-P vertical sounding,
//!   Ocean CTD depth profile, Regional 1950-2100 time-series chart, CMIP6 ensemble spread boxplot,
//!   and Zarr v3 JSON subset exporter.
//! - Bottom Temporal Controller (`temporal_dimension_controller`): Live/Historical/CMIP mode selector,
//!   1950-2100 timeline scrubber, Play/Pause animation, and SQLite/Rayon telemetry status bar.

use crate::coastlines::{CONTINENT_POLYLINES, REFERENCE_OBSERVATORIES, TELECONNECTION_BASINS};
use crate::colormaps::{
    default_layer_style, shade_basemap_with_slope_rgb, ColormapPalette,
};
use crate::interp::{
    detect_mslp_extrema, detect_thermal_fronts, equirectangular_scales,
    extract_marching_squares_contours, fibonacci_sphere_points, haversine_km,
    project_latlon_to_ndc, sample_grid_bilinear, slerp_great_circle, unproject_ndc_to_latlon,
    zoom_camera_toward_ndc, CameraState, ContourSegment, FrontSegment, MapProjectionKind,
    PressureExtrema,
};
use crate::physics::{
    compare_ssp_pathways, compute_land_cryosphere_profile_from_grids_at_doy,
    compute_ocean_ctd_profile_from_grids, compute_vertical_sounding_from_grids,
    elevation_slope_m_per_deg, regional_cmip6_delta_t_c, snapshot_regional_delta_t_c,
    synthetic_planetary_elevation_m, ClimateRegion, LandCryosphereProfile, SimulationConfig,
};
use crate::schema::{HubSchema, SspScenario};
use crate::worker::{RunningWorkers, TelemetrySnapshot};
use anyhow::Result;
use chrono::{Datelike, Timelike};
use egui::{Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmip6PlotMetric {
    Co2Ppm,
    DeltaTC,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuickScenarioPreset {
    BaselinePresent,
    ModerateSsp245,
    HighEmissionsSsp585,
}

impl QuickScenarioPreset {
    pub const ALL: [QuickScenarioPreset; 3] = [
        QuickScenarioPreset::BaselinePresent,
        QuickScenarioPreset::ModerateSsp245,
        QuickScenarioPreset::HighEmissionsSsp585,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            QuickScenarioPreset::BaselinePresent => "Baseline (Now)",
            QuickScenarioPreset::ModerateSsp245 => "Moderate SSP2-4.5",
            QuickScenarioPreset::HighEmissionsSsp585 => "High SSP5-8.5",
        }
    }

    pub fn tooltip(&self) -> &'static str {
        match self {
            QuickScenarioPreset::BaselinePresent => {
                "Present-day live telemetry baseline (SSP2-4.5, current UTC date, default layers)"
            }
            QuickScenarioPreset::ModerateSsp245 => {
                "Mid-century moderate emissions (SSP2-4.5, Year 2050) highlighting CMIP6 ΔT, SST & fronts"
            }
            QuickScenarioPreset::HighEmissionsSsp585 => {
                "End-of-century high emissions (SSP5-8.5, Year 2100) highlighting warming, SST, WBGT & surge"
            }
        }
    }
}

fn parse_lat_lon_query(input: &str) -> Option<(f64, f64)> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parts: Vec<&str> = if trimmed.contains(',') {
        trimmed.split(',').map(str::trim).collect()
    } else {
        trimmed.split_whitespace().collect()
    };
    if parts.len() != 2 {
        return None;
    }
    let la = parts[0].parse::<f64>().ok()?;
    let lo = parts[1].parse::<f64>().ok()?;
    if !la.is_finite() || !lo.is_finite() {
        return None;
    }
    Some((la.clamp(-90.0, 90.0), lo.clamp(-180.0, 180.0)))
}

#[derive(Debug, Clone)]
struct LayerUiState {
    visible: bool,
    opacity: f32,
    palette: ColormapPalette,
    min_range: f32,
    max_range: f32,
}

#[derive(Debug, Clone, Copy)]
struct LagrangianParticle {
    lat: f64,
    lon: f64,
    age: u16,
    max_age: u16,
}

pub struct TerraTwinGuiApp {
    workers: RunningWorkers,
    sim_cfg: SimulationConfig,
    projection: MapProjectionKind,
    camera: CameraState,
    basemap_id: String,
    anomaly_mode: bool,

    left_sidebar_open: bool,
    right_drawer_open: bool,
    about_open: bool,
    command_palette_open: bool,
    command_palette_just_opened: bool,
    command_palette_query: String,
    ssp_compare_year: f64,
    crosshair_inspector: bool,
    transect_mode: bool,
    transect_points: Vec<(f64, f64)>,

    pinned_coord: (f64, f64),
    coord_search_buf: String,
    layer_filter_query: String,
    temporal_mode_id: String,
    playing_timeline: bool,
    playback_timestamps: Vec<String>,
    playback_index: Option<usize>,
    playback_last_advance: std::time::Instant,
    last_timeline_refresh: std::time::Instant,
    export_status_msg: String,

    cmip6_metric: Cmip6PlotMetric,
    cmip6_region: ClimateRegion,

    layer_states: HashMap<String, LayerUiState>,
    solo_restore_vis: Option<HashMap<String, bool>>,
    particles: Vec<LagrangianParticle>,
    map_texture: Option<TextureHandle>,
    last_tex_size: (usize, usize),
    cached_snapshot: Option<TelemetrySnapshot>,
    cached_isobars: Vec<ContourSegment>,
    cached_extrema: Vec<PressureExtrema>,
    cached_fronts: Vec<FrontSegment>,
    raster_dirty: bool,
    last_raster_gen_key: u64,
    open_visible_groups_only: bool,
}

impl TerraTwinGuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, workers: RunningWorkers) -> Self {
        Self::new_headless(workers)
    }

    pub fn new_headless(workers: RunningWorkers) -> Self {
        let schema = workers.coordinator.schema().clone();

        let cam_def = &schema.main_viewport.camera_defaults;
        let camera = CameraState {
            center_lon: cam_def.longitude,
            center_lat: cam_def.latitude,
            zoom: 1.0,
            pitch_deg: cam_def.pitch_degrees,
            bearing_deg: cam_def.bearing_degrees,
            vertical_exaggeration: cam_def.vertical_exaggeration,
        };

        let mut layer_states = HashMap::new();
        for (_, layer) in schema.all_layers() {
            let (palette, min_range, max_range) = default_layer_style(&layer.id);
            layer_states.insert(
                layer.id.clone(),
                LayerUiState {
                    visible: layer.default_visible,
                    opacity: layer.default_opacity,
                    palette,
                    min_range,
                    max_range,
                },
            );
        }

        let default_basemap = schema
            .main_viewport
            .base_maps
            .iter()
            .find(|b| b.default)
            .map(|b| b.id.clone())
            .unwrap_or_else(|| "bathymetry_topography_shaded".to_string());

        let seed_pts = fibonacci_sphere_points(260);
        let particles = seed_pts
            .into_iter()
            .enumerate()
            .map(|(idx, (lat, lon))| LagrangianParticle {
                lat,
                lon,
                age: ((idx * 7) % 90) as u16,
                max_age: 95 + ((idx * 13) % 55) as u16,
            })
            .collect();

        Self {
            workers,
            sim_cfg: SimulationConfig::default(),
            projection: MapProjectionKind::from_id(&schema.main_viewport.default_projection),
            camera,
            basemap_id: default_basemap,
            anomaly_mode: false,
            left_sidebar_open: true,
            right_drawer_open: true,
            about_open: false,
            command_palette_open: false,
            command_palette_just_opened: false,
            command_palette_query: String::new(),
            ssp_compare_year: 2100.0,
            crosshair_inspector: true,
            transect_mode: false,
            transect_points: Vec::new(),
            pinned_coord: (cam_def.latitude, cam_def.longitude),
            coord_search_buf: format!("{:.4}, {:.4}", cam_def.latitude, cam_def.longitude),
            layer_filter_query: String::new(),
            temporal_mode_id: "realtime_telemetry".to_string(),
            playing_timeline: false,
            playback_timestamps: Vec::new(),
            playback_index: None,
            playback_last_advance: std::time::Instant::now(),
            last_timeline_refresh: std::time::Instant::now() - std::time::Duration::from_secs(10),
            export_status_msg: String::new(),
            cmip6_metric: Cmip6PlotMetric::Co2Ppm,
            cmip6_region: ClimateRegion::Global,
            layer_states,
            solo_restore_vis: None,
            particles,
            map_texture: None,
            last_tex_size: (0, 0),
            cached_snapshot: None,
            cached_isobars: Vec::new(),
            cached_extrema: Vec::new(),
            cached_fronts: Vec::new(),
            raster_dirty: true,
            last_raster_gen_key: 0,
            open_visible_groups_only: false,
        }
    }

    pub fn reset_camera(&mut self) {
        let cam_def = &self.workers.coordinator.schema().main_viewport.camera_defaults;
        self.camera = CameraState {
            center_lon: cam_def.longitude,
            center_lat: cam_def.latitude,
            zoom: 1.0,
            pitch_deg: cam_def.pitch_degrees,
            bearing_deg: cam_def.bearing_degrees,
            vertical_exaggeration: cam_def.vertical_exaggeration,
        };
        self.raster_dirty = true;
    }

    pub fn apply_quick_preset(&mut self, preset: QuickScenarioPreset) {
        let schema = self.workers.coordinator.schema().clone();
        self.solo_restore_vis = None;
        self.playing_timeline = false;
        match preset {
            QuickScenarioPreset::BaselinePresent => {
                let now = chrono::Utc::now();
                self.sim_cfg.scenario = SspScenario::Ssp2_45;
                self.sim_cfg.sim_year = now.year() as f64 + (now.ordinal() as f64 / 365.25);
                self.sim_cfg.day_of_year = now.ordinal() as f64;
                self.sim_cfg.hour_utc = now.hour() as f64 + (now.minute() as f64 / 60.0);
                self.sim_cfg.time_step_secs = 600.0;
                self.temporal_mode_id = "realtime_telemetry".to_string();
                self.anomaly_mode = false;
                for (_, lyr) in schema.all_layers() {
                    if let Some(st) = self.layer_states.get_mut(&lyr.id) {
                        st.visible = lyr.default_visible;
                    }
                }
            }
            QuickScenarioPreset::ModerateSsp245 => {
                self.sim_cfg.scenario = SspScenario::Ssp2_45;
                self.sim_cfg.sim_year = 2050.0;
                self.sim_cfg.time_step_secs = 3600.0;
                self.temporal_mode_id = "cmip_projections".to_string();
                self.cmip6_metric = Cmip6PlotMetric::DeltaTC;
                for st in self.layer_states.values_mut() {
                    st.visible = false;
                }
                for id in [
                    "cmip6_downscaled_ensemble",
                    "sst_marine_heatwaves",
                    "synoptic_fronts_isobars",
                ] {
                    if let Some(st) = self.layer_states.get_mut(id) {
                        st.visible = true;
                    }
                }
            }
            QuickScenarioPreset::HighEmissionsSsp585 => {
                self.sim_cfg.scenario = SspScenario::Ssp5_85;
                self.sim_cfg.sim_year = 2100.0;
                self.sim_cfg.time_step_secs = 3600.0;
                self.temporal_mode_id = "cmip_projections".to_string();
                self.cmip6_metric = Cmip6PlotMetric::DeltaTC;
                for st in self.layer_states.values_mut() {
                    st.visible = false;
                }
                for id in [
                    "cmip6_downscaled_ensemble",
                    "sst_marine_heatwaves",
                    "wet_bulb_globe_temp_heat_stress",
                    "coastal_inundation_storm_surge",
                    "sea_ice_concentration_drift",
                ] {
                    if let Some(st) = self.layer_states.get_mut(id) {
                        st.visible = true;
                    }
                }
            }
        }
        self.push_sim_update();
    }

    fn step_timeline_by(&mut self, dir: i32) {
        if let Some(cur_idx) = self.playback_index {
            if dir < 0 && cur_idx > 0 {
                self.load_replay_snapshot_at_idx(cur_idx - 1);
            } else if dir > 0 && cur_idx + 1 < self.playback_timestamps.len() {
                self.load_replay_snapshot_at_idx(cur_idx + 1);
            }
            return;
        }
        let now = chrono::Utc::now();
        let present_year = now.year() as f64 + (now.ordinal() as f64 / 365.25);
        match self.temporal_mode_id.as_str() {
            "realtime_telemetry" => {
                let lo_yr = present_year - 7.0 / 365.25;
                let hi_yr = present_year + 10.0 / 365.25;
                let step_yr = (dir as f64) * (1.0 / (365.25 * 24.0));
                self.sim_cfg.sim_year = (self.sim_cfg.sim_year + step_yr).clamp(lo_yr, hi_yr);
                self.sim_cfg.hour_utc = (self.sim_cfg.hour_utc + dir as f64).rem_euclid(24.0);
            }
            "historical_reanalysis" => {
                self.sim_cfg.sim_year =
                    (self.sim_cfg.sim_year + dir as f64).clamp(1950.0, present_year);
            }
            _ => {
                self.sim_cfg.sim_year =
                    (self.sim_cfg.sim_year + dir as f64).clamp(2026.0, 2100.0);
            }
        }
        self.push_sim_update();
    }

    fn push_sim_update(&mut self) {
        self.playback_index = None;
        self.workers
            .coordinator
            .request_simulation_update(self.sim_cfg.clone());
        self.raster_dirty = true;
    }

    fn refresh_playback_timestamps(&mut self) {
        if let Ok(ts) = self
            .workers
            .coordinator
            .db()
            .list_available_timestamps(self.sim_cfg.scenario, 240)
        {
            self.playback_timestamps = ts;
            self.last_timeline_refresh = std::time::Instant::now();
        }
    }

    fn load_replay_snapshot_at_idx(&mut self, idx: usize) {
        let Some(ts) = self.playback_timestamps.get(idx).cloned() else {
            return;
        };
        let db = self.workers.coordinator.db();
        let schema = self.workers.coordinator.schema();
        if let Ok(grids) = db.load_layer_bundle_at_timestamp(
            schema,
            self.sim_cfg.scenario,
            self.sim_cfg.atmospheric_level_hpa,
            self.sim_cfg.ocean_depth_m,
            &ts,
        ) {
            let readouts = db
                .load_readouts_at_timestamp(self.sim_cfg.scenario, &ts)
                .ok()
                .flatten()
                .unwrap_or(crate::db::GlobalLiveReadouts {
                    timestamp_utc: ts.clone(),
                    scenario: self.sim_cfg.scenario.code().to_string(),
                    global_2m_temp_anomaly_c: 0.0,
                    mauna_loa_co2_ppm: self.sim_cfg.scenario.co2_ppm_at_year(self.sim_cfg.sim_year),
                    enso_nino34_c: 0.0,
                    global_sst_anomaly_c: 0.0,
                });
            let db_stats = db.summary_stats().unwrap_or(crate::db::DbStatsSummary {
                db_path: String::new(),
                registered_layers: 34,
                raw_observation_count: 0,
                simulated_grid_count: grids.len() as i64,
                readout_snapshots: 0,
                sync_events: 0,
                latest_simulation_utc: Some(ts.clone()),
            });
            self.cached_snapshot = Some(TelemetrySnapshot {
                readouts,
                grids,
                config: self.sim_cfg.clone(),
                db_stats,
                last_ingest_status: format!("4D SQLite Replay: {}", ts),
                last_step_ms: 0,
            });
            self.playback_index = Some(idx);
            self.raster_dirty = true;
        }
    }

    fn advect_particles(&mut self) {
        let Some(ref snap) = self.cached_snapshot else {
            return;
        };
        // Prefer active vector layer: wind_vector_jet_streams -> atmospheric_rivers_ivt -> thermohaline_currents_amoc
        let vector_grid = ["wind_vector_jet_streams", "atmospheric_rivers_ivt", "thermohaline_currents_amoc"]
            .iter()
            .find(|&&id| {
                self.layer_states
                    .get(id)
                    .map(|s| s.visible)
                    .unwrap_or(false)
            })
            .and_then(|&id| snap.grids.get(id));

        let Some(grid) = vector_grid else {
            return;
        };
        let (Some(ref u_arr), Some(ref v_arr)) = (&grid.u_component, &grid.v_component) else {
            return;
        };

        let speed_scale = if grid.layer_id == "thermohaline_currents_amoc" {
            1.4
        } else if grid.layer_id == "atmospheric_rivers_ivt" {
            0.0025
        } else {
            0.045
        };

        let seed_pts = fibonacci_sphere_points(self.particles.len().max(1));
        for (idx, p) in self.particles.iter_mut().enumerate() {
            p.age += 1;
            if p.age >= p.max_age || p.lat.abs() > 88.5 {
                let (slat, slon) = seed_pts[idx % seed_pts.len()];
                p.lat = slat + ((p.max_age as f64 % 5.0) - 2.5) * 0.4;
                p.lon = slon + ((idx as f64 % 7.0) - 3.5) * 0.4;
                p.age = 0;
                continue;
            }
            let u = sample_grid_bilinear(u_arr, grid.width, grid.height, p.lat, p.lon) as f64;
            let v = sample_grid_bilinear(v_arr, grid.width, grid.height, p.lat, p.lon) as f64;
            if u.is_finite() && v.is_finite() {
                let cos_lat = p.lat.to_radians().cos().max(0.18);
                p.lon = ((p.lon + (u * speed_scale) / cos_lat + 540.0).rem_euclid(360.0)) - 180.0;
                p.lat = (p.lat + v * speed_scale).clamp(-88.5, 88.5);
            } else {
                p.age = p.max_age;
            }
        }
    }

    fn refresh_synoptic_overlays(&mut self) {
        let Some(ref snap) = self.cached_snapshot else {
            self.cached_isobars.clear();
            self.cached_extrema.clear();
            self.cached_fronts.clear();
            return;
        };
        if let Some(mslp) = snap.grids.get("synoptic_fronts_isobars") {
            self.cached_isobars = extract_marching_squares_contours(
                &mslp.values,
                mslp.width,
                mslp.height,
                4.0,
                960.0,
                1048.0,
            );
            self.cached_extrema = detect_mslp_extrema(&mslp.values, mslp.width, mslp.height, 1.4);
        } else {
            self.cached_isobars.clear();
            self.cached_extrema.clear();
        }

        if let (Some(temp_g), Some(wind_g)) = (
            snap.grids.get("cmip6_downscaled_ensemble"),
            snap.grids.get("wind_vector_jet_streams"),
        ) {
            if let (Some(ref u), Some(ref v)) = (&wind_g.u_component, &wind_g.v_component) {
                self.cached_fronts = detect_thermal_fronts(
                    &temp_g.values,
                    u,
                    v,
                    temp_g.width,
                    temp_g.height,
                    8.5,
                );
            }
        }
    }

    fn rebuild_composite_texture(&mut self, ctx: &egui::Context, tex_w: usize, tex_h: usize) {
        self.refresh_synoptic_overlays();
        let aspect = tex_w as f64 / tex_h.max(1) as f64;
        let proj = self.projection;
        let cam = self.camera;
        let basemap_id = self.basemap_id.clone();
        let anomaly_mode = self.anomaly_mode;

        // Collect active visible scalar layers with their grids
        struct RenderLayer<'a> {
            grid: &'a crate::db::StoredGridRecord,
            opacity: f32,
            palette: ColormapPalette,
            min_v: f32,
            max_v: f32,
        }

        let mut active_layers = Vec::new();
        if let Some(ref snap) = self.cached_snapshot {
            for (_, layer_spec) in self.workers.coordinator.schema().all_layers() {
                // Vector-only layers render strictly as contours/polygons in render_central_map, never as opaque raster
                if layer_spec.id == "synoptic_fronts_isobars"
                    || layer_spec.id == "teleconnection_oscillation_masks"
                {
                    continue;
                }
                if let Some(ui_st) = self.layer_states.get(&layer_spec.id) {
                    if ui_st.visible {
                        if let Some(g) = snap.grids.get(&layer_spec.id) {
                            active_layers.push(RenderLayer {
                                grid: g,
                                opacity: ui_st.opacity,
                                palette: ui_st.palette,
                                min_v: ui_st.min_range,
                                max_v: ui_st.max_range,
                            });
                        }
                    }
                }
            }
        }

        let mut pixels = vec![Color32::from_rgb(7, 11, 20); tex_w * tex_h];
        pixels
            .par_chunks_mut(tex_w)
            .enumerate()
            .for_each(|(py, row_slice)| {
                let ndc_y = 1.0 - 2.0 * ((py as f64 + 0.5) / tex_h as f64);
                for (px, out_px) in row_slice.iter_mut().enumerate() {
                    let ndc_x = 2.0 * ((px as f64 + 0.5) / tex_w as f64) - 1.0;
                    let Some((lat, lon)) = unproject_ndc_to_latlon(proj, &cam, ndc_x, ndc_y, aspect)
                    else {
                        *out_px = if proj == MapProjectionKind::Globe3dWgs84 {
                            let radius = 0.86 * cam.zoom;
                            let sx = (ndc_x * aspect) / radius;
                            let sy = ndc_y / radius;
                            let rho = (sx * sx + sy * sy).sqrt();
                            if rho <= 1.06 {
                                let halo = ((-((rho - 1.0).max(0.0)) * 55.0).exp() * 0.42) as f32;
                                Color32::from_rgb(
                                    (6.0 + 48.0 * halo) as u8,
                                    (10.0 + 145.0 * halo) as u8,
                                    (18.0 + 225.0 * halo) as u8,
                                )
                            } else {
                                Color32::from_rgb(6, 10, 18)
                            }
                        } else {
                            Color32::from_rgb(6, 10, 18)
                        };
                        continue;
                    };

                    let elev_m = synthetic_planetary_elevation_m(lat, lon);
                    let is_ocean_px = elev_m < 0.0;
                    let (dx_elev, dy_elev) = elevation_slope_m_per_deg(lat, lon);
                    let base_rgb = shade_basemap_with_slope_rgb(
                        &basemap_id,
                        lat,
                        lon,
                        elev_m,
                        dx_elev,
                        dy_elev,
                    );
                    let mut r = base_rgb[0] as f32;
                    let mut g = base_rgb[1] as f32;
                    let mut b = base_rgb[2] as f32;

                    for lyr in &active_layers {
                        let id = lyr.grid.layer_id.as_str();
                        let ocean_only = matches!(
                            id,
                            "sst_marine_heatwaves"
                                | "subsurface_ocean_heat_salinity_3d"
                                | "thermohaline_currents_amoc"
                                | "swot_sea_surface_height_altimetry"
                                | "ocean_acidification_carbonate"
                                | "marine_chlorophyll_hypoxia"
                                | "sea_ice_concentration_drift"
                                | "sea_ice_thickness_freeboard_age"
                        );
                        let land_only = matches!(
                            id,
                            "gravimetric_ice_sheet_mass_balance"
                                | "alpine_glacier_velocity_rgi"
                                | "permafrost_thermokarst_subsidence"
                                | "canopy_height_aboveground_biomass"
                                | "sif_photosynthesis_evaporative_stress"
                                | "soil_moisture_groundwater_grace"
                                | "swot_surface_water_river_discharge"
                                | "active_wildfires_frp_burn_severity"
                                | "dynamic_land_cover_deforestation"
                                | "coastal_inundation_storm_surge"
                                | "compound_hazard_fire_weather_index"
                                | "critical_infrastructure_socioeconomic_exposure"
                        );
                        if (ocean_only && !is_ocean_px) || (land_only && is_ocean_px) {
                            continue;
                        }

                        let raw_val = sample_grid_bilinear(
                            &lyr.grid.values,
                            lyr.grid.width,
                            lyr.grid.height,
                            lat,
                            lon,
                        );
                        if !raw_val.is_finite() {
                            continue;
                        }
                        // Skip near-zero transparent sparse fields (radar dBZ, lightning, fires, sea ice, flood, permafrost, ice sheets)
                        if (id == "volumetric_doppler_radar" && raw_val < 5.0)
                            || (id == "sea_ice_concentration_drift" && raw_val < 8.0)
                            || (id == "sea_ice_thickness_freeboard_age" && raw_val < 0.08)
                            || (id == "permafrost_thermokarst_subsidence" && raw_val < 0.08)
                            || (id == "gravimetric_ice_sheet_mass_balance" && raw_val.abs() < 1.5)
                            || (id == "active_wildfires_frp_burn_severity" && raw_val < 2.0)
                            || (id == "coastal_inundation_storm_surge" && raw_val < 0.08)
                        {
                            continue;
                        }

                        let mut alpha = lyr.opacity.clamp(0.0, 1.0);
                        if id == "geo_multispectral_satellite" {
                            let cloud_frac = if let Some(ref cf_arr) = lyr.grid.u_component {
                                sample_grid_bilinear(
                                    cf_arr,
                                    lyr.grid.width,
                                    lyr.grid.height,
                                    lat,
                                    lon,
                                )
                            } else {
                                ((288.0 - raw_val) / 48.0).clamp(0.0, 1.0)
                            };
                            if !cloud_frac.is_finite() || cloud_frac < 0.08 {
                                continue;
                            }
                            let cf_norm = ((cloud_frac - 0.08) / 0.92).clamp(0.0, 1.0);
                            alpha *= cf_norm.powf(0.85);
                        } else if id == "wind_vector_jet_streams" {
                            let jet_mask = ((raw_val - 16.0) / 28.0).clamp(0.0, 1.0);
                            if jet_mask <= 0.02 {
                                continue;
                            }
                            alpha *= jet_mask * 0.72;
                        } else if id == "atmospheric_rivers_ivt" {
                            let ivt_mask = ((raw_val - 220.0) / 400.0).clamp(0.0, 1.0);
                            if ivt_mask <= 0.02 {
                                continue;
                            }
                            alpha *= ivt_mask;
                        }

                        let (norm, pal) = if anomaly_mode {
                            let span = (lyr.max_v - lyr.min_v).abs().max(1e-3) * 0.45;
                            let anom = raw_val - lyr.grid.mean_val;
                            (
                                ((anom / span) * 0.5 + 0.5).clamp(0.0, 1.0),
                                ColormapPalette::CoolWarmDiverging,
                            )
                        } else {
                            let denom = (lyr.max_v - lyr.min_v).max(1e-6);
                            (((raw_val - lyr.min_v) / denom).clamp(0.0, 1.0), lyr.palette)
                        };

                        let c_rgb = pal.sample_rgb(norm);
                        r = r * (1.0 - alpha) + (c_rgb[0] as f32) * alpha;
                        g = g * (1.0 - alpha) + (c_rgb[1] as f32) * alpha;
                        b = b * (1.0 - alpha) + (c_rgb[2] as f32) * alpha;
                    }

                    *out_px = if proj == MapProjectionKind::Globe3dWgs84 {
                        let radius = 0.86 * cam.zoom;
                        let sx = (ndc_x * aspect) / radius;
                        let sy = ndc_y / radius;
                        let [sr, sg, sb] = apply_globe_lambert_fresnel([r, g, b], sx, sy);
                        Color32::from_rgb(sr, sg, sb)
                    } else {
                        Color32::from_rgb(r as u8, g as u8, b as u8)
                    };
                }
            });

        let img = ColorImage {
            size: [tex_w, tex_h],
            pixels,
        };
        if let Some(ref mut handle) = self.map_texture {
            handle.set(img, TextureOptions::LINEAR);
        } else {
            self.map_texture = Some(ctx.load_texture("planetary_composite", img, TextureOptions::LINEAR));
        }
        self.raster_dirty = false;
    }

    fn render_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("terratwin_top_bar")
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(13, 19, 31))
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .selectable_label(self.left_sidebar_open, "Layers")
                        .on_hover_text("Toggle Left Predictive Layer Tree sidebar (L)")
                        .clicked()
                    {
                        self.left_sidebar_open = !self.left_sidebar_open;
                    }
                    ui.colored_label(
                        Color32::from_rgb(98, 210, 255),
                        egui::RichText::new(concat!("TerraTwin HPC v", env!("CARGO_PKG_VERSION")))
                            .strong(),
                    );
                    ui.separator();

                    // Live Readout Badges
                    if let Some(ref snap) = self.cached_snapshot {
                        let r = &snap.readouts;
                        draw_readout_badge(
                            ui,
                            "2m Temp Anom",
                            &format!("{:+.2} °C", r.global_2m_temp_anomaly_c),
                            Color32::from_rgb(250, 130, 88),
                            "Global cosine-area-weighted 2m surface air temperature anomaly vs 1850–1900 pre-industrial baseline (ECMWF ERA5 / GISTEMP + TerraTwin coupled QGPV/EBM solver)",
                        );
                        draw_readout_badge(
                            ui,
                            "Mauna Loa CO2",
                            &format!("{:.1} ppm", r.mauna_loa_co2_ppm),
                            Color32::from_rgb(245, 200, 75),
                            "Atmospheric CO2 dry-air mole fraction (ppm) anchored to NOAA GML Mauna Loa Observatory & Meinshausen CMIP6 SSP pathways",
                        );
                        draw_readout_badge(
                            ui,
                            "ENSO Niño 3.4",
                            &format!("{:+.2} °C", r.enso_nino34_c),
                            Color32::from_rgb(120, 215, 250),
                            "ENSO Niño 3.4 equatorial Pacific SST anomaly (5°S–5°N, 170°W–120°W) from NOAA CPC & 1.5-layer reduced-gravity ocean solver",
                        );
                        draw_readout_badge(
                            ui,
                            "Global SST Anom",
                            &format!("{:+.2} °C", r.global_sst_anomaly_c),
                            Color32::from_rgb(255, 160, 105),
                            "Global 60°S–60°N Sea Surface Temperature anomaly relative to climatological baseline (NOAA OISST v2.1 / CoastWatch ERDDAP)",
                        );
                    }
                    ui.separator();

                    // Projection Switcher
                    let mut proj_changed = false;
                    egui::ComboBox::from_label("Projection")
                        .selected_text(self.projection.label())
                        .show_ui(ui, |ui| {
                            for p in MapProjectionKind::ALL {
                                if ui
                                    .selectable_value(&mut self.projection, p, p.label())
                                    .changed()
                                {
                                    proj_changed = true;
                                }
                            }
                        })
                        .response
                        .on_hover_text("Switch map projection (Keys 1–4)");
                    if proj_changed {
                        self.raster_dirty = true;
                    }

                    // SSP Scenario Selector
                    let mut ssp_changed = false;
                    egui::ComboBox::from_label("SSP")
                        .selected_text(self.sim_cfg.scenario.code())
                        .show_ui(ui, |ui| {
                            for s in SspScenario::ALL {
                                if ui
                                    .selectable_value(&mut self.sim_cfg.scenario, s, s.code())
                                    .changed()
                                {
                                    ssp_changed = true;
                                }
                            }
                        });
                    if ssp_changed {
                        self.push_sim_update();
                    }

                    // Coordinate Search / Jump
                    ui.label("Coord:");
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut self.coord_search_buf)
                            .id(egui::Id::new("terratwin_coord_search"))
                            .desired_width(125.0),
                    );
                    if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        || ui.button("Go").clicked()
                    {
                        if let Some((la, lo)) = parse_lat_lon_query(&self.coord_search_buf) {
                            self.pinned_coord = (la, lo);
                            self.camera.center_lat = la;
                            self.camera.center_lon = lo;
                            if self.export_status_msg.starts_with("Invalid coordinates") {
                                self.export_status_msg.clear();
                            }
                            self.raster_dirty = true;
                        } else {
                            self.export_status_msg = format!(
                                "Invalid coordinates '{}' (expected finite lat, lon)",
                                self.coord_search_buf.trim()
                            );
                        }
                    }

                    if ui
                        .selectable_label(self.crosshair_inspector, "Crosshair")
                        .clicked()
                    {
                        self.crosshair_inspector = !self.crosshair_inspector;
                    }
                    if ui
                        .selectable_label(self.transect_mode, "Transect A-B")
                        .on_hover_text("Click two map points to plot a great-circle A-B cross-section transect (Esc to cancel)")
                        .clicked()
                    {
                        self.transect_mode = !self.transect_mode;
                        if !self.transect_mode {
                            self.transect_points.clear();
                        }
                    }
                    if ui
                        .selectable_label(self.right_drawer_open, "Analytics")
                        .on_hover_text("Toggle Right Telemetry Analytics drawer (A)")
                        .clicked()
                    {
                        self.right_drawer_open = !self.right_drawer_open;
                    }
                    if ui
                        .button("Reset View")
                        .on_hover_text("Reset camera center, zoom, and bearing (R)")
                        .clicked()
                    {
                        self.reset_camera();
                    }
                    if ui
                        .selectable_label(self.about_open, "About")
                        .on_hover_text("Show TerraTwin version, physics engines, and citation info (F1)")
                        .clicked()
                    {
                        self.about_open = !self.about_open;
                    }
                    if ui
                        .selectable_label(self.command_palette_open, "Palette (Ctrl+K)")
                        .on_hover_text("Fuzzy quick-jump command palette for SSP pathways, layers, tipping points, and exports (Ctrl+K)")
                        .clicked()
                    {
                        self.command_palette_open = !self.command_palette_open;
                        self.command_palette_just_opened = self.command_palette_open;
                    }
                });
            });
    }

    fn export_ssp_delta_file(&mut self, as_csv: bool) {
        let report = compare_ssp_pathways(self.ssp_compare_year);
        let ext = if as_csv { "csv" } else { "json" };
        let filter_name = if as_csv { "CSV Table" } else { "JSON Report" };
        if let Some(out_file) = rfd::FileDialog::new()
            .set_title("Export SSP Pathway Delta Comparison")
            .add_filter(filter_name, &[ext])
            .set_file_name(format!("terratwin_ssp_delta_{:.0}.{ext}", report.year))
            .save_file()
        {
            let content = if as_csv {
                Ok(report.to_csv())
            } else {
                report.to_json_pretty()
            };
            match content.and_then(|body| Ok(std::fs::write(&out_file, body)?)) {
                Ok(()) => {
                    self.export_status_msg = format!(
                        "SSP delta {} ({:.0}) -> {}",
                        ext.to_ascii_uppercase(),
                        report.year,
                        out_file.display()
                    );
                }
                Err(e) => {
                    self.export_status_msg = format!("SSP delta export failed: {e}");
                }
            }
        }
    }

    fn apply_palette_command(&mut self, cmd: &PaletteAction) {
        let schema = self.workers.coordinator.schema().clone();
        match cmd {
            PaletteAction::SetSsp(ssp, year_opt) => {
                self.sim_cfg.scenario = *ssp;
                if let Some(yr) = year_opt {
                    self.sim_cfg.sim_year = *yr;
                    self.temporal_mode_id = "cmip_projections".to_string();
                    self.ssp_compare_year = *yr;
                }
                self.push_sim_update();
            }
            PaletteAction::ApplyPreset(preset) => {
                self.apply_quick_preset(*preset);
            }
            PaletteAction::FocusTippingPoint {
                lat,
                lon,
                zoom,
                layer_ids,
                region,
            } => {
                self.pinned_coord = (*lat, *lon);
                self.camera.center_lat = *lat;
                self.camera.center_lon = *lon;
                self.camera.zoom = *zoom;
                self.coord_search_buf = format!("{:.4}, {:.4}", lat, lon);
                self.cmip6_region = *region;
                for st in self.layer_states.values_mut() {
                    st.visible = false;
                }
                for id in *layer_ids {
                    if let Some(st) = self.layer_states.get_mut(*id) {
                        st.visible = true;
                    }
                }
                self.right_drawer_open = true;
                self.raster_dirty = true;
            }
            PaletteAction::FocusLayer(layer_id) => {
                self.solo_restore_vis = None;
                for (_, lyr) in schema.all_layers() {
                    if let Some(st) = self.layer_states.get_mut(&lyr.id) {
                        st.visible = lyr.id == *layer_id;
                    }
                }
                self.raster_dirty = true;
            }
            PaletteAction::SetProjection(proj) => {
                self.projection = *proj;
                self.raster_dirty = true;
            }
            PaletteAction::ToggleLeftDrawer => {
                self.left_sidebar_open = !self.left_sidebar_open;
            }
            PaletteAction::ToggleRightDrawer => {
                self.right_drawer_open = !self.right_drawer_open;
            }
            PaletteAction::ResetCamera => {
                self.reset_camera();
            }
            PaletteAction::ExportSspDeltaCsv => {
                self.export_ssp_delta_file(true);
            }
            PaletteAction::ExportSspDeltaJson => {
                self.export_ssp_delta_file(false);
            }
            PaletteAction::ExportVisibleZarr => {
                let (plat, plon) = self.pinned_coord;
                let first_vis = schema
                    .all_layers()
                    .into_iter()
                    .find(|(_, l)| {
                        self.layer_states
                            .get(&l.id)
                            .map(|s| s.visible)
                            .unwrap_or(false)
                    })
                    .map(|(g, l)| {
                        let lvl = if g.group_id == "physical_biogeochemical_oceanography" {
                            self.sim_cfg.ocean_depth_m
                        } else {
                            self.sim_cfg.atmospheric_level_hpa
                        };
                        (l.id.clone(), lvl)
                    })
                    .unwrap_or_else(|| {
                        (
                            "cmip6_downscaled_ensemble".to_string(),
                            self.sim_cfg.atmospheric_level_hpa,
                        )
                    });
                if let Some(out_dir) = rfd::FileDialog::new()
                    .set_title("Export Zarr v3 Climate Subset Store")
                    .set_file_name(format!("subset_{}.zarr", first_vis.0))
                    .save_file()
                {
                    match self.workers.coordinator.db().export_zarr_v3_store(
                        &first_vis.0,
                        self.sim_cfg.scenario,
                        first_vis.1,
                        (plat - 20.0).max(-90.0),
                        (plat + 20.0).min(90.0),
                        (plon - 25.0).max(-180.0),
                        (plon + 25.0).min(180.0),
                        (18, 36),
                        &out_dir,
                    ) {
                        Ok(sum) => {
                            self.export_status_msg = format!(
                                "Zarr v3 ({} chunks, {} cells) -> {}",
                                sum.chunks_written, sum.cells_exported, sum.store_dir
                            );
                        }
                        Err(e) => {
                            self.export_status_msg = format!("Zarr v3 export failed: {e}");
                        }
                    }
                }
            }
        }
    }

    fn build_palette_entries(&self) -> Vec<PaletteEntry> {
        let mut entries = vec![
            PaletteEntry {
                category: "SSP",
                title: "SSP1-1.9 Net-Zero Sustainability (2100)".into(),
                subtitle: "Low radiative forcing ~1.9 W/m2 with CO2 drawdown to ~393 ppm".into(),
                action: PaletteAction::SetSsp(SspScenario::Ssp1_19, Some(2100.0)),
            },
            PaletteEntry {
                category: "SSP",
                title: "SSP1-2.6 Paris-Aligned Low Emissions (2100)".into(),
                subtitle: "Compare baseline low-emissions pathway (~446 ppm CO2)".into(),
                action: PaletteAction::SetSsp(SspScenario::Ssp1_26, Some(2100.0)),
            },
            PaletteEntry {
                category: "SSP",
                title: "SSP2-4.5 Intermediate Pathway (2050)".into(),
                subtitle: "Middle-of-the-road mid-century forcing & warming".into(),
                action: PaletteAction::ApplyPreset(QuickScenarioPreset::ModerateSsp245),
            },
            PaletteEntry {
                category: "SSP",
                title: "SSP3-7.0 Regional Rivalry High Forcing (2100)".into(),
                subtitle: "High emissions trajectory reaching ~867 ppm CO2 by 2100".into(),
                action: PaletteAction::SetSsp(SspScenario::Ssp3_70, Some(2100.0)),
            },
            PaletteEntry {
                category: "SSP",
                title: "SSP5-8.5 Fossil-Fueled High Emissions (2100)".into(),
                subtitle: "Extreme heat stress, SLR, and Arctic sea-ice loss (~1135 ppm)".into(),
                action: PaletteAction::ApplyPreset(QuickScenarioPreset::HighEmissionsSsp585),
            },
            PaletteEntry {
                category: "SSP",
                title: "Baseline Present-Day Telemetry (Now)".into(),
                subtitle: "Reset simulation year to current UTC and restore default layers".into(),
                action: PaletteAction::ApplyPreset(QuickScenarioPreset::BaselinePresent),
            },
            PaletteEntry {
                category: "Tipping",
                title: "Arctic Permafrost & Sea-Ice Loss (72°N, 125°E)".into(),
                subtitle: "Focus Siberian Arctic permafrost thaw, sea-ice drift, and polar amplification".into(),
                action: PaletteAction::FocusTippingPoint {
                    lat: 72.0,
                    lon: 125.0,
                    zoom: 2.2,
                    layer_ids: &[
                        "permafrost_thermokarst_subsidence",
                        "sea_ice_concentration_drift",
                        "cmip6_downscaled_ensemble",
                    ],
                    region: ClimateRegion::Arctic,
                },
            },
            PaletteEntry {
                category: "Tipping",
                title: "Greenland Ice Sheet Mass Balance (72°N, -40°W)".into(),
                subtitle: "Focus GRACE gravimetric ice loss and Arctic polar projection".into(),
                action: PaletteAction::FocusTippingPoint {
                    lat: 72.0,
                    lon: -40.0,
                    zoom: 2.4,
                    layer_ids: &[
                        "gravimetric_ice_sheet_mass_balance",
                        "cmip6_downscaled_ensemble",
                    ],
                    region: ClimateRegion::Arctic,
                },
            },
            PaletteEntry {
                category: "Tipping",
                title: "West Antarctic Ice Sheet & Amundsen Sea (75°S, -110°W)".into(),
                subtitle: "Focus Antarctic basal melt, ice mass balance, and coastal SLR".into(),
                action: PaletteAction::FocusTippingPoint {
                    lat: -75.0,
                    lon: -110.0,
                    zoom: 2.3,
                    layer_ids: &[
                        "gravimetric_ice_sheet_mass_balance",
                        "sea_ice_concentration_drift",
                        "coastal_inundation_storm_surge",
                    ],
                    region: ClimateRegion::Antarctic,
                },
            },
            PaletteEntry {
                category: "Tipping",
                title: "AMOC Subpolar North Atlantic Gyre (56°N, -35°W)".into(),
                subtitle: "Focus thermohaline circulation slowdown and subsurface heat/salinity".into(),
                action: PaletteAction::FocusTippingPoint {
                    lat: 56.0,
                    lon: -35.0,
                    zoom: 2.1,
                    layer_ids: &[
                        "thermohaline_currents_amoc",
                        "subsurface_ocean_heat_salinity_3d",
                        "sst_marine_heatwaves",
                    ],
                    region: ClimateRegion::MidLatitudes,
                },
            },
            PaletteEntry {
                category: "Tipping",
                title: "Amazon Basin Canopy & Evaporative Stress (4°S, -63°W)".into(),
                subtitle: "Focus rainforest SIF photosynthesis, biomass, and fire weather".into(),
                action: PaletteAction::FocusTippingPoint {
                    lat: -4.0,
                    lon: -63.0,
                    zoom: 2.2,
                    layer_ids: &[
                        "sif_photosynthesis_evaporative_stress",
                        "canopy_height_aboveground_biomass",
                        "compound_hazard_fire_weather_index",
                    ],
                    region: ClimateRegion::Tropics,
                },
            },
            PaletteEntry {
                category: "Tipping",
                title: "Tropical Coral Reefs & Ocean Acidification (0°N, -145°W)".into(),
                subtitle: "Focus Niño 3.4 SST marine heatwaves and aragonite saturation state".into(),
                action: PaletteAction::FocusTippingPoint {
                    lat: 0.0,
                    lon: -145.0,
                    zoom: 1.9,
                    layer_ids: &[
                        "ocean_acidification_carbonate",
                        "sst_marine_heatwaves",
                        "teleconnection_oscillation_masks",
                    ],
                    region: ClimateRegion::Tropics,
                },
            },
            PaletteEntry {
                category: "Export",
                title: "Export SSP Delta Comparator CSV (SSP1-2.6 / 2-4.5 / 5-8.5)".into(),
                subtitle: format!("Save cross-pathway temperature, SLR, and CO2 table at year {:.0}", self.ssp_compare_year),
                action: PaletteAction::ExportSspDeltaCsv,
            },
            PaletteEntry {
                category: "Export",
                title: "Export SSP Delta Comparator JSON (SSP1-2.6 / 2-4.5 / 5-8.5)".into(),
                subtitle: format!("Save structured SSP delta JSON report at year {:.0}", self.ssp_compare_year),
                action: PaletteAction::ExportSspDeltaJson,
            },
            PaletteEntry {
                category: "Export",
                title: "Export Active Layer Zarr v3 Subset Store...".into(),
                subtitle: "Write Zarr v3 compressed directory store around pinned coordinate".into(),
                action: PaletteAction::ExportVisibleZarr,
            },
            PaletteEntry {
                category: "View",
                title: "Projection: 3D Globe (EPSG:4978)".into(),
                subtitle: "Switch viewport to shaded 3D orthographic globe".into(),
                action: PaletteAction::SetProjection(MapProjectionKind::Globe3dWgs84),
            },
            PaletteEntry {
                category: "View",
                title: "Projection: 2D Equirectangular (EPSG:4326)".into(),
                subtitle: "Switch viewport to full-Earth geographic grid".into(),
                action: PaletteAction::SetProjection(MapProjectionKind::Equirectangular2d),
            },
            PaletteEntry {
                category: "View",
                title: "Projection: Arctic Polar Stereographic (EPSG:3995)".into(),
                subtitle: "Switch viewport to North Polar projection".into(),
                action: PaletteAction::SetProjection(MapProjectionKind::ArcticPolarStereo),
            },
            PaletteEntry {
                category: "View",
                title: "Projection: Antarctic Polar Stereographic (EPSG:3031)".into(),
                subtitle: "Switch viewport to South Polar projection".into(),
                action: PaletteAction::SetProjection(MapProjectionKind::AntarcticPolarStereo),
            },
            PaletteEntry {
                category: "View",
                title: "Toggle Left Predictive Layer Tree".into(),
                subtitle: "Show or collapse the 34-layer sidebar (L)".into(),
                action: PaletteAction::ToggleLeftDrawer,
            },
            PaletteEntry {
                category: "View",
                title: "Toggle Right Analytics & SSP Comparator Drawer".into(),
                subtitle: "Show or collapse telemetry soundings, CMIP6, and SSP delta table (A)".into(),
                action: PaletteAction::ToggleRightDrawer,
            },
            PaletteEntry {
                category: "View",
                title: "Reset Camera Center & Zoom".into(),
                subtitle: "Restore default global camera view (R)".into(),
                action: PaletteAction::ResetCamera,
            },
        ];

        for (grp, lyr) in self.workers.coordinator.schema().all_layers() {
            entries.push(PaletteEntry {
                category: "Layer",
                title: format!("Solo Layer: {}", lyr.name),
                subtitle: format!("{} [{} | {}]", grp.group_name, lyr.id, lyr.units),
                action: PaletteAction::FocusLayer(lyr.id.clone()),
            });
        }
        entries
    }

    fn render_command_palette(&mut self, ctx: &egui::Context) {
        if !self.command_palette_open {
            return;
        }
        let mut keep_open = true;
        let enter_pressed = ctx.input(|i| i.key_pressed(egui::Key::Enter));
        let entries = self.build_palette_entries();
        let q = self.command_palette_query.trim();
        let mut scored: Vec<(i32, usize)> = entries
            .iter()
            .enumerate()
            .filter_map(|(idx, e)| {
                let hay = format!("{} {} {}", e.category, e.title, e.subtitle);
                fuzzy_match_score(q, &hay).map(|s| (s, idx))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        let mut activated: Option<PaletteAction> = None;
        egui::Window::new("Quick-Jump Climate & SSP Palette (Ctrl+K)")
            .open(&mut keep_open)
            .collapsible(false)
            .resizable(false)
            .default_width(560.0)
            .anchor(Align2::CENTER_TOP, egui::vec2(0.0, 56.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Search:").strong());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.command_palette_query)
                            .id(egui::Id::new("terratwin_command_palette_input"))
                            .hint_text("Type SSP pathway, climate layer, tipping point, or export...")
                            .desired_width(430.0),
                    );
                    if self.command_palette_just_opened {
                        resp.request_focus();
                        self.command_palette_just_opened = false;
                    }
                    if ui.small_button("Clear").clicked() {
                        self.command_palette_query.clear();
                        resp.request_focus();
                    }
                });
                ui.separator();

                let first_match = scored.first().map(|&(_, idx)| entries[idx].action.clone());
                egui::ScrollArea::vertical()
                    .max_height(360.0)
                    .show(ui, |ui| {
                        if scored.is_empty() {
                            ui.small("No matching SSP pathways, layers, tipping points, or exports.");
                        } else {
                            for &(_, idx) in scored.iter().take(16) {
                                let entry = &entries[idx];
                                let resp = ui
                                    .selectable_label(
                                        false,
                                        format!("[{}] {}", entry.category, entry.title),
                                    )
                                    .on_hover_text(&entry.subtitle);
                                ui.small(
                                    egui::RichText::new(&entry.subtitle)
                                        .color(Color32::from_rgb(148, 163, 184)),
                                );
                                ui.add_space(2.0);
                                if resp.clicked() {
                                    activated = Some(entry.action.clone());
                                }
                            }
                        }
                    });

                if activated.is_none() && enter_pressed {
                    activated = first_match;
                }
            });

        if !keep_open {
            self.command_palette_open = false;
            self.command_palette_just_opened = false;
        }
        if let Some(action) = activated {
            self.apply_palette_command(&action);
            self.command_palette_open = false;
            self.command_palette_just_opened = false;
        }
    }

    fn render_about_window(&mut self, ctx: &egui::Context) {
        if !self.about_open {
            return;
        }
        let mut open = self.about_open;
        egui::Window::new("About TerraTwin")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.set_min_width(380.0);
                ui.heading("TerraTwin HPC Climate Modeling & Telemetry Hub");
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.add_space(4.0);
                ui.strong("Copyright CRV Science Network");
                ui.label("License: GNU General Public License v3.0 or later (GPL-3.0-or-later)");
                ui.add_space(6.0);
                ui.label(
                    "Multi-threaded Rayon coupled climate physics engine, 34 predictive global layers, \
                     SQLite WAL 4D replay, and Zarr v3 scientific subset exporter.",
                );
                ui.add_space(6.0);
                ui.hyperlink_to("CRV Science Network (https://crvscience.com)", "https://crvscience.com");
                ui.hyperlink_to("CRV Science Public Apps (https://crvscience.com/apps)", "https://crvscience.com/apps");
            });
        self.about_open = open;
    }

    fn render_left_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("layer_tree_controller")
            .resizable(true)
            .default_width(340.0)
            .show_animated(ctx, self.left_sidebar_open, |ui| {
                ui.add_space(4.0);
                ui.heading("Predictive Layer Tree (34)");

                ui.horizontal(|ui| {
                    if ui
                        .checkbox(&mut self.anomaly_mode, "Anomaly Mode (vs Mean)")
                        .on_hover_text("Render signed spatial deviations relative to each layer's global mean using a CoolWarm diverging colormap")
                        .changed()
                    {
                        self.raster_dirty = true;
                    }
                });

                // Basemap Selector
                let basemaps = self.workers.coordinator.schema().main_viewport.base_maps.clone();
                let basemap_label = basemaps
                    .iter()
                    .find(|b| b.id == self.basemap_id)
                    .map(|b| b.name.as_str())
                    .unwrap_or(&self.basemap_id);
                egui::ComboBox::from_label("Basemap")
                    .selected_text(basemap_label)
                    .show_ui(ui, |ui| {
                        for bm in &basemaps {
                            if ui
                                .selectable_value(&mut self.basemap_id, bm.id.clone(), &bm.name)
                                .changed()
                            {
                                self.raster_dirty = true;
                            }
                        }
                    });

                ui.separator();
                ui.label(egui::RichText::new("Quick Scenario Presets").strong());
                let mut clicked_preset: Option<QuickScenarioPreset> = None;
                ui.horizontal_wrapped(|ui| {
                    for preset in QuickScenarioPreset::ALL {
                        let active = match preset {
                            QuickScenarioPreset::BaselinePresent => {
                                self.temporal_mode_id == "realtime_telemetry"
                                    && self.sim_cfg.scenario == SspScenario::Ssp2_45
                            }
                            QuickScenarioPreset::ModerateSsp245 => {
                                self.temporal_mode_id == "cmip_projections"
                                    && self.sim_cfg.scenario == SspScenario::Ssp2_45
                                    && (self.sim_cfg.sim_year - 2050.0).abs() < 1.0
                            }
                            QuickScenarioPreset::HighEmissionsSsp585 => {
                                self.temporal_mode_id == "cmip_projections"
                                    && self.sim_cfg.scenario == SspScenario::Ssp5_85
                                    && (self.sim_cfg.sim_year - 2100.0).abs() < 1.0
                            }
                        };
                        if ui
                            .selectable_label(active, preset.label())
                            .on_hover_text(preset.tooltip())
                            .clicked()
                        {
                            clicked_preset = Some(preset);
                        }
                    }
                });
                if let Some(p) = clicked_preset {
                    self.apply_quick_preset(p);
                }

                ui.separator();
                ui.label(egui::RichText::new("Camera & Bearing").strong());
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Slider::new(&mut self.camera.bearing_deg, -180.0..=180.0)
                                .suffix("°")
                                .text("Bearing"),
                        )
                        .on_hover_text("Rotate camera bearing (or Shift+Drag on the globe/polar map)")
                        .changed()
                    {
                        self.raster_dirty = true;
                    }
                    if ui.small_button("0°").on_hover_text("Reset bearing to North-up").clicked() {
                        self.camera.bearing_deg = 0.0;
                        self.raster_dirty = true;
                    }
                });

                ui.separator();
                ui.label(egui::RichText::new("Vertical Slice Engine").strong());
                let mut slice_changed = false;
                slice_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.sim_cfg.atmospheric_level_hpa, 10.0..=1000.0)
                            .logarithmic(true)
                            .text("Atm (hPa)"),
                    )
                    .on_hover_text("Atmospheric isobaric pressure level (1000 hPa surface to 10 hPa stratosphere)")
                    .changed();
                slice_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.sim_cfg.ocean_depth_m, 0.0..=6000.0)
                            .text("Ocean (m)"),
                    )
                    .on_hover_text("Subsurface ocean depth slice (0 m sea surface to 6000 m abyssal)")
                    .changed();
                if slice_changed {
                    self.push_sim_update();
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.layer_filter_query)
                            .id(egui::Id::new("terratwin_layer_filter"))
                            .hint_text("Filter 34 layers...")
                            .desired_width(190.0),
                    );
                    if !self.layer_filter_query.is_empty() && ui.small_button("✕").clicked() {
                        self.layer_filter_query.clear();
                    }
                });

                let filter_lc = self.layer_filter_query.trim().to_ascii_lowercase();
                let groups = self.workers.coordinator.schema().layer_groups.clone();
                let mut solo_clicked_id: Option<String> = None;
                let visible_count = self.layer_states.values().filter(|s| s.visible).count();
                let matching_count: usize = groups
                    .iter()
                    .flat_map(|g| g.layers.iter().map(move |l| (g, l)))
                    .filter(|(g, l)| layer_matches_filter(g, l, &filter_lc))
                    .count();
                if !filter_lc.is_empty() {
                    ui.small(format!("Showing {} of 34 layers", matching_count));
                }

                egui::ScrollArea::vertical().show(ui, |ui| {
                    for grp in &groups {
                        let matching_layers: Vec<_> = grp
                            .layers
                            .iter()
                            .filter(|l| layer_matches_filter(grp, l, &filter_lc))
                            .collect();
                        if matching_layers.is_empty() {
                            continue;
                        }
                        let has_vis = matching_layers.iter().any(|l| {
                            self.layer_states
                                .get(&l.id)
                                .map(|s| s.visible)
                                .unwrap_or(false)
                        });
                        let is_open = if !filter_lc.is_empty() {
                            true
                        } else if self.open_visible_groups_only {
                            has_vis
                        } else {
                            grp.default_expanded
                        };
                        egui::CollapsingHeader::new(
                            egui::RichText::new(&grp.group_name).strong(),
                        )
                        .default_open(is_open)
                        .show(ui, |ui| {
                            for lyr in matching_layers {
                                let Some(st) = self.layer_states.get_mut(&lyr.id) else {
                                    continue;
                                };
                                let is_soloed = st.visible && visible_count == 1;
                                ui.group(|ui| {
                                    ui.horizontal(|ui| {
                                        if ui.checkbox(&mut st.visible, &lyr.name).changed() {
                                            self.raster_dirty = true;
                                        }
                                        if ui
                                            .selectable_label(is_soloed, "👁 Solo")
                                            .on_hover_text("Solo: show only this layer (click again to restore)")
                                            .clicked()
                                        {
                                            solo_clicked_id = Some(lyr.id.clone());
                                        }
                                        ui.menu_button("Info", |ui| {
                                            ui.set_max_width(300.0);
                                            ui.strong(&lyr.name);
                                            ui.label(&lyr.description);
                                            ui.separator();
                                            ui.label(format!("Units: {}", lyr.units));
                                            ui.label(format!(
                                                "Resolution: {} | {}",
                                                lyr.spatial_resolution, lyr.temporal_resolution
                                            ));
                                            ui.label(format!("Format: {}", lyr.data_format));
                                            ui.label("Public Sources:");
                                            for src in &lyr.public_sources {
                                                ui.label(format!(" • {}", src));
                                            }
                                        });
                                    });

                                    if st.visible {
                                        ui.horizontal(|ui| {
                                            if ui
                                                .add(
                                                    egui::Slider::new(&mut st.opacity, 0.08..=1.0)
                                                        .text("Opacity"),
                                                )
                                                .changed()
                                            {
                                                self.raster_dirty = true;
                                            }
                                        });
                                        egui::ComboBox::new(format!("pal_{}", lyr.id), "Palette")
                                            .selected_text(st.palette.label())
                                            .show_ui(ui, |ui| {
                                                for pal in ColormapPalette::ALL {
                                                    if ui
                                                        .selectable_value(
                                                            &mut st.palette,
                                                            pal,
                                                            pal.label(),
                                                        )
                                                        .changed()
                                                    {
                                                        self.raster_dirty = true;
                                                    }
                                                }
                                            });
                                    }
                                });
                            }
                        });
                    }
                });
                if let Some(target_id) = solo_clicked_id {
                    let currently_solo = self
                        .layer_states
                        .get(&target_id)
                        .map(|s| s.visible)
                        .unwrap_or(false)
                        && visible_count == 1;
                    if currently_solo {
                        if let Some(prev) = self.solo_restore_vis.take() {
                            for (id, st) in &mut self.layer_states {
                                if let Some(&v) = prev.get(id) {
                                    st.visible = v;
                                }
                            }
                        }
                    } else {
                        let prev: HashMap<String, bool> = self
                            .layer_states
                            .iter()
                            .map(|(k, v)| (k.clone(), v.visible))
                            .collect();
                        self.solo_restore_vis = Some(prev);
                        for (id, st) in &mut self.layer_states {
                            st.visible = *id == target_id;
                        }
                    }
                    self.raster_dirty = true;
                }
            });
    }

    fn render_right_analytics_drawer(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("telemetry_analytics_dock")
            .resizable(true)
            .default_width(365.0)
            .show_animated(ctx, self.right_drawer_open, |ui| {
                let (plat, plon) = self.pinned_coord;
                ui.heading("Telemetry Analytics Dock");
                ui.label(format!(
                    "Target Pin: {:.4}° N, {:.4}° E | Elev: {:.0} m",
                    plat,
                    plon,
                    synthetic_planetary_elevation_m(plat, plon)
                ));

                // Quick Observatory Presets
                ui.horizontal_wrapped(|ui| {
                    for &(name, la, lo, _) in REFERENCE_OBSERVATORIES {
                        if ui.small_button(name).clicked() {
                            self.pinned_coord = (la, lo);
                            self.camera.center_lat = la;
                            self.camera.center_lon = lo;
                            self.coord_search_buf = format!("{:.4}, {:.4}", la, lo);
                            self.raster_dirty = true;
                        }
                    }
                });

                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // 1. Active Layer Point Readouts
                    ui.strong("Active Layer Point Telemetry");
                    if let Some(ref snap) = self.cached_snapshot {
                        for (_, lyr) in self.workers.coordinator.schema().all_layers() {
                            let vis = self
                                .layer_states
                                .get(&lyr.id)
                                .map(|s| s.visible)
                                .unwrap_or(false);
                            if vis {
                                if let Some(g) = snap.grids.get(&lyr.id) {
                                    let val = sample_grid_bilinear(
                                        &g.values, g.width, g.height, plat, plon,
                                    );
                                    if val.is_finite() {
                                        ui.label(format!(
                                            "• {}: {:.2} {}",
                                            lyr.name, val, lyr.units
                                        ));
                                    }
                                }
                            }
                        }
                    }

                    ui.separator();
                    let schema_ref = self.workers.coordinator.schema();
                    let grids_opt = self.cached_snapshot.as_ref().map(|s| &s.grids);
                    let elev_m = synthetic_planetary_elevation_m(plat, plon);

                    // 2. Skew-T / Log-P Vertical Atmospheric Profile (1000 -> 10 hPa)
                    ui.strong("Atmospheric Vertical Sounding (1000–10 hPa)");
                    let sounding =
                        compute_vertical_sounding_from_grids(schema_ref, grids_opt, plat, plon);
                    draw_sounding_canvas(ui, &sounding);

                    ui.separator();
                    // 3. Ocean Water Column CTD (elev < 0) OR Land Cryosphere & Permafrost (elev >= 0)
                    if elev_m < 0.0 {
                        ui.strong("Ocean Water-Column CTD Profile (0–6000 m)");
                        let ctd =
                            compute_ocean_ctd_profile_from_grids(schema_ref, grids_opt, plat, plon);
                        draw_ctd_canvas(ui, &ctd);
                    } else {
                        ui.strong("Land Cryosphere & Permafrost Profile (0–20 m)");
                        let snap_doy = self
                            .cached_snapshot
                            .as_ref()
                            .map(|s| s.config.day_of_year)
                            .unwrap_or(self.sim_cfg.day_of_year);
                        let cryo = compute_land_cryosphere_profile_from_grids_at_doy(
                            grids_opt,
                            plat,
                            plon,
                            elev_m,
                            self.sim_cfg.scenario,
                            self.sim_cfg.sim_year,
                            snap_doy,
                        );
                        let mut summary_parts = vec![format!("T2m {:.1} °C", cryo.surface_t2m_c)];
                        if let Some(thaw) = cryo.active_layer_m {
                            summary_parts.push(format!("Thaw {:.2} m", thaw));
                        }
                        if let Some(smb) = cryo.ice_sheet_smb_cm {
                            summary_parts.push(format!("SMB {:+.1} cm", smb));
                        }
                        if let Some(vel) = cryo.glacier_vel_m_yr {
                            summary_parts.push(format!("Ice Vel {:.0} m/yr", vel));
                        }
                        if let Some(sm) = cryo.soil_moisture {
                            summary_parts.push(format!("Soil {:.2} m³/m³", sm));
                        }
                        ui.small(summary_parts.join(" | "));
                        draw_land_cryosphere_canvas(ui, &cryo);
                    }

                    ui.separator();
                    // 4. Regional CO2 & Warming Projection + CMIP6 Ensemble Spread Boxplot
                    ui.strong("CMIP6 1950–2100 Trajectory & Ensemble Spread");
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.cmip6_metric,
                            Cmip6PlotMetric::Co2Ppm,
                            "CO₂ (ppm)",
                        );
                        ui.selectable_value(
                            &mut self.cmip6_metric,
                            Cmip6PlotMetric::DeltaTC,
                            "Regional ΔT (°C)",
                        );
                        if self.cmip6_metric == Cmip6PlotMetric::DeltaTC {
                            egui::ComboBox::new("cmip6_region_combo", "")
                                .selected_text(self.cmip6_region.label())
                                .width(132.0)
                                .show_ui(ui, |ui| {
                                    for reg in ClimateRegion::ALL {
                                        ui.selectable_value(
                                            &mut self.cmip6_region,
                                            reg,
                                            reg.label(),
                                        );
                                    }
                                });
                        }
                    });
                    let live_snapshot_val = self.cached_snapshot.as_ref().and_then(|snap| {
                        match self.cmip6_metric {
                            Cmip6PlotMetric::Co2Ppm => Some(snap.readouts.mauna_loa_co2_ppm),
                            Cmip6PlotMetric::DeltaTC => snap
                                .grids
                                .get("cmip6_downscaled_ensemble")
                                .and_then(|g| {
                                    snapshot_regional_delta_t_c(g, self.cmip6_region, plat, plon)
                                }),
                        }
                    });
                    draw_cmip6_projection_canvas(
                        ui,
                        self.sim_cfg.scenario,
                        self.sim_cfg.sim_year,
                        self.cmip6_metric,
                        self.cmip6_region,
                        plat,
                        live_snapshot_val,
                    );

                    ui.separator();
                    ui.strong("SSP Pathway Delta Comparator (1-2.6 vs 2-4.5 vs 5-8.5)");
                    ui.horizontal_wrapped(|ui| {
                        ui.small("Year:");
                        if ui.selectable_label((self.ssp_compare_year - 2050.0).abs() < 0.25, "2050").clicked() {
                            self.ssp_compare_year = 2050.0;
                        }
                        if ui.selectable_label((self.ssp_compare_year - 2100.0).abs() < 0.25, "2100").clicked() {
                            self.ssp_compare_year = 2100.0;
                        }
                        if ui.small_button("Sync Live").clicked() {
                            self.ssp_compare_year = self.sim_cfg.sim_year.clamp(1950.0, 2100.0);
                        }
                    });
                    let ssp_report = compare_ssp_pathways(self.ssp_compare_year);
                    egui::Grid::new("terratwin_ssp_delta_grid")
                        .striped(true)
                        .spacing([8.0, 3.0])
                        .show(ui, |ui| {
                            ui.small(egui::RichText::new("SSP").strong());
                            ui.small(egui::RichText::new("CO2").strong());
                            ui.small(egui::RichText::new("ΔT").strong());
                            ui.small(egui::RichText::new("SLR").strong());
                            ui.small(egui::RichText::new("pH").strong());
                            ui.end_row();
                            for row in &ssp_report.rows {
                                ui.small(egui::RichText::new(&row.scenario).monospace());
                                ui.small(format!("{:.0} ({:+.0})", row.co2_ppm, row.delta_co2_vs_ssp126_ppm));
                                ui.small(format!("{:+.2}° ({:+.2})", row.transient_temp_anomaly_c, row.delta_temp_vs_ssp126_c));
                                ui.small(format!("{:.1}cm ({:+.1})", row.thermosteric_slr_cm, row.delta_slr_vs_ssp126_cm));
                                ui.small(format!("{:.3}", row.surface_ocean_ph));
                                ui.end_row();
                            }
                        });
                    let mut export_ssp_fmt: Option<bool> = None;
                    ui.horizontal(|ui| {
                        if ui.small_button("Export SSP JSON...").clicked() {
                            export_ssp_fmt = Some(false);
                        }
                        if ui.small_button("Export SSP CSV...").clicked() {
                            export_ssp_fmt = Some(true);
                        }
                    });

                    if self.transect_points.len() == 2 {
                        ui.separator();
                        let (a_la, a_lon) = self.transect_points[0];
                        let (b_la, b_lon) = self.transect_points[1];
                        let dist_km = haversine_km(a_la, a_lon, b_la, b_lon);
                        ui.strong(format!("Cross-Section Transect A-B ({:.0} km)", dist_km));
                        draw_transect_profile(
                            ui,
                            schema_ref,
                            &self.cached_snapshot,
                            &self.layer_states,
                            (a_la, a_lon),
                            (b_la, b_lon),
                        );
                    }

                    ui.separator();
                    ui.strong("Zarr v3 / JSON Subset Exporter");
                    let first_vis_pair = schema_ref
                        .all_layers()
                        .into_iter()
                        .find(|(_, l)| {
                            self.layer_states
                                .get(&l.id)
                                .map(|s| s.visible)
                                .unwrap_or(false)
                        })
                        .map(|(g, l)| {
                            let lvl = if g.group_id == "physical_biogeochemical_oceanography" {
                                self.sim_cfg.ocean_depth_m
                            } else {
                                self.sim_cfg.atmospheric_level_hpa
                            };
                            (l.id.clone(), lvl)
                        })
                        .unwrap_or_else(|| {
                            (
                                "cmip6_downscaled_ensemble".to_string(),
                                self.sim_cfg.atmospheric_level_hpa,
                            )
                        });
                    if let Some(as_csv) = export_ssp_fmt {
                        self.export_ssp_delta_file(as_csv);
                    }
                    ui.horizontal_wrapped(|ui| {
                        if ui.button("Export Zarr v3 Store...").clicked() {
                            if let Some(out_dir) = rfd::FileDialog::new()
                                .set_title("Export Zarr v3 Climate Subset Store")
                                .set_file_name(format!("subset_{}.zarr", first_vis_pair.0))
                                .save_file()
                            {
                                match self.workers.coordinator.db().export_zarr_v3_store(
                                    &first_vis_pair.0,
                                    self.sim_cfg.scenario,
                                    first_vis_pair.1,
                                    (plat - 20.0).max(-90.0),
                                    (plat + 20.0).min(90.0),
                                    (plon - 25.0).max(-180.0),
                                    (plon + 25.0).min(180.0),
                                    (18, 36),
                                    &out_dir,
                                ) {
                                    Ok(sum) => {
                                        self.export_status_msg = format!(
                                            "Zarr v3 ({} chunks, {} cells) -> {}",
                                            sum.chunks_written, sum.cells_exported, sum.store_dir
                                        );
                                    }
                                    Err(e) => {
                                        self.export_status_msg =
                                            format!("Zarr v3 export failed: {}", e);
                                    }
                                }
                            }
                        }

                        if ui.button("Export JSON Subset...").clicked() {
                            if let Some(out_file) = rfd::FileDialog::new()
                                .set_title("Export JSON Climate Subset")
                                .add_filter("JSON Subset", &["json"])
                                .set_file_name(format!("subset_{}.json", first_vis_pair.0))
                                .save_file()
                            {
                                match self.workers.coordinator.db().export_layer_subset_json(
                                    &first_vis_pair.0,
                                    self.sim_cfg.scenario,
                                    first_vis_pair.1,
                                    (plat - 20.0).max(-90.0),
                                    (plat + 20.0).min(90.0),
                                    (plon - 25.0).max(-180.0),
                                    (plon + 25.0).min(180.0),
                                    &out_file,
                                ) {
                                    Ok(cells) => {
                                        self.export_status_msg = format!(
                                            "JSON subset ({} cells) -> {}",
                                            cells,
                                            out_file.display()
                                        );
                                    }
                                    Err(e) => {
                                        self.export_status_msg =
                                            format!("JSON export failed: {}", e);
                                    }
                                }
                            }
                        }

                        if ui.button("Open JSON / Schema...").clicked() {
                            if let Some(in_file) = rfd::FileDialog::new()
                                .set_title("Inspect Climate JSON / Schema")
                                .add_filter("JSON", &["json"])
                                .pick_file()
                            {
                                match std::fs::read_to_string(&in_file) {
                                    Ok(raw) => {
                                        match serde_json::from_str::<serde_json::Value>(&raw) {
                                            Ok(val) => {
                                                if let Some(app_ver) = val
                                                    .get("application")
                                                    .and_then(|a| a.get("version"))
                                                    .and_then(|v| v.as_str())
                                                {
                                                    let n_grp = val
                                                        .get("layerGroups")
                                                        .and_then(|g| g.as_array())
                                                        .map(|a| a.len())
                                                        .unwrap_or(0);
                                                    self.export_status_msg = format!(
                                                        "Schema v{} ({} groups) loaded from {}",
                                                        app_ver,
                                                        n_grp,
                                                        in_file.display()
                                                    );
                                                } else if let Some(cnt) =
                                                    val.get("cell_count").and_then(|c| c.as_u64())
                                                {
                                                    let lyr = val
                                                        .get("layer_id")
                                                        .and_then(|l| l.as_str())
                                                        .unwrap_or("unknown");
                                                    self.export_status_msg = format!(
                                                        "Subset JSON ({}, {} cells) <- {}",
                                                        lyr,
                                                        cnt,
                                                        in_file.display()
                                                    );
                                                } else {
                                                    self.export_status_msg = format!(
                                                        "Loaded valid JSON ({} bytes) <- {}",
                                                        raw.len(),
                                                        in_file.display()
                                                    );
                                                }
                                            }
                                            Err(e) => {
                                                self.export_status_msg =
                                                    format!("Invalid JSON file: {}", e);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        self.export_status_msg =
                                            format!("Failed reading file: {}", e);
                                    }
                                }
                            }
                        }
                    });
                    if !self.export_status_msg.is_empty() {
                        ui.small(&self.export_status_msg);
                    }
                });
            });
    }

    fn render_bottom_dock(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("temporal_dimension_controller")
            .frame(
                egui::Frame::none()
                    .fill(Color32::from_rgb(13, 19, 31))
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .button(if self.playing_timeline { "Pause" } else { "Play" })
                        .on_hover_text("Play or pause the 4D replay / climate timeline animation (Space)")
                        .clicked()
                    {
                        self.playing_timeline = !self.playing_timeline;
                    }

                    let modes = self.workers.coordinator.schema().ui_layout.bottom_dock.modes.clone();
                    let active_mode = modes
                        .iter()
                        .find(|m| m.id == self.temporal_mode_id);
                    let mode_label = active_mode
                        .map(|m| m.label.as_str())
                        .unwrap_or(&self.temporal_mode_id);
                    let mode_step = active_mode.map(|m| m.step.as_str()).unwrap_or("10m");
                    let prev_mode = self.temporal_mode_id.clone();
                    egui::ComboBox::from_label("Mode")
                        .selected_text(mode_label)
                        .show_ui(ui, |ui| {
                            for m in &modes {
                                ui.selectable_value(
                                    &mut self.temporal_mode_id,
                                    m.id.clone(),
                                    &m.label,
                                );
                            }
                        });
                    if self.temporal_mode_id != prev_mode {
                        let now = chrono::Utc::now();
                        let present_year = now.year() as f64 + (now.ordinal() as f64 / 365.25);
                        match self.temporal_mode_id.as_str() {
                            "realtime_telemetry" => {
                                self.sim_cfg.sim_year = present_year;
                                self.sim_cfg.day_of_year = now.ordinal() as f64;
                                self.sim_cfg.hour_utc =
                                    now.hour() as f64 + (now.minute() as f64 / 60.0);
                                self.sim_cfg.time_step_secs = 600.0;
                            }
                            "historical_reanalysis" => {
                                self.sim_cfg.time_step_secs = 3600.0;
                                self.sim_cfg.sim_year =
                                    self.sim_cfg.sim_year.clamp(1950.0, present_year);
                            }
                            "cmip_projections" => {
                                self.sim_cfg.time_step_secs = 3600.0;
                                if self.sim_cfg.sim_year < 2026.0 {
                                    self.sim_cfg.sim_year = 2050.0;
                                } else {
                                    self.sim_cfg.sim_year =
                                        self.sim_cfg.sim_year.clamp(2026.0, 2100.0);
                                }
                            }
                            _ => {}
                        }
                        self.playback_index = None;
                        self.push_sim_update();
                    }
                    ui.small(format!("[Step: {}]", mode_step));

                    // Live vs 4D SQLite Replay Scrubber
                    let is_replay = self.playback_index.is_some();
                    if ui
                        .selectable_label(!is_replay, "Live")
                        .clicked()
                    {
                        self.playback_index = None;
                        self.playing_timeline = false;
                        self.raster_dirty = true;
                    }
                    if ui
                        .selectable_label(is_replay, "4D Replay")
                        .clicked()
                    {
                        self.refresh_playback_timestamps();
                        if !self.playback_timestamps.is_empty() {
                            let last_i = self.playback_timestamps.len() - 1;
                            self.load_replay_snapshot_at_idx(last_i);
                        }
                    }

                    if let Some(cur_idx) = self.playback_index {
                        let total = self.playback_timestamps.len().max(1);
                        if ui.small_button("◀").clicked() && cur_idx > 0 {
                            self.load_replay_snapshot_at_idx(cur_idx - 1);
                        }
                        let mut slider_idx = cur_idx;
                        if ui
                            .add(
                                egui::Slider::new(&mut slider_idx, 0..=(total - 1))
                                    .text(format!("Tick {}/{}", cur_idx + 1, total)),
                            )
                            .changed()
                        {
                            self.load_replay_snapshot_at_idx(slider_idx);
                        }
                        if ui.small_button("▶").clicked() && cur_idx + 1 < total {
                            self.load_replay_snapshot_at_idx(cur_idx + 1);
                        }
                        if let Some(ts) = self.playback_timestamps.get(cur_idx) {
                            ui.monospace(ts);
                        }
                    } else {
                        let now = chrono::Utc::now();
                        let present_year = now.year() as f64 + (now.ordinal() as f64 / 365.25);
                        let mut time_changed = false;
                        match self.temporal_mode_id.as_str() {
                            "realtime_telemetry" => {
                                let lo_yr = present_year - 7.0 / 365.25;
                                let hi_yr = present_year + 10.0 / 365.25;
                                self.sim_cfg.sim_year = self.sim_cfg.sim_year.clamp(lo_yr, hi_yr);
                                time_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut self.sim_cfg.sim_year, lo_yr..=hi_yr)
                                            .text("Window"),
                                    )
                                    .changed();
                                time_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut self.sim_cfg.hour_utc, 0.0..=23.9)
                                            .text("UTC"),
                                    )
                                    .changed();
                            }
                            "historical_reanalysis" => {
                                time_changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.sim_cfg.sim_year,
                                            1950.0..=present_year,
                                        )
                                        .text("Year"),
                                    )
                                    .changed();
                                time_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut self.sim_cfg.day_of_year, 1.0..=365.0)
                                            .text("Day"),
                                    )
                                    .changed();
                            }
                            _ => {
                                time_changed |= ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.sim_cfg.sim_year,
                                            2026.0..=2100.0,
                                        )
                                        .text("Year"),
                                    )
                                    .changed();
                                time_changed |= ui
                                    .add(
                                        egui::Slider::new(&mut self.sim_cfg.day_of_year, 1.0..=365.0)
                                            .text("Day"),
                                    )
                                    .changed();
                            }
                        }
                        if time_changed {
                            self.push_sim_update();
                        }
                    }

                    ui.separator();
                    if let Some(ref snap) = self.cached_snapshot {
                        ui.small(format!(
                            "Rayon 34-Layer Step: {} ms | SQLite Grids: {} | Raw Obs: {} | {}",
                            snap.last_step_ms,
                            snap.db_stats.simulated_grid_count,
                            snap.db_stats.raw_observation_count,
                            snap.last_ingest_status
                        ));
                    } else {
                        ui.small(self.workers.coordinator.current_ingest_status());
                    }
                    if !self.export_status_msg.is_empty() {
                        ui.separator();
                        ui.small(
                            egui::RichText::new(&self.export_status_msg)
                                .color(Color32::from_rgb(245, 200, 95)),
                        );
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.small("Copyright CRV Science Network");
                    });
                });
            });
    }

    fn render_central_map(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::from_rgb(6, 10, 18)))
            .show(ctx, |ui| {
                let avail = ui.available_size();
                let (response, painter) = ui.allocate_painter(avail, Sense::click_and_drag());
                let rect = response.rect;
                let aspect = (rect.width() / rect.height().max(1.0)) as f64;

                // Pan / Globe Rotate (or Shift+Drag Bearing Rotate) on drag
                if response.dragged() {
                    let delta = response.drag_delta();
                    let shift_held = ui.input(|i| i.modifiers.shift);
                    if shift_held {
                        self.camera.bearing_deg =
                            ((self.camera.bearing_deg + delta.x as f64 * 0.45 + 180.0)
                                .rem_euclid(360.0))
                                - 180.0;
                    } else {
                        let deg_per_px = match self.projection {
                            MapProjectionKind::Equirectangular2d => {
                                let (_, scale_y) = equirectangular_scales(aspect);
                                180.0
                                    / (rect.height().max(120.0) as f64 * self.camera.zoom * scale_y)
                            }
                            _ => 180.0 / (rect.height().max(120.0) as f64 * self.camera.zoom),
                        };
                        let b = if self.projection == MapProjectionKind::Globe3dWgs84 {
                            self.camera.bearing_deg.to_radians()
                        } else {
                            0.0
                        };
                        let sx = delta.x as f64;
                        let sy = -(delta.y as f64);
                        let raw_x = sx * b.cos() + sy * b.sin();
                        let raw_y = -sx * b.sin() + sy * b.cos();
                        self.camera.center_lon =
                            ((self.camera.center_lon - raw_x * deg_per_px + 540.0)
                                .rem_euclid(360.0))
                                - 180.0;
                        self.camera.center_lat =
                            (self.camera.center_lat - raw_y * deg_per_px).clamp(-88.0, 88.0);
                    }
                    self.raster_dirty = true;
                }

                // Cursor-centered zoom on mouse wheel
                if response.hovered() {
                    let scroll = ui.input(|i| i.raw_scroll_delta.y);
                    if scroll.abs() > 0.1 {
                        let factor = (1.0 + scroll as f64 * 0.0018).clamp(0.82, 1.22);
                        let cursor_ndc = response.hover_pos().map(|pos| {
                            (
                                2.0 * ((pos.x - rect.left()) / rect.width()) as f64 - 1.0,
                                1.0 - 2.0 * ((pos.y - rect.top()) / rect.height()) as f64,
                            )
                        });
                        zoom_camera_toward_ndc(
                            self.projection,
                            &mut self.camera,
                            factor,
                            cursor_ndc,
                            aspect,
                        );
                        self.raster_dirty = true;
                    }
                }

                // Click to pin coordinate or define A->B transect
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let ndc_x = 2.0 * ((pos.x - rect.left()) / rect.width()) as f64 - 1.0;
                        let ndc_y = 1.0 - 2.0 * ((pos.y - rect.top()) / rect.height()) as f64;
                        if let Some((lat, lon)) = unproject_ndc_to_latlon(
                            self.projection,
                            &self.camera,
                            ndc_x,
                            ndc_y,
                            aspect,
                        ) {
                            self.pinned_coord = (lat, lon);
                            self.coord_search_buf = format!("{:.4}, {:.4}", lat, lon);
                            if self.transect_mode {
                                if self.transect_points.len() >= 2 {
                                    self.transect_points.clear();
                                }
                                self.transect_points.push((lat, lon));
                            }
                        }
                    }
                }

                // Re-rasterize HiDPI-responsive composite texture when dirty or resized
                let ppp = ctx.pixels_per_point().clamp(1.0, 2.5);
                let tex_w = ((rect.width() * ppp * 0.85) as usize).clamp(480, 1920);
                let tex_h = ((rect.height() * ppp * 0.85) as usize).clamp(300, 1280);
                if (tex_w, tex_h) != self.last_tex_size {
                    self.last_tex_size = (tex_w, tex_h);
                    self.raster_dirty = true;
                }
                if self.raster_dirty || self.map_texture.is_none() {
                    self.rebuild_composite_texture(ctx, tex_w, tex_h);
                }

                if let Some(ref tex) = self.map_texture {
                    painter.image(
                        tex.id(),
                        rect,
                        Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }

                let ndc_to_screen = |nx: f32, ny: f32| -> Pos2 {
                    egui::pos2(
                        rect.left() + (nx * 0.5 + 0.5) * rect.width(),
                        rect.top() + (0.5 - ny * 0.5) * rect.height(),
                    )
                };

                // 3D Globe atmosphere rim halo or 2:1 Equirectangular plate carrée frame
                if self.projection == MapProjectionKind::Globe3dWgs84 {
                    let r_px = 0.86 * (self.camera.zoom as f32) * (rect.height() * 0.5);
                    painter.circle_stroke(
                        rect.center(),
                        r_px,
                        Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(90, 195, 255, 130)),
                    );
                } else if self.projection == MapProjectionKind::Equirectangular2d {
                    let (scale_x, scale_y) = equirectangular_scales(aspect);
                    let nx_max = (self.camera.zoom * scale_x) as f32;
                    let ny_top = (((90.0 - self.camera.center_lat) / 90.0)
                        * self.camera.zoom
                        * scale_y) as f32;
                    let ny_bot = (((-90.0 - self.camera.center_lat) / 90.0)
                        * self.camera.zoom
                        * scale_y) as f32;
                    let tl = ndc_to_screen(-nx_max, ny_top);
                    let br = ndc_to_screen(nx_max, ny_bot);
                    let frame_rect = Rect::from_min_max(tl, br).intersect(rect);
                    if frame_rect.is_positive() {
                        painter.rect_stroke(
                            frame_rect,
                            0.0,
                            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(90, 145, 200, 110)),
                        );
                    }
                }

                // Graticule (30 deg parallels & meridians + degree labels)
                let grat_stroke =
                    Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(180, 210, 240, 38));
                let grat_label_col = Color32::from_rgba_unmultiplied(185, 214, 244, 165);
                let mut placed_grat_labels: Vec<Pos2> = Vec::new();
                let label_bounds = rect.shrink(22.0);
                for lat_g in (-60..=60).step_by(30) {
                    let mut prev: Option<Pos2> = None;
                    for lon_g in (-180..=180).step_by(6) {
                        if let Some((nx, ny)) = project_latlon_to_ndc(
                            self.projection,
                            &self.camera,
                            lat_g as f64,
                            lon_g as f64,
                            0.0,
                            aspect,
                        ) {
                            let pt = ndc_to_screen(nx, ny);
                            if let Some(p0) = prev {
                                if p0.distance(pt) < rect.width() * 0.35 {
                                    painter.line_segment([p0, pt], grat_stroke);
                                }
                            }
                            prev = Some(pt);
                        } else {
                            prev = None;
                        }
                    }

                    let candidate_pt = if self.projection == MapProjectionKind::Equirectangular2d {
                        let (_, scale_y) = equirectangular_scales(aspect);
                        let ny = (((lat_g as f64 - self.camera.center_lat) / 90.0)
                            * self.camera.zoom
                            * scale_y) as f32;
                        let (scale_x, _) = equirectangular_scales(aspect);
                        let left_x = ndc_to_screen(-(self.camera.zoom * scale_x) as f32, ny)
                            .x
                            .max(rect.left() + 8.0)
                            + 4.0;
                        let y = ndc_to_screen(0.0, ny).y;
                        let pt = egui::pos2(left_x, y);
                        if label_bounds.contains(pt) { Some(pt) } else { None }
                    } else {
                        [
                            self.camera.center_lon - 48.0,
                            self.camera.center_lon + 48.0,
                            self.camera.center_lon,
                            0.0,
                            -60.0,
                            60.0,
                        ]
                        .into_iter()
                        .find_map(|lon_try| {
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                lat_g as f64,
                                lon_try,
                                0.0,
                                aspect,
                            )
                            .map(|(nx, ny)| ndc_to_screen(nx, ny))
                            .filter(|pt| label_bounds.contains(*pt))
                        })
                    };
                    if let Some(pt) = candidate_pt {
                        if !placed_grat_labels.iter().any(|p| p.distance(pt) < 34.0) {
                            placed_grat_labels.push(pt);
                            painter.text(
                                pt + egui::vec2(2.0, -2.0),
                                Align2::LEFT_BOTTOM,
                                format_graticule_lat(lat_g),
                                FontId::monospace(9.0),
                                grat_label_col,
                            );
                        }
                    }
                }
                for lon_g in (-180..180).step_by(30) {
                    let mut prev: Option<Pos2> = None;
                    for lat_g in (-84..=84).step_by(6) {
                        if let Some((nx, ny)) = project_latlon_to_ndc(
                            self.projection,
                            &self.camera,
                            lat_g as f64,
                            lon_g as f64,
                            0.0,
                            aspect,
                        ) {
                            let pt = ndc_to_screen(nx, ny);
                            if let Some(p0) = prev {
                                if p0.distance(pt) < rect.width() * 0.35 {
                                    painter.line_segment([p0, pt], grat_stroke);
                                }
                            }
                            prev = Some(pt);
                        } else {
                            prev = None;
                        }
                    }

                    let cand_lats: &[f64] = match self.projection {
                        MapProjectionKind::ArcticPolarStereo => &[60.0, 30.0],
                        MapProjectionKind::AntarcticPolarStereo => &[-60.0, -30.0],
                        MapProjectionKind::Equirectangular2d => &[self.camera.center_lat.clamp(-60.0, 60.0), 0.0],
                        MapProjectionKind::Globe3dWgs84 => &[0.0, 30.0, -30.0, 60.0, -60.0],
                    };
                    let candidate_pt = if self.projection == MapProjectionKind::Equirectangular2d {
                        let (_, scale_y) = equirectangular_scales(aspect);
                        let bot_ny = (((-90.0 - self.camera.center_lat) / 90.0)
                            * self.camera.zoom
                            * scale_y) as f32;
                        let bot_y = ndc_to_screen(0.0, bot_ny).y.min(rect.bottom() - 42.0) - 4.0;
                        project_latlon_to_ndc(
                            self.projection,
                            &self.camera,
                            cand_lats[0],
                            lon_g as f64,
                            0.0,
                            aspect,
                        )
                        .map(|(nx, _)| egui::pos2(ndc_to_screen(nx, 0.0).x, bot_y))
                        .filter(|pt| label_bounds.contains(*pt))
                    } else {
                        cand_lats.iter().find_map(|&lat_try| {
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                lat_try,
                                lon_g as f64,
                                0.0,
                                aspect,
                            )
                            .map(|(nx, ny)| ndc_to_screen(nx, ny))
                            .filter(|pt| label_bounds.contains(*pt))
                        })
                    };
                    if let Some(pt) = candidate_pt {
                        if !placed_grat_labels.iter().any(|p| p.distance(pt) < 38.0) {
                            placed_grat_labels.push(pt);
                            painter.text(
                                pt + egui::vec2(2.0, -2.0),
                                Align2::LEFT_BOTTOM,
                                format_graticule_lon(lon_g),
                                FontId::monospace(9.0),
                                grat_label_col,
                            );
                        }
                    }
                }

                // Marching-Squares 4 hPa MSLP Isobars + hPa Labels & WMO Cold/Warm Front Pips
                if self
                    .layer_states
                    .get("synoptic_fronts_isobars")
                    .map(|s| s.visible)
                    .unwrap_or(false)
                {
                    let iso_thin =
                        Stroke::new(0.85_f32, Color32::from_rgba_unmultiplied(125, 205, 240, 112));
                    let iso_major =
                        Stroke::new(1.15_f32, Color32::from_rgba_unmultiplied(168, 228, 255, 160));
                    let extrema_pts: Vec<Pos2> = self
                        .cached_extrema
                        .iter()
                        .filter_map(|ext| {
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                ext.lat,
                                ext.lon,
                                0.0,
                                aspect,
                            )
                            .map(|(nx, ny)| ndc_to_screen(nx, ny))
                        })
                        .collect();
                    let mut placed_iso_labels: Vec<(Pos2, f32)> = Vec::new();
                    let iso_label_bounds = rect.shrink(30.0);

                    for seg in &self.cached_isobars {
                        if let (Some((nx1, ny1)), Some((nx2, ny2))) = (
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                seg.lat1,
                                seg.lon1,
                                0.0,
                                aspect,
                            ),
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                seg.lat2,
                                seg.lon2,
                                0.0,
                                aspect,
                            ),
                        ) {
                            let p1 = ndc_to_screen(nx1, ny1);
                            let p2 = ndc_to_screen(nx2, ny2);
                            let seg_len = p1.distance(p2);
                            if seg_len < rect.width() * 0.25 {
                                let is_major = ((seg.level - 1012.0).abs() < 0.5)
                                    || ((seg.level - 1000.0).abs() < 0.5);
                                painter.line_segment(
                                    [p1, p2],
                                    if is_major { iso_major } else { iso_thin },
                                );
                                if seg_len >= 3.0 {
                                    let mid = egui::pos2(0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y));
                                    let clear_of_extrema =
                                        !extrema_pts.iter().any(|ep| ep.distance(mid) < 30.0);
                                    let clear_same_level = !placed_iso_labels.iter().any(|(lp, lv)| {
                                        (*lv - seg.level).abs() < 0.5 && lp.distance(mid) < 160.0
                                    });
                                    let clear_any = !placed_iso_labels.iter().any(|(lp, _)| {
                                        (lp.x - mid.x).abs() < 48.0 && (lp.y - mid.y).abs() < 18.0
                                    });
                                    if iso_label_bounds.contains(mid)
                                        && clear_of_extrema
                                        && clear_same_level
                                        && clear_any
                                    {
                                        placed_iso_labels.push((mid, seg.level));
                                        let bg = Rect::from_center_size(mid, egui::vec2(44.0, 12.0));
                                        painter.rect_filled(
                                            bg,
                                            2.0,
                                            Color32::from_rgba_unmultiplied(10, 18, 30, 175),
                                        );
                                        painter.text(
                                            mid,
                                            Align2::CENTER_CENTER,
                                            format!("{:.0} hPa", seg.level),
                                            FontId::monospace(8.5),
                                            Color32::from_rgba_unmultiplied(178, 232, 255, 225),
                                        );
                                    }
                                }
                            }
                        }
                    }

                    let cold_stroke =
                        Stroke::new(1.8_f32, Color32::from_rgba_unmultiplied(70, 175, 255, 215));
                    let warm_stroke =
                        Stroke::new(1.8_f32, Color32::from_rgba_unmultiplied(255, 90, 75, 215));
                    let mut cold_pips: Vec<Pos2> = Vec::new();
                    let mut warm_pips: Vec<Pos2> = Vec::new();
                    for fseg in &self.cached_fronts {
                        if let (Some((nx1, ny1)), Some((nx2, ny2))) = (
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                fseg.lat1,
                                fseg.lon1,
                                0.0,
                                aspect,
                            ),
                            project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                fseg.lat2,
                                fseg.lon2,
                                0.0,
                                aspect,
                            ),
                        ) {
                            let p1 = ndc_to_screen(nx1, ny1);
                            let p2 = ndc_to_screen(nx2, ny2);
                            let seg_len = p1.distance(p2);
                            if seg_len < rect.width() * 0.2 {
                                painter.line_segment(
                                    [p1, p2],
                                    if fseg.is_cold_front {
                                        cold_stroke
                                    } else {
                                        warm_stroke
                                    },
                                );
                                let mid = egui::pos2(0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y));
                                let pip_list = if fseg.is_cold_front {
                                    &mut cold_pips
                                } else {
                                    &mut warm_pips
                                };
                                if seg_len >= 3.5 && !pip_list.iter().any(|p| p.distance(mid) < 14.0) {
                                    pip_list.push(mid);
                                    draw_wmo_front_pip(&painter, p1, p2, fseg.is_cold_front);
                                }
                            }
                        }
                    }

                    for ext in &self.cached_extrema {
                        if let Some((nx, ny)) = project_latlon_to_ndc(
                            self.projection,
                            &self.camera,
                            ext.lat,
                            ext.lon,
                            0.0,
                            aspect,
                        ) {
                            let pt = ndc_to_screen(nx, ny);
                            let col = if ext.is_low {
                                Color32::from_rgb(255, 95, 95)
                            } else {
                                Color32::from_rgb(105, 195, 255)
                            };
                            let letter = if ext.is_low { "L" } else { "H" };
                            painter.text(
                                pt,
                                Align2::CENTER_CENTER,
                                letter,
                                FontId::proportional(14.5),
                                col,
                            );
                            painter.text(
                                egui::pos2(pt.x, pt.y + 11.0),
                                Align2::CENTER_TOP,
                                format!("{:.0}", ext.pressure_hpa),
                                FontId::monospace(9.5),
                                Color32::from_rgb(235, 242, 252),
                            );
                        }
                    }
                }

                // Teleconnection Basin Bounding Polygons
                if self
                    .layer_states
                    .get("teleconnection_oscillation_masks")
                    .map(|s| s.visible)
                    .unwrap_or(false)
                {
                    let mask_stroke = Stroke::new(1.6_f32, Color32::from_rgb(255, 215, 85));
                    for &(_, label, min_la, max_la, min_lo, max_lo) in TELECONNECTION_BASINS {
                        let corners = [
                            (min_la, min_lo),
                            (min_la, max_lo),
                            (max_la, max_lo),
                            (max_la, min_lo),
                            (min_la, min_lo),
                        ];
                        let mut prev: Option<Pos2> = None;
                        for &(la, lo) in &corners {
                            if let Some((nx, ny)) = project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                la,
                                lo,
                                0.0,
                                aspect,
                            ) {
                                let pt = ndc_to_screen(nx, ny);
                                if let Some(p0) = prev {
                                    painter.line_segment([p0, pt], mask_stroke);
                                }
                                prev = Some(pt);
                            }
                        }
                        if let Some((nx, ny)) = project_latlon_to_ndc(
                            self.projection,
                            &self.camera,
                            0.5 * (min_la + max_la),
                            0.5 * (min_lo + max_lo),
                            0.0,
                            aspect,
                        ) {
                            painter.text(
                                ndc_to_screen(nx, ny),
                                Align2::CENTER_CENTER,
                                label,
                                FontId::monospace(10.5),
                                Color32::from_rgb(255, 232, 130),
                            );
                        }
                    }
                }

                // Vector Continental Coastlines (rendered ABOVE raster overlays & isobars: 2.8px dark obsidian casing + 1.35px warm gold/ivory shoreline)
                let coast_casing =
                    Stroke::new(2.8_f32, Color32::from_rgba_unmultiplied(8, 12, 20, 235));
                let coast_stroke =
                    Stroke::new(1.35_f32, Color32::from_rgba_unmultiplied(245, 226, 172, 242));
                for pass in 0..2 {
                    let stroke = if pass == 0 { coast_casing } else { coast_stroke };
                    for poly in CONTINENT_POLYLINES {
                        let mut prev: Option<(Pos2, f64, f64)> = None;
                        for &(lat, lon) in *poly {
                            if let Some((nx, ny)) = project_latlon_to_ndc(
                                self.projection,
                                &self.camera,
                                lat,
                                lon,
                                0.0,
                                aspect,
                            ) {
                                let pt = ndc_to_screen(nx, ny);
                                if let Some((p0, la0, lo0)) = prev {
                                    let is_polar_seam = (la0 < -89.5 && lat < -89.5)
                                        || ((lo0.abs() > 179.9 && lon.abs() > 179.9)
                                            && (la0 < -80.0 || lat < -80.0));
                                    if !is_polar_seam && p0.distance(pt) < rect.width() * 0.35 {
                                        painter.line_segment([p0, pt], stroke);
                                    }
                                }
                                prev = Some((pt, lat, lon));
                            } else {
                                prev = None;
                            }
                        }
                    }
                }

                // Animated Lagrangian Streamline Particles
                let any_vectors = ["wind_vector_jet_streams", "atmospheric_rivers_ivt", "thermohaline_currents_amoc"]
                    .iter()
                    .any(|&id| self.layer_states.get(id).map(|s| s.visible).unwrap_or(false));
                if any_vectors {
                    for p in &self.particles {
                        if let Some((nx, ny)) = project_latlon_to_ndc(
                            self.projection,
                            &self.camera,
                            p.lat,
                            p.lon,
                            0.0,
                            aspect,
                        ) {
                            let pt = ndc_to_screen(nx, ny);
                            let fade = 1.0 - (p.age as f32 / p.max_age.max(1) as f32);
                            let alpha = (fade * 215.0) as u8;
                            painter.circle_filled(
                                pt,
                                1.9,
                                Color32::from_rgba_unmultiplied(150, 245, 255, alpha),
                            );
                        }
                    }
                }

                // Reference Observatories
                for &(name, la, lo, _) in REFERENCE_OBSERVATORIES {
                    if let Some((nx, ny)) =
                        project_latlon_to_ndc(self.projection, &self.camera, la, lo, 0.0, aspect)
                    {
                        let pt = ndc_to_screen(nx, ny);
                        painter.circle_filled(pt, 3.5, Color32::from_rgb(80, 250, 175));
                        painter.text(
                            egui::pos2(pt.x + 5.0, pt.y - 5.0),
                            Align2::LEFT_BOTTOM,
                            name,
                            FontId::proportional(10.5),
                            Color32::from_rgb(195, 245, 220),
                        );
                    }
                }

                // Pinned Target Crosshair
                if self.crosshair_inspector {
                    let (plat, plon) = self.pinned_coord;
                    if let Some((nx, ny)) = project_latlon_to_ndc(
                        self.projection,
                        &self.camera,
                        plat,
                        plon,
                        0.0,
                        aspect,
                    ) {
                        let pt = ndc_to_screen(nx, ny);
                        let ch_stroke = Stroke::new(1.5_f32, Color32::from_rgb(255, 95, 95));
                        painter.circle_stroke(pt, 7.5, ch_stroke);
                        painter.line_segment(
                            [egui::pos2(pt.x - 12.0, pt.y), egui::pos2(pt.x + 12.0, pt.y)],
                            ch_stroke,
                        );
                        painter.line_segment(
                            [egui::pos2(pt.x, pt.y - 12.0), egui::pos2(pt.x, pt.y + 12.0)],
                            ch_stroke,
                        );
                    }
                }

                // A->B Great-Circle SLERP Transect (64 segments)
                if !self.transect_points.is_empty() {
                    if let Some((ax, ay)) = project_latlon_to_ndc(
                        self.projection,
                        &self.camera,
                        self.transect_points[0].0,
                        self.transect_points[0].1,
                        0.0,
                        aspect,
                    ) {
                        let pa = ndc_to_screen(ax, ay);
                        painter.circle_filled(pa, 5.2, Color32::from_rgb(255, 210, 60));
                        painter.circle_stroke(
                            pa,
                            5.2,
                            Stroke::new(1.2_f32, Color32::from_rgb(10, 14, 24)),
                        );
                        painter.text(
                            pa + egui::vec2(7.0, -7.0),
                            Align2::LEFT_BOTTOM,
                            "A",
                            FontId::proportional(12.0),
                            Color32::from_rgb(255, 225, 110),
                        );
                    }
                }
                if self.transect_points.len() == 2 {
                    let a = self.transect_points[0];
                    let b = self.transect_points[1];
                    let arc = slerp_great_circle(a, b, 64);
                    let casing = Stroke::new(4.0_f32, Color32::from_rgba_unmultiplied(8, 12, 20, 210));
                    let fg = Stroke::new(2.3_f32, Color32::from_rgb(255, 210, 60));

                    if self.projection == MapProjectionKind::Equirectangular2d {
                        let (scale_x, scale_y) = equirectangular_scales(aspect);
                        let eq_to_screen = |lat: f64, dlon: f64| -> Pos2 {
                            let nx = ((dlon / 180.0) * self.camera.zoom * scale_x) as f32;
                            let ny = (((lat - self.camera.center_lat) / 90.0)
                                * self.camera.zoom
                                * scale_y) as f32;
                            ndc_to_screen(nx, ny)
                        };
                        for win in arc.windows(2) {
                            let p0 = win[0];
                            let p1 = win[1];
                            let dlon0 = ((p0.1 - self.camera.center_lon + 540.0).rem_euclid(360.0))
                                - 180.0;
                            let dlon1 = ((p1.1 - self.camera.center_lon + 540.0).rem_euclid(360.0))
                                - 180.0;
                            if (dlon1 - dlon0).abs() > 180.0 {
                                let step_dlon = ((dlon1 - dlon0 + 540.0).rem_euclid(360.0)) - 180.0;
                                if step_dlon.abs() > 1e-6 {
                                    let edge_dlon = if step_dlon > 0.0 { 180.0 } else { -180.0 };
                                    let frac = ((edge_dlon - dlon0) / step_dlon).clamp(0.0, 1.0);
                                    let lat_seam = p0.0 + frac * (p1.0 - p0.0);
                                    let s0 = eq_to_screen(p0.0, dlon0);
                                    let s_edge_a = eq_to_screen(lat_seam, edge_dlon);
                                    let s_edge_b = eq_to_screen(lat_seam, -edge_dlon);
                                    let s1 = eq_to_screen(p1.0, dlon1);
                                    painter.line_segment([s0, s_edge_a], casing);
                                    painter.line_segment([s0, s_edge_a], fg);
                                    painter.line_segment([s_edge_b, s1], casing);
                                    painter.line_segment([s_edge_b, s1], fg);
                                }
                            } else {
                                let s0 = eq_to_screen(p0.0, dlon0);
                                let s1 = eq_to_screen(p1.0, dlon1);
                                painter.line_segment([s0, s1], casing);
                                painter.line_segment([s0, s1], fg);
                            }
                        }
                    } else {
                        for win in arc.windows(2) {
                            let p0 = win[0];
                            let p1 = win[1];
                            if let (Some((nx0, ny0)), Some((nx1, ny1))) = (
                                project_latlon_to_ndc(
                                    self.projection,
                                    &self.camera,
                                    p0.0,
                                    p0.1,
                                    0.0,
                                    aspect,
                                ),
                                project_latlon_to_ndc(
                                    self.projection,
                                    &self.camera,
                                    p1.0,
                                    p1.1,
                                    0.0,
                                    aspect,
                                ),
                            ) {
                                let sp0 = ndc_to_screen(nx0, ny0);
                                let sp1 = ndc_to_screen(nx1, ny1);
                                if sp0.distance(sp1) < rect.width() * 0.35 {
                                    painter.line_segment([sp0, sp1], casing);
                                    painter.line_segment([sp0, sp1], fg);
                                }
                            }
                        }
                    }

                    if let Some((bx, by)) = project_latlon_to_ndc(
                        self.projection,
                        &self.camera,
                        b.0,
                        b.1,
                        0.0,
                        aspect,
                    ) {
                        let pb = ndc_to_screen(bx, by);
                        painter.circle_filled(pb, 5.2, Color32::from_rgb(255, 140, 60));
                        painter.circle_stroke(
                            pb,
                            5.2,
                            Stroke::new(1.2_f32, Color32::from_rgb(10, 14, 24)),
                        );
                        painter.text(
                            pb + egui::vec2(7.0, -7.0),
                            Align2::LEFT_BOTTOM,
                            "B",
                            FontId::proportional(12.0),
                            Color32::from_rgb(255, 180, 110),
                        );
                    }
                }

                // Bottom-Right Quantitative Viewport Colorbars
                draw_viewport_colorbars(
                    &painter,
                    rect,
                    self.workers.coordinator.schema(),
                    &self.layer_states,
                    self.anomaly_mode,
                );

                // Hover HUD Tooltip with Cursor-Sampled Active Scalar Readout
                if let Some(hover_pos) = response.hover_pos() {
                    let ndc_x = 2.0 * ((hover_pos.x - rect.left()) / rect.width()) as f64 - 1.0;
                    let ndc_y = 1.0 - 2.0 * ((hover_pos.y - rect.top()) / rect.height()) as f64;
                    if let Some((hlat, hlon)) = unproject_ndc_to_latlon(
                        self.projection,
                        &self.camera,
                        ndc_x,
                        ndc_y,
                        aspect,
                    ) {
                        let elev = synthetic_planetary_elevation_m(hlat, hlon);
                        let scalar_suffix = self
                            .cached_snapshot
                            .as_ref()
                            .and_then(|snap| {
                                sample_active_scalar_at(
                                    self.workers.coordinator.schema(),
                                    snap,
                                    &self.layer_states,
                                    hlat,
                                    hlon,
                                    self.anomaly_mode,
                                )
                            })
                            .map(|(name, val, units)| {
                                if self.anomaly_mode {
                                    if units.is_empty() {
                                        format!(" | {}: {:+.2} (anom)", truncate_ui_label(&name, 20), val)
                                    } else {
                                        format!(
                                            " | {}: {:+.2} {} (anom)",
                                            truncate_ui_label(&name, 18),
                                            val,
                                            truncate_ui_label(&units, 8)
                                        )
                                    }
                                } else if units.is_empty() {
                                    format!(" | {}: {:.2}", truncate_ui_label(&name, 22), val)
                                } else {
                                    format!(
                                        " | {}: {:.2} {}",
                                        truncate_ui_label(&name, 20),
                                        val,
                                        truncate_ui_label(&units, 10)
                                    )
                                }
                            })
                            .unwrap_or_default();
                        let hud_text = format!(
                            "Cursor: {:+.2}°, {:+.2}° | Elev: {:+.0} m{}",
                            hlat, hlon, elev, scalar_suffix
                        );
                        let font_id = FontId::monospace(10.5);
                        let galley = painter.layout_no_wrap(
                            hud_text.clone(),
                            font_id.clone(),
                            Color32::from_rgb(210, 230, 250),
                        );
                        let max_hud_w = (rect.width() - 352.0).max(260.0);
                        let hud_w = (galley.size().x + 16.0).min(max_hud_w);
                        let hud_rect = Rect::from_min_size(
                            egui::pos2(rect.left() + 12.0, rect.bottom() - 34.0),
                            egui::vec2(hud_w, 24.0),
                        );
                        painter.rect_filled(
                            hud_rect,
                            4.0,
                            Color32::from_rgba_unmultiplied(10, 16, 26, 215),
                        );
                        painter.with_clip_rect(hud_rect).text(
                            hud_rect.left_center() + egui::vec2(8.0, 0.0),
                            Align2::LEFT_CENTER,
                            hud_text,
                            font_id,
                            Color32::from_rgb(210, 230, 250),
                        );
                    }
                }
            });
    }
}

impl TerraTwinGuiApp {
    pub fn ui_step(&mut self, ctx: &egui::Context) {
        let wants_kb = ctx.wants_keyboard_input();
        let (
            toggle_palette,
            f1_pressed,
            esc_pressed,
            space_pressed,
            r_pressed,
            l_pressed,
            a_pressed,
            k1_pressed,
            k2_pressed,
            k3_pressed,
            k4_pressed,
            left_pressed,
            right_pressed,
            has_cmd_mod,
        ) = ctx.input_mut(|i| {
            let cmd_k = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::K);
            let tk = i.consume_shortcut(&cmd_k);
            let cmd = i.modifiers.ctrl || i.modifiers.alt || i.modifiers.command;
            (
                tk,
                i.key_pressed(egui::Key::F1),
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::R),
                i.key_pressed(egui::Key::L),
                i.key_pressed(egui::Key::A),
                i.key_pressed(egui::Key::Num1),
                i.key_pressed(egui::Key::Num2),
                i.key_pressed(egui::Key::Num3),
                i.key_pressed(egui::Key::Num4),
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                cmd,
            )
        });
        if toggle_palette {
            self.command_palette_open = !self.command_palette_open;
            self.command_palette_just_opened = self.command_palette_open;
        }
        if f1_pressed {
            self.about_open = !self.about_open;
        }
        if esc_pressed {
            if self.command_palette_open {
                self.command_palette_open = false;
                self.command_palette_just_opened = false;
            } else if self.about_open {
                self.about_open = false;
            } else if self.transect_mode || !self.transect_points.is_empty() {
                self.transect_mode = false;
                self.transect_points.clear();
            } else if !self.layer_filter_query.is_empty() {
                self.layer_filter_query.clear();
            }
        }
        if !wants_kb && !has_cmd_mod {
            if space_pressed {
                self.playing_timeline = !self.playing_timeline;
            }
            if r_pressed {
                self.reset_camera();
            }
            if l_pressed {
                self.left_sidebar_open = !self.left_sidebar_open;
            }
            if a_pressed {
                self.right_drawer_open = !self.right_drawer_open;
            }
            let new_proj = if k1_pressed {
                Some(MapProjectionKind::Globe3dWgs84)
            } else if k2_pressed {
                Some(MapProjectionKind::Equirectangular2d)
            } else if k3_pressed {
                Some(MapProjectionKind::ArcticPolarStereo)
            } else if k4_pressed {
                Some(MapProjectionKind::AntarcticPolarStereo)
            } else {
                None
            };
            if let Some(p) = new_proj {
                if self.projection != p {
                    self.projection = p;
                    self.raster_dirty = true;
                }
            }
            if left_pressed {
                self.step_timeline_by(-1);
            } else if right_pressed {
                self.step_timeline_by(1);
            }
        }

        if self.last_timeline_refresh.elapsed() > std::time::Duration::from_secs(3) {
            self.refresh_playback_timestamps();
        }

        // Only overwrite cached_snapshot with live background updates when in Live mode
        if self.playback_index.is_none() {
            if let Some(snap) = self.workers.coordinator.latest_snapshot() {
                let gen_key = (snap.readouts.mauna_loa_co2_ppm.to_bits())
                    ^ (snap.db_stats.simulated_grid_count as u64)
                    ^ (snap.config.atmospheric_level_hpa.to_bits());
                if gen_key != self.last_raster_gen_key {
                    self.last_raster_gen_key = gen_key;
                    self.raster_dirty = true;
                }
                self.cached_snapshot = Some(snap);
            }
        }

        // Advance either 4D SQLite Replay ticks or mode-aware Live simulation clock when playing
        if self.playing_timeline {
            if let Some(cur_idx) = self.playback_index {
                if !self.playback_timestamps.is_empty()
                    && self.playback_last_advance.elapsed() >= std::time::Duration::from_millis(550)
                {
                    let next_idx = (cur_idx + 1) % self.playback_timestamps.len();
                    self.load_replay_snapshot_at_idx(next_idx);
                    self.playback_last_advance = std::time::Instant::now();
                }
            } else if self.playback_last_advance.elapsed() >= std::time::Duration::from_millis(350)
            {
                let now = chrono::Utc::now();
                let present_year = now.year() as f64 + (now.ordinal() as f64 / 365.25);
                match self.temporal_mode_id.as_str() {
                    "realtime_telemetry" => {
                        let lo_yr = present_year - 7.0 / 365.25;
                        let hi_yr = present_year + 10.0 / 365.25;
                        self.sim_cfg.time_step_secs = 600.0;
                        self.sim_cfg.advance_tick();
                        if self.sim_cfg.sim_year > hi_yr || self.sim_cfg.sim_year < lo_yr {
                            self.sim_cfg.sim_year = lo_yr;
                        }
                    }
                    "historical_reanalysis" => {
                        self.sim_cfg.time_step_secs = 3600.0;
                        self.sim_cfg.sim_year += 0.5;
                        if self.sim_cfg.sim_year > present_year {
                            self.sim_cfg.sim_year = 1950.0;
                        }
                    }
                    _ => {
                        self.sim_cfg.time_step_secs = 3600.0;
                        self.sim_cfg.sim_year += 0.5;
                        if self.sim_cfg.sim_year > 2100.0 || self.sim_cfg.sim_year < 2026.0 {
                            self.sim_cfg.sim_year = 2026.0;
                        }
                    }
                }
                self.push_sim_update();
                self.playback_last_advance = std::time::Instant::now();
            }
        }

        self.advect_particles();

        self.render_top_bar(ctx);
        self.render_bottom_dock(ctx);
        self.render_left_sidebar(ctx);
        self.render_right_analytics_drawer(ctx);
        self.render_central_map(ctx);
        self.render_command_palette(ctx);
        self.render_about_window(ctx);

        ctx.request_repaint_after(std::time::Duration::from_millis(45));
    }
}

impl eframe::App for TerraTwinGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui_step(ctx);
    }
}

fn draw_readout_badge(
    ui: &mut egui::Ui,
    label: &str,
    value: &str,
    color: Color32,
    tooltip: &str,
) {
    let ir = egui::Frame::none()
        .fill(Color32::from_rgb(22, 30, 46))
        .rounding(4.0)
        .inner_margin(egui::Margin::symmetric(7.0, 3.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small(label);
                ui.colored_label(color, egui::RichText::new(value).monospace().strong());
            });
        });
    ui.interact(ir.response.rect, ui.id().with(label), Sense::hover())
        .on_hover_text(tooltip);
}

#[inline]
fn apply_globe_lambert_fresnel(rgb: [f32; 3], sx: f64, sy: f64) -> [u8; 3] {
    let rho2 = (sx * sx + sy * sy).min(1.0);
    let nz = (1.0 - rho2).max(0.0).sqrt() as f32;
    let nx = sx as f32;
    let ny = sy as f32;
    // Upper-left solar key light direction normalized: (-0.42, 0.52, 0.744)
    let mu_sun = (-0.42 * nx + 0.52 * ny + 0.744 * nz).clamp(0.0, 1.0);
    let lambert = 0.60 + 0.44 * mu_sun;
    // Eddington/Minnaert limb darkening at oblique grazing angles
    let limb_dark = 0.76 + 0.24 * nz.powf(0.45);
    // Fresnel Rayleigh atmospheric limb scatter near grazing angles
    let fresnel = (1.0 - nz).powf(3.0);
    let atm_scatter = 0.34 * fresnel;
    let lit_r = rgb[0] * lambert * limb_dark;
    let lit_g = rgb[1] * lambert * limb_dark;
    let lit_b = rgb[2] * lambert * limb_dark;
    [
        (lit_r * (1.0 - atm_scatter) + 88.0 * atm_scatter).clamp(0.0, 255.0) as u8,
        (lit_g * (1.0 - atm_scatter) + 185.0 * atm_scatter).clamp(0.0, 255.0) as u8,
        (lit_b * (1.0 - atm_scatter) + 255.0 * atm_scatter).clamp(0.0, 255.0) as u8,
    ]
}

#[inline]
fn layer_matches_filter(
    grp: &crate::schema::LayerGroupSpec,
    lyr: &crate::schema::LayerSpec,
    filter_lc: &str,
) -> bool {
    if filter_lc.is_empty() {
        return true;
    }
    lyr.name.to_ascii_lowercase().contains(filter_lc)
        || lyr.id.to_ascii_lowercase().contains(filter_lc)
        || lyr.description.to_ascii_lowercase().contains(filter_lc)
        || lyr.units.to_ascii_lowercase().contains(filter_lc)
        || grp.group_name.to_ascii_lowercase().contains(filter_lc)
        || lyr
            .public_sources
            .iter()
            .any(|s| s.to_ascii_lowercase().contains(filter_lc))
}

#[inline]
fn format_graticule_lat(lat: i32) -> String {
    if lat > 0 {
        format!("{}°N", lat)
    } else if lat < 0 {
        format!("{}°S", -lat)
    } else {
        "0°".to_string()
    }
}

#[inline]
fn format_graticule_lon(lon: i32) -> String {
    if lon == 0 {
        "0°".to_string()
    } else if lon == -180 || lon == 180 {
        "180°".to_string()
    } else if lon > 0 {
        format!("{}°E", lon)
    } else {
        format!("{}°W", -lon)
    }
}

/// Render WMO cold front triangular pips (pointing toward warm air) or warm front semicircular
/// pips (bulging toward cold air) along segment `p1 -> p2`.
fn draw_wmo_front_pip(painter: &egui::Painter, p1: Pos2, p2: Pos2, is_cold_front: bool) {
    let d = p2 - p1;
    let len = d.length();
    if len < 1.0e-3 {
        return;
    }
    let t = d / len;
    let mid = egui::pos2(0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y));
    // Screen-space warm-side normal (+grad_S(T)) given tangent (-dtdy, dtdx) and y-down screen coords
    let n_warm = egui::vec2(-t.y, t.x);
    if is_cold_front {
        let col = Color32::from_rgba_unmultiplied(70, 175, 255, 225);
        let pts = vec![mid - t * 3.6, mid + n_warm * 6.5, mid + t * 3.6];
        painter.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
    } else {
        let col = Color32::from_rgba_unmultiplied(255, 90, 75, 225);
        let n_cold = -n_warm;
        let r = 4.2_f32;
        let mut pts = Vec::with_capacity(7);
        for k in 0..=6 {
            let theta = std::f32::consts::PI * (k as f32 / 6.0);
            pts.push(mid + (-t * theta.cos() + n_cold * theta.sin()) * r);
        }
        painter.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
    }
}

fn sample_active_scalar_at(
    schema: &HubSchema,
    snap: &TelemetrySnapshot,
    layer_states: &HashMap<String, LayerUiState>,
    lat: f64,
    lon: f64,
    anomaly_mode: bool,
) -> Option<(String, f32, String)> {
    for (_, lyr) in schema.all_layers().into_iter().rev() {
        if lyr.id == "synoptic_fronts_isobars" || lyr.id == "teleconnection_oscillation_masks" {
            continue;
        }
        let vis = layer_states.get(&lyr.id).map(|s| s.visible).unwrap_or(false);
        if !vis {
            continue;
        }
        if let Some(grid) = snap.grids.get(&lyr.id) {
            let raw = sample_grid_bilinear(&grid.values, grid.width, grid.height, lat, lon);
            if raw.is_finite() {
                let val = if anomaly_mode {
                    raw - grid.mean_val
                } else {
                    raw
                };
                return Some((lyr.name.clone(), val, lyr.units.clone()));
            }
        }
    }
    None
}

fn draw_sounding_canvas(ui: &mut egui::Ui, sounding: &[(f64, f64, f64, f64)]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 150.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 20, 32));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));

    let plot_left = rect.left() + 34.0;
    let plot_right = rect.right() - 8.0;
    let plot_top = rect.top() + 21.0;
    let plot_bottom = rect.bottom() - 16.0;

    let map_pt = |p_hpa: f64, temp_c: f64| -> Pos2 {
        let y_frac = ((1000.0_f64.ln() - p_hpa.ln()) / (1000.0_f64.ln() - 10.0_f64.ln()))
            .clamp(0.0, 1.0) as f32;
        let x_frac = ((temp_c + 80.0) / 125.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_left + x_frac * (plot_right - plot_left),
            plot_bottom - y_frac * (plot_bottom - plot_top),
        )
    };

    for p_tick in [1000.0, 300.0, 10.0] {
        let py = map_pt(p_tick, 0.0).y;
        painter.line_segment(
            [egui::pos2(plot_left, py), egui::pos2(plot_right, py)],
            Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(120, 145, 180, 45)),
        );
        painter.text(
            egui::pos2(plot_left - 4.0, py),
            Align2::RIGHT_CENTER,
            format!("{:.0}", p_tick),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }
    for t_tick in [-60.0, 0.0, 30.0] {
        let px = map_pt(500.0, t_tick).x;
        painter.text(
            egui::pos2(px, rect.bottom() - 3.0),
            Align2::CENTER_BOTTOM,
            format!("{:+.0}°C", t_tick),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }

    for win in sounding.windows(2) {
        let (p0, t0, td0, _) = win[0];
        let (p1, t1, td1, _) = win[1];
        painter.line_segment(
            [map_pt(p0, t0), map_pt(p1, t1)],
            Stroke::new(2.0_f32, Color32::from_rgb(250, 95, 85)),
        );
        painter.line_segment(
            [map_pt(p0, td0), map_pt(p1, td1)],
            Stroke::new(1.6_f32, Color32::from_rgb(85, 205, 250)),
        );
    }
    painter.text(
        rect.left_top() + egui::vec2(8.0, 4.0),
        Align2::LEFT_TOP,
        "Red: Temp (°C) | Cyan: Dewpoint (°C) | hPa",
        FontId::monospace(9.0),
        Color32::from_rgb(190, 205, 225),
    );
}

fn draw_ctd_canvas(ui: &mut egui::Ui, ctd: &[(f64, f64, f64)]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 140.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(12, 22, 36));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));

    let plot_left = rect.left() + 36.0;
    let plot_right = rect.right() - 8.0;
    let plot_top = rect.top() + 21.0;
    let plot_bottom = rect.bottom() - 16.0;

    let map_temp = |z_m: f64, t_c: f64| -> Pos2 {
        let y_frac = (z_m / 6000.0).clamp(0.0, 1.0) as f32;
        let x_frac = ((t_c + 2.0) / 34.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_left + x_frac * (plot_right - plot_left),
            plot_top + y_frac * (plot_bottom - plot_top),
        )
    };
    let map_sal = |z_m: f64, sal: f64| -> Pos2 {
        let y_frac = (z_m / 6000.0).clamp(0.0, 1.0) as f32;
        let x_frac = ((sal - 33.0) / 4.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_left + x_frac * (plot_right - plot_left),
            plot_top + y_frac * (plot_bottom - plot_top),
        )
    };

    for z_tick in [0.0, 3000.0, 6000.0] {
        let py = map_temp(z_tick, 0.0).y;
        painter.line_segment(
            [egui::pos2(plot_left, py), egui::pos2(plot_right, py)],
            Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(120, 145, 180, 45)),
        );
        painter.text(
            egui::pos2(plot_left - 4.0, py),
            Align2::RIGHT_CENTER,
            format!("{:.0}m", z_tick),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }
    for t_tick in [0.0, 15.0, 30.0] {
        let px = map_temp(0.0, t_tick).x;
        painter.text(
            egui::pos2(px, rect.bottom() - 3.0),
            Align2::CENTER_BOTTOM,
            format!("{:.0}°C", t_tick),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }

    for win in ctd.windows(2) {
        let (z0, t0, s0) = win[0];
        let (z1, t1, s1) = win[1];
        painter.line_segment(
            [map_temp(z0, t0), map_temp(z1, t1)],
            Stroke::new(2.0_f32, Color32::from_rgb(255, 165, 70)),
        );
        painter.line_segment(
            [map_sal(z0, s0), map_sal(z1, s1)],
            Stroke::new(1.6_f32, Color32::from_rgb(100, 235, 190)),
        );
    }
    painter.text(
        rect.left_top() + egui::vec2(8.0, 4.0),
        Align2::LEFT_TOP,
        "Orange: Ocean Temp (°C) | Teal: Salinity (PSU)",
        FontId::monospace(9.0),
        Color32::from_rgb(190, 205, 225),
    );
}

fn draw_land_cryosphere_canvas(ui: &mut egui::Ui, cryo: &LandCryosphereProfile) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 140.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 22, 34));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(48, 64, 88)));

    let plot_left = rect.left() + 34.0;
    let plot_right = rect.right() - 8.0;
    let plot_top = rect.top() + 21.0;
    let plot_bottom = rect.bottom() - 16.0;

    let t_min = cryo
        .levels
        .iter()
        .map(|l| l.1)
        .fold(-5.0_f64, f64::min)
        .min(-5.0)
        .floor();
    let t_max = cryo
        .levels
        .iter()
        .map(|l| l.1)
        .fold(5.0_f64, f64::max)
        .max(5.0)
        .ceil();
    let t_span = (t_max - t_min).max(5.0);

    let map_temp = |z_m: f64, t_c: f64| -> Pos2 {
        let y_frac = (z_m / 20.0).clamp(0.0, 1.0) as f32;
        let x_frac = ((t_c - t_min) / t_span).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_left + x_frac * (plot_right - plot_left),
            plot_top + y_frac * (plot_bottom - plot_top),
        )
    };
    let map_moist = |z_m: f64, w: f64| -> Pos2 {
        let y_frac = (z_m / 20.0).clamp(0.0, 1.0) as f32;
        let x_frac = (w / 0.50).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_left + x_frac * (plot_right - plot_left),
            plot_top + y_frac * (plot_bottom - plot_top),
        )
    };

    for z_tick in [0.0, 10.0, 20.0] {
        let py = map_temp(z_tick, t_min).y;
        painter.line_segment(
            [egui::pos2(plot_left, py), egui::pos2(plot_right, py)],
            Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(120, 145, 180, 45)),
        );
        painter.text(
            egui::pos2(plot_left - 4.0, py),
            Align2::RIGHT_CENTER,
            format!("{:.0}m", z_tick),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }

    // 0 °C Freezing Isotherm
    if t_min <= 0.0 && t_max >= 0.0 {
        let x0 = map_temp(0.0, 0.0).x;
        painter.line_segment(
            [egui::pos2(x0, plot_top), egui::pos2(x0, plot_bottom)],
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(150, 220, 255, 115)),
        );
    }

    if let Some(thaw_m) = cryo.active_layer_m {
        let ty = map_temp(thaw_m.clamp(0.0, 20.0), t_min).y;
        painter.line_segment(
            [egui::pos2(plot_left, ty), egui::pos2(plot_right, ty)],
            Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(255, 210, 80, 170)),
        );
    }

    for (frac, val) in [(0.0, t_min), (0.5, 0.5 * (t_min + t_max)), (1.0, t_max)] {
        let px = plot_left + frac * (plot_right - plot_left);
        painter.text(
            egui::pos2(px, rect.bottom() - 3.0),
            if frac < 0.2 {
                Align2::LEFT_BOTTOM
            } else if frac > 0.8 {
                Align2::RIGHT_BOTTOM
            } else {
                Align2::CENTER_BOTTOM
            },
            format!("{:+.0}°C", val),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }

    for win in cryo.levels.windows(2) {
        let (z0, t0, w0) = win[0];
        let (z1, t1, w1) = win[1];
        painter.line_segment(
            [map_temp(z0, t0), map_temp(z1, t1)],
            Stroke::new(2.0_f32, Color32::from_rgb(255, 145, 85)),
        );
        painter.line_segment(
            [map_moist(z0, w0), map_moist(z1, w1)],
            Stroke::new(1.6_f32, Color32::from_rgb(110, 220, 250)),
        );
    }
    painter.text(
        rect.left_top() + egui::vec2(8.0, 4.0),
        Align2::LEFT_TOP,
        "Orange: Ground T (°C) | Cyan: Unfrozen H₂O",
        FontId::monospace(9.0),
        Color32::from_rgb(190, 205, 225),
    );
}

fn draw_cmip6_projection_canvas(
    ui: &mut egui::Ui,
    active_ssp: SspScenario,
    cur_year: f64,
    metric: Cmip6PlotMetric,
    region: ClimateRegion,
    pinned_lat: f64,
    live_snapshot_val: Option<f64>,
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 165.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 20, 32));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));

    let plot_left = rect.left() + 40.0;
    let plot_right = rect.right() - 34.0;
    let plot_top = rect.top() + 21.0;
    let plot_bottom = rect.bottom() - 18.0;

    let (y_min, y_max, y_ticks): (f64, f64, [f64; 3]) = match metric {
        Cmip6PlotMetric::Co2Ppm => (300.0, 1180.0, [300.0, 740.0, 1150.0]),
        Cmip6PlotMetric::DeltaTC => {
            let max_dt = regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2100.0, region, pinned_lat)
                .max(2.0);
            let ceil_dt = (max_dt * 1.12).ceil().max(2.0);
            (0.0, ceil_dt, [0.0, 0.5 * ceil_dt, ceil_dt])
        }
    };
    let y_span = (y_max - y_min).max(1.0);

    let eval_ssp = |ssp: SspScenario, yr: f64| -> f64 {
        match metric {
            Cmip6PlotMetric::Co2Ppm => ssp.co2_ppm_at_year(yr),
            Cmip6PlotMetric::DeltaTC => regional_cmip6_delta_t_c(ssp, yr, region, pinned_lat),
        }
    };

    let map_yr_val = |yr: f64, val: f64| -> Pos2 {
        let xf = ((yr - 1950.0) / 150.0).clamp(0.0, 1.0) as f32;
        let yf = ((val - y_min) / y_span).clamp(0.0, 1.0) as f32;
        egui::pos2(
            plot_left + xf * (plot_right - plot_left),
            plot_bottom - yf * (plot_bottom - plot_top),
        )
    };

    // Y-axis ticks & grid lines
    for yt in y_ticks {
        let py = map_yr_val(1950.0, yt).y;
        painter.line_segment(
            [egui::pos2(plot_left, py), egui::pos2(plot_right, py)],
            Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(120, 145, 180, 45)),
        );
        let lbl = match metric {
            Cmip6PlotMetric::Co2Ppm => format!("{:.0}", yt),
            Cmip6PlotMetric::DeltaTC => format!("+{:.1}°", yt),
        };
        painter.text(
            egui::pos2(plot_left - 4.0, py),
            Align2::RIGHT_CENTER,
            lbl,
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }

    // X-axis year ticks
    for yr_tick in [1950, 2000, 2050, 2100] {
        let px = map_yr_val(yr_tick as f64, y_min).x;
        painter.line_segment(
            [egui::pos2(px, plot_bottom), egui::pos2(px, plot_bottom + 3.0)],
            Stroke::new(0.8_f32, Color32::from_rgb(120, 145, 180)),
        );
        painter.text(
            egui::pos2(px, rect.bottom() - 2.0),
            Align2::CENTER_BOTTOM,
            format!("{}", yr_tick),
            FontId::monospace(8.5),
            Color32::from_rgb(155, 175, 200),
        );
    }

    // Compact 5-SSP color legend at top
    let mut leg_x = rect.left() + 6.0;
    for ssp in SspScenario::ALL {
        let col = ssp_color(ssp);
        let short = match ssp {
            SspScenario::Ssp1_19 => "1-1.9",
            SspScenario::Ssp1_26 => "1-2.6",
            SspScenario::Ssp2_45 => "2-4.5",
            SspScenario::Ssp3_70 => "3-7.0",
            SspScenario::Ssp5_85 => "5-8.5",
        };
        painter.line_segment(
            [
                egui::pos2(leg_x, rect.top() + 9.0),
                egui::pos2(leg_x + 10.0, rect.top() + 9.0),
            ],
            Stroke::new(if ssp == active_ssp { 2.4_f32 } else { 1.4_f32 }, col),
        );
        painter.text(
            egui::pos2(leg_x + 12.0, rect.top() + 9.0),
            Align2::LEFT_CENTER,
            short,
            FontId::monospace(8.5),
            if ssp == active_ssp {
                Color32::WHITE
            } else {
                Color32::from_rgb(170, 190, 215)
            },
        );
        leg_x += 48.0;
    }

    for ssp in SspScenario::ALL {
        let is_sel = ssp == active_ssp;
        let col = ssp_color(ssp);
        let stroke = Stroke::new(if is_sel { 2.4_f32 } else { 1.0_f32 }, col);
        let mut prev: Option<Pos2> = None;
        for yr_step in (1950..=2100).step_by(5) {
            let yr = yr_step as f64;
            let pt = map_yr_val(yr, eval_ssp(ssp, yr));
            if let Some(p0) = prev {
                painter.line_segment([p0, pt], stroke);
            }
            prev = Some(pt);
        }
    }

    // Current year vertical marker
    let x_cur = map_yr_val(cur_year, y_min).x;
    painter.line_segment(
        [
            egui::pos2(x_cur, plot_top),
            egui::pos2(x_cur, plot_bottom),
        ],
        Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 150)),
    );

    if let Some(live_val) = live_snapshot_val {
        if live_val.is_finite() {
            let live_pt = map_yr_val(cur_year, live_val);
            painter.circle_filled(live_pt, 4.2, Color32::WHITE);
            painter.circle_stroke(live_pt, 4.2, Stroke::new(1.4_f32, ssp_color(active_ssp)));
        }
    }

    // 2100 Ensemble Spread Boxplot bar on right margin
    let bx = rect.right() - 16.0;
    let p10 = map_yr_val(2100.0, eval_ssp(SspScenario::Ssp1_19, 2100.0)).y;
    let p25 = map_yr_val(2100.0, eval_ssp(SspScenario::Ssp1_26, 2100.0)).y;
    let p50 = map_yr_val(2100.0, eval_ssp(SspScenario::Ssp2_45, 2100.0)).y;
    let p75 = map_yr_val(2100.0, eval_ssp(SspScenario::Ssp3_70, 2100.0)).y;
    let p90 = map_yr_val(2100.0, eval_ssp(SspScenario::Ssp5_85, 2100.0)).y;
    painter.line_segment(
        [egui::pos2(bx, p90), egui::pos2(bx, p10)],
        Stroke::new(1.5_f32, Color32::LIGHT_GRAY),
    );
    let box_rect = Rect::from_min_max(egui::pos2(bx - 7.0, p75), egui::pos2(bx + 7.0, p25));
    painter.rect_filled(box_rect, 2.0, Color32::from_rgba_unmultiplied(90, 175, 245, 120));
    painter.rect_stroke(box_rect, 2.0, Stroke::new(1.2_f32, Color32::WHITE));
    painter.line_segment(
        [egui::pos2(bx - 7.0, p50), egui::pos2(bx + 7.0, p50)],
        Stroke::new(2.0_f32, Color32::from_rgb(255, 215, 70)),
    );
}

fn ssp_color(ssp: SspScenario) -> Color32 {
    match ssp {
        SspScenario::Ssp1_19 => Color32::from_rgb(70, 200, 245),
        SspScenario::Ssp1_26 => Color32::from_rgb(95, 225, 145),
        SspScenario::Ssp2_45 => Color32::from_rgb(245, 205, 80),
        SspScenario::Ssp3_70 => Color32::from_rgb(245, 130, 70),
        SspScenario::Ssp5_85 => Color32::from_rgb(235, 65, 75),
    }
}

fn draw_transect_profile(
    ui: &mut egui::Ui,
    schema: &HubSchema,
    snap_opt: &Option<TelemetrySnapshot>,
    layer_states: &HashMap<String, LayerUiState>,
    a: (f64, f64),
    b: (f64, f64),
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 110.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 20, 32));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));
    let Some(snap) = snap_opt else { return };

    let slerp_pts = slerp_great_circle(a, b, 64);
    let n = slerp_pts.len();
    let Some((id, grid)) = snap
        .grids
        .iter()
        .filter(|(id, _)| layer_states.get(*id).map(|s| s.visible).unwrap_or(false))
        .max_by_key(|(_, g)| {
            slerp_pts
                .iter()
                .filter(|&&(la, lo)| sample_grid_bilinear(&g.values, g.width, g.height, la, lo).is_finite())
                .count()
        })
    else {
        return;
    };

    let mut vals = Vec::with_capacity(n);
    for &(la, lo) in &slerp_pts {
        vals.push(sample_grid_bilinear(&grid.values, grid.width, grid.height, la, lo));
    }
    let min_v = grid.min_val;
    let span = (grid.max_val - min_v).max(1e-5);
    let mut prev: Option<Pos2> = None;
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            prev = None;
            continue;
        }
        let xf = i as f32 / (n - 1) as f32;
        let yf = ((v - min_v) / span).clamp(0.0, 1.0);
        let pt = egui::pos2(
            rect.left() + 8.0 + xf * (rect.width() - 16.0),
            rect.bottom() - 12.0 - yf * (rect.height() - 30.0),
        );
        if let Some(p0) = prev {
            painter.line_segment([p0, pt], Stroke::new(2.0_f32, Color32::from_rgb(255, 210, 65)));
        }
        prev = Some(pt);
    }

    let (disp_name, units) = schema
        .find_layer(id)
        .map(|l| (l.name.as_str(), l.units.as_str()))
        .unwrap_or((id.as_str(), ""));
    let dist_km = haversine_km(a.0, a.1, b.0, b.1);
    let short_name = truncate_ui_label(disp_name, 30);
    let title = if units.is_empty() {
        format!("{} | {:.0} km", short_name, dist_km)
    } else {
        format!("{} [{}] | {:.0} km", short_name, truncate_ui_label(units, 12), dist_km)
    };
    painter.text(
        rect.left_top() + egui::vec2(8.0, 4.0),
        Align2::LEFT_TOP,
        title,
        FontId::monospace(9.0),
        Color32::LIGHT_GRAY,
    );
    painter.text(
        rect.left_bottom() + egui::vec2(8.0, -2.0),
        Align2::LEFT_BOTTOM,
        format!("A ({:.1}°, {:.1}°)", a.0, a.1),
        FontId::monospace(8.0),
        Color32::from_rgb(255, 210, 60),
    );
    painter.text(
        rect.right_bottom() + egui::vec2(-8.0, -2.0),
        Align2::RIGHT_BOTTOM,
        format!("B ({:.1}°, {:.1}°)", b.0, b.1),
        FontId::monospace(8.0),
        Color32::from_rgb(255, 140, 60),
    );
}

fn truncate_ui_label(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        s.to_string()
    } else if max_chars > 1 {
        let mut out: String = s.chars().take(max_chars - 1).collect();
        out.push('…');
        out
    } else {
        String::new()
    }
}

fn format_colorbar_tick(val: f32, span: f32) -> String {
    let abs = val.abs();
    if abs >= 1e5 || (abs > 0.0 && abs < 1e-2) {
        format!("{:.1e}", val)
    } else if span < 2.0 {
        format!("{:.2}", val)
    } else if span < 40.0 {
        format!("{:.1}", val)
    } else {
        format!("{:.0}", val)
    }
}

fn draw_viewport_colorbars(
    painter: &egui::Painter,
    rect: Rect,
    schema: &HubSchema,
    layer_states: &HashMap<String, LayerUiState>,
    anomaly_mode: bool,
) {
    let mut visible_scalar = Vec::new();
    for (_, lyr) in schema.all_layers() {
        if lyr.id == "synoptic_fronts_isobars" || lyr.id == "teleconnection_oscillation_masks" {
            continue;
        }
        if let Some(st) = layer_states.get(&lyr.id) {
            if st.visible {
                visible_scalar.push((lyr, st));
            }
        }
    }
    if visible_scalar.is_empty() {
        return;
    }
    visible_scalar.reverse();
    let total_vis = visible_scalar.len();
    let shown = &visible_scalar[..total_vis.min(2)];

    let card_w = 320.0_f32;
    let card_h = 48.0_f32;
    let gap = 6.0_f32;
    let extra_pill_h = if total_vis > 2 { 18.0_f32 } else { 0.0_f32 };
    let mut cur_bottom = rect.bottom() - 10.0;

    if extra_pill_h > 0.0 {
        let pill_rect = Rect::from_min_max(
            egui::pos2(rect.right() - 12.0 - 140.0, cur_bottom - 16.0),
            egui::pos2(rect.right() - 12.0, cur_bottom),
        );
        painter.rect_filled(
            pill_rect,
            4.0,
            Color32::from_rgba_unmultiplied(10, 16, 26, 215),
        );
        painter.text(
            pill_rect.center(),
            Align2::CENTER_CENTER,
            format!("+{} more visible layers", total_vis - 2),
            FontId::monospace(9.0),
            Color32::from_rgb(175, 200, 230),
        );
        cur_bottom -= extra_pill_h + 2.0;
    }

    for (lyr, st) in shown.iter().rev() {
        let card_rect = Rect::from_min_max(
            egui::pos2(rect.right() - 12.0 - card_w, cur_bottom - card_h),
            egui::pos2(rect.right() - 12.0, cur_bottom),
        );
        painter.rect_filled(
            card_rect,
            4.0,
            Color32::from_rgba_unmultiplied(10, 16, 26, 224),
        );
        painter.rect_stroke(
            card_rect,
            4.0,
            Stroke::new(1.0_f32, Color32::from_rgb(52, 70, 96)),
        );

        let (pal, lo, hi, suffix) = if anomaly_mode {
            let span = (st.max_range - st.min_range).abs().max(1e-3) * 0.45;
            (ColormapPalette::CoolWarmDiverging, -span, span, " [Anom]")
        } else {
            (st.palette, st.min_range, st.max_range, "")
        };
        let span = (hi - lo).abs().max(1e-6);

        let title_raw = if lyr.units.is_empty() {
            format!("{}{}", lyr.name, suffix)
        } else {
            format!("{} [{}]{}", lyr.name, lyr.units, suffix)
        };
        let title = truncate_ui_label(&title_raw, 48);
        let card_painter = painter.with_clip_rect(card_rect);
        card_painter.text(
            egui::pos2(card_rect.left() + 8.0, card_rect.top() + 4.0),
            Align2::LEFT_TOP,
            title,
            FontId::proportional(10.0),
            Color32::from_rgb(225, 238, 252),
        );

        let bar_left = card_rect.left() + 12.0;
        let bar_right = card_rect.right() - 12.0;
        let bar_top = card_rect.top() + 19.0;
        let bar_bot = bar_top + 10.0;
        let steps = 64usize;
        let step_w = (bar_right - bar_left) / steps as f32;
        for i in 0..steps {
            let t = i as f32 / (steps - 1) as f32;
            let [r, g, b] = pal.sample_rgb(t);
            let sub = Rect::from_min_max(
                egui::pos2(bar_left + i as f32 * step_w, bar_top),
                egui::pos2(bar_left + (i + 1) as f32 * step_w + 0.5, bar_bot),
            );
            painter.rect_filled(sub, 0.0, Color32::from_rgb(r, g, b));
        }
        let bar_rect = Rect::from_min_max(
            egui::pos2(bar_left, bar_top),
            egui::pos2(bar_right, bar_bot),
        );
        painter.rect_stroke(bar_rect, 0.0, Stroke::new(0.8_f32, Color32::from_rgb(90, 115, 145)));

        // 5-tick colorbar scale with explicit highlighted 0 marker in anomaly mode
        for idx in 0..5usize {
            let frac = idx as f32 * 0.25;
            let tx = bar_left + frac * (bar_right - bar_left);
            let val = lo + frac * (hi - lo);
            let is_zero_anom = anomaly_mode && idx == 2;
            let is_near_zero = is_zero_anom || val.abs() < 1.0e-4 * span;
            if is_zero_anom {
                painter.line_segment(
                    [egui::pos2(tx, bar_top - 1.5), egui::pos2(tx, bar_bot + 3.5)],
                    Stroke::new(1.4_f32, Color32::WHITE),
                );
            } else {
                painter.line_segment(
                    [egui::pos2(tx, bar_bot), egui::pos2(tx, bar_bot + 2.5)],
                    Stroke::new(0.9_f32, Color32::from_rgb(175, 195, 220)),
                );
            }
            let tick_lbl = if is_near_zero {
                "0".to_string()
            } else {
                format_colorbar_tick(val, span)
            };
            let align = match idx {
                0 => Align2::LEFT_TOP,
                4 => Align2::RIGHT_TOP,
                _ => Align2::CENTER_TOP,
            };
            let lbl_col = if is_zero_anom {
                Color32::WHITE
            } else {
                Color32::from_rgb(190, 210, 232)
            };
            painter.text(
                egui::pos2(tx, bar_bot + 3.5),
                align,
                tick_lbl,
                FontId::monospace(8.5),
                lbl_col,
            );
        }

        cur_bottom -= card_h + gap;
    }
}

#[derive(Debug, Clone)]
enum PaletteAction {
    SetSsp(SspScenario, Option<f64>),
    ApplyPreset(QuickScenarioPreset),
    FocusTippingPoint {
        lat: f64,
        lon: f64,
        zoom: f64,
        layer_ids: &'static [&'static str],
        region: ClimateRegion,
    },
    FocusLayer(String),
    SetProjection(MapProjectionKind),
    ToggleLeftDrawer,
    ToggleRightDrawer,
    ResetCamera,
    ExportSspDeltaCsv,
    ExportSspDeltaJson,
    ExportVisibleZarr,
}

#[derive(Debug, Clone)]
struct PaletteEntry {
    category: &'static str,
    title: String,
    subtitle: String,
    action: PaletteAction,
}

fn fuzzy_match_score(query: &str, text: &str) -> Option<i32> {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return Some(0);
    }
    let t = text.to_ascii_lowercase();
    if let Some(pos) = t.find(&q) {
        return Some(1000 - pos as i32);
    }
    let mut t_chars = t.char_indices();
    let mut score = 200i32;
    let mut prev_idx: Option<usize> = None;
    for qc in q.chars() {
        if qc.is_whitespace() {
            continue;
        }
        let found = t_chars.find(|&(_, tc)| tc == qc);
        match found {
            Some((idx, _)) => {
                if let Some(prev) = prev_idx {
                    score -= (idx.saturating_sub(prev + 1)) as i32;
                }
                prev_idx = Some(idx);
            }
            None => return None,
        }
    }
    Some(score)
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Encode an RGBA buffer as a DEFLATE-compressed PNG using PNG Sub filtering (`1`)
/// and `flate2` best zlib compression.
pub fn write_rgba_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let row_bytes = width as usize * 4;
    if rgba.len() != row_bytes * height as usize {
        anyhow::bail!("Unexpected RGBA buffer length {}", rgba.len());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in rgba.chunks_exact(row_bytes) {
        // Filter type 1 (Sub): each byte minus the byte 4 positions left (same RGBA channel)
        raw.push(1);
        for (i, &b) in row.iter().enumerate() {
            let left = if i >= 4 { row[i - 4] } else { 0 };
            raw.push(b.wrapping_sub(left));
        }
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&raw)?;
    let zlib = encoder.finish()?;

    let push_chunk = |out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let mut crc = 0xFFFF_FFFFu32;
        for &b in tag.iter().chain(data.iter()) {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
            }
        }
        out.extend_from_slice(&(crc ^ 0xFFFF_FFFF).to_be_bytes());
    };
    let mut png = Vec::with_capacity(zlib.len() + 64);
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
    let mut ihdr = [0u8; 13];
    ihdr[0..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8] = 8;
    ihdr[9] = 6;
    push_chunk(&mut png, b"IHDR", &ihdr);
    push_chunk(&mut png, b"IDAT", &zlib);
    push_chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)?;
    Ok(())
}

/// Render an offscreen 1600x1000 RGBA PNG screenshot without opening any OS GUI window.
pub fn render_headless_screenshot(
    workers: RunningWorkers,
    out_path: &std::path::Path,
    view: u8,
) -> Result<()> {
    use eframe::egui_wgpu::{wgpu, Renderer, ScreenDescriptor};

    let width: u32 = 1600;
    let height: u32 = 1000;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = block_on_future(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| anyhow::anyhow!("No GPU adapter available for offscreen rendering"))?;
    let (device, queue) = block_on_future(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("terratwin-offscreen-device"),
            ..Default::default()
        },
        None,
    ))
    .map_err(|e| anyhow::anyhow!("Failed to request offscreen GPU device: {e}"))?;

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terratwin-offscreen-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bytes_per_row = width * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terratwin-offscreen-staging"),
        size: (bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut renderer = Renderer::new(&device, format, None, 1, false);
    let egui_ctx = egui::Context::default();
    egui_ctx.set_pixels_per_point(1.0);
    let mut app = TerraTwinGuiApp::new_headless(workers);

    // Wait for initial 288x144 Rayon simulation snapshot across all view presets
    for _ in 0..45 {
        if app.workers.coordinator.latest_snapshot().is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    app.raster_dirty = true;

    if view == 2 {
        app.projection = MapProjectionKind::Equirectangular2d;
        app.camera.center_lon = 0.0;
        app.camera.center_lat = 0.0;
        app.camera.zoom = 1.0;
        app.basemap_id = "bathymetry_topography_shaded".to_string();
        app.anomaly_mode = false;
        app.open_visible_groups_only = true;
        app.pinned_coord = (74.0, -40.0);
        app.coord_search_buf = "74.0000, -40.0000".to_string();
        for st in app.layer_states.values_mut() {
            st.visible = false;
        }
        for (id, op) in [
            ("sea_ice_concentration_drift", 0.94f32),
            ("sea_ice_thickness_freeboard_age", 0.78),
            ("gravimetric_ice_sheet_mass_balance", 0.78),
            ("permafrost_thermokarst_subsidence", 0.82),
        ] {
            if let Some(st) = app.layer_states.get_mut(id) {
                st.visible = true;
                st.opacity = op;
            }
        }
        app.raster_dirty = true;
    } else if view == 3 {
        app.temporal_mode_id = "cmip_projections".to_string();
        app.sim_cfg.time_step_secs = 3600.0;
        app.cmip6_metric = Cmip6PlotMetric::DeltaTC;
        app.cmip6_region = ClimateRegion::Global;
        app.sim_cfg.scenario = SspScenario::Ssp5_85;
        app.sim_cfg.sim_year = 2085.0;
        app.sim_cfg.ocean_depth_m = 450.0;
        app.anomaly_mode = false;
        app.open_visible_groups_only = true;
        app.projection = MapProjectionKind::Globe3dWgs84;
        app.camera.center_lat = 26.0;
        app.camera.center_lon = -52.0;
        app.camera.zoom = 1.0;
        app.basemap_id = "bathymetry_topography_shaded".to_string();
        app.pinned_coord = (36.0, -56.0);
        app.coord_search_buf = "36.0000, -56.0000".to_string();
        app.transect_mode = true;
        app.transect_points = vec![(26.0, -76.0), (54.0, -14.0)];
        for st in app.layer_states.values_mut() {
            st.visible = false;
        }
        for (id, op) in [
            ("toa_radiative_energy_budget", 0.30f32),
            ("sst_marine_heatwaves", 0.85),
            ("coastal_inundation_storm_surge", 0.95),
            ("cmip6_downscaled_ensemble", 0.45),
        ] {
            if let Some(st) = app.layer_states.get_mut(id) {
                st.visible = true;
                st.opacity = op;
            }
        }
        app.push_sim_update();
        for _ in 0..45 {
            std::thread::sleep(std::time::Duration::from_millis(120));
            if let Some(snap) = app.workers.coordinator.latest_snapshot() {
                if snap.config.scenario == SspScenario::Ssp5_85
                    && (snap.config.sim_year - 2085.0).abs() < 0.5
                    && (snap.config.ocean_depth_m - 450.0).abs() < 1.0
                {
                    break;
                }
            }
        }
        app.raster_dirty = true;
    }

    let warmup_delays_ms = [350u64, 250, 150, 150];
    for (frame_idx, delay_ms) in warmup_delays_ms.into_iter().enumerate() {
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            viewports: std::iter::once((
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(1.0),
                    ..Default::default()
                },
            ))
            .collect(),
            time: Some(frame_idx as f64 / 60.0),
            predicted_dt: 1.0 / 60.0,
            ..Default::default()
        };
        let full_output = egui_ctx.run(raw_input, |ctx| {
            app.ui_step(ctx);
        });
        for (id, image_delta) in &full_output.textures_delta.set {
            renderer.update_texture(&device, &queue, *id, image_delta);
        }
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: 1.0,
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terratwin-offscreen-encoder"),
        });
        let user_cmd_bufs = renderer.update_buffers(
            &device,
            &queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terratwin-offscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            let mut render_pass = render_pass.forget_lifetime();
            renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }
        if frame_idx == 3 {
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(user_cmd_bufs.into_iter().chain(std::iter::once(encoder.finish())));
        device.poll(wgpu::Maintain::Wait);
        for id in &full_output.textures_delta.free {
            renderer.free_texture(id);
        }
    }

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|e| anyhow::anyhow!("GPU readback channel error: {e}"))?
        .map_err(|e| anyhow::anyhow!("GPU buffer map failed: {e}"))?;
    let mapped = slice.get_mapped_range();
    write_rgba_png(out_path, width, height, &mapped)?;
    drop(mapped);
    staging.unmap();
    Ok(())
}

/// Launch the native hardware-accelerated desktop GUI window.
pub fn launch_native_gui(workers: RunningWorkers) -> Result<()> {
    let icon_rgba = include_bytes!(concat!(env!("OUT_DIR"), "/terratwin_icon_64.rgba")).to_vec();
    let icon = egui::IconData {
        rgba: icon_rgba,
        width: 64,
        height: 64,
    };

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1520.0, 900.0])
            .with_min_inner_size([1024.0, 660.0])
            .with_icon(Arc::new(icon))
            .with_title(format!(
                "TerraTwin HPC Climate Modeling & Telemetry Hub (v{})",
                env!("CARGO_PKG_VERSION")
            )),
        ..Default::default()
    };

    eframe::run_native(
        "TerraTwin HPC Climate Modeling & Telemetry Hub",
        native_options,
        Box::new(|cc| Ok(Box::new(TerraTwinGuiApp::new(cc, workers)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe native window error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ClimateDatabase;
    use crate::worker::BackgroundCoordinator;

    fn run_key_step(app: &mut TerraTwinGuiApp, ctx: &egui::Context, key: egui::Key) {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        let _ = ctx.run(raw_input, |c| {
            app.ui_step(c);
        });
    }

    #[test]
    fn parse_lat_lon_query_guards_nan_inf_and_clamps() {
        assert_eq!(parse_lat_lon_query("NaN, 0"), None);
        assert_eq!(parse_lat_lon_query("inf, 10"), None);
        assert_eq!(parse_lat_lon_query("10 -infinity"), None);
        assert_eq!(parse_lat_lon_query("45.52 -122.68"), Some((45.52, -122.68)));
        assert_eq!(parse_lat_lon_query("45.52, -122.68"), Some((45.52, -122.68)));
        assert_eq!(parse_lat_lon_query("120.0, -250.0"), Some((90.0, -180.0)));
        assert_eq!(parse_lat_lon_query("-120.0 250.0"), Some((-90.0, 180.0)));
        assert_eq!(parse_lat_lon_query(""), None);
        assert_eq!(parse_lat_lon_query("10, 20, 30"), None);
    }

    #[test]
    fn headless_keyboard_shortcuts_f1_escape_and_space_transitions() {
        let db_path = std::env::temp_dir().join(format!(
            "terratwin_gui_shortcut_test_{}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&db_path);
        let schema = HubSchema::load(None).expect("embedded schema");
        let db = ClimateDatabase::open(&db_path).expect("temp db");
        let coord = BackgroundCoordinator::new(schema, db, SimulationConfig::default());
        let workers = RunningWorkers::without_threads(coord);

        let mut app = TerraTwinGuiApp::new_headless(workers);
        app.transect_mode = true;
        app.transect_points = vec![(12.5, -45.0)];
        let ctx = egui::Context::default();

        assert!(!app.about_open);
        assert!(!app.playing_timeline);

        run_key_step(&mut app, &ctx, egui::Key::F1);
        assert!(app.about_open);

        run_key_step(&mut app, &ctx, egui::Key::Space);
        assert!(app.playing_timeline);

        run_key_step(&mut app, &ctx, egui::Key::Space);
        assert!(!app.playing_timeline);

        run_key_step(&mut app, &ctx, egui::Key::Escape);
        assert!(!app.about_open);
        assert!(app.transect_mode);
        assert_eq!(app.transect_points.len(), 1);

        run_key_step(&mut app, &ctx, egui::Key::Escape);
        assert!(!app.transect_mode);
        assert!(app.transect_points.is_empty());

        // Projections 1..=4
        run_key_step(&mut app, &ctx, egui::Key::Num2);
        assert_eq!(app.projection, MapProjectionKind::Equirectangular2d);
        run_key_step(&mut app, &ctx, egui::Key::Num3);
        assert_eq!(app.projection, MapProjectionKind::ArcticPolarStereo);
        run_key_step(&mut app, &ctx, egui::Key::Num4);
        assert_eq!(app.projection, MapProjectionKind::AntarcticPolarStereo);
        run_key_step(&mut app, &ctx, egui::Key::Num1);
        assert_eq!(app.projection, MapProjectionKind::Globe3dWgs84);

        // L & A drawer toggles
        let left_init = app.left_sidebar_open;
        let right_init = app.right_drawer_open;
        run_key_step(&mut app, &ctx, egui::Key::L);
        assert_eq!(app.left_sidebar_open, !left_init);
        run_key_step(&mut app, &ctx, egui::Key::A);
        assert_eq!(app.right_drawer_open, !right_init);
        run_key_step(&mut app, &ctx, egui::Key::L);
        run_key_step(&mut app, &ctx, egui::Key::A);

        // R resets camera center, zoom, and bearing
        app.camera.center_lat = 48.0;
        app.camera.center_lon = 120.0;
        app.camera.zoom = 3.5;
        app.camera.bearing_deg = 55.0;
        run_key_step(&mut app, &ctx, egui::Key::R);
        assert!((app.camera.zoom - 1.0).abs() < 1e-9);
        assert!((app.camera.bearing_deg - 0.0).abs() < 1e-9);
        assert!((app.camera.center_lat - 20.0).abs() < 1e-6);

        // ArrowLeft / ArrowRight step simulation year in projection mode
        app.temporal_mode_id = "cmip_projections".to_string();
        app.sim_cfg.sim_year = 2050.0;
        run_key_step(&mut app, &ctx, egui::Key::ArrowRight);
        assert!((app.sim_cfg.sim_year - 2051.0).abs() < 1e-6);
        run_key_step(&mut app, &ctx, egui::Key::ArrowLeft);
        assert!((app.sim_cfg.sim_year - 2050.0).abs() < 1e-6);

        // Presets configure scenario, year, and layers cleanly
        app.apply_quick_preset(QuickScenarioPreset::HighEmissionsSsp585);
        assert_eq!(app.sim_cfg.scenario, SspScenario::Ssp5_85);
        assert!((app.sim_cfg.sim_year - 2100.0).abs() < 1e-6);
        assert!(app.layer_states["coastal_inundation_storm_surge"].visible);
        app.apply_quick_preset(QuickScenarioPreset::ModerateSsp245);
        assert_eq!(app.sim_cfg.scenario, SspScenario::Ssp2_45);
        assert!((app.sim_cfg.sim_year - 2050.0).abs() < 1e-6);
        app.apply_quick_preset(QuickScenarioPreset::BaselinePresent);
        assert_eq!(app.temporal_mode_id, "realtime_telemetry");

        drop(app);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }

    #[test]
    fn shortcuts_suppressed_when_text_input_focused_and_fidelity_helpers() {
        let db_path = std::env::temp_dir().join(format!(
            "terratwin_gui_focus_test_{}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&db_path);
        let schema = HubSchema::load(None).expect("embedded schema");
        let db = ClimateDatabase::open(&db_path).expect("temp db");
        let coord = BackgroundCoordinator::new(schema.clone(), db, SimulationConfig::default());
        let workers = RunningWorkers::without_threads(coord);
        let mut app = TerraTwinGuiApp::new_headless(workers);
        let ctx = egui::Context::default();

        // Render frame 0, focus layer search TextEdit, then verify ui_step ignores shortcuts while focused
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                ..Default::default()
            },
            |c| {
                app.ui_step(c);
                c.memory_mut(|m| m.request_focus(egui::Id::new("terratwin_layer_filter")));
            },
        );
        app.camera.zoom = 2.5;
        app.temporal_mode_id = "cmip_projections".to_string();
        app.sim_cfg.sim_year = 2050.0;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                events: vec![
                    egui::Event::Key {
                        key: egui::Key::R,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::Key {
                        key: egui::Key::Num2,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::Key {
                        key: egui::Key::L,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::Key {
                        key: egui::Key::ArrowRight,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            |c| {
                assert!(c.wants_keyboard_input());
                app.ui_step(c);
            },
        );
        assert!((app.camera.zoom - 2.5).abs() < 1e-9);
        assert_eq!(app.projection, MapProjectionKind::Globe3dWgs84);
        assert!(app.left_sidebar_open);
        assert!((app.sim_cfg.sim_year - 2050.0).abs() < 1e-9);

        // Layer filter matching across all 34 layers
        let matched: usize = schema
            .layer_groups
            .iter()
            .flat_map(|g| g.layers.iter().map(move |l| (g, l)))
            .filter(|(g, l)| layer_matches_filter(g, l, "methane"))
            .count();
        assert_eq!(matched, 1);

        // Graticule formatting
        assert_eq!(format_graticule_lat(60), "60°N");
        assert_eq!(format_graticule_lat(-30), "30°S");
        assert_eq!(format_graticule_lat(0), "0°");
        assert_eq!(format_graticule_lon(60), "60°E");
        assert_eq!(format_graticule_lon(-120), "120°W");
        assert_eq!(format_graticule_lon(0), "0°");

        // Lambertian globe shading is brighter on subsolar side (-0.42, 0.52) than dark limb (0.85, -0.45)
        let subsolar = apply_globe_lambert_fresnel([160.0, 160.0, 160.0], -0.42, 0.52);
        let dark_limb = apply_globe_lambert_fresnel([160.0, 160.0, 160.0], 0.80, -0.45);
        assert!(subsolar[0] > dark_limb[0] + 20);

        // Cursor-sampled active scalar readout
        let w = 12usize;
        let h = 6usize;
        let mut grids = HashMap::new();
        grids.insert(
            "cmip6_downscaled_ensemble".to_string(),
            crate::db::StoredGridRecord {
                layer_id: "cmip6_downscaled_ensemble".to_string(),
                scenario: "SSP2-4.5".to_string(),
                timestamp_utc: "2050-01-01T00:00:00Z".to_string(),
                vertical_level: 850.0,
                width: w,
                height: h,
                min_val: 2.0,
                max_val: 4.0,
                mean_val: 3.0,
                values: vec![3.5_f32; w * h],
                u_component: None,
                v_component: None,
            },
        );
        let snap = TelemetrySnapshot {
            config: app.sim_cfg.clone(),
            readouts: crate::db::GlobalLiveReadouts {
                timestamp_utc: "2050-01-01T00:00:00Z".to_string(),
                scenario: "SSP2-4.5".to_string(),
                global_2m_temp_anomaly_c: 2.0,
                mauna_loa_co2_ppm: 500.0,
                enso_nino34_c: 0.0,
                global_sst_anomaly_c: 1.5,
            },
            grids,
            db_stats: crate::db::DbStatsSummary {
                db_path: String::new(),
                registered_layers: 34,
                raw_observation_count: 0,
                simulated_grid_count: 1,
                readout_snapshots: 1,
                sync_events: 0,
                latest_simulation_utc: None,
            },
            last_step_ms: 10,
            last_ingest_status: String::new(),
        };
        app.apply_quick_preset(QuickScenarioPreset::ModerateSsp245);
        let (name, val, _) =
            sample_active_scalar_at(&schema, &snap, &app.layer_states, 10.0, -20.0, false).unwrap();
        assert!(name.contains("CMIP"));
        assert!((val - 3.5).abs() < 1e-4);
        let (_, anom_val, _) =
            sample_active_scalar_at(&schema, &snap, &app.layer_states, 10.0, -20.0, true).unwrap();
        assert!((anom_val - 0.5).abs() < 1e-4);

        drop(app);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }

    #[test]
    fn command_palette_ctrl_k_fuzzy_match_and_tipping_point_jump() {
        let db_path = std::env::temp_dir().join(format!(
            "terratwin_gui_palette_test_{}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&db_path);
        let schema = HubSchema::load(None).expect("embedded schema");
        let db = ClimateDatabase::open(&db_path).expect("temp db");
        let coord = BackgroundCoordinator::new(schema, db, SimulationConfig::default());
        let workers = RunningWorkers::without_threads(coord);
        let mut app = TerraTwinGuiApp::new_headless(workers);
        let ctx = egui::Context::default();

        assert!(fuzzy_match_score("permafrost", "Arctic Permafrost & Sea-Ice Loss").is_some());
        assert!(fuzzy_match_score("apsl", "Arctic Permafrost & Sea-Ice Loss").is_some());
        assert!(fuzzy_match_score("zzzz", "Arctic Permafrost & Sea-Ice Loss").is_none());

        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                events: vec![egui::Event::Key {
                    key: egui::Key::K,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::COMMAND,
                }],
                ..Default::default()
            },
            |c| app.ui_step(c),
        );
        assert!(app.command_palette_open, "Ctrl+K must open command palette");

        app.command_palette_query = "permafrost".to_string();
        let entries = app.build_palette_entries();
        let best = entries
            .iter()
            .filter_map(|e| {
                let hay = format!("{} {} {}", e.category, e.title, e.subtitle);
                fuzzy_match_score(&app.command_palette_query, &hay).map(|s| (s, e))
            })
            .max_by_key(|x| x.0)
            .expect("matching tipping point entry")
            .1
            .clone();
        app.apply_palette_command(&best.action);
        assert!((app.pinned_coord.0 - 72.0).abs() < 1e-6);
        assert!(app.layer_states["permafrost_thermokarst_subsidence"].visible);

        // Escape dismisses palette
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280.0, 800.0),
                )),
                events: vec![egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |c| app.ui_step(c),
        );
        assert!(!app.command_palette_open, "Esc must close command palette");

        drop(app);
        let _ = std::fs::remove_file(&db_path);
        let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
        let _ = std::fs::remove_file(db_path.with_extension("db-shm"));
    }
}
