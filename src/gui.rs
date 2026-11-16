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
    detect_mslp_extrema, detect_thermal_fronts, extract_marching_squares_contours,
    fibonacci_sphere_points, haversine_km, project_latlon_to_ndc, sample_grid_bilinear,
    unproject_ndc_to_latlon, CameraState, ContourSegment, FrontSegment, MapProjectionKind,
    PressureExtrema,
};
use crate::physics::{
    elevation_slope_m_per_deg, synthetic_planetary_elevation_m, ClimateSimulationEngine,
    SimulationConfig,
};
use crate::schema::SspScenario;
use crate::worker::{RunningWorkers, TelemetrySnapshot};
use anyhow::Result;
use egui::{Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

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
    sim_engine_local: ClimateSimulationEngine,
    sim_cfg: SimulationConfig,
    projection: MapProjectionKind,
    camera: CameraState,
    basemap_id: String,
    anomaly_mode: bool,

    left_sidebar_open: bool,
    right_drawer_open: bool,
    crosshair_inspector: bool,
    transect_mode: bool,
    transect_points: Vec<(f64, f64)>,

    pinned_coord: (f64, f64),
    coord_search_buf: String,
    temporal_mode_id: String,
    playing_timeline: bool,
    playback_timestamps: Vec<String>,
    playback_index: Option<usize>,
    playback_last_advance: std::time::Instant,
    last_timeline_refresh: std::time::Instant,
    export_status_msg: String,

    layer_states: HashMap<String, LayerUiState>,
    solo_restore_vis: Option<HashMap<String, bool>>,
    particles: Vec<LagrangianParticle>,
    map_texture: Option<TextureHandle>,
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
        let db = workers.coordinator.db().clone();
        let sim_engine_local = ClimateSimulationEngine::new(schema.clone(), db);

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
            sim_engine_local,
            sim_cfg: SimulationConfig::default(),
            projection: MapProjectionKind::from_id(&schema.main_viewport.default_projection),
            camera,
            basemap_id: default_basemap,
            anomaly_mode: false,
            left_sidebar_open: true,
            right_drawer_open: true,
            crosshair_inspector: true,
            transect_mode: false,
            transect_points: Vec::new(),
            pinned_coord: (cam_def.latitude, cam_def.longitude),
            coord_search_buf: format!("{:.4}, {:.4}", cam_def.latitude, cam_def.longitude),
            temporal_mode_id: "realtime_telemetry".to_string(),
            playing_timeline: false,
            playback_timestamps: Vec::new(),
            playback_index: None,
            playback_last_advance: std::time::Instant::now(),
            last_timeline_refresh: std::time::Instant::now() - std::time::Duration::from_secs(10),
            export_status_msg: String::new(),
            layer_states,
            solo_restore_vis: None,
            particles,
            map_texture: None,
            cached_snapshot: None,
            cached_isobars: Vec::new(),
            cached_extrema: Vec::new(),
            cached_fronts: Vec::new(),
            raster_dirty: true,
            last_raster_gen_key: 0,
            open_visible_groups_only: false,
        }
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
                        *out_px = Color32::from_rgb(6, 10, 18);
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

                    *out_px = Color32::from_rgb(r as u8, g as u8, b as u8);
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
                        .clicked()
                    {
                        self.left_sidebar_open = !self.left_sidebar_open;
                    }
                    ui.colored_label(
                        Color32::from_rgb(98, 210, 255),
                        egui::RichText::new("TerraTwin HPC v2.4").strong(),
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
                        );
                        draw_readout_badge(
                            ui,
                            "Mauna Loa CO2",
                            &format!("{:.1} ppm", r.mauna_loa_co2_ppm),
                            Color32::from_rgb(245, 200, 75),
                        );
                        draw_readout_badge(
                            ui,
                            "ENSO Niño 3.4",
                            &format!("{:+.2} °C", r.enso_nino34_c),
                            Color32::from_rgb(120, 215, 250),
                        );
                        draw_readout_badge(
                            ui,
                            "Global SST Anom",
                            &format!("{:+.2} °C", r.global_sst_anomaly_c),
                            Color32::from_rgb(255, 160, 105),
                        );
                    }
                    ui.separator();

                    // Projection Switcher
                    let mut proj_changed = false;
                    egui::ComboBox::from_label("Projection")
                        .selected_text(self.projection.id())
                        .show_ui(ui, |ui| {
                            for p in MapProjectionKind::ALL {
                                if ui
                                    .selectable_value(&mut self.projection, p, p.label())
                                    .changed()
                                {
                                    proj_changed = true;
                                }
                            }
                        });
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
                            .desired_width(125.0),
                    );
                    if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
                        || ui.button("Go").clicked()
                    {
                        let parts: Vec<&str> = self.coord_search_buf.split(',').collect();
                        if parts.len() == 2 {
                            if let (Ok(la), Ok(lo)) = (
                                parts[0].trim().parse::<f64>(),
                                parts[1].trim().parse::<f64>(),
                            ) {
                                self.pinned_coord = (la.clamp(-90.0, 90.0), lo.clamp(-180.0, 180.0));
                                self.camera.center_lat = self.pinned_coord.0;
                                self.camera.center_lon = self.pinned_coord.1;
                                self.raster_dirty = true;
                            }
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
                        .clicked()
                    {
                        self.transect_mode = !self.transect_mode;
                        if !self.transect_mode {
                            self.transect_points.clear();
                        }
                    }
                    if ui
                        .selectable_label(self.right_drawer_open, "Analytics")
                        .clicked()
                    {
                        self.right_drawer_open = !self.right_drawer_open;
                    }
                });
            });
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
                        .changed()
                    {
                        self.raster_dirty = true;
                    }
                });

                // Basemap Selector
                let basemaps = self.workers.coordinator.schema().main_viewport.base_maps.clone();
                egui::ComboBox::from_label("Basemap")
                    .selected_text(&self.basemap_id)
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
                ui.label(egui::RichText::new("Vertical Slice Engine").strong());
                let mut slice_changed = false;
                slice_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.sim_cfg.atmospheric_level_hpa, 10.0..=1000.0)
                            .logarithmic(true)
                            .text("Atm (hPa)"),
                    )
                    .changed();
                slice_changed |= ui
                    .add(
                        egui::Slider::new(&mut self.sim_cfg.ocean_depth_m, 0.0..=6000.0)
                            .text("Ocean (m)"),
                    )
                    .changed();
                if slice_changed {
                    self.push_sim_update();
                }

                ui.separator();
                let groups = self.workers.coordinator.schema().layer_groups.clone();
                let mut solo_clicked_id: Option<String> = None;
                let visible_count = self.layer_states.values().filter(|s| s.visible).count();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for grp in &groups {
                        let has_vis = grp.layers.iter().any(|l| {
                            self.layer_states
                                .get(&l.id)
                                .map(|s| s.visible)
                                .unwrap_or(false)
                        });
                        let is_open = if self.open_visible_groups_only {
                            has_vis
                        } else {
                            grp.default_expanded
                        };
                        egui::CollapsingHeader::new(
                            egui::RichText::new(&grp.group_name).strong(),
                        )
                        .default_open(is_open)
                        .show(ui, |ui| {
                            for lyr in &grp.layers {
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
                                        egui::ComboBox::from_label(format!("Pal ({})", lyr.id))
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
                    // 2. Skew-T / Log-P Vertical Atmospheric Profile (1000 -> 10 hPa)
                    ui.strong("Atmospheric Vertical Sounding (1000–10 hPa)");
                    let sounding = self
                        .sim_engine_local
                        .compute_vertical_sounding(&self.sim_cfg, plat, plon);
                    draw_sounding_canvas(ui, &sounding);

                    ui.separator();
                    // 3. Ocean Water Column CTD Profile (0 -> 6000 m)
                    ui.strong("Ocean Water-Column CTD Profile (0–6000 m)");
                    let ctd = self
                        .sim_engine_local
                        .compute_ocean_ctd_profile(&self.sim_cfg, plat, plon);
                    draw_ctd_canvas(ui, &ctd);

                    ui.separator();
                    // 4. Regional CO2 & Warming Projection + CMIP6 Ensemble Spread Boxplot
                    ui.strong("CMIP6 1950–2100 Trajectory & Ensemble Spread");
                    draw_cmip6_projection_canvas(ui, self.sim_cfg.scenario, self.sim_cfg.sim_year);

                    if self.transect_points.len() == 2 {
                        ui.separator();
                        let (a_lat, a_lon) = self.transect_points[0];
                        let (b_lat, b_lon) = self.transect_points[1];
                        let dist_km = haversine_km(a_lat, a_lon, b_lat, b_lon);
                        ui.strong(format!("Cross-Section Transect A-B ({:.0} km)", dist_km));
                        draw_transect_profile(ui, &self.cached_snapshot, &self.layer_states, (a_lat, a_lon), (b_lat, b_lon));
                    }

                    ui.separator();
                    ui.strong("Zarr v3 / JSON Subset Exporter");
                    if ui.button("Export Visible Top Layer Subset").clicked() {
                        let first_vis = self
                            .workers
                            .coordinator
                            .schema()
                            .all_layers()
                            .into_iter()
                            .find(|(_, l)| {
                                self.layer_states
                                    .get(&l.id)
                                    .map(|s| s.visible)
                                    .unwrap_or(false)
                            })
                            .map(|(_, l)| l.id.clone())
                            .unwrap_or_else(|| "cmip6_downscaled_ensemble".to_string());
                        let out_dir = PathBuf::from(format!("subset_{}.zarr", first_vis));
                        match self.workers.coordinator.db().export_zarr_v3_store(
                            &first_vis,
                            self.sim_cfg.scenario,
                            self.sim_cfg.atmospheric_level_hpa,
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
                                self.export_status_msg = format!("Zarr v3 export failed: {}", e);
                            }
                        }
                    }
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
                        .clicked()
                    {
                        self.playing_timeline = !self.playing_timeline;
                    }

                    let modes = self.workers.coordinator.schema().ui_layout.bottom_dock.modes.clone();
                    egui::ComboBox::from_label("Mode")
                        .selected_text(&self.temporal_mode_id)
                        .show_ui(ui, |ui| {
                            for m in &modes {
                                ui.selectable_value(
                                    &mut self.temporal_mode_id,
                                    m.id.clone(),
                                    &m.label,
                                );
                            }
                        });

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
                        .selectable_label(is_replay, "4D SQLite Replay")
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
                        let mut time_changed = false;
                        time_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.sim_cfg.sim_year, 1950.0..=2100.0)
                                    .text("Year"),
                            )
                            .changed();
                        time_changed |= ui
                            .add(
                                egui::Slider::new(&mut self.sim_cfg.day_of_year, 1.0..=365.0)
                                    .text("Day"),
                            )
                            .changed();
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

                // Pan / Globe Rotate on drag
                if response.dragged() {
                    let delta = response.drag_delta();
                    let deg_per_px = 180.0 / (rect.height().max(120.0) as f64 * self.camera.zoom);
                    self.camera.center_lon =
                        ((self.camera.center_lon - delta.x as f64 * deg_per_px + 540.0)
                            .rem_euclid(360.0))
                            - 180.0;
                    self.camera.center_lat =
                        (self.camera.center_lat + delta.y as f64 * deg_per_px).clamp(-88.0, 88.0);
                    self.raster_dirty = true;
                }

                // Zoom on mouse wheel
                if response.hovered() {
                    let scroll = ui.input(|i| i.raw_scroll_delta.y);
                    if scroll.abs() > 0.1 {
                        let factor = (1.0 + scroll as f64 * 0.0018).clamp(0.82, 1.22);
                        self.camera.zoom = (self.camera.zoom * factor).clamp(0.55, 8.5);
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

                // Re-rasterize composite texture when dirty (0.85x viewport buffer for crisp Natural Earth coastlines & cloud detail)
                let tex_w = ((rect.width() * 0.85) as usize).clamp(480, 1120);
                let tex_h = ((rect.height() * 0.85) as usize).clamp(300, 760);
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

                // 3D Globe atmosphere rim halo
                if self.projection == MapProjectionKind::Globe3dWgs84 {
                    let r_px = 0.86 * (self.camera.zoom as f32) * (rect.height() * 0.5);
                    painter.circle_stroke(
                        rect.center(),
                        r_px,
                        Stroke::new(2.0_f32, Color32::from_rgba_unmultiplied(90, 195, 255, 130)),
                    );
                }

                // Graticule (30 deg parallels & meridians)
                let grat_stroke =
                    Stroke::new(0.6_f32, Color32::from_rgba_unmultiplied(180, 210, 240, 38));
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
                }

                // Marching-Squares 4 hPa MSLP Isobars (thin translucent cyan-slate below coastlines) & Thermal Fronts
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
                            if p1.distance(p2) < rect.width() * 0.25 {
                                let is_major = ((seg.level - 1012.0).abs() < 0.5)
                                    || ((seg.level - 1000.0).abs() < 0.5);
                                painter.line_segment(
                                    [p1, p2],
                                    if is_major { iso_major } else { iso_thin },
                                );
                            }
                        }
                    }

                    let cold_stroke =
                        Stroke::new(1.8_f32, Color32::from_rgba_unmultiplied(70, 175, 255, 215));
                    let warm_stroke =
                        Stroke::new(1.8_f32, Color32::from_rgba_unmultiplied(255, 90, 75, 215));
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
                            if p1.distance(p2) < rect.width() * 0.2 {
                                painter.line_segment(
                                    [p1, p2],
                                    if fseg.is_cold_front {
                                        cold_stroke
                                    } else {
                                        warm_stroke
                                    },
                                );
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

                // A->B Transect Line
                if self.transect_points.len() == 2 {
                    let (a_la, a_lo) = self.transect_points[0];
                    let (b_la, b_lo) = self.transect_points[1];
                    if let (Some((ax, ay)), Some((bx, by))) = (
                        project_latlon_to_ndc(self.projection, &self.camera, a_la, a_lo, 0.0, aspect),
                        project_latlon_to_ndc(self.projection, &self.camera, b_la, b_lo, 0.0, aspect),
                    ) {
                        let pa = ndc_to_screen(ax, ay);
                        let pb = ndc_to_screen(bx, by);
                        painter.line_segment([pa, pb], Stroke::new(2.4_f32, Color32::from_rgb(255, 210, 60)));
                        painter.circle_filled(pa, 4.5, Color32::from_rgb(255, 210, 60));
                        painter.circle_filled(pb, 4.5, Color32::from_rgb(255, 140, 60));
                    }
                }

                // Hover HUD Tooltip
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
                        let hud_rect = Rect::from_min_size(
                            egui::pos2(rect.left() + 12.0, rect.bottom() - 34.0),
                            egui::vec2(310.0, 24.0),
                        );
                        painter.rect_filled(
                            hud_rect,
                            4.0,
                            Color32::from_rgba_unmultiplied(10, 16, 26, 210),
                        );
                        painter.text(
                            hud_rect.left_center() + egui::vec2(8.0, 0.0),
                            Align2::LEFT_CENTER,
                            format!(
                                "Cursor: {:+.2}°, {:+.2}° | Elev: {:+.0} m",
                                hlat,
                                hlon,
                                synthetic_planetary_elevation_m(hlat, hlon)
                            ),
                            FontId::monospace(11.0),
                            Color32::from_rgb(210, 230, 250),
                        );
                    }
                }
            });
    }
}

impl TerraTwinGuiApp {
    pub fn ui_step(&mut self, ctx: &egui::Context) {
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

        // Advance either 4D SQLite Replay ticks or Live simulation year when playing
        if self.playing_timeline {
            if let Some(cur_idx) = self.playback_index {
                if !self.playback_timestamps.is_empty()
                    && self.playback_last_advance.elapsed() >= std::time::Duration::from_millis(550)
                {
                    let next_idx = (cur_idx + 1) % self.playback_timestamps.len();
                    self.load_replay_snapshot_at_idx(next_idx);
                    self.playback_last_advance = std::time::Instant::now();
                }
            } else {
                self.sim_cfg.sim_year += 0.25;
                if self.sim_cfg.sim_year > 2100.0 {
                    self.sim_cfg.sim_year = 1950.0;
                }
                self.push_sim_update();
            }
        }

        self.advect_particles();

        self.render_top_bar(ctx);
        self.render_bottom_dock(ctx);
        self.render_left_sidebar(ctx);
        self.render_right_analytics_drawer(ctx);
        self.render_central_map(ctx);

        ctx.request_repaint_after(std::time::Duration::from_millis(45));
    }
}

impl eframe::App for TerraTwinGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui_step(ctx);
    }
}

fn draw_readout_badge(ui: &mut egui::Ui, label: &str, value: &str, color: Color32) {
    egui::Frame::none()
        .fill(Color32::from_rgb(22, 30, 46))
        .rounding(4.0)
        .inner_margin(egui::Margin::symmetric(7.0, 3.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.small(label);
                ui.colored_label(color, egui::RichText::new(value).monospace().strong());
            });
        });
}

fn draw_sounding_canvas(ui: &mut egui::Ui, sounding: &[(f64, f64, f64, f64)]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 145.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 20, 32));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));

    let map_pt = |p_hpa: f64, temp_c: f64| -> Pos2 {
        let y_frac = ((1000.0_f64.ln() - p_hpa.ln()) / (1000.0_f64.ln() - 10.0_f64.ln()))
            .clamp(0.0, 1.0) as f32;
        let x_frac = ((temp_c + 80.0) / 125.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + 8.0 + x_frac * (rect.width() - 16.0),
            rect.bottom() - 10.0 - y_frac * (rect.height() - 20.0),
        )
    };

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
        rect.left_top() + egui::vec2(8.0, 6.0),
        Align2::LEFT_TOP,
        "Red: Temp (°C) | Cyan: Dewpoint (°C)",
        FontId::monospace(9.5),
        Color32::from_rgb(190, 205, 225),
    );
}

fn draw_ctd_canvas(ui: &mut egui::Ui, ctd: &[(f64, f64, f64)]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 135.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(12, 22, 36));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));

    let map_temp = |z_m: f64, t_c: f64| -> Pos2 {
        let y_frac = (z_m / 6000.0).clamp(0.0, 1.0) as f32;
        let x_frac = ((t_c + 2.0) / 34.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + 8.0 + x_frac * (rect.width() - 16.0),
            rect.top() + 18.0 + y_frac * (rect.height() - 26.0),
        )
    };
    let map_sal = |z_m: f64, sal: f64| -> Pos2 {
        let y_frac = (z_m / 6000.0).clamp(0.0, 1.0) as f32;
        let x_frac = ((sal - 33.0) / 4.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + 8.0 + x_frac * (rect.width() - 16.0),
            rect.top() + 18.0 + y_frac * (rect.height() - 26.0),
        )
    };

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
        rect.left_top() + egui::vec2(8.0, 5.0),
        Align2::LEFT_TOP,
        "Orange: Ocean Temp (°C) | Teal: Salinity (PSU)",
        FontId::monospace(9.5),
        Color32::from_rgb(190, 205, 225),
    );
}

fn draw_cmip6_projection_canvas(ui: &mut egui::Ui, active_ssp: SspScenario, cur_year: f64) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 145.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 20, 32));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0_f32, Color32::from_rgb(42, 56, 78)));

    let map_yr_co2 = |yr: f64, ppm: f64| -> Pos2 {
        let xf = ((yr - 1950.0) / 150.0).clamp(0.0, 1.0) as f32;
        let yf = ((ppm - 300.0) / 860.0).clamp(0.0, 1.0) as f32;
        egui::pos2(
            rect.left() + 8.0 + xf * (rect.width() - 55.0),
            rect.bottom() - 10.0 - yf * (rect.height() - 24.0),
        )
    };

    for ssp in SspScenario::ALL {
        let is_sel = ssp == active_ssp;
        let col = match ssp {
            SspScenario::Ssp1_19 => Color32::from_rgb(70, 200, 245),
            SspScenario::Ssp1_26 => Color32::from_rgb(95, 225, 145),
            SspScenario::Ssp2_45 => Color32::from_rgb(245, 205, 80),
            SspScenario::Ssp3_70 => Color32::from_rgb(245, 130, 70),
            SspScenario::Ssp5_85 => Color32::from_rgb(235, 65, 75),
        };
        let stroke = Stroke::new(if is_sel { 2.4_f32 } else { 1.0_f32 }, col);
        let mut prev: Option<Pos2> = None;
        for yr_step in (1950..=2100).step_by(5) {
            let yr = yr_step as f64;
            let pt = map_yr_co2(yr, ssp.co2_ppm_at_year(yr));
            if let Some(p0) = prev {
                painter.line_segment([p0, pt], stroke);
            }
            prev = Some(pt);
        }
    }

    // Current year vertical marker
    let x_cur = map_yr_co2(cur_year, 400.0).x;
    painter.line_segment(
        [
            egui::pos2(x_cur, rect.top() + 14.0),
            egui::pos2(x_cur, rect.bottom() - 8.0),
        ],
        Stroke::new(1.2_f32, Color32::from_rgba_unmultiplied(255, 255, 255, 150)),
    );

    // 2100 Ensemble Spread Boxplot bar on right margin
    let bx = rect.right() - 24.0;
    let p10 = map_yr_co2(2100.0, SspScenario::Ssp1_19.co2_ppm_at_year(2100.0)).y;
    let p25 = map_yr_co2(2100.0, SspScenario::Ssp1_26.co2_ppm_at_year(2100.0)).y;
    let p50 = map_yr_co2(2100.0, SspScenario::Ssp2_45.co2_ppm_at_year(2100.0)).y;
    let p75 = map_yr_co2(2100.0, SspScenario::Ssp3_70.co2_ppm_at_year(2100.0)).y;
    let p90 = map_yr_co2(2100.0, SspScenario::Ssp5_85.co2_ppm_at_year(2100.0)).y;
    painter.line_segment([egui::pos2(bx, p90), egui::pos2(bx, p10)], Stroke::new(1.5_f32, Color32::LIGHT_GRAY));
    let box_rect = Rect::from_min_max(egui::pos2(bx - 8.0, p75), egui::pos2(bx + 8.0, p25));
    painter.rect_filled(box_rect, 2.0, Color32::from_rgba_unmultiplied(90, 175, 245, 120));
    painter.rect_stroke(box_rect, 2.0, Stroke::new(1.2_f32, Color32::WHITE));
    painter.line_segment(
        [egui::pos2(bx - 8.0, p50), egui::pos2(bx + 8.0, p50)],
        Stroke::new(2.0_f32, Color32::from_rgb(255, 215, 70)),
    );
}

fn draw_transect_profile(
    ui: &mut egui::Ui,
    snap_opt: &Option<TelemetrySnapshot>,
    layer_states: &HashMap<String, LayerUiState>,
    a: (f64, f64),
    b: (f64, f64),
) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(325.0, 105.0), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(14, 20, 32));
    let Some(snap) = snap_opt else { return };
    let n = 48usize;
    let Some((id, grid)) = snap
        .grids
        .iter()
        .filter(|(id, _)| layer_states.get(*id).map(|s| s.visible).unwrap_or(false))
        .max_by_key(|(_, g)| {
            (0..n)
                .filter(|&i| {
                    let t = i as f64 / (n - 1) as f64;
                    let la = a.0 + (b.0 - a.0) * t;
                    let lo = a.1 + (b.1 - a.1) * t;
                    sample_grid_bilinear(&g.values, g.width, g.height, la, lo).is_finite()
                })
                .count()
        })
    else {
        return;
    };

    let mut vals = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let la = a.0 + (b.0 - a.0) * t;
        let lo = a.1 + (b.1 - a.1) * t;
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
            rect.bottom() - 8.0 - yf * (rect.height() - 24.0),
        );
        if let Some(p0) = prev {
            painter.line_segment([p0, pt], Stroke::new(2.0_f32, Color32::from_rgb(255, 210, 65)));
        }
        prev = Some(pt);
    }
    painter.text(
        rect.left_top() + egui::vec2(8.0, 4.0),
        Align2::LEFT_TOP,
        format!("Transect: {}", id),
        FontId::monospace(9.5),
        Color32::LIGHT_GRAY,
    );
}

fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    fn noop_clone(_: *const ()) -> std::task::RawWaker {
        noop_raw_waker()
    }
    fn noop(_: *const ()) {}
    fn noop_raw_waker() -> std::task::RawWaker {
        static VTABLE: std::task::RawWakerVTable =
            std::task::RawWakerVTable::new(noop_clone, noop, noop, noop);
        std::task::RawWaker::new(std::ptr::null(), &VTABLE)
    }
    let waker = unsafe { std::task::Waker::from_raw(noop_raw_waker()) };
    let mut cx = std::task::Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(val) => return val,
            std::task::Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn write_rgba_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
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
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let (mut s1, mut s2) = (1u32, 0u32);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    let mut zlib = Vec::with_capacity(raw.len() + (raw.len() / 65535 + 1) * 5 + 6);
    zlib.extend_from_slice(&[0x78, 0x01]);
    let n_chunks = raw.len().div_ceil(65535).max(1);
    for (idx, chunk) in raw.chunks(65535).enumerate() {
        zlib.push(if idx + 1 == n_chunks { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(chunk);
    }
    zlib.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

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
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1520.0, 900.0])
            .with_min_inner_size([1024.0, 660.0])
            .with_title("TerraTwin HPC Climate Modeling & Telemetry Hub (v2.4.0)"),
        ..Default::default()
    };

    eframe::run_native(
        "TerraTwin HPC Climate Modeling & Telemetry Hub",
        native_options,
        Box::new(|cc| Ok(Box::new(TerraTwinGuiApp::new(cc, workers)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe native window error: {}", e))
}
