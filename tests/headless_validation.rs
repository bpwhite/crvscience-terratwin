use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Map, Value};
use terratwin_climate_hub::db::ClimateDatabase;
use terratwin_climate_hub::physics::{
    co2_radiative_forcing_wm2, eos80_seawater_density, ClimateSimulationEngine, SimulationConfig,
};
use terratwin_climate_hub::schema::{HubSchema, SspScenario};

static SUMMARY_LOCK: Mutex<()> = Mutex::new(());

const EARTH_RADIUS_KM: f64 = 6_371.0;
const SEAWATER_CP_J_KG_K: f64 = 3_985.0;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn validation_dir() -> PathBuf {
    let dir = repo_root().join("validation");
    fs::create_dir_all(&dir).expect("Failed creating validation directory");
    dir
}

fn temp_db_path(tag: &str) -> PathBuf {
    let pid = std::process::id();
    std::env::temp_dir().join(format!("terratwin_headless_val_{tag}_{pid}.db"))
}

#[inline]
fn cell_lat_deg(row: usize, height: usize) -> f64 {
    90.0 - (row as f64 + 0.5) * (180.0 / height as f64)
}

#[inline]
fn cell_lon_deg(col: usize, width: usize) -> f64 {
    -180.0 + (col as f64 + 0.5) * (360.0 / width as f64)
}

#[inline]
fn cell_area_km2(row: usize, width: usize, height: usize) -> f64 {
    let lat = cell_lat_deg(row, height).to_radians();
    let dlat = PI / (height as f64);
    let dlon = (2.0 * PI) / (width as f64);
    (EARTH_RADIUS_KM * EARTH_RADIUS_KM) * dlat * dlon * lat.cos().abs().max(1e-6)
}

fn merge_summary_section(section_key: &str, section_val: Value) {
    let _guard = SUMMARY_LOCK.lock().expect("Summary mutex poisoned");
    let vdir = validation_dir();
    let local_path = vdir.join("terratwin_validation_summary.json");

    let mut root: Map<String, Value> = if local_path.exists() {
        fs::read_to_string(&local_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default()
    } else {
        Map::new()
    };

    root.insert("repository".into(), json!("crvscience-terratwin"));
    root.insert("version".into(), json!(env!("CARGO_PKG_VERSION")));
    root.insert("slug".into(), json!("terratwin-coupled-climate-ssp"));
    root.insert("preprint_id".into(), json!("APP-PREPRINT-10"));
    root.insert("concept_doi".into(), json!("10.5281/zenodo.24222392"));
    root.insert(
        "preprint_url".into(),
        json!("https://crvscience.com/apps/preprints/terratwin-coupled-climate-ssp"),
    );
    root.insert(section_key.into(), section_val);

    let serialized = serde_json::to_string_pretty(&Value::Object(root))
        .expect("Failed serializing validation summary");
    assert!(
        serialized.len() < 384 * 1024,
        "terratwin_validation_summary.json exceeded 384 KB limit: {} bytes",
        serialized.len()
    );
    fs::write(&local_path, &serialized).expect("Failed writing local summary JSON");

    // Mirror to app-preprints/releases when running inside the CRVScience monorepo
    if let Some(rel_dir) = repo_root()
        .ancestors()
        .map(|anc| anc.join("app-preprints").join("releases"))
        .find(|p| p.exists())
    {
        let mirror = rel_dir.join("terratwin-coupled-climate-ssp-validation-telemetry.json");
        fs::write(&mirror, &serialized).expect("Failed mirroring summary JSON");
    }
}

fn area_weighted_band_mean(
    values: &[f32],
    width: usize,
    height: usize,
    lat_min: f64,
    lat_max: f64,
) -> Option<f64> {
    let mut sum_w = 0.0;
    let mut sum_v = 0.0;
    for r in 0..height {
        let lat = cell_lat_deg(r, height);
        if lat < lat_min || lat > lat_max {
            continue;
        }
        let area = cell_area_km2(r, width, height);
        for c in 0..width {
            let v = values[r * width + c] as f64;
            if v.is_finite() {
                sum_v += v * area;
                sum_w += area;
            }
        }
    }
    if sum_w > 0.0 {
        Some(sum_v / sum_w)
    } else {
        None
    }
}

#[test]
fn t1_coupled_ssp_radiative_forcing_and_temp_trajectories() {
    let db_path = temp_db_path("t1");
    let _ = fs::remove_file(&db_path);

    let schema = HubSchema::load(None).expect("Embedded schema must load");
    let db = ClimateDatabase::open(&db_path).expect("Open temp SQLite DB");
    db.sync_schema_catalog(&schema).expect("Sync schema catalog");
    let engine = ClimateSimulationEngine::new(schema, db.clone());

    let scenarios = [
        SspScenario::Ssp1_19,
        SspScenario::Ssp2_45,
        SspScenario::Ssp3_70,
        SspScenario::Ssp5_85,
    ];
    let years: Vec<u32> = (1950..=2100).step_by(5).collect();

    let mut csv = String::from(
        "scenario,year,co2_ppm,radiative_forcing_wm2,global_2m_temp_anomaly_c,global_sst_anomaly_c,arctic_warming_c,tropics_warming_c\n",
    );
    let mut end_2100 = Map::new();

    for scen in scenarios {
        let mut prev_t2m_2050 = f64::NEG_INFINITY;
        let mut t2m_1950 = 0.0_f64;
        let mut t2m_2100 = 0.0_f64;
        let mut rf_2100 = 0.0_f64;
        let mut co2_2100 = 0.0_f64;

        for &yr in &years {
            // Diagnostic-annualized mid-spring/mid-autumn pair cancels seasonal solar tilt bias
            let mut sum_glob_t = 0.0;
            let mut sum_sst = 0.0;
            let mut sum_arc = 0.0;
            let mut sum_trop = 0.0;
            let mut co2_ppm = 0.0;
            let mut forcing_wm2 = 0.0;

            for doy in [105.0, 288.0] {
                let cfg = SimulationConfig {
                    grid_width: 48,
                    grid_height: 24,
                    scenario: scen,
                    sim_year: yr as f64 + doy / 365.25,
                    day_of_year: doy,
                    hour_utc: 12.0,
                    atmospheric_level_hpa: 850.0,
                    ocean_depth_m: 0.0,
                    time_step_secs: 3600.0,
                    thermal_diffusivity_m2_s: 4.5e5,
                    prognostic_relaxation_alpha: 0.55,
                };
                let readout = engine
                    .step_all_layers(&cfg)
                    .expect("step_all_layers failed");
                co2_ppm = readout.mauna_loa_co2_ppm;
                forcing_wm2 = co2_radiative_forcing_wm2(co2_ppm);
                sum_glob_t += readout.global_2m_temp_anomaly_c;
                sum_sst += readout.global_sst_anomaly_c;

                let anom_grid = db
                    .load_grid_at_timestamp(
                        "cmip6_downscaled_ensemble",
                        scen,
                        cfg.atmospheric_level_hpa,
                        &readout.timestamp_utc,
                    )
                    .expect("load cmip6 grid")
                    .expect("cmip6 grid missing");
                sum_arc += area_weighted_band_mean(
                    &anom_grid.values,
                    anom_grid.width,
                    anom_grid.height,
                    66.5,
                    90.0,
                )
                .expect("Arctic band missing");
                sum_trop += area_weighted_band_mean(
                    &anom_grid.values,
                    anom_grid.width,
                    anom_grid.height,
                    -23.5,
                    23.5,
                )
                .expect("Tropics band missing");
            }

            let glob_t = sum_glob_t * 0.5;
            let glob_sst = sum_sst * 0.5;
            let arc_t = sum_arc * 0.5;
            let trop_t = sum_trop * 0.5;

            csv.push_str(&format!(
                "{},{},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                scen.code(),
                yr,
                co2_ppm,
                forcing_wm2,
                glob_t,
                glob_sst,
                arc_t,
                trop_t
            ));

            if yr == 1950 {
                t2m_1950 = glob_t;
            }
            if yr == 2050 {
                prev_t2m_2050 = glob_t;
            }
            if yr == 2100 {
                t2m_2100 = glob_t;
                rf_2100 = forcing_wm2;
                co2_2100 = co2_ppm;
            }
        }

        assert!(
            t2m_2100 > t2m_1950,
            "{}: 2100 anomaly ({:.3} C) must exceed 1950 ({:.3} C)",
            scen.code(),
            t2m_2100,
            t2m_1950
        );
        if scen == SspScenario::Ssp1_19 {
            // SSP1-1.9 peaks mid-century before net-negative drawdown
            assert!(
                prev_t2m_2050 >= t2m_2100 - 0.15,
                "SSP1-1.9 should stabilize/draw down after mid-century"
            );
        }

        end_2100.insert(
            scen.code().into(),
            json!({
                "nominal_target_forcing_2100_wm2": scen.forcing_2100_wm2(),
                "co2_ppm_2100": (co2_2100 * 100.0).round() / 100.0,
                "co2_radiative_forcing_2100_wm2": (rf_2100 * 1000.0).round() / 1000.0,
                "global_2m_temp_anomaly_1950_c": (t2m_1950 * 1000.0).round() / 1000.0,
                "global_2m_temp_anomaly_2050_c": (prev_t2m_2050 * 1000.0).round() / 1000.0,
                "global_2m_temp_anomaly_2100_c": (t2m_2100 * 1000.0).round() / 1000.0,
            }),
        );
    }

    // Verify strict SSP ordering in 2100 across pathways
    let get_t2100 = |code: &str| -> f64 {
        end_2100[code]["global_2m_temp_anomaly_2100_c"]
            .as_f64()
            .unwrap()
    };
    let t_ssp1 = get_t2100("SSP1-1.9");
    let t_ssp2 = get_t2100("SSP2-4.5");
    let t_ssp3 = get_t2100("SSP3-7.0");
    let t_ssp5 = get_t2100("SSP5-8.5");
    assert!(
        t_ssp1 < t_ssp2 && t_ssp2 < t_ssp3 && t_ssp3 < t_ssp5,
        "2100 warming must increase across SSPs: {t_ssp1:.3} < {t_ssp2:.3} < {t_ssp3:.3} < {t_ssp5:.3}"
    );

    assert!(csv.len() < 384 * 1024, "CSV exceeded 384 KB cap");
    let csv_path = validation_dir().join("ssp_projections_1950_2100.csv");
    fs::write(&csv_path, &csv).expect("Failed writing ssp_projections_1950_2100.csv");

    merge_summary_section(
        "t1_ssp_trajectories_1950_2100",
        json!({
            "csv_artifact": "validation/ssp_projections_1950_2100.csv",
            "csv_bytes": csv.len(),
            "step_years": 5,
            "row_count": scenarios.len() * years.len(),
            "endpoints_2100": Value::Object(end_2100),
        }),
    );

    let _ = fs::remove_file(&db_path);
}

#[test]
fn t2_multilayer_cryosphere_ocean_slr_and_biome_aridity() {
    let db_path = temp_db_path("t2");
    let _ = fs::remove_file(&db_path);

    let schema = HubSchema::load(None).expect("Embedded schema must load");
    let db = ClimateDatabase::open(&db_path).expect("Open temp SQLite DB");
    db.sync_schema_catalog(&schema).expect("Sync schema catalog");
    let engine = ClimateSimulationEngine::new(schema, db.clone());

    let scenarios = [
        SspScenario::Ssp1_19,
        SspScenario::Ssp2_45,
        SspScenario::Ssp3_70,
        SspScenario::Ssp5_85,
    ];

    let mut per_scenario = Map::new();
    let mut slr_2100_vec = Vec::new();
    let mut sea_ice_2100_vec = Vec::new();
    let mut ohc_2100_vec = Vec::new();
    let mut fwi_land_2100_vec = Vec::new();

    for scen in scenarios {
        // Warmup ticks with monthly prognostic time-steps before dual-season 2100 sampling
        for (yr, doy) in [
            (2000.0, 45.0),
            (2040.0, 45.0),
            (2075.0, 45.0),
        ] {
            let cfg = SimulationConfig {
                grid_width: 144,
                grid_height: 72,
                scenario: scen,
                sim_year: yr + doy / 365.25,
                day_of_year: doy,
                hour_utc: 12.0,
                atmospheric_level_hpa: 850.0,
                ocean_depth_m: 100.0,
                time_step_secs: 3600.0,
                thermal_diffusivity_m2_s: 4.5e5,
                prognostic_relaxation_alpha: 0.75,
            };
            engine.step_all_layers(&cfg).unwrap();
        }

        // Sample 2100 across NH winter / SH summer (doy=45) and NH summer / SH winter (doy=225)
        let mut sum_ice_area_mkm2 = 0.0;
        let mut sum_ice_ext_mkm2 = 0.0;
        let mut sum_nh_ice_mkm2 = 0.0;
        let mut sum_ice_thick_m = 0.0;
        let mut sum_slr_cm = 0.0;
        let mut sum_surge_m = 0.0;
        let mut sum_ohc_gj_m2 = 0.0;
        let mut sum_sub_ocean_c = 0.0;
        let mut sum_sm = 0.0;
        let mut sum_sif = 0.0;
        let mut sum_fwi = 0.0;
        let mut sum_wbgt = 0.0;
        let mut sum_arid_pct = 0.0;

        let seasonal_doys = [45.0_f64, 225.0_f64];
        for &doy in &seasonal_doys {
            let cfg_2100 = SimulationConfig {
                grid_width: 144,
                grid_height: 72,
                scenario: scen,
                sim_year: 2100.0 + doy / 365.25,
                day_of_year: doy,
                hour_utc: 12.0,
                atmospheric_level_hpa: 850.0,
                ocean_depth_m: 100.0,
                time_step_secs: 3600.0,
                thermal_diffusivity_m2_s: 4.5e5,
                prognostic_relaxation_alpha: 0.85,
            };
            let ts = engine.step_all_layers(&cfg_2100).unwrap().timestamp_utc;

            // 1. Cryosphere sea-ice area (continuous), extent (>= 15% conc), & mean thickness
            let ice_conc = db
                .load_grid_at_timestamp("sea_ice_concentration_drift", scen, 0.0, &ts)
                .unwrap()
                .expect("sea_ice_concentration_drift missing");
            let ice_thick = db
                .load_grid_at_timestamp("sea_ice_thickness_freeboard_age", scen, 0.0, &ts)
                .unwrap()
                .expect("sea_ice_thickness_freeboard_age missing");
            let mut sea_ice_area_km2 = 0.0;
            let mut sea_ice_extent_km2 = 0.0;
            let mut nh_sea_ice_extent_km2 = 0.0;
            let mut thick_area_sum = 0.0;
            for r in 0..ice_conc.height {
                let lat = cell_lat_deg(r, ice_conc.height);
                let area = cell_area_km2(r, ice_conc.width, ice_conc.height);
                for c in 0..ice_conc.width {
                    let idx = r * ice_conc.width + c;
                    let conc = ice_conc.values[idx] as f64;
                    if conc > 0.0 {
                        sea_ice_area_km2 += area * (conc / 100.0);
                    }
                    if conc >= 15.0 {
                        sea_ice_extent_km2 += area;
                        if lat > 0.0 {
                            nh_sea_ice_extent_km2 += area;
                        }
                        thick_area_sum += (ice_thick.values[idx] as f64) * area;
                    }
                }
            }
            sum_ice_area_mkm2 += sea_ice_area_km2 / 1.0e6;
            sum_ice_ext_mkm2 += sea_ice_extent_km2 / 1.0e6;
            sum_nh_ice_mkm2 += nh_sea_ice_extent_km2 / 1.0e6;
            sum_ice_thick_m += if sea_ice_extent_km2 > 0.0 {
                thick_area_sum / sea_ice_extent_km2
            } else {
                0.0
            };

            // 2. Thermosteric + dynamic SSH sea-level rise & coastal surge
            let ssh = db
                .load_grid_at_timestamp("swot_sea_surface_height_altimetry", scen, 0.0, &ts)
                .unwrap()
                .expect("swot_sea_surface_height_altimetry missing");
            let surge = db
                .load_grid_at_timestamp("coastal_inundation_storm_surge", scen, 0.0, &ts)
                .unwrap()
                .expect("coastal_inundation_storm_surge missing");
            sum_slr_cm +=
                area_weighted_band_mean(&ssh.values, ssh.width, ssh.height, -90.0, 90.0)
                    .expect("Ocean SSH missing");
            let mut coastal_surge_sum = 0.0;
            let mut coastal_area = 0.0;
            for r in 0..surge.height {
                let area = cell_area_km2(r, surge.width, surge.height);
                for c in 0..surge.width {
                    let v = surge.values[r * surge.width + c] as f64;
                    if v > 0.0 {
                        coastal_surge_sum += v * area;
                        coastal_area += area;
                    }
                }
            }
            sum_surge_m += if coastal_area > 0.0 {
                coastal_surge_sum / coastal_area
            } else {
                0.0
            };

            // 3. Upper-ocean heat content (0..2000 m trapezoid integration over sampled global ocean CTD stations)
            let ocean_stations = [
                (0.0, -140.0),
                (30.0, -45.0),
                (-30.0, 80.0),
                (45.0, -165.0),
                (-55.0, 15.0),
                (68.0, 5.0),
            ];
            let mut station_ohc_gj_m2 = Vec::new();
            for &(lat, lon) in &ocean_stations {
                let ctd = engine.compute_ocean_ctd_profile(&cfg_2100, lat, lon);
                let mut ohc_j_m2 = 0.0;
                for w in ctd.windows(2) {
                    let (z0, t0, s0) = w[0];
                    let (z1, t1, s1) = w[1];
                    if z1 > 2000.0 {
                        continue;
                    }
                    let dz = z1 - z0;
                    let rho = eos80_seawater_density(0.5 * (t0 + t1), 0.5 * (s0 + s1), 0.5 * (z0 + z1));
                    let t_mid_k = 0.5 * (t0 + t1) + 273.15;
                    ohc_j_m2 += rho * SEAWATER_CP_J_KG_K * (t_mid_k - 273.15) * dz;
                }
                station_ohc_gj_m2.push(ohc_j_m2 / 1.0e9);
            }
            sum_ohc_gj_m2 +=
                station_ohc_gj_m2.iter().sum::<f64>() / (station_ohc_gj_m2.len() as f64);

            // Check subsurface 3D ocean grid exists & is finite over ocean
            let sub_ocean = db
                .load_grid_at_timestamp("subsurface_ocean_heat_salinity_3d", scen, 100.0, &ts)
                .unwrap()
                .expect("subsurface_ocean_heat_salinity_3d missing");
            sum_sub_ocean_c +=
                area_weighted_band_mean(&sub_ocean.values, sub_ocean.width, sub_ocean.height, -90.0, 90.0)
                    .unwrap();

            // 4. Biome aridity & fire weather stress index over land
            let sm_grid = db
                .load_grid_at_timestamp("soil_moisture_groundwater_grace", scen, 0.0, &ts)
                .unwrap()
                .expect("soil_moisture_groundwater_grace missing");
            let sif_grid = db
                .load_grid_at_timestamp("sif_photosynthesis_evaporative_stress", scen, 0.0, &ts)
                .unwrap()
                .expect("sif_photosynthesis_evaporative_stress missing");
            let fwi_grid = db
                .load_grid_at_timestamp("compound_hazard_fire_weather_index", scen, 0.0, &ts)
                .unwrap()
                .expect("compound_hazard_fire_weather_index missing");
            let wbgt_grid = db
                .load_grid_at_timestamp("wet_bulb_globe_temp_heat_stress", scen, 0.0, &ts)
                .unwrap()
                .expect("wet_bulb_globe_temp_heat_stress missing");

            let mut sm_sum = 0.0;
            let mut sif_sum = 0.0;
            let mut fwi_sum = 0.0;
            let mut wbgt_sum = 0.0;
            let mut arid_frac_area = 0.0;
            let mut land_area = 0.0;
            for r in 0..sm_grid.height {
                let area = cell_area_km2(r, sm_grid.width, sm_grid.height);
                for c in 0..sm_grid.width {
                    let idx = r * sm_grid.width + c;
                    let sm = sm_grid.values[idx] as f64;
                    if sm.is_finite() {
                        let fwi = fwi_grid.values[idx] as f64;
                        let sif = sif_grid.values[idx] as f64;
                        let wbgt = wbgt_grid.values[idx] as f64;
                        sm_sum += sm * area;
                        sif_sum += sif * area;
                        fwi_sum += fwi * area;
                        wbgt_sum += wbgt * area;
                        if sm < 0.22 {
                            arid_frac_area += area;
                        }
                        land_area += area;
                    }
                }
            }
            sum_sm += sm_sum / land_area;
            sum_sif += sif_sum / land_area;
            sum_fwi += fwi_sum / land_area;
            sum_wbgt += wbgt_sum / land_area;
            sum_arid_pct += 100.0 * arid_frac_area / land_area;
        }

        let n_seas = seasonal_doys.len() as f64;
        let sea_ice_area_mkm2 = sum_ice_area_mkm2 / n_seas;
        let sea_ice_extent_mkm2 = sum_ice_ext_mkm2 / n_seas;
        let nh_sea_ice_mkm2 = sum_nh_ice_mkm2 / n_seas;
        let mean_ice_thickness_m = sum_ice_thick_m / n_seas;
        let mean_slr_cm = sum_slr_cm / n_seas;
        let mean_coastal_surge_m = sum_surge_m / n_seas;
        let mean_ohc_gj_m2 = sum_ohc_gj_m2 / n_seas;
        let sub_ocean_mean_c = sum_sub_ocean_c / n_seas;
        let mean_sm = sum_sm / n_seas;
        let mean_sif = sum_sif / n_seas;
        let mean_fwi = sum_fwi / n_seas;
        let mean_wbgt = sum_wbgt / n_seas;
        let arid_biome_pct = sum_arid_pct / n_seas;
        let biome_aridity_index = mean_fwi / (mean_sm * 100.0).max(1.0);

        slr_2100_vec.push(mean_slr_cm);
        sea_ice_2100_vec.push(sea_ice_area_mkm2);
        ohc_2100_vec.push(mean_ohc_gj_m2);
        fwi_land_2100_vec.push(mean_fwi);

        per_scenario.insert(
            scen.code().into(),
            json!({
                "global_sea_ice_area_mkm2": (sea_ice_area_mkm2 * 1000.0).round() / 1000.0,
                "global_sea_ice_extent_mkm2": (sea_ice_extent_mkm2 * 1000.0).round() / 1000.0,
                "nh_sea_ice_extent_mkm2": (nh_sea_ice_mkm2 * 1000.0).round() / 1000.0,
                "mean_sea_ice_thickness_m": (mean_ice_thickness_m * 1000.0).round() / 1000.0,
                "thermosteric_slr_mean_cm": (mean_slr_cm * 1000.0).round() / 1000.0,
                "mean_coastal_surge_m": (mean_coastal_surge_m * 1000.0).round() / 1000.0,
                "upper_ocean_ohc_0_2000m_gj_m2": (mean_ohc_gj_m2 * 1000.0).round() / 1000.0,
                "subsurface_100m_temp_mean_c": (sub_ocean_mean_c * 1000.0).round() / 1000.0,
                "land_soil_moisture_m3_m3": (mean_sm * 10000.0).round() / 10000.0,
                "land_sif_photosynthesis": (mean_sif * 1000.0).round() / 1000.0,
                "land_fire_weather_index": (mean_fwi * 1000.0).round() / 1000.0,
                "land_wbgt_heat_stress_c": (mean_wbgt * 1000.0).round() / 1000.0,
                "arid_biome_land_pct": (arid_biome_pct * 100.0).round() / 100.0,
                "biome_aridity_index": (biome_aridity_index * 10000.0).round() / 10000.0,
            }),
        );
    }

    // Physical monotonicity checks across SSP1-1.9 -> SSP2-4.5 -> SSP3-7.0 -> SSP5-8.5
    assert!(
        slr_2100_vec[0] < slr_2100_vec[1]
            && slr_2100_vec[1] < slr_2100_vec[2]
            && slr_2100_vec[2] < slr_2100_vec[3],
        "Sea-level rise in 2100 must increase monotonically with forcing: {:?}",
        slr_2100_vec
    );
    assert!(
        sea_ice_2100_vec[3] < sea_ice_2100_vec[0],
        "SSP5-8.5 sea-ice extent ({:.2} Mkm2) must be lower than SSP1-1.9 ({:.2} Mkm2)",
        sea_ice_2100_vec[3],
        sea_ice_2100_vec[0]
    );
    assert!(
        ohc_2100_vec[3] > ohc_2100_vec[0],
        "SSP5-8.5 ocean heat content ({:.2} GJ/m2) must exceed SSP1-1.9 ({:.2} GJ/m2)",
        ohc_2100_vec[3],
        ohc_2100_vec[0]
    );
    assert!(
        fwi_land_2100_vec[3] > fwi_land_2100_vec[0],
        "SSP5-8.5 fire-weather aridity stress ({:.2}) must exceed SSP1-1.9 ({:.2})",
        fwi_land_2100_vec[3],
        fwi_land_2100_vec[0]
    );

    merge_summary_section(
        "t2_multilayer_telemetry_2100",
        json!({
            "scenarios": Value::Object(per_scenario),
        }),
    );

    let _ = fs::remove_file(&db_path);
}

#[test]
fn t3_arctic_amplification_polar_to_tropics_ratio() {
    let db_path = temp_db_path("t3");
    let _ = fs::remove_file(&db_path);

    let schema = HubSchema::load(None).expect("Embedded schema must load");
    let db = ClimateDatabase::open(&db_path).expect("Open temp SQLite DB");
    db.sync_schema_catalog(&schema).expect("Sync schema catalog");
    let engine = ClimateSimulationEngine::new(schema, db.clone());

    let mut ratios = Map::new();
    for scen in [SspScenario::Ssp2_45, SspScenario::Ssp3_70, SspScenario::Ssp5_85] {
        // Sample equinox/annual-mean phase across symmetric opposing seasons to isolate forced polar amplification
        let mut arctic_sum = 0.0;
        let mut tropics_sum = 0.0;
        let mut global_sum = 0.0;
        let mut antarctic_sum = 0.0;
        let seasons = [46.0, 136.0, 228.0, 319.0];

        for &doy in &seasons {
            let cfg = SimulationConfig {
                grid_width: 72,
                grid_height: 36,
                scenario: scen,
                sim_year: 2100.0 + doy / 365.25,
                day_of_year: doy,
                hour_utc: 12.0,
                atmospheric_level_hpa: 850.0,
                ocean_depth_m: 0.0,
                time_step_secs: 3600.0,
                thermal_diffusivity_m2_s: 4.5e5,
                prognostic_relaxation_alpha: 1.0,
            };
            let readout = engine.step_all_layers(&cfg).unwrap();
            let anom = db
                .load_grid_at_timestamp(
                    "cmip6_downscaled_ensemble",
                    scen,
                    850.0,
                    &readout.timestamp_utc,
                )
                .unwrap()
                .expect("cmip6_downscaled_ensemble grid missing");

            arctic_sum +=
                area_weighted_band_mean(&anom.values, anom.width, anom.height, 66.5, 90.0).unwrap();
            tropics_sum +=
                area_weighted_band_mean(&anom.values, anom.width, anom.height, -23.5, 23.5).unwrap();
            global_sum +=
                area_weighted_band_mean(&anom.values, anom.width, anom.height, -90.0, 90.0).unwrap();
            antarctic_sum +=
                area_weighted_band_mean(&anom.values, anom.width, anom.height, -90.0, -66.5).unwrap();
        }

        let n = seasons.len() as f64;
        let arctic_dt = arctic_sum / n;
        let tropics_dt = tropics_sum / n;
        let global_dt = global_sum / n;
        let antarctic_dt = antarctic_sum / n;
        let arctic_to_tropics = arctic_dt / tropics_dt.max(1e-6);
        let arctic_to_global = arctic_dt / global_dt.max(1e-6);

        assert!(
            arctic_to_tropics >= 1.8 && arctic_to_tropics <= 5.0,
            "{}: Arctic/Tropics ratio {:.3} out of physical bounds [1.8, 5.0] (Arctic={:.2}C, Tropics={:.2}C)",
            scen.code(),
            arctic_to_tropics,
            arctic_dt,
            tropics_dt
        );
        assert!(
            arctic_dt > global_dt && global_dt > tropics_dt,
            "{}: expected Arctic ({:.2}C) > Global ({:.2}C) > Tropics ({:.2}C)",
            scen.code(),
            arctic_dt,
            global_dt,
            tropics_dt
        );

        ratios.insert(
            scen.code().into(),
            json!({
                "arctic_gt_66_5n_warming_c": (arctic_dt * 1000.0).round() / 1000.0,
                "tropics_pm_23_5_warming_c": (tropics_dt * 1000.0).round() / 1000.0,
                "global_mean_warming_c": (global_dt * 1000.0).round() / 1000.0,
                "antarctic_lt_66_5s_warming_c": (antarctic_dt * 1000.0).round() / 1000.0,
                "arctic_to_tropics_amplification_ratio": (arctic_to_tropics * 1000.0).round() / 1000.0,
                "arctic_to_global_amplification_ratio": (arctic_to_global * 1000.0).round() / 1000.0,
            }),
        );
    }

    // Also inspect cell longitude helper sanity for unused-fn prevention
    let _lon0 = cell_lon_deg(0, 72);

    merge_summary_section(
        "t3_arctic_amplification_2100",
        json!({
            "latitudinal_bands": Value::Object(ratios),
        }),
    );

    let _ = fs::remove_file(&db_path);
}
