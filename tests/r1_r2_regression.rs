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

use std::collections::HashMap;
use terratwin_climate_hub::{
    db::{ClimateDatabase, StoredGridRecord},
    interp::{
        equirectangular_scales, haversine_km, project_latlon_to_ndc, slerp_great_circle,
        slerp_latlon, unproject_ndc_to_latlon, CameraState, MapProjectionKind,
    },
    physics::{
        compute_land_cryosphere_profile_from_grids,
        compute_land_cryosphere_profile_from_grids_at_doy, compute_ocean_ctd_profile_from_grids,
        compute_vertical_sounding_from_grids, regional_cmip6_delta_t_c, seasonal_surface_swing_c,
        snapshot_regional_delta_t_c, spherical_diffusive_heat_step,
        synthetic_planetary_elevation_m, ClimateRegion, ClimateSimulationEngine, SimulationConfig,
    },
    schema::{HubSchema, SspScenario},
};

#[test]
fn equirectangular_preserves_2_to_1_aspect_and_roundtrips() {
    for &aspect in &[0.8_f64, 0.97, 1.6, 2.5] {
        for &zoom in &[0.75_f64, 1.0, 2.0] {
            let cam = CameraState {
                center_lon: 0.0,
                center_lat: 0.0,
                zoom,
                pitch_deg: 0.0,
                bearing_deg: 0.0,
                vertical_exaggeration: 1.0,
            };

            let (sx, sy) = equirectangular_scales(aspect);
            assert!((sx * aspect / sy - 2.0).abs() < 1e-9);

            let (nx1, ny1) = project_latlon_to_ndc(
                MapProjectionKind::Equirectangular2d,
                &cam,
                -30.0,
                -60.0,
                0.0,
                aspect,
            )
            .expect("(-30, -60) must project");
            let (nx2, ny2) = project_latlon_to_ndc(
                MapProjectionKind::Equirectangular2d,
                &cam,
                30.0,
                60.0,
                0.0,
                aspect,
            )
            .expect("(30, 60) must project");

            // dlon = 120 deg, dlat = 60 deg -> 2:1 in pixel space (aspect * dnx / dny == 2.0)
            let pixel_ratio = ((nx2 - nx1) as f64 * aspect) / (ny2 - ny1) as f64;
            assert!(
                (pixel_ratio - 2.0).abs() < 1e-4,
                "Expected 2:1 pixel ratio for aspect={} zoom={}, got {}",
                aspect,
                zoom,
                pixel_ratio
            );

            let (lat_r, lon_r) = unproject_ndc_to_latlon(
                MapProjectionKind::Equirectangular2d,
                &cam,
                nx2 as f64,
                ny2 as f64,
                aspect,
            )
            .expect("Projected point must unproject");
            assert!(haversine_km(30.0, 60.0, lat_r, lon_r) < 1e-2);
        }
    }

    // Outside latitude [-90, 90] and |dlon| > 180 must return None (no duplicate tiling)
    let wide_cam = CameraState {
        center_lon: 0.0,
        center_lat: 0.0,
        zoom: 0.6,
        pitch_deg: 0.0,
        bearing_deg: 0.0,
        vertical_exaggeration: 1.0,
    };
    assert!(
        unproject_ndc_to_latlon(MapProjectionKind::Equirectangular2d, &wide_cam, 0.99, 0.0, 2.5)
            .is_none()
    );
    assert!(
        unproject_ndc_to_latlon(MapProjectionKind::Equirectangular2d, &wide_cam, 0.0, 0.99, 0.97)
            .is_none()
    );
}

#[test]
fn slerp_great_circle_64_segments_and_degenerate_cases() {
    let arc = slerp_great_circle((26.0, -76.0), (54.0, -14.0), 64);
    assert_eq!(arc.len(), 65);
    let mid = arc[32];
    assert!(
        mid.0 > 40.0 + 2.0,
        "Great-circle midpoint latitude {} should bow poleward above linear 40.0",
        mid.0
    );

    // Equal arc-length steps along the 64 segments
    let total_km = haversine_km(26.0, -76.0, 54.0, -14.0);
    let step_km = total_km / 64.0;
    for win in arc.windows(2) {
        let seg_km = haversine_km(win[0].0, win[0].1, win[1].0, win[1].1);
        assert!(
            (seg_km - step_km).abs() < 1.5,
            "SLERP segment {} km != expected {} km",
            seg_km,
            step_km
        );
    }

    // Antimeridian crossing: midpoint of (0, 170) to (0, -170) is (0, ±180), never (0, 0)
    let anti_mid = slerp_latlon((0.0, 170.0), (0.0, -170.0), 0.5);
    assert!(anti_mid.0.abs() < 1e-6);
    assert!(anti_mid.1.abs() > 179.9, "Expected ±180, got {}", anti_mid.1);

    // Coincident endpoints -> zero NaNs, finite identical point
    let same = slerp_great_circle((45.0, -120.0), (45.0, -120.0), 64);
    assert_eq!(same.len(), 65);
    for &(la, lo) in &same {
        assert!(la.is_finite() && lo.is_finite());
        assert!((la - 45.0).abs() < 1e-4 && (lo + 120.0).abs() < 1e-4);
    }

    // Antipodal endpoints -> deterministic finite great circle with zero NaNs
    let antipodal = slerp_great_circle((0.0, 0.0), (0.0, 180.0), 64);
    assert_eq!(antipodal.len(), 65);
    for &(la, lo) in &antipodal {
        assert!(la.is_finite() && lo.is_finite());
        assert!((-90.0..=90.0).contains(&la) && (-180.0..=180.0).contains(&lo));
    }
}

#[test]
fn in_memory_soundings_land_cryosphere_and_ocean_ctd() {
    let schema = HubSchema::load(None).unwrap();
    let w = 36usize;
    let h = 18usize;
    let n = w * h;

    let mut grids = HashMap::new();
    let make_grid = |id: &str, val: f32| StoredGridRecord {
        layer_id: id.to_string(),
        scenario: "SSP2-4.5".to_string(),
        timestamp_utc: "2026-11-18T00:00:00Z".to_string(),
        vertical_level: 0.0,
        width: w,
        height: h,
        min_val: val,
        max_val: val,
        mean_val: val,
        values: vec![val; n],
        u_component: None,
        v_component: None,
    };

    grids.insert(
        "wet_bulb_globe_temp_heat_stress".to_string(),
        make_grid("wet_bulb_globe_temp_heat_stress", 24.5),
    );
    grids.insert(
        "sst_marine_heatwaves".to_string(),
        make_grid("sst_marine_heatwaves", 21.0),
    );
    grids.insert(
        "cmip6_downscaled_ensemble".to_string(),
        make_grid("cmip6_downscaled_ensemble", 1.8),
    );
    grids.insert(
        "gravimetric_ice_sheet_mass_balance".to_string(),
        make_grid("gravimetric_ice_sheet_mass_balance", -18.5),
    );
    grids.insert(
        "permafrost_thermokarst_subsidence".to_string(),
        make_grid("permafrost_thermokarst_subsidence", 1.4),
    );

    let sounding = compute_vertical_sounding_from_grids(&schema, Some(&grids), 20.0, -20.0);
    assert_eq!(sounding.len(), 10);
    assert!((sounding[0].1 - 24.5).abs() < 0.1);

    let ctd = compute_ocean_ctd_profile_from_grids(&schema, Some(&grids), 36.0, -56.0);
    assert_eq!(ctd.len(), 10);
    assert!((ctd[0].1 - 21.0).abs() < 0.1);

    // Greenland (74N, -40E) is land >= 0 m; Ocean (36N, -56E) is < 0 m
    let greenland_elev = synthetic_planetary_elevation_m(74.0, -40.0);
    let ocean_elev = synthetic_planetary_elevation_m(36.0, -56.0);
    assert!(greenland_elev >= 0.0);
    assert!(ocean_elev < 0.0);

    let cryo_greenland = compute_land_cryosphere_profile_from_grids(
        Some(&grids),
        74.0,
        -40.0,
        greenland_elev,
        SspScenario::Ssp2_45,
        2026.5,
    );
    assert!(cryo_greenland.ice_sheet_smb_cm.is_some());
    assert_eq!(cryo_greenland.levels.len(), 10);
    assert!(cryo_greenland.levels.iter().all(|l| l.1.is_finite() && l.2.is_finite()));

    let siberia_elev = synthetic_planetary_elevation_m(67.0, 115.0);
    assert!(siberia_elev >= 0.0);
    let cryo_siberia = compute_land_cryosphere_profile_from_grids(
        Some(&grids),
        67.0,
        115.0,
        siberia_elev,
        SspScenario::Ssp2_45,
        2026.5,
    );
    assert!(cryo_siberia.active_layer_m.is_some());
}

#[test]
fn regional_cmip6_delta_t_monotonicity_and_snapshot_agreement() {
    let dt_arctic = regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2100.0, ClimateRegion::Arctic, 0.0);
    let dt_global = regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2100.0, ClimateRegion::Global, 0.0);
    let dt_tropics =
        regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2100.0, ClimateRegion::Tropics, 0.0);
    assert!(dt_tropics > 0.0);
    assert!(dt_arctic > dt_global && dt_global > dt_tropics);

    let ssp119 = regional_cmip6_delta_t_c(SspScenario::Ssp1_19, 2100.0, ClimateRegion::Global, 0.0);
    let ssp245 = regional_cmip6_delta_t_c(SspScenario::Ssp2_45, 2100.0, ClimateRegion::Global, 0.0);
    let ssp370 = regional_cmip6_delta_t_c(SspScenario::Ssp3_70, 2100.0, ClimateRegion::Global, 0.0);
    let ssp585 = regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2100.0, ClimateRegion::Global, 0.0);
    assert!(ssp119 < ssp245 && ssp245 < ssp370 && ssp370 < ssp585);

    // Uniform 3.25 °C grid must yield 3.25 °C across every ClimateRegion
    let grid = StoredGridRecord {
        layer_id: "cmip6_downscaled_ensemble".to_string(),
        scenario: "SSP5-8.5".to_string(),
        timestamp_utc: "2085-01-01T00:00:00Z".to_string(),
        vertical_level: 850.0,
        width: 36,
        height: 18,
        min_val: 3.25,
        max_val: 3.25,
        mean_val: 3.25,
        values: vec![3.25; 36 * 18],
        u_component: None,
        v_component: None,
    };
    for reg in ClimateRegion::ALL {
        let val = snapshot_regional_delta_t_c(&grid, reg, 36.0, -56.0).unwrap();
        assert!((val - 3.25).abs() < 1e-4);
    }
}

#[test]
fn schema_version_2_5_0_and_human_readable_labels() {
    assert_eq!(env!("CARGO_PKG_VERSION"), "2.5.0");
    let schema = HubSchema::load(None).unwrap();
    assert_eq!(schema.application.version, "2.5.0");

    for proj in MapProjectionKind::ALL {
        assert!(!proj.label().contains('_'));
    }
    for bm in &schema.main_viewport.base_maps {
        assert!(!bm.name.contains('_'));
    }
    for mode in &schema.ui_layout.bottom_dock.modes {
        assert!(!mode.label.contains('_'));
    }
    for reg in ClimateRegion::ALL {
        assert!(!reg.label().contains('_'));
    }
}

#[test]
fn json_export_creates_parent_dirs_and_propagates_errors() {
    let tmp_root = std::env::temp_dir().join("terratwin_r1_r2_export_test");
    let _ = std::fs::remove_dir_all(&tmp_root);
    std::fs::create_dir_all(&tmp_root).unwrap();

    let db_path = tmp_root.join("test.db");
    let db = ClimateDatabase::open(&db_path).unwrap();
    let schema = HubSchema::load(None).unwrap();
    db.sync_schema_catalog(&schema).unwrap();

    let grid = StoredGridRecord {
        layer_id: "sst_marine_heatwaves".to_string(),
        scenario: "SSP2-4.5".to_string(),
        timestamp_utc: "2026-11-18T00:00:00Z".to_string(),
        vertical_level: 0.0,
        width: 12,
        height: 6,
        min_val: 10.0,
        max_val: 20.0,
        mean_val: 15.0,
        values: vec![15.0; 72],
        u_component: None,
        v_component: None,
    };
    db.upsert_simulated_grids(&[grid]).unwrap();

    let nested_json = tmp_root.join("nested").join("dir").join("subset.json");
    let exported = db
        .export_layer_subset_json(
            "sst_marine_heatwaves",
            SspScenario::Ssp2_45,
            0.0,
            -90.0,
            90.0,
            -180.0,
            180.0,
            &nested_json,
        )
        .unwrap();
    assert_eq!(exported, 72);
    assert!(nested_json.is_file());

    // Passing a path whose parent is an existing regular file must return Err cleanly
    let bad_path = nested_json.join("impossible.json");
    let err_res = db.export_layer_subset_json(
        "sst_marine_heatwaves",
        SspScenario::Ssp2_45,
        0.0,
        -90.0,
        90.0,
        -180.0,
        180.0,
        &bad_path,
    );
    assert!(err_res.is_err());

    let _ = std::fs::remove_dir_all(&tmp_root);
}

#[cfg(feature = "gui")]
#[test]
fn deflate_png_writer_compresses_1600x1000_under_1mb() {
    let w = 1600u32;
    let h = 1000u32;
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = (((x + y) * 127) / (w + h)) as u8;
            rgba.extend_from_slice(&[r, g, b, 255]);
        }
    }

    let png_path = std::env::temp_dir().join("terratwin_deflate_test.png");
    let _ = std::fs::remove_file(&png_path);
    terratwin_climate_hub::gui::write_rgba_png(&png_path, w, h, &rgba).unwrap();
    let bytes = std::fs::read(&png_path).unwrap();
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert!(
        bytes.len() < 1_000_000,
        "Compressed 1600x1000 PNG should be < 1 MB, got {} bytes",
        bytes.len()
    );
    let _ = std::fs::remove_file(&png_path);
}

#[test]
fn cfl_stability_under_large_time_step_secs() {
    let w = 288usize;
    let h = 144usize;
    let field: Vec<f32> = (0..(w * h))
        .map(|i| 15.0 + 10.0 * ((i % w) as f32 / w as f32))
        .collect();
    let diffused = spherical_diffusive_heat_step(&field, None, w, h, 4.5e5, 2_592_000.0);
    assert_eq!(diffused.len(), w * h);
    assert!(diffused.iter().all(|v| v.is_finite() && (0.0..=40.0).contains(v)));

    let tmp_root = std::env::temp_dir().join("terratwin_r1_r2_cfl_test");
    let _ = std::fs::remove_dir_all(&tmp_root);
    std::fs::create_dir_all(&tmp_root).unwrap();
    let db = ClimateDatabase::open(&tmp_root.join("cfl.db")).unwrap();
    let schema = HubSchema::load(None).unwrap();
    db.sync_schema_catalog(&schema).unwrap();

    let engine = ClimateSimulationEngine::new(schema, db);
    let mut cfg = SimulationConfig::default();
    cfg.scenario = SspScenario::Ssp5_85;
    cfg.sim_year = 2085.0;
    cfg.ocean_depth_m = 450.0;
    cfg.time_step_secs = 3600.0;

    let r1 = engine.step_all_layers(&cfg).unwrap();
    let r2 = engine.step_all_layers(&cfg).unwrap();
    assert!(r1.global_2m_temp_anomaly_c.is_finite());
    assert!(
        r2.global_2m_temp_anomaly_c.is_finite()
            && (0.5..=8.0).contains(&r2.global_2m_temp_anomaly_c),
        "Expected bounded SSP5-8.5 2085 synoptic anomaly, got {}",
        r2.global_2m_temp_anomaly_c
    );

    // Even an extreme 30-day dt must remain CFL-stable and finite (never explode to -1e35)
    cfg.time_step_secs = 2_592_000.0;
    let r3 = engine.step_all_layers(&cfg).unwrap();
    assert!(
        r3.global_2m_temp_anomaly_c.is_finite()
            && (-5.0..=25.0).contains(&r3.global_2m_temp_anomaly_c),
        "Expected CFL-stable bounded anomaly for dt=2_592_000s, got {}",
        r3.global_2m_temp_anomaly_c
    );

    let _ = std::fs::remove_dir_all(&tmp_root);
}

#[test]
fn cmip6_snapshot_deseasonalized_across_solstices() {
    let tmp_root = std::env::temp_dir().join("terratwin_r1_r2_solstice_deseas_test");
    let _ = std::fs::remove_dir_all(&tmp_root);
    std::fs::create_dir_all(&tmp_root).unwrap();

    let schema = HubSchema::load(None).expect("Embedded schema must load");
    let db = ClimateDatabase::open(&tmp_root.join("solstice.db")).unwrap();
    db.sync_schema_catalog(&schema).unwrap();
    let engine = ClimateSimulationEngine::new(schema, db.clone());

    let pinned_lat = 36.0;
    let pinned_lon = -56.0;

    for &doy in &[322.0_f64, 172.0_f64] {
        // Fresh DB state per solstice season so carryover only tests within-season prognostic ticks
        let mut cfg = SimulationConfig::default();
        cfg.grid_width = 72;
        cfg.grid_height = 36;
        cfg.scenario = SspScenario::Ssp5_85;
        cfg.sim_year = 2085.0;
        cfg.day_of_year = doy;
        cfg.ocean_depth_m = 450.0;
        cfg.time_step_secs = 3600.0;

        // Run two consecutive steps so both cold-start and prognostic `prev_t2m` reconstruction execute
        let _r1 = engine.step_all_layers(&cfg).unwrap();
        let r2 = engine.step_all_layers(&cfg).unwrap();

        let expected_global =
            regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2085.0, ClimateRegion::Global, 0.0);
        assert!(
            (r2.global_2m_temp_anomaly_c - expected_global).abs() < 1.5,
            "doy={}: global_2m_temp_anomaly_c {:.3} differs from analytical {:.3}",
            doy,
            r2.global_2m_temp_anomaly_c,
            expected_global
        );

        let grid = db
            .load_grid_at_timestamp(
                "cmip6_downscaled_ensemble",
                SspScenario::Ssp5_85,
                cfg.atmospheric_level_hpa,
                &r2.timestamp_utc,
            )
            .unwrap()
            .expect("cmip6_downscaled_ensemble grid missing");

        let dt_arctic =
            snapshot_regional_delta_t_c(&grid, ClimateRegion::Arctic, pinned_lat, pinned_lon)
                .unwrap();
        let dt_global =
            snapshot_regional_delta_t_c(&grid, ClimateRegion::Global, pinned_lat, pinned_lon)
                .unwrap();
        let dt_tropics =
            snapshot_regional_delta_t_c(&grid, ClimateRegion::Tropics, pinned_lat, pinned_lon)
                .unwrap();

        assert!(
            dt_tropics > 0.0 && dt_arctic > dt_global && dt_global > dt_tropics,
            "doy={}: expected Arctic ({:.3}) > Global ({:.3}) > Tropics ({:.3}) > 0",
            doy,
            dt_arctic,
            dt_global,
            dt_tropics
        );

        for reg in ClimateRegion::ALL {
            let snap_dt = snapshot_regional_delta_t_c(&grid, reg, pinned_lat, pinned_lon).unwrap();
            let model_dt =
                regional_cmip6_delta_t_c(SspScenario::Ssp5_85, 2085.0, reg, pinned_lat);
            assert!(
                (snap_dt - model_dt).abs() < 1.5,
                "doy={}, region={:?}: snapshot ΔT ({:.3}) vs analytical ({:.3}) exceeded 1.5 °C",
                doy,
                reg,
                snap_dt,
                model_dt
            );
        }
    }

    // Verify seasonal_surface_swing_c and LandCryosphereProfile seasonal surface + 20m damping
    let greenland_elev = synthetic_planetary_elevation_m(74.0, -40.0);
    let annual = compute_land_cryosphere_profile_from_grids(
        None,
        74.0,
        -40.0,
        greenland_elev,
        SspScenario::Ssp5_85,
        2085.0,
    );
    let winter = compute_land_cryosphere_profile_from_grids_at_doy(
        None,
        74.0,
        -40.0,
        greenland_elev,
        SspScenario::Ssp5_85,
        2085.0,
        322.0,
    );
    let summer = compute_land_cryosphere_profile_from_grids_at_doy(
        None,
        74.0,
        -40.0,
        greenland_elev,
        SspScenario::Ssp5_85,
        2085.0,
        172.0,
    );
    assert!(summer.surface_t2m_c > annual.surface_t2m_c && annual.surface_t2m_c > winter.surface_t2m_c);
    assert!(
        (winter.surface_t2m_c - annual.surface_t2m_c - seasonal_surface_swing_c(74.0, false, 322.0))
            .abs()
            < 1e-6
    );
    // At 20m depth, seasonal swing damps out (>99%) toward annual-mean firn temperature + geothermal gradient
    assert!((winter.levels.last().unwrap().1 - annual.levels.last().unwrap().1).abs() < 0.05);

    let _ = std::fs::remove_dir_all(&tmp_root);
}
