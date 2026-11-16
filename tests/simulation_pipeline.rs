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

//! End-to-end integration tests for TerraTwin HPC Climate Modeling & Telemetry Hub.

use std::collections::HashSet;
use std::path::PathBuf;
use terratwin_climate_hub::{
    coastlines::{CONTINENT_POLYLINES, REFERENCE_OBSERVATORIES, TELECONNECTION_BASINS},
    colormaps::{default_layer_style, shade_basemap_rgb, ColormapPalette},
    db::{
        pack_f32_compressed, pack_f32_le, read_zarr_v3_chunk_f32, unpack_f32_auto, unpack_f32_le,
        ClimateDatabase, RawTelemetrySample, RetentionPolicy, ZarrCompressionCodec,
    },
    ingest::{
        parse_erddap_oisst_csv, parse_firms_csv_hotspots, parse_noaa_cpc_nao,
        parse_noaa_enso_nino34,
    },
    interp::{
        assimilate_3dvar_cg, barnes_assimilate_grid, detect_mslp_extrema, detect_thermal_fronts,
        extract_marching_squares_contours, haversine_km, project_latlon_to_ndc,
        sample_grid_bicubic_monotone, spherical_area_weighted_mean, unproject_ndc_to_latlon,
        CameraState, GeoSamplePoint, MapProjectionKind, ObsSamplePoint, ThreeDVarConfig,
    },
    physics::{
        bolton_equivalent_potential_temp_k, bolton_saturation_vapor_pressure_hpa,
        compute_surface_wind_stress, compute_wind_stress_curl,
        co2_radiative_forcing_wm2, compute_outdoor_wbgt,
        compute_two_layer_qgpv_from_streamfunctions, diagnose_rotational_wind_from_streamfunction,
        invert_two_layer_qgpv_streamfunctions, planetary_coriolis_f, semi_lagrangian_advect_qmsl,
        solve_seawater_carbonate_chemistry, solve_spherical_linear_balance_mslp,
        solve_stommel_munk_gyre, spherical_diffusive_heat_step,
        step_ocean_reduced_gravity_and_gyre, step_semtner_sea_ice_thickness, step_two_layer_qgpv,
        step_van_wagner_fwi, stull_wet_bulb_celsius, synthetic_planetary_elevation_m,
        ClimateSimulationEngine, SimulationConfig,
    },
    schema::{HubSchema, SspScenario},
    worker::run_single_headless_cycle,
};

#[test]
fn parses_all_34_layers_and_schema_metadata() {
    let schema = HubSchema::load(None).expect("Embedded schema must parse");
    assert_eq!(schema.application.id, "planetary-climate-telemetry-hub");
    assert_eq!(schema.layer_groups.len(), 6);
    assert_eq!(schema.all_layers().len(), 34);
    assert_eq!(schema.main_viewport.supported_projections.len(), 4);
    assert_eq!(
        schema
            .main_viewport
            .vertical_slice_engine
            .atmospheric_levels_hpa
            .len(),
        10
    );
    assert_eq!(
        schema
            .main_viewport
            .vertical_slice_engine
            .ocean_depth_levels_meters
            .len(),
        10
    );
}

#[test]
fn verifies_firms_erddap_enso_and_nao_parsers() {
    // 1. CoastWatch ERDDAP OISST v2.1 CSV (including row-2 units & NaN land mask)
    let erddap_csv = "\
time,zlev,latitude,longitude,sst
UTC,m,degrees_north,degrees_east,Celsius
2026-11-11T12:00:00Z,0.0,44.625,-124.875,13.42
2026-11-11T12:00:00Z,0.0,44.625,-121.125,NaN
2026-11-11T12:00:00Z,0.0,0.125,-140.125,27.85
";
    let oisst_samples = parse_erddap_oisst_csv(erddap_csv, "2026-11-11T00:00:00Z").unwrap();
    assert_eq!(oisst_samples.len(), 2);
    assert!((oisst_samples[0].primary_value - 13.42).abs() < 1e-6);

    // 2. NASA FIRMS VIIRS 24h active fire CSV (filters low confidence, aggregates 1x1 deg FRP)
    let firms_csv = "\
latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
44.42,-121.85,342.1,0.4,0.4,2026-08-19,2045,N21,VIIRS,h,2.0NRT,298.0,48.5,D
44.48,-121.82,336.4,0.4,0.4,2026-08-19,2045,N21,VIIRS,n,2.0NRT,295.2,31.5,D
35.10,-118.20,305.0,0.4,0.4,2026-08-19,2045,N21,VIIRS,l,2.0NRT,289.0,12.0,D
";
    let fire_clusters = parse_firms_csv_hotspots(
        firms_csv,
        "https://firms.modaps.eosdis.nasa.gov/",
        "2026-08-19T00:00:00Z",
        50,
    )
    .unwrap();
    assert_eq!(fire_clusters.len(), 1);
    assert!((fire_clusters[0].primary_value - 80.0).abs() < 1e-5);

    // 3. NOAA PSL 13-column & CPC 5-column ENSO Niño 3.4 ASCII feeds
    let psl_nino = "\
  1950 2026
  2025  -0.62 -0.45 -0.18  0.05  0.14  0.22  0.31  0.44  0.52  0.61  0.68  0.72
  2026   0.64  0.49  0.28  0.12 -0.08 -0.24 -0.38 -0.51 -0.63 -0.74 -99.99 -99.99
";
    assert!((parse_noaa_enso_nino34(psl_nino).unwrap() - (-0.74)).abs() < 1e-6);

    let cpc_nino = "\
 YR   MON  TOTAL ClimAdjust   ANOM
 2026   9  26.12      26.75  -0.63
 2026  10  25.94      26.72  -0.78
";
    assert!((parse_noaa_enso_nino34(cpc_nino).unwrap() - (-0.78)).abs() < 1e-6);

    // 4. NOAA CPC Monthly NAO ASCII Table
    let cpc_nao = "\
 YEAR   JAN   FEB   MAR   APR   MAY   JUN   JUL   AUG   SEP   OCT   NOV   DEC
 2025  1.12  0.85  0.42 -0.19 -0.54  0.11  0.33  0.28 -0.14  0.46  0.91  1.24
 2026  1.05  0.76  0.34 -0.22  0.18  0.44 -0.09  0.62  0.88  1.19 -99.9 -99.9
";
    assert!((parse_noaa_cpc_nao(cpc_nao).unwrap() - 1.19).abs() < 1e-6);
}

#[test]
fn verifies_coupled_climate_ocean_cryosphere_and_fwi_physics() {
    let es_0 = bolton_saturation_vapor_pressure_hpa(0.0);
    assert!((es_0 - 6.112).abs() < 0.01);
    let theta_e = bolton_equivalent_potential_temp_k(26.0, 0.016, 1008.0);
    assert!(theta_e > 335.0 && theta_e < 360.0);

    let tw = stull_wet_bulb_celsius(20.0, 50.0);
    assert!(tw > 13.2 && tw < 14.2);
    assert!(
        compute_outdoor_wbgt(32.0, 70.0, 850.0, 2.0) > compute_outdoor_wbgt(32.0, 70.0, 0.0, 2.0)
    );

    let df_2x = co2_radiative_forcing_wm2(556.0);
    assert!((df_2x - 3.708).abs() < 0.05);

    let preind_carb = solve_seawater_carbonate_chemistry(18.0, 35.0, 280.0, 0.0);
    let high_co2_carb = solve_seawater_carbonate_chemistry(18.0, 35.0, 800.0, 0.0);
    assert!(preind_carb.ph_total > 8.10 && high_co2_carb.ph_total < 7.82);
    assert!(high_co2_carb.omega_aragonite < preind_carb.omega_aragonite);

    let grown = step_semtner_sea_ice_thickness(1.2, -25.0, -1.8, 0.0, 86_400.0);
    let melted = step_semtner_sea_ice_thickness(1.2, 4.0, 2.5, 320.0, 86_400.0);
    assert!(grown > 1.2 && melted < 1.2);

    let dry_fwi = step_van_wagner_fwi(34.0, 16.0, 9.0, 0.0, 90.0, 55.0, 320.0);
    let wet_fwi = step_van_wagner_fwi(18.0, 82.0, 2.0, 18.0, 90.0, 55.0, 320.0);
    assert!(dry_fwi.fwi > wet_fwi.fwi + 20.0);
}

#[test]
fn verifies_two_layer_qgpv_helmholtz_inversion_and_cyclone_evolution() {
    let w = 72usize;
    let h = 36usize;
    let n = w * h;

    // 1. Verify exact forward->inverse normal-mode elliptic round-trip (psi_1, psi_2 -> q_1, q_2 -> psi_1, psi_2)
    let mut psi1_true = vec![0.0_f32; n];
    let mut psi2_true = vec![0.0_f32; n];
    let mut elev = vec![0.0_f64; n];
    let mut t2m = vec![15.0_f64; n];

    for r in 0..h {
        let lat = 90.0 - (r as f64 + 0.5) * (180.0 / h as f64);
        let phi = lat.to_radians();
        for c in 0..w {
            let lon = -180.0 + (c as f64 + 0.5) * (360.0 / w as f64);
            let lam = lon.to_radians();
            let idx = r * w + c;
            elev[idx] = synthetic_planetary_elevation_m(lat, lon);
            t2m[idx] = 27.0 * phi.cos() - 15.0 * phi.sin().powi(2);

            psi1_true[idx] = (2.4e7 * (2.0 * phi).sin() * (4.0 * lam).cos()
                - 3.5e7 * phi.sin()) as f32;
            psi2_true[idx] = (1.1e7 * (2.0 * phi).sin() * (4.0 * lam + 0.5).cos()
                - 1.2e7 * phi.sin()) as f32;
        }
    }

    let (q1, q2) =
        compute_two_layer_qgpv_from_streamfunctions(&psi1_true, &psi2_true, &elev, w, h);
    let (psi1_rec, _psi2_rec) = invert_two_layer_qgpv_streamfunctions(&q1, &q2, &elev, w, h);

    let max_p1 = psi1_true
        .iter()
        .fold(0.0_f32, |a, &v| a.max(v.abs()))
        .max(1.0);
    let max_err_p1 = psi1_true
        .iter()
        .zip(psi1_rec.iter())
        .fold(0.0_f32, |a, (&t, &r)| a.max((t - r).abs()));
    assert!(
        max_err_p1 / max_p1 < 2.5e-2,
        "2-layer QGPV modal Helmholtz inversion relative error must be < 2.5%, got {}",
        max_err_p1 / max_p1
    );

    // 2. Verify isolated NH positive PV cyclone anomaly produces:
    //    - negative streamfunction bullseye
    //    - counter-clockwise rotational winds (southerly v > 0 to east, northerly v < 0 to west)
    //    - closed low-pressure MSLP minimum (< 1005 hPa) via spherical Linear Balance Equation
    let cyc_row = 9usize; // ~42.5 deg N
    let cyc_col = 36usize; // ~2.5 deg E
    let mut q1_cyc = vec![0.0_f32; n];
    let mut q2_cyc = vec![0.0_f32; n];
    let flat_ocean = vec![-2000.0_f64; n];

    for r in 0..h {
        let lat = 90.0 - (r as f64 + 0.5) * (180.0 / h as f64);
        let f_cor = planetary_coriolis_f(lat);
        for c in 0..w {
            let dlat = (r as f64 - cyc_row as f64) * 5.0;
            let dlon = (c as f64 - cyc_col as f64) * 5.0;
            let bump = 1.1e-4 * (-(dlat * dlat + dlon * dlon) / (12.0 * 12.0)).exp();
            q1_cyc[r * w + c] = (f_cor + 0.6 * bump) as f32;
            q2_cyc[r * w + c] = (f_cor + bump) as f32;
        }
    }

    let (_, psi2_cyc) =
        invert_two_layer_qgpv_streamfunctions(&q1_cyc, &q2_cyc, &flat_ocean, w, h);
    let (_, v2_cyc) = diagnose_rotational_wind_from_streamfunction(&psi2_cyc, w, h);
    let mslp_cyc = solve_spherical_linear_balance_mslp(&psi2_cyc, w, h);

    assert!(psi2_cyc[cyc_row * w + cyc_col] < psi2_cyc[cyc_row * w + (cyc_col + 8)]);
    assert!(v2_cyc[cyc_row * w + (cyc_col + 2)] > 2.0); // Southerly on east flank of NH cyclone
    assert!(v2_cyc[cyc_row * w + (cyc_col - 2)] < -2.0); // Northerly on west flank of NH cyclone
    assert!(mslp_cyc[cyc_row * w + cyc_col] < 1005.0);

    // 3. Verify prognostic QGPV time-stepping evolves synoptic cyclones dynamically
    let st0 = step_two_layer_qgpv(None, &elev, &t2m, w, h, 3600.0, 0.2, 1.3, 0.4, 0.6);
    let st1 = step_two_layer_qgpv(
        Some((&st0.q1, &st0.q2)),
        &elev,
        &t2m,
        w,
        h,
        6.0 * 3600.0,
        0.2,
        1.3,
        0.4,
        0.6,
    );
    let mslp_diff = st0
        .mslp_hpa
        .iter()
        .zip(st1.mslp_hpa.iter())
        .fold(0.0_f32, |a, (&p0, &p1)| a.max((p1 - p0).abs()));
    assert!(
        mslp_diff > 0.15,
        "Synoptic cyclones must dynamically evolve MSLP across QGPV ticks"
    );
}


#[test]
fn verifies_reduced_gravity_kelvin_wave_and_stommel_munk_gyre() {
    let w = 72usize;
    let h = 36usize;
    let n = w * h;

    // Idealized rectangular Pacific-like ocean basin bounded by western/eastern continents
    let mut is_ocean = vec![true; n];
    for r in 0..h {
        for c in 0..10 {
            is_ocean[r * w + c] = false;
        }
        for c in 62..w {
            is_ocean[r * w + c] = false;
        }
    }

    // 1. Equatorial Kelvin wave propagation: westerly wind burst at col ~20 deepens thermocline eastward
    let mut u10_burst = vec![0.0_f64; n];
    let v10_zero = vec![0.0_f64; n];
    for r in 16..20 {
        for c in 18..24 {
            u10_burst[r * w + c] = 14.0;
        }
    }
    let st0 = step_ocean_reduced_gravity_and_gyre(
        None, None, None, None, &u10_burst, &v10_zero, &is_ocean, w, h, 86_400.0, 0.0,
    );
    let st1 = step_ocean_reduced_gravity_and_gyre(
        Some(&st0.h_tc_anom_m),
        Some(&st0.psi_gyre),
        Some(&st0.u_sw),
        Some(&st0.v_sw),
        &u10_burst,
        &v10_zero,
        &is_ocean,
        w,
        h,
        3.0 * 86_400.0,
        0.0,
    );
    // Eastward of the forcing burst (col 28, equator row 18), thermocline anomaly must deepen (> 0)
    assert!(st1.h_tc_anom_m[18 * w + 28] > st0.h_tc_anom_m[18 * w + 28]);

    // 2. Stommel-Munk westward intensification under NH subtropical wind-stress curl
    let mut u10_gyre = vec![0.0_f64; n];
    for r in 0..h {
        let lat = 90.0 - (r as f64 + 0.5) * (180.0 / h as f64);
        let u_wind = -10.0 * ((lat - 30.0) / 15.0).sin(); // Trades south of 30N, westerlies north of 30N
        for c in 0..w {
            u10_gyre[r * w + c] = u_wind;
        }
    }
    let (tx, ty) = compute_surface_wind_stress(&u10_gyre, &v10_zero, &is_ocean);
    let curl = compute_wind_stress_curl(&tx, &ty, &is_ocean, w, h);
    assert!(curl[12 * w + 36].abs() > 1e-10);
    let (psi_gyre, _, v_gyre) =
        solve_stommel_munk_gyre(None, &tx, &ty, &is_ocean, w, h, 96);
    // Land mask boundary must stay strictly zero
    assert_eq!(psi_gyre[12 * w + 5], 0.0);
    // Western boundary (col 11) meridional speed must exceed eastern interior (col 52)
    assert!(v_gyre[12 * w + 11].abs() > v_gyre[12 * w + 52].abs());
}

#[test]
fn verifies_cqmsl_bicubic_monotone_advection_and_mass_conservation() {
    let w = 72usize;
    let h = 36usize;
    let mut plume = vec![0.0_f32; w * h];
    plume[18 * w + 36] = 12.0;
    plume[18 * w + 37] = 8.0;

    let s = sample_grid_bicubic_monotone(&plume, w, h, -2.5, 3.2);
    assert!(s >= 0.0 && s <= 12.0);

    let mean_before = spherical_area_weighted_mean(&plume, w, h);
    let u = vec![28.0_f32; w * h];
    let v = vec![0.0_f32; w * h];
    let advected = semi_lagrangian_advect_qmsl(&plume, &u, &v, w, h, 12.0 * 3600.0, true, true);
    let mean_after = spherical_area_weighted_mean(&advected, w, h);

    assert!((mean_before - mean_after).abs() < 1e-5);
    assert!(advected.iter().all(|&x| x >= 0.0));
    assert!(advected[18 * w + 39] > advected[18 * w + 34]);

    let diffused = spherical_diffusive_heat_step(&plume, None, w, h, 8.0e5, 6.0 * 3600.0);
    let max_after = diffused.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(max_after < 12.0 && max_after > 1.0);
}

#[test]
fn verifies_3dvar_cg_quality_weighting_and_flow_anisotropy() {
    let w = 36usize;
    let h = 18usize;
    let n = w * h;
    let bg = vec![10.0_f32; n];

    // High-quality obs (flag=2, 22.0) vs degraded-quality obs (flag=-1, 22.0) at identical separation
    let obs_hq = vec![ObsSamplePoint {
        lat: 45.0,
        lon: -120.0,
        value: 22.0,
        quality_flag: 2,
        sigma_obs: None,
    }];
    let obs_lq = vec![ObsSamplePoint {
        lat: 45.0,
        lon: -120.0,
        value: 22.0,
        quality_flag: -1,
        sigma_obs: None,
    }];
    let cfg = ThreeDVarConfig::default();
    let res_hq = assimilate_3dvar_cg(&bg, w, h, &obs_hq, None, &cfg);
    let res_lq = assimilate_3dvar_cg(&bg, w, h, &obs_lq, None, &cfg);

    assert!(res_hq.rel_residual < 1.0e-5);
    let max_hq = res_hq.analysis.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let max_lq = res_lq.analysis.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_hq > max_lq + 2.0,
        "High-quality 3D-Var observation must pull analysis closer than low-quality obs"
    );

    // Flow-aligned anisotropy under strong zonal jet (u=35 m/s, v=0): zonal increment > meridional increment at equal distance
    let u_jet = vec![35.0_f32; n];
    let v_jet = vec![0.0_f32; n];
    let obs_eq = vec![ObsSamplePoint {
        lat: 0.0,
        lon: 0.0,
        value: 20.0,
        quality_flag: 1,
        sigma_obs: None,
    }];
    let res_aniso = assimilate_3dvar_cg(&bg, w, h, &obs_eq, Some((&u_jet, &v_jet)), &cfg);
    let zonal_pt = res_aniso.increment[9 * w + 20]; // +20 deg lon along jet
    let merid_pt = res_aniso.increment[5 * w + 18]; // +20 deg lat across jet
    assert!(
        zonal_pt > merid_pt,
        "Flow-aligned 3D-Var increment must stretch further along the zonal jet ({}) than across it ({})",
        zonal_pt,
        merid_pt
    );
}

#[test]
fn verifies_marching_squares_isobars_extrema_and_thermal_fronts() {
    let w = 72usize;
    let h = 36usize;
    let n = w * h;
    let mut mslp = vec![1016.0_f32; n];
    let mut temp = vec![10.0_f32; n];
    let u = vec![20.0_f32; n];
    let v = vec![0.0_f32; n];

    for r in 0..h {
        let lat = 90.0 - (r as f64 + 0.5) * (180.0 / h as f64);
        for c in 0..w {
            let lon = -180.0 + (c as f64 + 0.5) * (360.0 / w as f64);
            let d_low = ((lat - 45.0).powi(2) + (lon + 40.0).powi(2)).sqrt();
            let d_high = ((lat - 35.0).powi(2) + (lon - 40.0).powi(2)).sqrt();
            mslp[r * w + c] += (-28.0 * (-d_low * d_low / 180.0).exp()
                + 18.0 * (-d_high * d_high / 200.0).exp()) as f32;
            temp[r * w + c] = (18.0 * ((lon + 30.0) / 5.0).tanh()) as f32;
        }
    }

    let isobars = extract_marching_squares_contours(&mslp, w, h, 4.0, 980.0, 1040.0);
    assert!(isobars.iter().any(|s| (s.level - 1004.0).abs() < 0.1));

    let extrema = detect_mslp_extrema(&mslp, w, h, 1.0);
    assert!(extrema.iter().any(|e| e.is_low && e.pressure_hpa < 995.0));
    assert!(extrema.iter().any(|e| !e.is_low && e.pressure_hpa > 1028.0));

    let fronts = detect_thermal_fronts(&temp, &u, &v, w, h, 8.0);
    assert!(fronts.iter().any(|f| f.is_cold_front));
}

#[test]
fn verifies_projections_and_barnes_assimilation() {
    let cam = CameraState::default();
    for proj in [
        MapProjectionKind::Globe3dWgs84,
        MapProjectionKind::Equirectangular2d,
    ] {
        let (nx, ny) = project_latlon_to_ndc(proj, &cam, 20.0, -20.0, 0.0, 1.6)
            .expect("Center camera point must be visible");
        let (lat_r, lon_r) = unproject_ndc_to_latlon(proj, &cam, nx as f64, ny as f64, 1.6)
            .expect("Center NDC must unproject");
        assert!(haversine_km(20.0, -20.0, lat_r, lon_r) < 5.0);
    }

    let bg = vec![10.0_f32; 36 * 18];
    let obs = vec![GeoSamplePoint {
        lat: 40.7128,
        lon: -74.0061,
        value: 22.0,
    }];
    let assimilated = barnes_assimilate_grid(&bg, 36, 18, &obs, 1200.0, 0.4);
    let max_v = assimilated
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(max_v > 14.0 && max_v <= 22.5);
}

#[test]
fn executes_multi_tick_coupled_sqlite_simulation_for_all_34_layers() {
    let tmp_db: PathBuf = std::env::temp_dir().join("terratwin_coupled_ticks_test.db");
    let _ = std::fs::remove_file(&tmp_db);

    let schema = HubSchema::load(None).unwrap();
    let db = ClimateDatabase::open(&tmp_db).unwrap();
    assert_eq!(db.sync_schema_catalog(&schema).unwrap(), 34);
    assert_eq!(
        unpack_f32_le(&pack_f32_le(&[-1.85_f32, 15.25])),
        vec![-1.85_f32, 15.25]
    );

    // Inject live-formatted ENSO, GloFAS river discharge, and NASA FIRMS samples into SQLite
    db.insert_raw_samples(&[
        RawTelemetrySample {
            layer_id: "teleconnection_oscillation_masks".to_string(),
            source_name: "NOAA CPC ENSO".to_string(),
            source_url: "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data"
                .to_string(),
            timestamp_utc: "2026-11-11T00:00:00Z".to_string(),
            latitude: 0.0,
            longitude: -145.0,
            vertical_level: 0.0,
            primary_value: -0.82,
            secondary_value: Some(1.15),
            quality_flag: 1,
        },
        RawTelemetrySample {
            layer_id: "swot_surface_water_river_discharge".to_string(),
            source_name: "Open-Meteo GloFAS".to_string(),
            source_url: "https://flood-api.open-meteo.com/v1/flood".to_string(),
            timestamp_utc: "2026-11-11T00:00:00Z".to_string(),
            latitude: -1.947,
            longitude: -55.511,
            vertical_level: 0.0,
            primary_value: 168_500.0,
            secondary_value: None,
            quality_flag: 1,
        },
    ])
    .unwrap();

    let mut cfg = SimulationConfig {
        grid_width: 72,
        grid_height: 36,
        scenario: SspScenario::Ssp2_45,
        sim_year: 2026.86,
        day_of_year: 315.0,
        hour_utc: 0.0,
        atmospheric_level_hpa: 250.0,
        ocean_depth_m: 100.0,
        time_step_secs: 3600.0,
        thermal_diffusivity_m2_s: 4.5e5,
        prognostic_relaxation_alpha: 0.16,
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let r0 = rt
        .block_on(run_single_headless_cycle(&schema, &db, &cfg, false))
        .expect("Tick 0 must succeed");
    assert!((r0.enso_nino34_c - (-0.82)).abs() < 1e-6);

    let syn0 = db
        .load_latest_grid("synoptic_fronts_isobars", cfg.scenario, 1000.0)
        .unwrap()
        .expect("Tick 0 synoptic MSLP + QGPV state must exist in SQLite");
    assert!(syn0.u_component.is_some() && syn0.v_component.is_some());

    let engine = ClimateSimulationEngine::new(schema.clone(), db.clone());
    cfg.advance_tick();
    let r1 = engine
        .step_all_layers(&cfg)
        .expect("Coupled Tick 1 must succeed");
    assert_ne!(r0.timestamp_utc, r1.timestamp_utc);
    assert_eq!(db.summary_stats().unwrap().simulated_grid_count, 68);

    let syn1 = db
        .load_latest_grid("synoptic_fronts_isobars", cfg.scenario, 1000.0)
        .unwrap()
        .expect("Tick 1 synoptic MSLP + QGPV state must exist in SQLite");
    let max_dp = syn0
        .values
        .iter()
        .zip(syn1.values.iter())
        .fold(0.0_f32, |a, (&p0, &p1)| a.max((p1 - p0).abs()));
    assert!(
        max_dp > 0.02,
        "Prognostic 2-layer QGPV must evolve SQLite MSLP across ticks"
    );

    // Verify 4D SQLite timeline playback queries & chunked Zarr v3 binary directory export
    let timestamps = db.list_available_timestamps(cfg.scenario, 10).unwrap();
    assert_eq!(timestamps.len(), 2);
    let replay_bundle = db
        .load_layer_bundle_at_timestamp(&schema, cfg.scenario, 250.0, 100.0, &timestamps[0])
        .unwrap();
    assert_eq!(replay_bundle.len(), 34);
    let replay_readouts = db
        .load_readouts_at_timestamp(cfg.scenario, &timestamps[0])
        .unwrap()
        .expect("Tick 0 readouts must exist");
    assert_eq!(replay_readouts.timestamp_utc, timestamps[0]);

    let zarr_dir = std::env::temp_dir().join("terratwin_test_export.zarr");
    let _ = std::fs::remove_dir_all(&zarr_dir);
    let zarr_sum = db
        .export_zarr_v3_store(
            "synoptic_fronts_isobars",
            cfg.scenario,
            1000.0,
            -45.0,
            45.0,
            -90.0,
            90.0,
            (9, 18),
            &zarr_dir,
        )
        .unwrap();
    assert!(zarr_dir.join("zarr.json").exists());
    assert!(zarr_dir.join("synoptic_fronts_isobars/zarr.json").exists());
    assert!(zarr_dir.join("synoptic_fronts_isobars/c/0/0").exists());
    assert!(zarr_dir.join("latitude/c/0").exists());
    assert!(zarr_dir.join("longitude/c/0").exists());
    assert!(zarr_sum.chunks_written >= 4 && zarr_sum.cells_exported > 0);
    let _ = std::fs::remove_dir_all(&zarr_dir);

    let _ = std::fs::remove_file(&tmp_db);
}

#[test]
fn verifies_byte_shuffled_lz4_blobs_tick_pruning_and_zarr_v3_codecs() {
    // 1. Lossless byte-shuffled LZ4 round-trip (including NaN land masks) & >2x compression ratio
    let mut smooth_grid = Vec::with_capacity(144 * 72);
    for i in 0..(144 * 72) {
        if i % 17 == 0 {
            smooth_grid.push(f32::NAN);
        } else {
            smooth_grid.push(1013.25 + 12.0 * ((i as f32) * 0.02).sin());
        }
    }
    let raw_bytes = pack_f32_le(&smooth_grid);
    let comp_bytes = pack_f32_compressed(&smooth_grid);
    assert!(
        comp_bytes.len() * 2 < raw_bytes.len(),
        "Byte-shuffled LZ4 must achieve >2x compression on smooth climate grids ({} vs {})",
        comp_bytes.len(),
        raw_bytes.len()
    );

    let decoded = unpack_f32_auto(&comp_bytes);
    assert_eq!(decoded.len(), smooth_grid.len());
    for (&orig, &dec) in smooth_grid.iter().zip(decoded.iter()) {
        if orig.is_nan() {
            assert!(dec.is_nan());
        } else {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }
    // Legacy raw little-endian f32 blobs must also decode transparently via unpack_f32_auto
    assert_eq!(unpack_f32_auto(&raw_bytes).len(), smooth_grid.len());

    // 2. Automatic SQLite tick retention ring-buffer pruning + Zarr v3 Zstd & Gzip chunk decoding
    let tmp_db: PathBuf = std::env::temp_dir().join("terratwin_retention_zarr_codec_test.db");
    let _ = std::fs::remove_file(&tmp_db);
    let schema = HubSchema::load(None).unwrap();
    let db = ClimateDatabase::open(&tmp_db).unwrap();
    db.sync_schema_catalog(&schema).unwrap();
    db.set_retention_policy(RetentionPolicy {
        max_ticks_per_scenario: 2,
        max_raw_samples_per_layer: 64,
        max_sync_log_rows: 32,
    });

    let mut cfg = SimulationConfig {
        grid_width: 36,
        grid_height: 18,
        scenario: SspScenario::Ssp2_45,
        sim_year: 2026.86,
        day_of_year: 315.0,
        hour_utc: 0.0,
        atmospheric_level_hpa: 500.0,
        ocean_depth_m: 0.0,
        time_step_secs: 3600.0,
        thermal_diffusivity_m2_s: 4.0e5,
        prognostic_relaxation_alpha: 0.16,
    };
    let engine = ClimateSimulationEngine::new(schema.clone(), db.clone());
    for _ in 0..4 {
        engine.step_all_layers(&cfg).unwrap();
        cfg.advance_tick();
    }

    // Ring buffer max_ticks_per_scenario = 2 must automatically prune ticks 0 & 1, keeping ticks 2 & 3!
    let retained_ts = db.list_available_timestamps(cfg.scenario, 10).unwrap();
    assert_eq!(retained_ts.len(), 2);
    assert_eq!(db.summary_stats().unwrap().simulated_grid_count, 68);

    // Export with both Zstd and Gzip codecs and verify exact decoded chunk values
    for (codec, suffix) in [
        (ZarrCompressionCodec::Zstd { level: 3, checksum: true }, "zstd"),
        (ZarrCompressionCodec::Gzip { level: 4 }, "gzip"),
    ] {
        let zdir = std::env::temp_dir().join(format!("terratwin_codec_{}.zarr", suffix));
        let _ = std::fs::remove_dir_all(&zdir);
        let summary = db
            .export_zarr_v3_store_with_codec(
                "synoptic_fronts_isobars",
                cfg.scenario,
                1000.0,
                -90.0,
                90.0,
                -180.0,
                180.0,
                (9, 18),
                codec,
                &zdir,
            )
            .unwrap();
        assert!(summary.compressed_bytes < summary.uncompressed_bytes);
        let chunk00 = read_zarr_v3_chunk_f32(&zdir.join("synoptic_fronts_isobars"), 0, 0).unwrap();
        assert_eq!(chunk00.len(), 9 * 18);
        assert!(chunk00.iter().all(|v| v.is_finite() && *v > 940.0 && *v < 1060.0));
        let _ = std::fs::remove_dir_all(&zdir);
    }

    let prune_rep = db.prune_expired_history(true).unwrap();
    assert!(prune_rep.grid_blob_bytes_remaining > 0);
    let _ = std::fs::remove_file(&tmp_db);
}

#[test]
fn colormaps_palettes_and_all_34_schema_layers_render_valid_rgb() {
    for pal in ColormapPalette::ALL {
        assert!(!pal.label().is_empty());
        let c_low = pal.sample_rgb(-0.5);
        let c0 = pal.sample_rgb(0.0);
        let c_mid = pal.sample_rgb(0.5);
        let c1 = pal.sample_rgb(1.0);
        let c_high = pal.sample_rgb(1.5);
        assert_eq!(c_low, c0, "{:?} must clamp below 0.0", pal);
        assert_eq!(c_high, c1, "{:?} must clamp above 1.0", pal);
        assert!(
            c0 != c_mid || c_mid != c1,
            "{:?} must exhibit RGB variation across [0, 1]",
            pal
        );
    }

    let schema = HubSchema::load(None).expect("Embedded schema must load");
    let layers = schema.all_layers();
    assert_eq!(layers.len(), 34);
    for (_, layer) in layers {
        let (pal, min_v, max_v) = default_layer_style(&layer.id);
        assert!(min_v.is_finite() && max_v.is_finite() && min_v < max_v);
        assert_ne!(
            (pal, min_v, max_v),
            (ColormapPalette::Viridis, 0.0, 100.0),
            "Layer {} must have a dedicated scientific palette range",
            layer.id
        );
    }

    for basemap_id in [
        "bathymetry_topography_shaded",
        "dark_telemetry_vector",
        "viirs_nightlights_baseline",
    ] {
        let ocean = shade_basemap_rgb(basemap_id, 15.0, -40.0, -3500.0);
        let land = shade_basemap_rgb(basemap_id, 35.0, -90.0, 1200.0);
        assert_ne!(ocean, land, "Basemap {} must distinguish ocean and land", basemap_id);
    }
}

#[test]
fn coastlines_basins_and_reference_observatories_invariants() {
    assert_eq!(CONTINENT_POLYLINES.len(), 7);
    for (idx, poly) in CONTINENT_POLYLINES.iter().enumerate() {
        assert!(poly.len() >= 10, "Continent {} must contain >= 10 vertices", idx);
        let first = poly.first().unwrap();
        let last = poly.last().unwrap();
        assert!(
            (first.0 - last.0).abs() < 1e-4 && (first.1 - last.1).abs() < 1e-4,
            "Continent polyline {} must form a closed ring",
            idx
        );
        for &(lat, lon) in *poly {
            assert!(
                (-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon),
                "Coastline vertex ({}, {}) out of bounds",
                lat,
                lon
            );
        }
    }

    assert_eq!(TELECONNECTION_BASINS.len(), 5);
    let mut basin_ids = HashSet::new();
    for &(id, label, min_lat, max_lat, min_lon, max_lon) in TELECONNECTION_BASINS {
        assert!(basin_ids.insert(id), "Duplicate teleconnection basin id {}", id);
        assert!(!label.is_empty());
        assert!(min_lat < max_lat && min_lon < max_lon);
        assert!((-90.0..=90.0).contains(&min_lat) && (-90.0..=90.0).contains(&max_lat));
        assert!((-180.0..=180.0).contains(&min_lon) && (-180.0..=180.0).contains(&max_lon));
    }

    assert_eq!(REFERENCE_OBSERVATORIES.len(), 20);
    let mut station_names = HashSet::new();
    for &(name, lat, lon, role) in REFERENCE_OBSERVATORIES {
        assert!(station_names.insert(name), "Duplicate station {}", name);
        assert!(!role.is_empty());
        assert!((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon));
    }
}

#[test]
fn zarr_v3_raw_none_codec_and_regional_bbox_subset_exact_values() {
    let pid = std::process::id();
    let tmp_db = std::env::temp_dir().join(format!("terratwin_zarr_none_bbox_{pid}.db"));
    let zdir = std::env::temp_dir().join(format!("terratwin_zarr_none_bbox_{pid}.zarr"));
    let _ = std::fs::remove_file(&tmp_db);
    let _ = std::fs::remove_dir_all(&zdir);

    let schema = HubSchema::load(None).unwrap();
    let db = ClimateDatabase::open(&tmp_db).unwrap();
    db.sync_schema_catalog(&schema).unwrap();

    let cfg = SimulationConfig {
        grid_width: 72,
        grid_height: 36,
        scenario: SspScenario::Ssp2_45,
        sim_year: 2026.86,
        day_of_year: 315.0,
        hour_utc: 0.0,
        atmospheric_level_hpa: 850.0,
        ocean_depth_m: 0.0,
        time_step_secs: 3600.0,
        thermal_diffusivity_m2_s: 4.5e5,
        prognostic_relaxation_alpha: 0.16,
    };
    let engine = ClimateSimulationEngine::new(schema.clone(), db.clone());
    engine.step_all_layers(&cfg).unwrap();

    let (min_lat, max_lat, min_lon, max_lon) = (-30.0, 30.0, -120.0, -60.0);
    let summary = db
        .export_zarr_v3_store_with_codec(
            "sst_marine_heatwaves",
            cfg.scenario,
            0.0,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            (6, 6),
            ZarrCompressionCodec::None,
            &zdir,
        )
        .unwrap();

    assert_eq!(summary.codec, "bytes");
    assert_eq!(summary.shape, [12, 12]);
    assert_eq!(summary.uncompressed_bytes, summary.compressed_bytes);

    let grid = db
        .load_latest_grid("sst_marine_heatwaves", cfg.scenario, 0.0)
        .unwrap()
        .unwrap();
    let mut expected_rows = Vec::new();
    for r in 0..grid.height {
        let lat = 90.0 - (r as f64 + 0.5) * (180.0 / grid.height as f64);
        if (min_lat..=max_lat).contains(&lat) {
            expected_rows.push(r);
        }
    }
    let mut expected_cols = Vec::new();
    for c in 0..grid.width {
        let lon = -180.0 + (c as f64 + 0.5) * (360.0 / grid.width as f64);
        if (min_lon..=max_lon).contains(&lon) {
            expected_cols.push(c);
        }
    }
    assert_eq!(expected_rows.len(), 12);
    assert_eq!(expected_cols.len(), 12);

    let chunk00 = read_zarr_v3_chunk_f32(&zdir.join("sst_marine_heatwaves"), 0, 0).unwrap();
    assert_eq!(chunk00.len(), 36);
    for lr in 0..6 {
        for lc in 0..6 {
            let expected = grid.values[expected_rows[lr] * grid.width + expected_cols[lc]];
            let actual = chunk00[lr * 6 + lc];
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    let _ = std::fs::remove_dir_all(&zdir);
    let _ = std::fs::remove_file(&tmp_db);
}

#[test]
fn sqlite_retention_policy_prunes_old_ticks_and_preserves_latest() {
    let pid = std::process::id();
    let tmp_db = std::env::temp_dir().join(format!("terratwin_explicit_prune_{pid}.db"));
    let _ = std::fs::remove_file(&tmp_db);

    let schema = HubSchema::load(None).unwrap();
    let db = ClimateDatabase::open(&tmp_db).unwrap();
    db.sync_schema_catalog(&schema).unwrap();
    db.set_retention_policy(RetentionPolicy {
        max_ticks_per_scenario: 10,
        max_raw_samples_per_layer: 64,
        max_sync_log_rows: 32,
    });

    let mut cfg = SimulationConfig {
        grid_width: 36,
        grid_height: 18,
        scenario: SspScenario::Ssp5_85,
        sim_year: 2030.0,
        day_of_year: 120.0,
        hour_utc: 0.0,
        atmospheric_level_hpa: 500.0,
        ocean_depth_m: 0.0,
        time_step_secs: 3600.0,
        thermal_diffusivity_m2_s: 4.0e5,
        prognostic_relaxation_alpha: 0.16,
    };
    let engine = ClimateSimulationEngine::new(schema.clone(), db.clone());
    let mut all_ts = Vec::new();
    for _ in 0..5 {
        let r = engine.step_all_layers(&cfg).unwrap();
        all_ts.push(r.timestamp_utc);
        cfg.advance_tick();
    }
    assert_eq!(db.list_available_timestamps(cfg.scenario, 10).unwrap().len(), 5);

    db.set_retention_policy(RetentionPolicy {
        max_ticks_per_scenario: 2,
        max_raw_samples_per_layer: 64,
        max_sync_log_rows: 32,
    });
    let report = db.prune_expired_history(true).unwrap();
    assert!(report.grids_deleted > 0);
    assert!(report.grid_blob_bytes_remaining > 0);

    let kept = db.list_available_timestamps(cfg.scenario, 10).unwrap();
    assert_eq!(kept, all_ts[3..5].to_vec());

    let _ = std::fs::remove_file(&tmp_db);
}
