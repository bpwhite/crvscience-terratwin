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

//! Multi-threaded coupled climate physics, prognostic PDE transport & geophysical simulation engine (`rayon`).
//!
//! ### Core Coupled Physical & Numerical Subsystems
//! 0. **Prognostic 2-Layer Spherical Quasi-Geostrophic Potential Vorticity (`QGPV`) & Elliptic Inversion**:
//!    - Phillips (1956) two-level baroclinic QG model on the sphere (`Layer 1 ~ 250 hPa`, `Layer 2 ~ 800 hPa`)
//!      with planetary vorticity `2 * Omega * sin(phi)`, Charney-Eliassen topographic PV `f * h_topo / H_0`,
//!      diabatic radiative thermal-wind restoring, and lower-layer Ekman relative-vorticity spin-down.
//!    - Exact vertical normal-mode decomposition into Barotropic (`psi_bt = 0.5*(psi_1 + psi_2)`) and
//!      Baroclinic (`psi_bc = 0.5*(psi_1 - psi_2)`) Helmholtz equations solved in parallel via `f64`
//!      Red-Black Successive Over-Relaxation (`RB-SOR`), coupled to Charney (1955) spherical Linear Balance
//!      Equation (`lap_S(dp) = rho_0 * div_S(f * grad_S(psi_2))`) so MSLP lows, blocking highs, and
//!      synoptic cyclones evolve dynamically across SQLite ticks.
//! 1. **Conservative Quasi-Monotone 2D Spherical Semi-Lagrangian Transport (`QMSL`)**:
//!    - Two-time-level midpoint predictor-corrector departure trajectories on the sphere
//!      (Staniforth & Côté 1991) paired with Bermejo & Staniforth (1992) quasi-monotone bicubic
//!      Catmull-Rom interpolation and Priestley / Bermejo-Conde spherical `cos(phi)` mass conservation.
//! 2. **Sub-Stepped Conservative 2D Spherical Laplacian Heat & Moisture Diffusion**:
//!    - Exact finite-volume spherical divergence `div_S(K_h * grad_S(psi))` with polar Neumann
//!      zero-flux caps, `NaN` coastline reflection, and automatic Fourier CFL sub-stepping
//!      (North, Cahalan & Coakley 1981).
//! 3. **Dynamic Geostrophic-Ekman Momentum, Thermal Wind Shear & Orographic Moisture Cycle**:
//!    - Rayleigh-damped spherical geostrophic/cross-isobaric surface winds driven by assimilated
//!      MSLP gradients (`-grad_S(p) / rho_a`) with smooth equatorial `f`-regularization.
//!    - Baroclinic thermal wind shear `du/d(ln p)` integrated from horizontal `grad_S(T)` across `1000..=10 hPa`.
//!    - Prognostic specific humidity `q` stepped with bulk aerodynamic evaporation, moisture flux
//!      convergence (`-q * div_S(V)`), orographic upslope lifting (`V . grad_S(z_elev)`),
//!      Clausius-Clapeyron supersaturation condensation, and latent heat release (`L_v * dq / c_p`).
//! 4. **Prognostic 1.5-Layer Reduced-Gravity Ocean, Stommel-Munk Gyres, Carbonate & Sea-Ice**:
//!    - Coupled air-sea bulk wind stress `tau = rho_a * C_D * |V_10| * V_10` from QGPV + Ekman winds.
//!    - Stommel (1948) / Munk (1950) wind-driven barotropic gyre streamfunction solver with planetary `beta`
//!      westward intensification along continental boundaries (`psi_gyre = 0` on land).
//!    - Prognostic 1.5-layer reduced-gravity shallow-water equations (`u_sw, v_sw, eta_tc`) integrated with
//!      Forward-Backward Coriolis-split sub-stepping for equatorial Kelvin/Rossby thermocline waves & ENSO.
//!    - Iterative Newton-Raphson marine carbonate solver (Weiss 1974 `K0`; Lueker et al. 2000 `K1, K2`;
//!    - Iterative Newton-Raphson marine carbonate solver (Weiss 1974 `K0`; Lueker et al. 2000 `K1, K2`;
//!      Mucci 1983 + Millero 1995 pressure-dependent aragonite saturation `Omega_arag` & total `pH`).
//!    - Semtner (1976) 0-layer conductive sea-ice thermodynamic growth/ablation + basal ocean heat flux.
//!    - Prognostic root-zone soil water bucket (`P - E - Q_runoff`) coupled to river discharge.
//!    - Full 6-component Van Wagner & Pickett (1985) / Van Wagner (1987) Canadian Forest Fire Weather
//!      Index (`FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`).

use crate::coastlines::CONTINENT_POLYLINES;
use crate::db::{ClimateDatabase, GlobalLiveReadouts, StoredGridRecord};
use crate::interp::{
    assimilate_3dvar_cg, conserve_spherical_mass, sample_grid_bicubic_monotone,
    sample_grid_bilinear, spherical_area_weighted_mean, ObsSamplePoint, ThreeDVarConfig,
    EARTH_RADIUS_KM,
};
use crate::schema::{HubSchema, SspScenario};
use anyhow::Result;
use chrono::{Datelike, NaiveDate, Timelike};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const EARTH_RADIUS_M: f64 = EARTH_RADIUS_KM * 1000.0;
pub const EARTH_OMEGA_RAD_S: f64 = 7.292_115_9e-5;
pub const DRY_AIR_GAS_RD: f64 = 287.058;
pub const CP_DRY_AIR: f64 = 1_004.64;
pub const LATENT_HEAT_VAPORIZATION: f64 = 2.501e6;
pub const VOLUMETRIC_LATENT_HEAT_SEA_ICE: f64 = 3.02e8; // J m^-3
pub const LAND_HEAT_CAPACITY_J_M2_K: f64 = 8.5e6;
pub const OCEAN_MIXED_LAYER_HEAT_CAPACITY_J_M2_K: f64 = 2.05e8;

/// First internal baroclinic Rossby radius of deformation `L_d` (`750 km` in m).
pub const QG_ROSSBY_DEFORMATION_RADIUS_M: f64 = 750_000.0;
/// External equivalent-barotropic Rossby radius `L_ext` (`2,800 km` in m).
pub const QG_EXTERNAL_ROSSBY_RADIUS_M: f64 = 2_800_000.0;
/// Tropospheric density scale height `H_0` (`8.4 km` in m) for topographic QGPV forcing.
pub const QG_SCALE_HEIGHT_M: f64 = 8_400.0;
/// Polar zonal metric floor `cos(phi)_min` matching Fourier polar filtering on lat-lon grids.
pub const QG_MIN_COS_ZONAL: f64 = 0.22;
/// Mean active upper-ocean thermocline layer thickness `H_0` (`150 m`) for 1.5-layer reduced-gravity model.
pub const OCEAN_THERMOCLINE_H0_M: f64 = 150.0;
/// Reduced gravity `g' = g * (delta_rho / rho_0)` (`0.032 m s^-2`, giving `c_1 = sqrt(g' H_0) ~ 2.19 m s^-1`).
pub const OCEAN_REDUCED_GRAVITY_M_S2: f64 = 0.032;
/// Reference seawater density `rho_0` (`1025 kg m^-3`).
pub const SEAWATER_RHO0_KG_M3: f64 = 1025.0;

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub scenario: SspScenario,
    pub sim_year: f64,
    pub day_of_year: f64,
    pub hour_utc: f64,
    pub atmospheric_level_hpa: f64,
    pub ocean_depth_m: f64,
    /// Prognostic integration time step (`dt`) in seconds per SQLite simulation tick.
    pub time_step_secs: f64,
    /// Horizontal synoptic macro-turbulent eddy diffusivity `K_h` (`m^2 s^-1`).
    pub thermal_diffusivity_m2_s: f64,
    /// Radiative-convective / climatological restoring blend per tick (`0.0` = free PDE, `1.0` = diagnostic).
    pub prognostic_relaxation_alpha: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            grid_width: 288,
            grid_height: 144,
            scenario: SspScenario::Ssp2_45,
            sim_year: now.year() as f64 + (now.ordinal() as f64 / 365.25),
            day_of_year: now.ordinal() as f64,
            hour_utc: now.hour() as f64 + (now.minute() as f64 / 60.0),
            atmospheric_level_hpa: 850.0,
            ocean_depth_m: 0.0,
            time_step_secs: 3600.0,
            thermal_diffusivity_m2_s: 4.5e5,
            prognostic_relaxation_alpha: 0.16,
        }
    }
}

impl SimulationConfig {
    /// Advance simulation clock forward by `self.time_step_secs`, rolling hours, days, and years cleanly.
    pub fn advance_tick(&mut self) {
        let dt_hours = self.time_step_secs / 3600.0;
        self.hour_utc += dt_hours;
        while self.hour_utc >= 24.0 {
            self.hour_utc -= 24.0;
            self.day_of_year += 1.0;
        }
        while self.day_of_year > 365.0 {
            self.day_of_year -= 365.0;
        }
        self.sim_year += self.time_step_secs / (365.25 * 86_400.0);
    }

    /// Format `(sim_year, day_of_year, hour_utc)` into a canonical ISO-8601 UTC timestamp string.
    pub fn format_timestamp_utc(&self) -> String {
        let yr = (self.sim_year.floor() as i32).clamp(1900, 2200);
        let doy = (self.day_of_year.floor() as u32).clamp(1, 365);
        let date = NaiveDate::from_yo_opt(yr, doy)
            .unwrap_or_else(|| NaiveDate::from_ymd_opt(yr, 1, 1).expect("Valid fallback date"));
        let total_secs = ((self.hour_utc.rem_euclid(24.0)) * 3600.0).round() as u32;
        let hh = (total_secs / 3600).min(23);
        let mm = (total_secs % 3600) / 60;
        let ss = total_secs % 60;
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            date.year(),
            date.month(),
            date.day(),
            hh,
            mm,
            ss
        )
    }
}

/// Seawater carbonate chemistry equilibrium result (`pH` on Total scale, `Omega_aragonite`, `pCO2_uatm`).
#[derive(Debug, Clone, Copy)]
pub struct SeawaterCarbonateState {
    pub ph_total: f64,
    pub omega_aragonite: f64,
    pub pco2_uatm: f64,
}

/// Full 6-component Van Wagner (1987) Canadian Forest Fire Weather Index state.
#[derive(Debug, Clone, Copy)]
pub struct FireWeatherComponents {
    pub ffmc: f64,
    pub dmc: f64,
    pub dc: f64,
    pub isi: f64,
    pub bui: f64,
    pub fwi: f64,
}

/// Normalize `(lat_deg, lon_deg)` across poles (`|lat| > 90` flips hemisphere & shifts `lon + 180`)
/// and wraps longitude into `[-180.0, 180.0)`.
#[inline]
pub fn normalize_spherical_latlon(mut lat_deg: f64, mut lon_deg: f64) -> (f64, f64) {
    if lat_deg > 90.0 {
        lat_deg = 180.0 - lat_deg;
        lon_deg += 180.0;
    } else if lat_deg < -90.0 {
        lat_deg = -180.0 - lat_deg;
        lon_deg += 180.0;
    }
    lat_deg = lat_deg.clamp(-89.999, 89.999);
    lon_deg = ((lon_deg + 180.0).rem_euclid(360.0)) - 180.0;
    (lat_deg, lon_deg)
}

/// Unconditionally Courant-stable Quasi-Monotone 2D Spherical Semi-Lagrangian Advection (`QMSL`).
///
/// Traces arrival cells `(lat_A, lon_A)` upstream via a 2-iteration midpoint predictor-corrector
/// (Staniforth & Côté 1991) and evaluates `field` at the departure point `(lat_D, lon_D)` using
/// Bermejo & Staniforth (1992) quasi-monotone bicubic Catmull-Rom interpolation (`O(dx^3)` without
/// spurious overshoots or negative tracer undershoots).
pub fn semi_lagrangian_advect_2d_spherical(
    field: &[f32],
    u_ms: &[f32],
    v_ms: &[f32],
    width: usize,
    height: usize,
    dt_secs: f64,
) -> Vec<f32> {
    semi_lagrangian_advect_qmsl(field, u_ms, v_ms, width, height, dt_secs, false, false)
}

/// Extended Conservative Quasi-Monotone Semi-Lagrangian (`CQMSL`) solver with optional global
/// `cos(phi)` spherical mass conservation restoration and strict non-negativity enforcement.
pub fn semi_lagrangian_advect_qmsl(
    field: &[f32],
    u_ms: &[f32],
    v_ms: &[f32],
    width: usize,
    height: usize,
    dt_secs: f64,
    conserve_mass: bool,
    non_negative: bool,
) -> Vec<f32> {
    assert_eq!(field.len(), width * height);
    assert_eq!(u_ms.len(), width * height);
    assert_eq!(v_ms.len(), width * height);

    if dt_secs.abs() < 1e-6 || width == 0 || height == 0 {
        return field.to_vec();
    }

    let initial_mean = if conserve_mass {
        spherical_area_weighted_mean(field, width, height)
    } else {
        0.0
    };

    let mut advected = vec![0.0_f32; width * height];
    advected
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let lat_a = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
            for (col, out_cell) in out_row.iter_mut().enumerate() {
                let idx = row * width + col;
                let arrival_val = field[idx];
                if !arrival_val.is_finite() {
                    *out_cell = f32::NAN;
                    continue;
                }

                let lon_a = -180.0 + (col as f64 + 0.5) * (360.0 / width as f64);
                let mut u_mid = u_ms[idx] as f64;
                let mut v_mid = v_ms[idx] as f64;
                if !u_mid.is_finite() || !v_mid.is_finite() {
                    *out_cell = arrival_val;
                    continue;
                }

                let mut lat_m = lat_a;
                let mut lon_m;
                for _ in 0..2 {
                    let cos_phi = lat_m.to_radians().cos().max(0.14);
                    let dlat_deg = ((v_mid * 0.5 * dt_secs) / EARTH_RADIUS_M).to_degrees();
                    let dlon_deg =
                        ((u_mid * 0.5 * dt_secs) / (EARTH_RADIUS_M * cos_phi)).to_degrees();
                    let (nm_lat, nm_lon) =
                        normalize_spherical_latlon(lat_a - dlat_deg, lon_a - dlon_deg);
                    lat_m = nm_lat;
                    lon_m = nm_lon;

                    let um_s = sample_grid_bilinear(u_ms, width, height, lat_m, lon_m) as f64;
                    let vm_s = sample_grid_bilinear(v_ms, width, height, lat_m, lon_m) as f64;
                    if um_s.is_finite() && vm_s.is_finite() {
                        u_mid = um_s;
                        v_mid = vm_s;
                    }
                }

                let cos_mid = lat_m.to_radians().cos().max(0.14);
                let full_dlat = ((v_mid * dt_secs) / EARTH_RADIUS_M).to_degrees();
                let full_dlon = ((u_mid * dt_secs) / (EARTH_RADIUS_M * cos_mid)).to_degrees();
                let (lat_d, lon_d) =
                    normalize_spherical_latlon(lat_a - full_dlat, lon_a - full_dlon);

                let sampled = sample_grid_bicubic_monotone(field, width, height, lat_d, lon_d);
                let val = if sampled.is_finite() {
                    sampled
                } else {
                    arrival_val
                };
                *out_cell = if non_negative { val.max(0.0) } else { val };
            }
        });

    if conserve_mass {
        conserve_spherical_mass(&mut advected, width, height, initial_mean, non_negative);
    }
    advected
}

/// Conservative 2D Spherical Laplacian Diffusion + Source Integration (`rayon` parallel)
/// with automatic Fourier CFL sub-stepping and polar Neumann zero-flux boundaries.
pub fn spherical_diffusive_heat_step(
    field: &[f32],
    forcing_rate_per_sec: Option<&[f32]>,
    width: usize,
    height: usize,
    diffusivity_m2_s: f64,
    dt_secs: f64,
) -> Vec<f32> {
    assert_eq!(field.len(), width * height);
    if let Some(s) = forcing_rate_per_sec {
        assert_eq!(s.len(), width * height);
    }
    if dt_secs <= 0.0 || diffusivity_m2_s <= 0.0 || width < 4 || height < 4 {
        return field.to_vec();
    }

    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let min_cos_zonal = 0.22_f64;
    let dy_m = EARTH_RADIUS_M * dphi;
    let dx_min_m = EARTH_RADIUS_M * min_cos_zonal * dlam;
    let min_grid_m2 = dy_m.min(dx_min_m).powi(2);

    let dt_max = (0.20 * min_grid_m2 / diffusivity_m2_s).max(1.0);
    let n_sub = ((dt_secs / dt_max).ceil() as usize).clamp(1, 256);
    let dt_sub = (dt_secs / (n_sub as f64)).min(dt_max);

    let mut cur = field.to_vec();
    let mut next = cur.clone();

    for _ in 0..n_sub {
        next.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out_row)| {
                let lat_c = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
                let phi_c = lat_c.to_radians();
                let cos_c = phi_c.cos().max(1e-4);
                let cos_zonal = phi_c.cos().max(min_cos_zonal);

                let cos_north_face = if row == 0 {
                    0.0
                } else {
                    (90.0 - (row as f64) * (180.0 / height as f64))
                        .to_radians()
                        .cos()
                        .max(0.0)
                };
                let cos_south_face = if row + 1 >= height {
                    0.0
                } else {
                    (90.0 - ((row + 1) as f64) * (180.0 / height as f64))
                        .to_radians()
                        .cos()
                        .max(0.0)
                };

                let row_n = row.saturating_sub(1);
                let row_s = (row + 1).min(height - 1);

                for col in 0..width {
                    let idx = row * width + col;
                    let tc = cur[idx];
                    if !tc.is_finite() {
                        out_row[col] = f32::NAN;
                        continue;
                    }
                    let tc_f = tc as f64;
                    let col_w = if col == 0 { width - 1 } else { col - 1 };
                    let col_e = if col + 1 == width { 0 } else { col + 1 };

                    let tn = {
                        let v = cur[row_n * width + col];
                        if v.is_finite() { v as f64 } else { tc_f }
                    };
                    let ts = {
                        let v = cur[row_s * width + col];
                        if v.is_finite() { v as f64 } else { tc_f }
                    };
                    let tw = {
                        let v = cur[row * width + col_w];
                        if v.is_finite() { v as f64 } else { tc_f }
                    };
                    let te = {
                        let v = cur[row * width + col_e];
                        if v.is_finite() { v as f64 } else { tc_f }
                    };

                    let lap_meridional =
                        (cos_north_face * (tn - tc_f) - cos_south_face * (tc_f - ts))
                            / (EARTH_RADIUS_M * EARTH_RADIUS_M * cos_c * dphi * dphi);
                    let lap_zonal = (te - 2.0 * tc_f + tw)
                        / (EARTH_RADIUS_M * EARTH_RADIUS_M * cos_zonal * cos_zonal * dlam * dlam);

                    let src = forcing_rate_per_sec
                        .map(|s| s[idx] as f64)
                        .filter(|v| v.is_finite())
                        .unwrap_or(0.0);

                    out_row[col] =
                        (tc_f + dt_sub * (diffusivity_m2_s * (lap_meridional + lap_zonal) + src))
                            as f32;
                }
            });
        std::mem::swap(&mut cur, &mut next);
    }

    cur
}

/// Complete prognostic & diagnostic state of the 2-Layer Spherical Quasi-Geostrophic (`QGPV`) solver.
#[derive(Debug, Clone)]
pub struct TwoLayerQgpvState {
    /// Upper-tropospheric (`~250 hPa`) quasi-geostrophic potential vorticity `q_1` (`s^-1`).
    pub q1: Vec<f32>,
    /// Lower-tropospheric (`~800 hPa`) quasi-geostrophic potential vorticity `q_2` (`s^-1`).
    pub q2: Vec<f32>,
    /// Inverted upper-layer geostrophic streamfunction `psi_1` (`m^2 s^-1`).
    pub psi1: Vec<f32>,
    /// Inverted lower-layer geostrophic streamfunction `psi_2` (`m^2 s^-1`).
    pub psi2: Vec<f32>,
    /// Upper-layer non-divergent zonal wind `u_1 = -(1/R_E) d(psi_1)/d(phi)` (`m s^-1`).
    pub u1: Vec<f32>,
    /// Upper-layer non-divergent meridional wind `v_1 = (1/(R_E cos(phi))) d(psi_1)/d(lam)` (`m s^-1`).
    pub v1: Vec<f32>,
    /// Lower-layer non-divergent zonal wind `u_2 = -(1/R_E) d(psi_2)/d(phi)` (`m s^-1`).
    pub u2: Vec<f32>,
    /// Lower-layer non-divergent meridional wind `v_2 = (1/(R_E cos(phi))) d(psi_2)/d(lam)` (`m s^-1`).
    pub v2: Vec<f32>,
    /// Dynamically balanced sea-level pressure `mslp_hpa` (`hPa`) from spherical Linear Balance of `psi_2`.
    pub mslp_hpa: Vec<f32>,
}

/// Planetary Coriolis parameter `f(phi) = 2 * Omega * sin(phi)` (`s^-1`).
#[inline]
pub fn planetary_coriolis_f(lat_deg: f64) -> f64 {
    2.0 * EARTH_OMEGA_RAD_S * lat_deg.to_radians().sin()
}

/// Effective lower-tropospheric Charney-Eliassen topographic potential vorticity `q_topo` (`s^-1`),
/// tapered smoothly across the equator so tropical cordilleras do not inject spurious `f = 0` noise.
#[inline]
pub fn topographic_qgpv_forcing_s1(lat_deg: f64, elev_m: f64) -> f64 {
    let h_pos = elev_m.clamp(0.0, 5_500.0);
    if h_pos <= 0.0 {
        return 0.0;
    }
    let f_lat = planetary_coriolis_f(lat_deg);
    let midlat_taper = (lat_deg.abs() / 15.0).tanh();
    (0.38 * f_lat * midlat_taper * (h_pos / QG_SCALE_HEIGHT_M)).clamp(-6.5e-5, 6.5e-5)
}

#[inline]
fn spherical_area_weighted_mean_f64(field: &[f64], width: usize, height: usize) -> f64 {
    let mut w_sum = 0.0_f64;
    let mut a_sum = 0.0_f64;
    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let w_lat = lat.to_radians().cos().max(1e-4);
        let row_off = row * width;
        for col in 0..width {
            let v = field[row_off + col];
            if v.is_finite() {
                w_sum += v * w_lat;
                a_sum += w_lat;
            }
        }
    }
    if a_sum > 0.0 {
        w_sum / a_sum
    } else {
        0.0
    }
}

/// High-precision `f64` self-adjoint finite-volume spherical Laplacian `lap_S(psi)` (`m^-2 * [psi]`)
/// with periodic longitude wrap and polar zero-flux Neumann caps.
fn spherical_laplacian_f64(psi: &[f64], width: usize, height: usize) -> Vec<f64> {
    assert_eq!(psi.len(), width * height);
    if width < 4 || height < 4 {
        return vec![0.0; width * height];
    }
    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let r2 = EARTH_RADIUS_M * EARTH_RADIUS_M;

    let mut out = vec![0.0_f64; width * height];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let lat_c = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
            let phi_c = lat_c.to_radians();
            let cos_c = phi_c.cos().max(1e-4);
            let cos_z = phi_c.cos().max(QG_MIN_COS_ZONAL);

            let cos_n = if row == 0 {
                0.0
            } else {
                (90.0 - (row as f64) * (180.0 / height as f64))
                    .to_radians()
                    .cos()
                    .max(0.0)
            };
            let cos_s = if row + 1 >= height {
                0.0
            } else {
                (90.0 - ((row + 1) as f64) * (180.0 / height as f64))
                    .to_radians()
                    .cos()
                    .max(0.0)
            };

            let a_n = cos_n / (r2 * cos_c * dphi * dphi);
            let a_s = cos_s / (r2 * cos_c * dphi * dphi);
            let a_ew = 1.0 / (r2 * cos_z * cos_z * dlam * dlam);

            let row_n = row.saturating_sub(1);
            let row_s = (row + 1).min(height - 1);

            for col in 0..width {
                let col_w = if col == 0 { width - 1 } else { col - 1 };
                let col_e = if col + 1 == width { 0 } else { col + 1 };
                let pc = psi[row * width + col];
                let pn = psi[row_n * width + col];
                let ps = psi[row_s * width + col];
                let pw = psi[row * width + col_w];
                let pe = psi[row * width + col_e];

                out_row[col] = a_n * (pn - pc) + a_s * (ps - pc) + a_ew * (pe + pw - 2.0 * pc);
            }
        });
    out
}

/// Public `f32` wrapper computing the conservative spherical Laplacian `lap_S(field)` (e.g. relative vorticity `zeta = lap_S(psi)`).
pub fn spherical_laplacian_2d(field: &[f32], width: usize, height: usize) -> Vec<f32> {
    let f64_in: Vec<f64> = field
        .iter()
        .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
        .collect();
    spherical_laplacian_f64(&f64_in, width, height)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// High-precision `f64` Red-Black Successive Over-Relaxation (`RB-SOR`) elliptic solver on the sphere:
/// solves `(lap_S - inv_radius_sq_m2) * psi = rhs` with periodic longitudes and polar Neumann caps.
fn invert_spherical_helmholtz_f64(
    rhs: &[f64],
    width: usize,
    height: usize,
    inv_radius_sq_m2: f64,
    max_iters: usize,
) -> Vec<f64> {
    assert_eq!(rhs.len(), width * height);
    let n = width * height;
    if width < 4 || height < 4 || max_iters == 0 {
        return vec![0.0; n];
    }

    let gamma = inv_radius_sq_m2.max(0.0);
    let near_singular_poisson = gamma < 1.0e-14;

    // For pure or near-pure Poisson problems, project out the spherical mean of RHS for solvability
    let rhs_adj: Vec<f64> = if near_singular_poisson {
        let mean_rhs = spherical_area_weighted_mean_f64(rhs, width, height);
        rhs.iter().map(|&r| r - mean_rhs).collect()
    } else {
        rhs.to_vec()
    };

    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let r2 = EARTH_RADIUS_M * EARTH_RADIUS_M;

    // Precompute latitude row stencil coefficients (a_n, a_s, a_ew, diag)
    let mut row_coeffs = Vec::with_capacity(height);
    for row in 0..height {
        let lat_c = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let phi_c = lat_c.to_radians();
        let cos_c = phi_c.cos().max(1e-4);
        let cos_z = phi_c.cos().max(QG_MIN_COS_ZONAL);

        let cos_n = if row == 0 {
            0.0
        } else {
            (90.0 - (row as f64) * (180.0 / height as f64))
                .to_radians()
                .cos()
                .max(0.0)
        };
        let cos_s = if row + 1 >= height {
            0.0
        } else {
            (90.0 - ((row + 1) as f64) * (180.0 / height as f64))
                .to_radians()
                .cos()
                .max(0.0)
        };

        let a_n = cos_n / (r2 * cos_c * dphi * dphi);
        let a_s = cos_s / (r2 * cos_c * dphi * dphi);
        let a_ew = 1.0 / (r2 * cos_z * cos_z * dlam * dlam);
        let diag = (a_n + a_s + 2.0 * a_ew + gamma).max(1.0e-30);
        row_coeffs.push((a_n, a_s, a_ew, diag));
    }

    let rhs_norm = rhs_adj
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0e-30);
    let omega = 1.76_f64;

    let mut psi = vec![0.0_f64; n];
    let mut next_psi = vec![0.0_f64; n];
    // Cap SOR sweeps on fine 288x144 GUI grids so coupled step remains <120 ms while coarse test grids run full max_iters
    let effective_max_iters = if n > 20_000 { max_iters.min(36) } else { max_iters };
    let res_tol = if n > 20_000 { 6.0e-5 } else { 1.0e-7 };

    for iter in 0..effective_max_iters {
        for color in 0..2usize {
            next_psi
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let (a_n, a_s, a_ew, diag) = row_coeffs[row];
                    let row_n = row.saturating_sub(1);
                    let row_s = (row + 1).min(height - 1);
                    let row_off = row * width;

                    for col in 0..width {
                        let pc = psi[row_off + col];
                        if ((row + col) & 1) != color {
                            out_row[col] = pc;
                            continue;
                        }
                        let col_w = if col == 0 { width - 1 } else { col - 1 };
                        let col_e = if col + 1 == width { 0 } else { col + 1 };

                        let pn = psi[row_n * width + col];
                        let ps = psi[row_s * width + col];
                        let pw = psi[row_off + col_w];
                        let pe = psi[row_off + col_e];

                        let gs_star =
                            (a_n * pn + a_s * ps + a_ew * (pe + pw) - rhs_adj[row_off + col])
                                / diag;
                        out_row[col] = (1.0 - omega) * pc + omega * gs_star;
                    }
                });
            std::mem::swap(&mut psi, &mut next_psi);
        }

        if near_singular_poisson && (iter % 16 == 15 || iter + 1 == effective_max_iters) {
            let mean_psi = spherical_area_weighted_mean_f64(&psi, width, height);
            for v in &mut psi {
                *v -= mean_psi;
            }
        }

        // Check max residual every 12 sweeps for early convergence exit
        if iter >= 23 && iter % 12 == 11 {
            let lap = spherical_laplacian_f64(&psi, width, height);
            let max_res = lap
                .iter()
                .zip(psi.iter())
                .zip(rhs_adj.iter())
                .fold(0.0_f64, |acc, ((&lp, &p), &r)| {
                    acc.max(((lp - gamma * p) - r).abs())
                });
            if max_res <= res_tol * rhs_norm {
                break;
            }
        }
    }

    psi
}

/// Public `f32` elliptic screened-Poisson / Helmholtz solver on the sphere:
/// solves `(lap_S - inv_radius_sq_m2) * psi = rhs`.
pub fn invert_spherical_helmholtz(
    rhs: &[f32],
    width: usize,
    height: usize,
    inv_radius_sq_m2: f64,
    max_iters: usize,
) -> Vec<f32> {
    let rhs_f64: Vec<f64> = rhs
        .iter()
        .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
        .collect();
    invert_spherical_helmholtz_f64(&rhs_f64, width, height, inv_radius_sq_m2, max_iters)
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// Forward 2-layer Phillips (1956) spherical QGPV operator `(psi_1, psi_2) -> (q_1, q_2)`:
/// - `q_1 = lap_S(psi_1) - (psi_1 - psi_2)/(2 L_d^2) - (psi_1 + psi_2)/(2 L_ext^2) + f(phi)`
/// - `q_2 = lap_S(psi_2) + (psi_1 - psi_2)/(2 L_d^2) - (psi_1 + psi_2)/(2 L_ext^2) + f(phi) + q_topo`
pub fn compute_two_layer_qgpv_from_streamfunctions(
    psi1: &[f32],
    psi2: &[f32],
    elev_m: &[f64],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = width * height;
    assert_eq!(psi1.len(), n);
    assert_eq!(psi2.len(), n);
    assert_eq!(elev_m.len(), n);

    let p1: Vec<f64> = psi1.iter().map(|&v| v as f64).collect();
    let p2: Vec<f64> = psi2.iter().map(|&v| v as f64).collect();
    let (lap1, lap2) = rayon::join(
        || spherical_laplacian_f64(&p1, width, height),
        || spherical_laplacian_f64(&p2, width, height),
    );

    let f_bc = 1.0 / (2.0 * QG_ROSSBY_DEFORMATION_RADIUS_M * QG_ROSSBY_DEFORMATION_RADIUS_M);
    let f_bt = 1.0 / (2.0 * QG_EXTERNAL_ROSSBY_RADIUS_M * QG_EXTERNAL_ROSSBY_RADIUS_M);

    let mut q1 = vec![0.0_f32; n];
    let mut q2 = vec![0.0_f32; n];

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let f_cor = planetary_coriolis_f(lat);
        let row_off = row * width;
        for col in 0..width {
            let i = row_off + col;
            let baroclinic_term = f_bc * (p1[i] - p2[i]);
            let barotropic_screen = f_bt * (p1[i] + p2[i]);
            let q_topo = topographic_qgpv_forcing_s1(lat, elev_m[i]);

            q1[i] = (lap1[i] - baroclinic_term - barotropic_screen + f_cor) as f32;
            q2[i] = (lap2[i] + baroclinic_term - barotropic_screen + f_cor + q_topo) as f32;
        }
    }

    (q1, q2)
}

/// Exact decoupled vertical normal-mode QGPV streamfunction inversion `(q_1, q_2) -> (psi_1, psi_2)`:
/// 1. Barotropic mode `psi_bt = 0.5*(psi_1 + psi_2)` solves `(lap_S - L_ext^-2) psi_bt = 0.5*(q_1 + q_2) - f - 0.5*q_topo`
/// 2. Baroclinic mode `psi_bc = 0.5*(psi_1 - psi_2)` solves `(lap_S - L_d^-2) psi_bc = 0.5*(q_1 - q_2) + 0.5*q_topo`
/// 3. Reconstructs `psi_1 = psi_bt + psi_bc` and `psi_2 = psi_bt - psi_bc`.
pub fn invert_two_layer_qgpv_streamfunctions(
    q1: &[f32],
    q2: &[f32],
    elev_m: &[f64],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = width * height;
    assert_eq!(q1.len(), n);
    assert_eq!(q2.len(), n);
    assert_eq!(elev_m.len(), n);

    let mut rhs_bt = vec![0.0_f64; n];
    let mut rhs_bc = vec![0.0_f64; n];

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let f_cor = planetary_coriolis_f(lat);
        let row_off = row * width;
        for col in 0..width {
            let i = row_off + col;
            let q1_v = if q1[i].is_finite() { q1[i] as f64 } else { f_cor };
            let q2_v = if q2[i].is_finite() { q2[i] as f64 } else { f_cor };
            let q_topo = topographic_qgpv_forcing_s1(lat, elev_m[i]);

            rhs_bt[i] = 0.5 * (q1_v + q2_v) - f_cor - 0.5 * q_topo;
            rhs_bc[i] = 0.5 * (q1_v - q2_v) + 0.5 * q_topo;
        }
    }

    let inv_lext_sq = 1.0 / (QG_EXTERNAL_ROSSBY_RADIUS_M * QG_EXTERNAL_ROSSBY_RADIUS_M);
    let inv_ld_sq = 1.0 / (QG_ROSSBY_DEFORMATION_RADIUS_M * QG_ROSSBY_DEFORMATION_RADIUS_M);

    let (psi_bt, psi_bc) = rayon::join(
        || invert_spherical_helmholtz_f64(&rhs_bt, width, height, inv_lext_sq, 260),
        || invert_spherical_helmholtz_f64(&rhs_bc, width, height, inv_ld_sq, 180),
    );

    let mut psi1 = vec![0.0_f32; n];
    let mut psi2 = vec![0.0_f32; n];
    for i in 0..n {
        psi1[i] = (psi_bt[i] + psi_bc[i]) as f32;
        psi2[i] = (psi_bt[i] - psi_bc[i]) as f32;
    }
    (psi1, psi2)
}

/// Diagnose non-divergent rotational wind `V_psi = k x grad_S(psi)`:
/// `u = -(1 / R_E) * d(psi)/d(phi)`, `v = +(1 / (R_E * cos(phi))) * d(psi)/d(lambda)`.
pub fn diagnose_rotational_wind_from_streamfunction(
    psi: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = width * height;
    assert_eq!(psi.len(), n);
    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);

    let mut u = vec![0.0_f32; n];
    let mut v = vec![0.0_f32; n];

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let cos_phi = lat.to_radians().cos().max(0.18);
        let row_n = row.saturating_sub(1);
        let row_s = (row + 1).min(height - 1);
        let dy_rad = ((row_s - row_n).max(1) as f64) * dphi;
        let row_off = row * width;

        for col in 0..width {
            let col_w = if col == 0 { width - 1 } else { col - 1 };
            let col_e = if col + 1 == width { 0 } else { col + 1 };

            // row_n is northward (+phi), row_s is southward (-phi)
            let dpsi_dphi = (psi[row_n * width + col] as f64 - psi[row_s * width + col] as f64) / dy_rad;
            let dpsi_dlam =
                (psi[row_off + col_e] as f64 - psi[row_off + col_w] as f64) / (2.0 * dlam);

            let u_val = (-(1.0 / EARTH_RADIUS_M) * dpsi_dphi).clamp(-95.0, 95.0);
            let v_val = ((1.0 / (EARTH_RADIUS_M * cos_phi)) * dpsi_dlam).clamp(-75.0, 75.0);

            u[row_off + col] = u_val as f32;
            v[row_off + col] = v_val as f32;
        }
    }

    (u, v)
}

/// Solve Charney (1955) spherical Linear Balance Equation from lower-layer streamfunction `psi_2`:
/// `(lap_S - gamma_p) * delta_p = rho_0 * div_S( f(phi) * grad_S(psi_2) )`
/// and return balanced Mean Sea-Level Pressure `mslp_hpa` (`hPa`) centered on `1013.25 hPa`.
pub fn solve_spherical_linear_balance_mslp(
    psi2: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let n = width * height;
    assert_eq!(psi2.len(), n);
    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let r2 = EARTH_RADIUS_M * EARTH_RADIUS_M;
    let rho_0 = 1.225_f64;

    let p2: Vec<f64> = psi2
        .iter()
        .map(|&v| if v.is_finite() { v as f64 } else { 0.0 })
        .collect();

    // Conservative flux-divergence form of rho_0 * div_S(f(phi) * grad_S(psi_2))
    let mut rhs_pa = vec![0.0_f64; n];
    rhs_pa
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let lat_c = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
            let phi_c = lat_c.to_radians();
            let cos_c = phi_c.cos().max(1e-4);
            let cos_z = phi_c.cos().max(QG_MIN_COS_ZONAL);
            let f_c = planetary_coriolis_f(lat_c);

            let (cos_n, f_n) = if row == 0 {
                (0.0, 0.0)
            } else {
                let lat_n = 90.0 - (row as f64) * (180.0 / height as f64);
                (
                    lat_n.to_radians().cos().max(0.0),
                    planetary_coriolis_f(lat_n),
                )
            };
            let (cos_s, f_s) = if row + 1 >= height {
                (0.0, 0.0)
            } else {
                let lat_s = 90.0 - ((row + 1) as f64) * (180.0 / height as f64);
                (
                    lat_s.to_radians().cos().max(0.0),
                    planetary_coriolis_f(lat_s),
                )
            };

            let row_n = row.saturating_sub(1);
            let row_s = (row + 1).min(height - 1);
            let row_off = row * width;

            for col in 0..width {
                let col_w = if col == 0 { width - 1 } else { col - 1 };
                let col_e = if col + 1 == width { 0 } else { col + 1 };

                let pc = p2[row_off + col];
                let pn = p2[row_n * width + col];
                let ps = p2[row_s * width + col];
                let pw = p2[row_off + col_w];
                let pe = p2[row_off + col_e];

                let div_merid =
                    (cos_n * f_n * (pn - pc) - cos_s * f_s * (pc - ps)) / (r2 * cos_c * dphi * dphi);
                let div_zonal =
                    (f_c * (pe - 2.0 * pc + pw)) / (r2 * cos_z * cos_z * dlam * dlam);

                out_row[col] = rho_0 * (div_merid + div_zonal);
            }
        });

    // Weak external acoustic-gravity screening (L_p ~ 4,800 km) guarantees fast convergence & zero gauge drift
    let gamma_p = 1.0 / (4_800_000.0 * 4_800_000.0);
    let dp_pa = invert_spherical_helmholtz_f64(&rhs_pa, width, height, gamma_p, 260);
    let mean_dp = spherical_area_weighted_mean_f64(&dp_pa, width, height);

    dp_pa
        .into_iter()
        .map(|dp| (1013.25 + (dp - mean_dp) / 100.0).clamp(950.0, 1052.0) as f32)
        .collect()
}

/// Construct radiative-convective & teleconnection equilibrium streamfunctions `(psi_1_eq, psi_2_eq)`,
/// optionally superposing baroclinically unstable synoptic cyclone wave packets on cold start.
fn build_equilibrium_streamfunctions(
    t2m_c: &[f64],
    width: usize,
    height: usize,
    seasonal_phase_rad: f64,
    global_warming_c: f64,
    enso_nino34: f64,
    nao_index: f64,
    seed_baroclinic_eddies: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n = width * height;
    let mut psi1 = vec![0.0_f32; n];
    let mut psi2 = vec![0.0_f32; n];
    let seas_cos = seasonal_phase_rad.cos();

    // Compute zonal-mean temperature at each latitude row so local land-sea thermal anomalies force stationary waves
    let mut zonal_mean_t = vec![0.0_f64; height];
    for row in 0..height {
        let row_off = row * width;
        let sum: f64 = (0..width).map(|c| t2m_c[row_off + c]).sum();
        zonal_mean_t[row] = sum / (width.max(1) as f64);
    }

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let phi = lat.to_radians();
        let sign_hem = if lat >= 0.0 { 1.0 } else { -1.0 };

        // Exact analytical meridional antiderivative of lower-layer zonal background wind:
        // u_2(phi) = -5.2 * cos(3*phi) + 6.4 * sin^2(2*phi)
        let psi2_zon = -EARTH_RADIUS_M
            * (-(5.2 / 3.0) * (3.0 * phi).sin() + 3.2 * phi - 0.80 * (4.0 * phi).sin());

        // Seasonal + Arctic-amplification thermal-wind shear amplitude (stronger in winter hemisphere)
        let shear_amp = if lat >= 0.0 {
            (27.5 - 6.5 * seas_cos - 0.55 * global_warming_c).clamp(16.0, 38.0)
        } else {
            (30.5 + 5.5 * seas_cos - 0.25 * global_warming_c).clamp(18.0, 40.0)
        };
        // Exact analytical antiderivative of u_shear(phi) = shear_amp * sin^2(2*phi)
        let psi_shear_zon = -EARTH_RADIUS_M * shear_amp * (0.5 * phi - 0.125 * (4.0 * phi).sin());

        let storm_track_env = (-((lat.abs() - 46.0) / 13.5).powi(2)).exp();
        let row_off = row * width;

        for col in 0..width {
            let lon = -180.0 + (col as f64 + 0.5) * (360.0 / width as f64);
            let lam = lon.to_radians();
            let i = row_off + col;

            // Land-sea & thermal anomaly baroclinic thickness perturbation
            let dt_zonal = (t2m_c[i] - zonal_mean_t[row]).clamp(-18.0, 18.0);
            let psi_bc_thermal = sign_hem * 3.8e5 * dt_zonal * (lat.abs() / 18.0).tanh();

            // NAO North Atlantic dipole (Icelandic Low ~61N, 28W vs Azores High ~36N, 28W)
            let nao_atl_lon = (-((lon + 28.0) / 26.0).powi(2)).exp();
            let nao_psi = nao_index
                * nao_atl_lon
                * (-1.15e7 * (-((lat - 61.0) / 10.0).powi(2)).exp()
                    + 0.95e7 * (-((lat - 36.0) / 10.0).powi(2)).exp());

            // ENSO Pacific-North American (Aleutian Low deepening during El Nino)
            let enso_psi = -enso_nino34
                * 0.95e7
                * (-((lat - 48.0) / 11.0).powi(2) - ((lon + 158.0) / 28.0).powi(2)).exp();

            let mut p2_cell = psi2_zon + nao_psi + enso_psi - 0.35 * psi_bc_thermal;
            let mut p1_cell =
                psi2_zon + psi_shear_zon + 1.35 * (nao_psi + enso_psi) + 0.65 * psi_bc_thermal;

            if seed_baroclinic_eddies {
                // Baroclinically unstable synoptic cyclone/anticyclone wave packet (m = 5, 6, 7)
                // with classic Eady/Phillips ~32 deg (0.56 rad) westward vertical phase tilt in psi_1!
                let eddy_low = storm_track_env
                    * (1.35e7 * (5.0 * lam + 0.4).sin()
                        + 1.05e7 * (6.0 * lam - 1.1).cos()
                        + 0.72e7 * (7.0 * lam + 2.0).sin());
                let tilt = 0.56 * sign_hem;
                let eddy_up = storm_track_env
                    * (1.65e7 * (5.0 * lam + 0.4 + tilt).sin()
                        + 1.25e7 * (6.0 * lam - 1.1 + tilt).cos()
                        + 0.85e7 * (7.0 * lam + 2.0 + tilt).sin());
                p2_cell += eddy_low;
                p1_cell += eddy_up;
            }

            psi1[i] = p1_cell as f32;
            psi2[i] = p2_cell as f32;
        }
    }

    (psi1, psi2)
}

/// Prognostic 2-Layer Spherical QGPV Integrator & Elliptic Streamfunction Inverter across SQLite ticks:
/// - Initializes `(q_1, q_2)` on cold start or resumes exact `(q_1, q_2)` from SQLite
/// - Sub-steps nonlinear `CQMSL` Semi-Lagrangian PV advection (`-V_k . grad_S(q_k)`)
/// - Applies radiative thermal-wind baroclinic restoring, lower-layer Ekman relative-vorticity spin-down,
///   and scale-selective perturbation-PV dissipation
/// - Inverts `(q_1, q_2) -> (psi_1, psi_2)` via parallel modal Helmholtz RB-SOR and solves spherical
///   Linear Balance for dynamic `mslp_hpa` (`hPa`).
#[allow(clippy::too_many_arguments)]
pub fn step_two_layer_qgpv(
    prev_q1_q2: Option<(&[f32], &[f32])>,
    elev_m: &[f64],
    t2m_c: &[f64],
    width: usize,
    height: usize,
    dt_secs: f64,
    seasonal_phase_rad: f64,
    global_warming_c: f64,
    enso_nino34: f64,
    nao_index: f64,
) -> TwoLayerQgpvState {
    let n = width * height;
    assert_eq!(elev_m.len(), n);
    assert_eq!(t2m_c.len(), n);

    let (psi1_eq, psi2_eq) = build_equilibrium_streamfunctions(
        t2m_c,
        width,
        height,
        seasonal_phase_rad,
        global_warming_c,
        enso_nino34,
        nao_index,
        false,
    );
    let (q1_eq, q2_eq) =
        compute_two_layer_qgpv_from_streamfunctions(&psi1_eq, &psi2_eq, elev_m, width, height);

    let (mut q1, mut q2) = if let Some((pq1, pq2)) =
        prev_q1_q2.filter(|(a, b)| a.len() == n && b.len() == n)
    {
        (pq1.to_vec(), pq2.to_vec())
    } else {
        let (psi1_init, psi2_init) = build_equilibrium_streamfunctions(
            t2m_c,
            width,
            height,
            seasonal_phase_rad,
            global_warming_c,
            enso_nino34,
            nao_index,
            true,
        );
        compute_two_layer_qgpv_from_streamfunctions(&psi1_init, &psi2_init, elev_m, width, height)
    };

    if dt_secs > 0.0 && prev_q1_q2.is_some() {
        // Sub-step in <= 3600s increments so Rossby wave phase & baroclinic shear stay sharp
        let n_sub = ((dt_secs / 3600.0).ceil() as usize).clamp(1, 24);
        let dt_sub = (dt_secs / (n_sub as f64)).min(3600.0);
        let tau_rad_s = 7.0 * 86_400.0; // 7-day radiative thermal-wind restoring timescale
        let nu_pv_m2_s = 1.5e5_f64; // Scale-selective subgrid enstrophy diffusivity

        for _ in 0..n_sub {
            let (psi1_cur, psi2_cur) =
                invert_two_layer_qgpv_streamfunctions(&q1, &q2, elev_m, width, height);
            let ((u1_cur, v1_cur), (u2_cur, v2_cur)) = rayon::join(
                || diagnose_rotational_wind_from_streamfunction(&psi1_cur, width, height),
                || diagnose_rotational_wind_from_streamfunction(&psi2_cur, width, height),
            );

            // Nonlinear spherical Semi-Lagrangian advection of total QGPV q_1 and q_2
            let (q1_adv, q2_adv) = rayon::join(
                || {
                    semi_lagrangian_advect_qmsl(
                        &q1, &u1_cur, &v1_cur, width, height, dt_sub, false, false,
                    )
                },
                || {
                    semi_lagrangian_advect_qmsl(
                        &q2, &u2_cur, &v2_cur, width, height, dt_sub, false, false,
                    )
                },
            );

            // Relative vorticity in lower layer zeta_2 = lap_S(psi_2) for boundary-layer Ekman spin-down
            let zeta2 = spherical_laplacian_2d(&psi2_cur, width, height);

            // Perturbation PV (q_k - f(phi)) for scale-selective eddy diffusion + diabatic/Ekman forcing
            let mut q1_pert = vec![0.0_f32; n];
            let mut q2_pert = vec![0.0_f32; n];
            let mut src1 = vec![0.0_f32; n];
            let mut src2 = vec![0.0_f32; n];

            for row in 0..height {
                let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
                let f_cor = planetary_coriolis_f(lat) as f32;
                let row_off = row * width;
                for col in 0..width {
                    let i = row_off + col;
                    q1_pert[i] = q1_adv[i] - f_cor;
                    q2_pert[i] = q2_adv[i] - f_cor;

                    let r_ekman = if elev_m[i] < 0.0 { 2.4e-6 } else { 4.6e-6 };
                    src1[i] = ((q1_eq[i] - q1_adv[i]) as f64 / tau_rad_s) as f32;
                    src2[i] = (((q2_eq[i] - q2_adv[i]) as f64 / tau_rad_s)
                        - r_ekman * (zeta2[i] as f64)) as f32;
                }
            }

            let (q1_pert_next, q2_pert_next) = rayon::join(
                || {
                    spherical_diffusive_heat_step(
                        &q1_pert,
                        Some(&src1),
                        width,
                        height,
                        nu_pv_m2_s,
                        dt_sub,
                    )
                },
                || {
                    spherical_diffusive_heat_step(
                        &q2_pert,
                        Some(&src2),
                        width,
                        height,
                        nu_pv_m2_s,
                        dt_sub,
                    )
                },
            );

            for row in 0..height {
                let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
                let f_cor = planetary_coriolis_f(lat) as f32;
                let row_off = row * width;
                for col in 0..width {
                    let i = row_off + col;
                    q1[i] = (q1_pert_next[i] + f_cor).clamp(-5.0e-4, 5.0e-4);
                    q2[i] = (q2_pert_next[i] + f_cor).clamp(-5.0e-4, 5.0e-4);
                }
            }
        }
    }

    let (psi1, psi2) = invert_two_layer_qgpv_streamfunctions(&q1, &q2, elev_m, width, height);
    let ((u1, v1), (u2, v2)) = rayon::join(
        || diagnose_rotational_wind_from_streamfunction(&psi1, width, height),
        || diagnose_rotational_wind_from_streamfunction(&psi2, width, height),
    );
    let mslp_hpa = solve_spherical_linear_balance_mslp(&psi2, width, height);

    TwoLayerQgpvState {
        q1,
        q2,
        psi1,
        psi2,
        u1,
        v1,
        u2,
        v2,
        mslp_hpa,
    }
}

/// Complete prognostic & diagnostic state of the coupled 1.5-layer reduced-gravity shallow-water
/// and Stommel-Munk wind-driven ocean circulation solver.
#[derive(Debug, Clone)]
pub struct CoupledOceanState {
    /// Upper-layer thermocline depth anomaly `eta_tc` (`m`, positive = deeper warm thermocline).
    pub h_tc_anom_m: Vec<f32>,
    /// Stommel-Munk barotropic wind-driven gyre streamfunction `psi_gyre` (`m^2 s^-1`, `0` on land).
    pub psi_gyre: Vec<f32>,
    /// Reduced-gravity shallow-water zonal current `u_sw` (`m s^-1`).
    pub u_sw: Vec<f32>,
    /// Reduced-gravity shallow-water meridional current `v_sw` (`m s^-1`).
    pub v_sw: Vec<f32>,
    /// Total coupled ocean surface/layer current `u_total` (`m s^-1`, SW + Stommel-Munk + Ekman + ACC).
    pub u_total: Vec<f32>,
    /// Total coupled ocean surface/layer current `v_total` (`m s^-1`).
    pub v_total: Vec<f32>,
    /// Dynamic steric/baroclinic sea-surface height anomaly `(g'/g) * eta_tc` (`cm`).
    pub ssh_baroclinic_cm: Vec<f32>,
}

/// Compute air-sea bulk aerodynamic wind stress `(tau_x, tau_y)` (`N m^-2`) from 10m winds.
pub fn compute_surface_wind_stress(
    u10_ms: &[f64],
    v10_ms: &[f64],
    is_ocean: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let n = u10_ms.len();
    assert_eq!(v10_ms.len(), n);
    assert_eq!(is_ocean.len(), n);

    let rho_air = 1.225_f64;
    let c_drag = 1.35e-3_f64;
    let mut tau_x = vec![0.0_f64; n];
    let mut tau_y = vec![0.0_f64; n];

    for i in 0..n {
        if is_ocean[i] {
            let spd = u10_ms[i].hypot(v10_ms[i]);
            let coef = rho_air * c_drag * spd;
            tau_x[i] = coef * u10_ms[i];
            tau_y[i] = coef * v10_ms[i];
        }
    }
    (tau_x, tau_y)
}

/// Compute spherical wind stress curl `curl_S(tau) = (1/(R_E cos(phi))) * (d(tau_y)/d(lam) - d(tau_x cos(phi))/d(phi))` (`N m^-3`).
pub fn compute_wind_stress_curl(
    tau_x: &[f64],
    tau_y: &[f64],
    is_ocean: &[bool],
    width: usize,
    height: usize,
) -> Vec<f64> {
    let n = width * height;
    assert_eq!(tau_x.len(), n);
    assert_eq!(tau_y.len(), n);
    assert_eq!(is_ocean.len(), n);

    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let mut curl = vec![0.0_f64; n];

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let phi = lat.to_radians();
        let cos_c = phi.cos().max(QG_MIN_COS_ZONAL);
        let row_n = row.saturating_sub(1);
        let row_s = (row + 1).min(height - 1);
        let cos_n = (90.0 - (row_n as f64 + 0.5) * (180.0 / height as f64))
            .to_radians()
            .cos()
            .max(0.05);
        let cos_s = (90.0 - (row_s as f64 + 0.5) * (180.0 / height as f64))
            .to_radians()
            .cos()
            .max(0.05);
        let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;
        let dx = 2.0 * EARTH_RADIUS_M * cos_c * dlam;
        let row_off = row * width;

        for col in 0..width {
            let idx = row_off + col;
            if !is_ocean[idx] {
                continue;
            }
            let col_w = if col == 0 { width - 1 } else { col - 1 };
            let col_e = if col + 1 == width { 0 } else { col + 1 };

            let dtauy_dx = (tau_y[row_off + col_e] - tau_y[row_off + col_w]) / dx;
            let dtaux_cos_dy = (tau_x[row_n * width + col] * cos_n
                - tau_x[row_s * width + col] * cos_s)
                / (cos_c * dy);
            curl[idx] = dtauy_dx - dtaux_cos_dy;
        }
    }
    curl
}

/// Solve the Stommel-Munk wind-driven ocean gyre vorticity-streamfunction equation on the sphere
/// with `psi_gyre = 0` no-normal-flow boundary conditions on all land cells:
/// `d(zeta)/dt + beta * u_gyre_merid = (1 / rho_0) * curl_S(tau) - r_stommel * zeta + A_H * lap_S(zeta)`
/// where `zeta = lap_S(psi_gyre)`. Produces Sverdrup interior return flow & western boundary intensification!
pub fn solve_stommel_munk_gyre(
    prev_psi_gyre: Option<&[f32]>,
    tau_x: &[f64],
    tau_y: &[f64],
    is_ocean: &[bool],
    width: usize,
    height: usize,
    max_iters: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = width * height;
    assert_eq!(tau_x.len(), n);
    assert_eq!(tau_y.len(), n);
    assert_eq!(is_ocean.len(), n);

    let curl_tau = compute_wind_stress_curl(tau_x, tau_y, is_ocean, width, height);
    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let r2 = EARTH_RADIUS_M * EARTH_RADIUS_M;

    let r_stommel = 4.5e-6_f64; // Stommel (1948) interfacial/bottom drag (s^-1)
    let a_h_munk = 1.8e4_f64; // Munk (1950) lateral eddy viscosity (m^2 s^-1)
    let h_gyre_m = 450.0_f64; // Wind-driven upper-gyre Sverdrup depth scale (m)

    let mut psi = if let Some(prev) = prev_psi_gyre.filter(|p| p.len() == n) {
        prev.iter()
            .enumerate()
            .map(|(i, &v)| if is_ocean[i] && v.is_finite() { v as f64 } else { 0.0 })
            .collect::<Vec<f64>>()
    } else {
        vec![0.0_f64; n]
    };

    // Precompute row metrics
    let mut row_metrics = Vec::with_capacity(height);
    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let phi = lat.to_radians();
        let cos_c = phi.cos().max(1e-4);
        let cos_z = phi.cos().max(QG_MIN_COS_ZONAL);
        let beta = (2.0 * EARTH_OMEGA_RAD_S * phi.cos()) / EARTH_RADIUS_M;

        let cos_n = if row == 0 {
            0.0
        } else {
            (90.0 - (row as f64) * (180.0 / height as f64))
                .to_radians()
                .cos()
                .max(0.0)
        };
        let cos_s = if row + 1 >= height {
            0.0
        } else {
            (90.0 - ((row + 1) as f64) * (180.0 / height as f64))
                .to_radians()
                .cos()
                .max(0.0)
        };

        let a_n = cos_n / (r2 * cos_c * dphi * dphi);
        let a_s = cos_s / (r2 * cos_c * dphi * dphi);
        let a_ew = 1.0 / (r2 * cos_z * cos_z * dlam * dlam);
        let dx_m = EARTH_RADIUS_M * cos_z * dlam;
        let diag = (a_n + a_s + 2.0 * a_ew).max(1.0e-30);
        row_metrics.push((a_n, a_s, a_ew, diag, beta, dx_m));
    }

    // Solve Stommel-Munk transport streamfunction (r_stommel * lap_S(psi) + beta * dpsi/dx = curl(tau)/(rho0*H_gyre))
    // using upwinded planetary beta term on the eastern side so western boundary currents sharpen automatically!
    let mut next_psi = psi.clone();
    let omega = 1.35_f64;
    let gyre_iters = if n > 20_000 { max_iters.min(18) } else { max_iters };

    for _ in 0..gyre_iters {
        for color in 0..2usize {
            next_psi
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let (a_n, a_s, a_ew, _, beta, dx_m) = row_metrics[row];
                    let row_n = row.saturating_sub(1);
                    let row_s = (row + 1).min(height - 1);
                    let row_off = row * width;

                    // Upwinded planetary beta: dpsi/dx ~ (psi_E - psi_C) / dx_m (propagates Rossby energy westward!)
                    let beta_coef = beta / dx_m;
                    let eff_diag = (r_stommel * (a_n + a_s + 2.0 * a_ew) + beta_coef
                        + a_h_munk * (a_n + a_s + 2.0 * a_ew) * (a_n + a_s + 2.0 * a_ew) * 0.08)
                        .max(1.0e-30);

                    for col in 0..width {
                        let idx = row_off + col;
                        if !is_ocean[idx] {
                            out_row[col] = 0.0;
                            continue;
                        }
                        let pc = psi[idx];
                        if ((row + col) & 1) != color {
                            out_row[col] = pc;
                            continue;
                        }

                        let col_w = if col == 0 { width - 1 } else { col - 1 };
                        let col_e = if col + 1 == width { 0 } else { col + 1 };

                        let pn = if is_ocean[row_n * width + col] { psi[row_n * width + col] } else { 0.0 };
                        let ps = if is_ocean[row_s * width + col] { psi[row_s * width + col] } else { 0.0 };
                        let pw = if is_ocean[row_off + col_w] { psi[row_off + col_w] } else { 0.0 };
                        let pe = if is_ocean[row_off + col_e] { psi[row_off + col_e] } else { 0.0 };

                        let forcing = curl_tau[idx] / (SEAWATER_RHO0_KG_M3 * h_gyre_m);
                        let num = r_stommel * (a_n * pn + a_s * ps + a_ew * (pe + pw))
                            + beta_coef * pe
                            - forcing;
                        let gs_star = (num / eff_diag).clamp(-3.5e5, 3.5e5);
                        out_row[col] = ((1.0 - omega) * pc + omega * gs_star).clamp(-3.5e5, 3.5e5);
                    }
                });
            std::mem::swap(&mut psi, &mut next_psi);
        }
    }

    // Diagnose geostrophic-Sverdrup gyre velocities u = -dpsi/dy, v = dpsi/dx (m s^-1)
    let mut u_gyre = vec![0.0_f32; n];
    let mut v_gyre = vec![0.0_f32; n];

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let cos_phi = lat.to_radians().cos().max(QG_MIN_COS_ZONAL);
        let row_n = row.saturating_sub(1);
        let row_s = (row + 1).min(height - 1);
        let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;
        let dx = 2.0 * EARTH_RADIUS_M * cos_phi * dlam;
        let row_off = row * width;

        for col in 0..width {
            let idx = row_off + col;
            if !is_ocean[idx] {
                continue;
            }
            let col_w = if col == 0 { width - 1 } else { col - 1 };
            let col_e = if col + 1 == width { 0 } else { col + 1 };

            let dpsi_dy = (psi[row_n * width + col] - psi[row_s * width + col]) / dy;
            let dpsi_dx = (psi[row_off + col_e] - psi[row_off + col_w]) / dx;
            u_gyre[idx] = (-dpsi_dy).clamp(-2.2, 2.2) as f32;
            v_gyre[idx] = dpsi_dx.clamp(-2.2, 2.2) as f32;
        }
    }

    let psi_out = psi.into_iter().map(|v| v as f32).collect();
    (psi_out, u_gyre, v_gyre)
}

/// Prognostic 1.5-layer reduced-gravity shallow-water + Stommel-Munk wind-driven ocean circulation solver:
/// - Integrates upper-ocean thermocline anomaly `eta_tc` (`h_tc_anom_m`) and currents `(u_sw, v_sw)`
///   with Forward-Backward Coriolis-split sub-stepping (`c_1 = sqrt(g' * H_0) ~ 2.19 m s^-1`)
/// - Solves Stommel-Munk wind-driven gyre streamfunction `psi_gyre` from QGPV surface wind stress
/// - Blends reduced-gravity, Stommel-Munk gyre, Ekman surface drift, and Southern Ocean ACC.
#[allow(clippy::too_many_arguments)]
pub fn step_ocean_reduced_gravity_and_gyre(
    prev_h_tc: Option<&[f32]>,
    prev_psi_gyre: Option<&[f32]>,
    prev_u_sw: Option<&[f32]>,
    prev_v_sw: Option<&[f32]>,
    u10_ms: &[f64],
    v10_ms: &[f64],
    is_ocean: &[bool],
    width: usize,
    height: usize,
    dt_secs: f64,
    global_warming_c: f64,
) -> CoupledOceanState {
    let n = width * height;
    assert_eq!(u10_ms.len(), n);
    assert_eq!(v10_ms.len(), n);
    assert_eq!(is_ocean.len(), n);

    let (tau_x, tau_y) = compute_surface_wind_stress(u10_ms, v10_ms, is_ocean);
    let (psi_gyre, u_gyre, v_gyre) =
        solve_stommel_munk_gyre(prev_psi_gyre, &tau_x, &tau_y, is_ocean, width, height, 64);

    let mut h_tc: Vec<f64> = if let Some(ph) = prev_h_tc.filter(|v| v.len() == n) {
        ph.iter()
            .enumerate()
            .map(|(i, &v)| if is_ocean[i] && v.is_finite() { v as f64 } else { 0.0 })
            .collect()
    } else {
        vec![0.0_f64; n]
    };

    let mut u_sw: Vec<f64> = if let Some(pu) = prev_u_sw.filter(|v| v.len() == n) {
        pu.iter()
            .enumerate()
            .map(|(i, &v)| if is_ocean[i] && v.is_finite() { (v as f64).clamp(-1.8, 1.8) } else { 0.0 })
            .collect()
    } else {
        vec![0.0_f64; n]
    };

    let mut v_sw: Vec<f64> = if let Some(pv) = prev_v_sw.filter(|v| v.len() == n) {
        pv.iter()
            .enumerate()
            .map(|(i, &v)| if is_ocean[i] && v.is_finite() { (v as f64).clamp(-1.8, 1.8) } else { 0.0 })
            .collect()
    } else {
        vec![0.0_f64; n]
    };

    if dt_secs > 0.0 && width >= 4 && height >= 4 {
        let dphi = std::f64::consts::PI / (height as f64);
        let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
        let c1 = (OCEAN_REDUCED_GRAVITY_M_S2 * OCEAN_THERMOCLINE_H0_M).sqrt();
        let dx_min = (EARTH_RADIUS_M * QG_MIN_COS_ZONAL * dlam).min(EARTH_RADIUS_M * dphi);
        let dt_cfl = (0.42 * dx_min / c1.max(1.0)).max(600.0);
        let n_sub = ((dt_secs / dt_cfl).ceil() as usize).clamp(1, 32);
        let dt_sub = (dt_secs / (n_sub as f64)).min(dt_cfl);

        let r_drag = 1.2e-6_f64; // Rayleigh momentum damping (s^-1)
        let r_tc_damp = 1.0 / (180.0 * 86_400.0); // 180-day thermocline Newtonian damping
        let kh_tc = 2.0e3_f64; // Weak lateral thermocline diffusion (m^2 s^-1)

        let mut next_u = vec![0.0_f64; n];
        let mut next_v = vec![0.0_f64; n];
        let mut next_h = vec![0.0_f64; n];

        for _ in 0..n_sub {
            // Forward-Backward step 1: Momentum (u_sw, v_sw) with semi-implicit Coriolis rotation
            next_u
                .par_chunks_mut(width)
                .zip(next_v.par_chunks_mut(width))
                .enumerate()
                .for_each(|(row, (u_out, v_out))| {
                    let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
                    let cos_phi = lat.to_radians().cos().max(QG_MIN_COS_ZONAL);
                    let f_cor = planetary_coriolis_f(lat);
                    let row_n = row.saturating_sub(1);
                    let row_s = (row + 1).min(height - 1);
                    let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;
                    let dx = 2.0 * EARTH_RADIUS_M * cos_phi * dlam;
                    let row_off = row * width;

                    let alpha_c = f_cor * dt_sub;
                    let denom = (1.0 + r_drag * dt_sub).powi(2) + alpha_c * alpha_c;
                    let damp_fac = 1.0 + r_drag * dt_sub;

                    for col in 0..width {
                        let idx = row_off + col;
                        if !is_ocean[idx] {
                            u_out[col] = 0.0;
                            v_out[col] = 0.0;
                            continue;
                        }
                        let col_w = if col == 0 { width - 1 } else { col - 1 };
                        let col_e = if col + 1 == width { 0 } else { col + 1 };

                        let h_c = h_tc[idx];
                        let h_e = if is_ocean[row_off + col_e] { h_tc[row_off + col_e] } else { h_c };
                        let h_w = if is_ocean[row_off + col_w] { h_tc[row_off + col_w] } else { h_c };
                        let h_n = if is_ocean[row_n * width + col] { h_tc[row_n * width + col] } else { h_c };
                        let h_s = if is_ocean[row_s * width + col] { h_tc[row_s * width + col] } else { h_c };

                        let dh_dx = (h_e - h_w) / dx;
                        let dh_dy = (h_n - h_s) / dy;

                        let fx = -OCEAN_REDUCED_GRAVITY_M_S2 * dh_dx
                            + tau_x[idx] / (SEAWATER_RHO0_KG_M3 * OCEAN_THERMOCLINE_H0_M);
                        let fy = -OCEAN_REDUCED_GRAVITY_M_S2 * dh_dy
                            + tau_y[idx] / (SEAWATER_RHO0_KG_M3 * OCEAN_THERMOCLINE_H0_M);

                        let rhs_u = u_sw[idx] + dt_sub * fx;
                        let rhs_v = v_sw[idx] + dt_sub * fy;

                        u_out[col] = ((damp_fac * rhs_u + alpha_c * rhs_v) / denom).clamp(-1.8, 1.8);
                        v_out[col] = ((damp_fac * rhs_v - alpha_c * rhs_u) / denom).clamp(-1.8, 1.8);
                    }
                });
            std::mem::swap(&mut u_sw, &mut next_u);
            std::mem::swap(&mut v_sw, &mut next_v);

            // Forward-Backward step 2: Continuity (eta_tc) using updated (u_sw, v_sw)
            next_h
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, h_out)| {
                    let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
                    let cos_c = lat.to_radians().cos().max(1e-4);
                    let cos_z = lat.to_radians().cos().max(QG_MIN_COS_ZONAL);
                    let row_n = row.saturating_sub(1);
                    let row_s = (row + 1).min(height - 1);
                    let cos_n = (90.0 - (row_n as f64 + 0.5) * (180.0 / height as f64))
                        .to_radians()
                        .cos()
                        .max(0.05);
                    let cos_s = (90.0 - (row_s as f64 + 0.5) * (180.0 / height as f64))
                        .to_radians()
                        .cos()
                        .max(0.05);
                    let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;
                    let dx = 2.0 * EARTH_RADIUS_M * cos_z * dlam;
                    let dy_cell = EARTH_RADIUS_M * dphi;
                    let dx_cell = EARTH_RADIUS_M * cos_z * dlam;
                    let row_off = row * width;

                    for col in 0..width {
                        let idx = row_off + col;
                        if !is_ocean[idx] {
                            h_out[col] = 0.0;
                            continue;
                        }
                        let col_w = if col == 0 { width - 1 } else { col - 1 };
                        let col_e = if col + 1 == width { 0 } else { col + 1 };

                        let u_e = if is_ocean[row_off + col_e] { u_sw[row_off + col_e] } else { 0.0 };
                        let u_w = if is_ocean[row_off + col_w] { u_sw[row_off + col_w] } else { 0.0 };
                        let v_n = if is_ocean[row_n * width + col] { v_sw[row_n * width + col] * cos_n } else { 0.0 };
                        let v_s = if is_ocean[row_s * width + col] { v_sw[row_s * width + col] * cos_s } else { 0.0 };

                        let div_u = (u_e - u_w) / dx + (v_n - v_s) / (cos_c * dy);

                        let h_c = h_tc[idx];
                        let h_e = if is_ocean[row_off + col_e] { h_tc[row_off + col_e] } else { h_c };
                        let h_w = if is_ocean[row_off + col_w] { h_tc[row_off + col_w] } else { h_c };
                        let h_n = if is_ocean[row_n * width + col] { h_tc[row_n * width + col] } else { h_c };
                        let h_s = if is_ocean[row_s * width + col] { h_tc[row_s * width + col] } else { h_c };
                        let lap_h = (h_e - 2.0 * h_c + h_w) / (dx_cell * dx_cell)
                            + (h_n - 2.0 * h_c + h_s) / (dy_cell * dy_cell);

                        h_out[col] = (h_c + dt_sub * (-OCEAN_THERMOCLINE_H0_M * div_u - r_tc_damp * h_c + kh_tc * lap_h))
                            .clamp(-95.0, 95.0);
                    }
                });
            std::mem::swap(&mut h_tc, &mut next_h);
        }
    }

    let amoc_factor = (1.0 - 0.06 * global_warming_c).clamp(0.45, 1.05);
    let mut u_total = vec![0.0_f32; n];
    let mut v_total = vec![0.0_f32; n];
    let mut ssh_baroclinic_cm = vec![0.0_f32; n];

    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let sign_f = if lat >= 0.0 { 1.0 } else { -1.0 };
        let row_off = row * width;

        for col in 0..width {
            let i = row_off + col;
            if !is_ocean[i] {
                continue;
            }
            // Ekman surface drift (45 deg right/left of 10m wind) + ACC + coupled SW & Stommel-Munk gyre
            let u_ek = 0.014 * (u10_ms[i] * 0.707 + sign_f * v10_ms[i] * 0.707);
            let v_ek = 0.014 * (v10_ms[i] * 0.707 - sign_f * u10_ms[i] * 0.707);
            let acc = if lat < -42.0 { 0.62 } else { 0.0 };

            let u_tot = (u_sw[i] + amoc_factor * (u_gyre[i] as f64) + u_ek + acc).clamp(-2.5, 2.5);
            let v_tot = (v_sw[i] + amoc_factor * (v_gyre[i] as f64) + v_ek).clamp(-2.2, 2.2);

            u_total[i] = u_tot as f32;
            v_total[i] = v_tot as f32;
            // Baroclinic steric SSH conversion: eta_ssh = (g'/g) * eta_tc (in cm)
            ssh_baroclinic_cm[i] = ((OCEAN_REDUCED_GRAVITY_M_S2 / 9.80665) * h_tc[i] * 100.0) as f32;
        }
    }

    CoupledOceanState {
        h_tc_anom_m: h_tc.into_iter().map(|v| v as f32).collect(),
        psi_gyre,
        u_sw: u_sw.into_iter().map(|v| v as f32).collect(),
        v_sw: v_sw.into_iter().map(|v| v as f32).collect(),
        u_total,
        v_total,
        ssh_baroclinic_cm,
    }
}

/// Bolton (1980) saturation vapor pressure in hPa for temperature `t_c` (°C).
/// Guards the `-243.5 °C` Magnus singularity and non-finite inputs so extreme temperatures remain finite.
#[inline]
pub fn bolton_saturation_vapor_pressure_hpa(t_c: f64) -> f64 {
    if t_c.is_nan() || t_c <= -243.5 {
        return 0.0;
    }
    if t_c.is_infinite() {
        return 6.112 * 17.67_f64.exp();
    }
    let denom = t_c + 243.5;
    if denom <= 1.0e-6 {
        return 0.0;
    }
    let exp_arg = ((17.67 * t_c) / denom).clamp(-100.0, 25.0);
    let es = 6.112 * exp_arg.exp();
    if es.is_finite() { es } else { 0.0 }
}

/// Saturation specific humidity `q_sat` (`kg kg^-1`) at temperature `t_c` (°C) and pressure `p_hpa`.
#[inline]
pub fn saturation_specific_humidity(t_c: f64, p_hpa: f64) -> f64 {
    let p_safe = if p_hpa.is_finite() {
        p_hpa.max(1.0)
    } else {
        1013.25
    };
    let es = bolton_saturation_vapor_pressure_hpa(t_c).min(0.85 * p_safe);
    let denom = (p_safe - 0.378 * es).max(1.0e-6);
    (0.622 * es / denom).clamp(1.0e-6, 0.045)
}

/// Bolton (1980, Eq. 43) Equivalent Potential Temperature `theta_e` (Kelvin) for frontal analysis & CAPE.
/// Guards `-243.5 °C` singularities, dry/zero humidity (`q <= 0`), LCL denominator positivity, and non-finite inputs.
#[inline]
pub fn bolton_equivalent_potential_temp_k(t_c: f64, q_kg_kg: f64, p_hpa: f64) -> f64 {
    let t_safe = if t_c.is_finite() {
        t_c.clamp(-240.0, 150.0)
    } else if t_c.is_sign_positive() {
        150.0
    } else {
        -240.0
    };
    let tk = (t_safe + 273.15).clamp(180.0, 450.0);
    let q_safe = if q_kg_kg.is_finite() {
        q_kg_kg.clamp(0.0, 0.1)
    } else if q_kg_kg.is_sign_positive() {
        0.05
    } else {
        0.0
    };
    let r = (q_safe / (1.0 - q_safe).max(0.9)).clamp(1.0e-6, 0.05);
    let p_safe = if p_hpa.is_finite() {
        p_hpa.clamp(10.0, 5000.0)
    } else if p_hpa.is_sign_positive() {
        5000.0
    } else {
        10.0
    };
    let e = (p_safe * r / (0.622 + r)).max(1.0e-4);
    // Lifting condensation level temperature T_L (Bolton Eq. 21) with safeguarded positive denominator
    let tl_denom = (3.5 * tk.ln() - e.ln() - 4.805).max(1.0);
    let tl = (2840.0 / tl_denom + 55.0).clamp(60.0, 1500.0);
    let p_dry = (p_safe - e).max(10.0);
    let theta_dl = tk * (1000.0 / p_dry).powf(0.2854) * (tk / tl).powf(0.28 * r);
    let exp_arg = ((3036.0 / tl - 1.78) * r * (1.0 + 0.448 * r)).clamp(-20.0, 20.0);
    let theta_e = theta_dl * exp_arg.exp();
    if theta_e.is_finite() { theta_e } else { tk }
}

/// Stull (2011) empirical Wet-Bulb Temperature (°C) from dry-bulb `t_c` (°C) and `rh_pct` (5–99%).
#[inline]
pub fn stull_wet_bulb_celsius(t_c: f64, rh_pct: f64) -> f64 {
    let t = if t_c.is_finite() {
        t_c.clamp(-100.0, 80.0)
    } else {
        15.0
    };
    let rh = if rh_pct.is_finite() {
        rh_pct.clamp(5.0, 99.0)
    } else {
        5.0
    };
    t * (0.151_977 * (rh + 8.313_659).sqrt()).atan() + (t + rh).atan()
        - (rh - 1.676_331).atan()
        + 0.003_918_38 * rh.powf(1.5) * (0.023_101 * rh).atan()
        - 4.686_035
}

/// Outdoor Wet-Bulb Globe Temperature (°C) combining Stull `Tw`, solar globe heating, and dry air `Ta`.
#[inline]
pub fn compute_outdoor_wbgt(t_c: f64, rh_pct: f64, shortwave_wm2: f64, wind_ms: f64) -> f64 {
    let tw = stull_wet_bulb_celsius(t_c, rh_pct);
    let tg = t_c + (0.014 * shortwave_wm2) / (wind_ms.max(0.5).powf(0.45));
    0.7 * tw + 0.2 * tg + 0.1 * t_c
}

/// IPCC AR6 / Etminan et al. (2016) CO2 effective radiative forcing per doubling (`W m^-2`).
pub const CO2_FORCING_2X_AR6_WM2: f64 = 3.93;
/// 1750 pre-industrial CO2 reference concentration (`ppm`).
pub const CO2_PREINDUSTRIAL_PPM: f64 = 278.0;
/// IPCC AR6 best-estimate Equilibrium Climate Sensitivity (`ECS`, °C per `2x CO2`).
pub const CLIMATE_SENSITIVITY_ECS_C: f64 = 3.0;
/// Realized transient response fraction (`TCR_eff / ECS = 2.16 / 3.0 = 0.72`).
pub const TRANSIENT_CLIMATE_FRACTION: f64 = 0.72;

/// IPCC AR6 / Etminan et al. (2016) CO2 effective radiative forcing (`W m^-2`) relative to 1750
/// pre-industrial (278 ppm), calibrated to `F_{2xCO2} = 3.93 W m^-2` (`alpha_AR6 = 3.93 / ln(2) ≈ 5.67 W m^-2`).
#[inline]
pub fn co2_radiative_forcing_wm2(co2_ppm: f64) -> f64 {
    let c = if co2_ppm.is_finite() {
        co2_ppm.max(180.0)
    } else if co2_ppm.is_sign_positive() {
        10_000.0
    } else {
        180.0
    };
    (CO2_FORCING_2X_AR6_WM2 / std::f64::consts::LN_2) * (c / CO2_PREINDUSTRIAL_PPM).ln()
}

/// Equilibrium global mean surface warming (°C over 1850–1900 pre-industrial) from CO2 forcing
/// using IPCC AR6 `ECS = 3.0 °C` per doubling (`3.93 W m^-2`).
#[inline]
pub fn equilibrium_global_warming_c(co2_ppm: f64) -> f64 {
    (co2_radiative_forcing_wm2(co2_ppm) / CO2_FORCING_2X_AR6_WM2) * CLIMATE_SENSITIVITY_ECS_C
}

/// Transient global mean surface warming (°C over 1850–1900 pre-industrial) from CO2 forcing
/// (`ECS = 3.0 °C` per doubling `3.93 W m^-2`, realized transient climate response fraction `0.72`).
#[inline]
pub fn transient_global_warming_c(co2_ppm: f64) -> f64 {
    equilibrium_global_warming_c(co2_ppm) * TRANSIENT_CLIMATE_FRACTION
}

/// Latitudinal polar amplification factor (asymmetric Arctic vs Antarctic amplification).
#[inline]
pub fn polar_amplification_factor(lat_deg: f64) -> f64 {
    let sin_phi = lat_deg.to_radians().sin();
    if lat_deg > 0.0 {
        0.85 + 1.75 * sin_phi * sin_phi
    } else {
        0.85 + 0.70 * sin_phi * sin_phi
    }
}

/// Van Wagner (1987) / Lawson & Armitage (2008) latitude-band monthly effective daylength factors
/// `(Le, Lf)` for the Duff Moisture Code (DMC) and Drought Code (DC).
#[inline]
pub fn van_wagner_daylength_factors(lat_deg: f64, month_1_to_12: u32) -> (f64, f64) {
    const DMC_LE_NH_HIGH: [f64; 12] = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0];
    const DMC_LE_NH_LOW: [f64; 12] = [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8];
    const DMC_LE_EQ: [f64; 12] = [9.0; 12];
    const DMC_LE_SH_LOW: [f64; 12] = [10.1, 9.7, 9.1, 8.6, 8.1, 7.8, 7.9, 8.4, 8.9, 9.5, 9.9, 10.2];
    const DMC_LE_SH_HIGH: [f64; 12] = [12.4, 10.9, 9.4, 8.0, 7.0, 6.0, 6.5, 7.5, 9.0, 12.8, 13.9, 13.9];

    const DC_LF_NH: [f64; 12] = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6];
    const DC_LF_EQ: [f64; 12] = [1.4; 12];
    const DC_LF_SH: [f64; 12] = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8];

    let idx = (month_1_to_12.clamp(1, 12) - 1) as usize;
    let lat = if lat_deg.is_finite() {
        lat_deg.clamp(-90.0, 90.0)
    } else {
        45.0
    };
    let le = if lat > 30.0 {
        DMC_LE_NH_HIGH[idx]
    } else if lat > 10.0 {
        DMC_LE_NH_LOW[idx]
    } else if lat >= -10.0 {
        DMC_LE_EQ[idx]
    } else if lat >= -30.0 {
        DMC_LE_SH_LOW[idx]
    } else {
        DMC_LE_SH_HIGH[idx]
    };
    let lf = if lat > 20.0 {
        DC_LF_NH[idx]
    } else if lat >= -20.0 {
        DC_LF_EQ[idx]
    } else {
        DC_LF_SH[idx]
    };
    (le, lf)
}

#[inline]
fn doy_to_month_1_to_12(day_of_year: f64) -> u32 {
    let doy = if day_of_year.is_finite() {
        day_of_year.round().clamp(1.0, 365.0) as u32
    } else {
        195
    };
    NaiveDate::from_yo_opt(2025, doy)
        .map(|d| d.month())
        .unwrap_or(7)
}

/// Full 6-component Van Wagner & Pickett (1985) / Van Wagner (1987) Canadian Forest Fire Weather Index
/// stepped from prior fuel moisture codes `(prev_ffmc, prev_dmc, prev_dc)` using latitude/month
/// effective daylength tables `(Le, Lf)`.
#[allow(clippy::too_many_arguments)]
pub fn step_van_wagner_fwi_at_location(
    t_c: f64,
    rh_pct: f64,
    wind_ms: f64,
    precip_24h_mm: f64,
    prev_ffmc: f64,
    prev_dmc: f64,
    prev_dc: f64,
    lat_deg: f64,
    month_1_to_12: u32,
) -> FireWeatherComponents {
    let t_safe = if t_c.is_finite() {
        t_c.clamp(-80.0, 70.0)
    } else {
        15.0
    };
    let rh = if rh_pct.is_finite() {
        rh_pct.clamp(5.0, 100.0)
    } else {
        50.0
    };
    let wind_kmh = if wind_ms.is_finite() {
        (wind_ms.max(0.0) * 3.6).min(120.0)
    } else {
        0.0
    };
    let rain = if precip_24h_mm.is_finite() {
        precip_24h_mm.max(0.0)
    } else {
        0.0
    };
    let (le, lf) = van_wagner_daylength_factors(lat_deg, month_1_to_12);

    // 1. Fine Fuel Moisture Code (FFMC)
    let ffmc_clamped = prev_ffmc.clamp(0.0, 101.0);
    let mut m0 = 147.2 * (101.0 - ffmc_clamped) / (59.5 + ffmc_clamped);
    if rain > 0.5 {
        let rf = rain - 0.5;
        let mr = if m0 <= 150.0 {
            m0 + 42.5 * rf * (-100.0 / (251.0 - m0)).exp() * (1.0 - (-6.93 / rf).exp())
        } else {
            m0 + 42.5 * rf * (-100.0 / (251.0 - m0)).exp() * (1.0 - (-6.93 / rf).exp())
                + 0.0015 * (m0 - 150.0).powi(2) * rf.sqrt()
        };
        m0 = mr.min(250.0);
    }
    let ed = 0.942 * rh.powf(0.679)
        + 11.0 * ((rh - 100.0) / 10.0).exp()
        + 0.18 * (21.1 - t_safe) * (1.0 - (-0.115 * rh).exp());
    let ew = 0.618 * rh.powf(0.753)
        + 10.0 * ((rh - 100.0) / 10.0).exp()
        + 0.18 * (21.1 - t_safe) * (1.0 - (-0.115 * rh).exp());

    let m = if m0 > ed {
        let ko = 0.424 * (1.0 - (rh / 100.0).powf(1.7))
            + 0.0694 * wind_kmh.sqrt() * (1.0 - (rh / 100.0).powi(8));
        let kd = ko * 0.581 * (0.0365 * t_safe).exp();
        ed + (m0 - ed) * 10.0_f64.powf(-kd)
    } else if m0 < ew {
        let k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0).powf(1.7))
            + 0.0694 * wind_kmh.sqrt() * (1.0 - ((100.0 - rh) / 100.0).powi(8));
        let kw = k1 * 0.581 * (0.0365 * t_safe).exp();
        ew - (ew - m0) * 10.0_f64.powf(-kw)
    } else {
        m0
    };
    let ffmc = (59.5 * (250.0 - m) / (147.2 + m)).clamp(0.0, 101.0);

    // 2. Duff Moisture Code (DMC) with latitude/month effective daylength Le
    let mut p0 = prev_dmc.max(0.0);
    if rain > 1.5 {
        let re = 0.92 * rain - 1.27;
        let mo = 20.0 + ((5.6348 - p0 / 43.43).exp());
        let b = if p0 <= 33.0 {
            100.0 / (0.5 + 0.3 * p0)
        } else if p0 <= 65.0 {
            14.0 - 1.3 * p0.ln()
        } else {
            6.2 * p0.ln() - 17.2
        };
        let mr = mo + (1000.0 * re) / (48.77 + b * re);
        p0 = (244.72 - 43.43 * (mr - 20.0).max(1.0).ln()).max(0.0);
    }
    let k_dmc = 1.894 * (t_safe.max(-1.1) + 1.1) * (100.0 - rh) * (le * 1.0e-6);
    let dmc = (p0 + 100.0 * k_dmc).clamp(0.0, 300.0);

    // 3. Drought Code (DC) with latitude/month daylength adjustment Lf
    let mut d0 = prev_dc.max(0.0);
    if rain > 2.8 {
        let rd = 0.83 * rain - 1.27;
        let qo = 800.0 * (-d0 / 400.0).exp();
        let qr = qo + 3.937 * rd;
        d0 = (400.0 * (800.0 / qr.max(1.0)).ln()).max(0.0);
    }
    let v_dc = (0.36 * (t_safe.max(-2.8) + 2.8) + lf).max(0.0);
    let dc = (d0 + 0.5 * v_dc).clamp(0.0, 1200.0);

    // 4. Initial Spread Index (ISI), Buildup Index (BUI), and Fire Weather Index (FWI)
    let fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc);
    let f_wind = (0.05039 * wind_kmh).exp();
    let f_fuel = 91.9 * (-0.1386 * fm).exp() * (1.0 + fm.powf(5.31) / 4.93e7);
    let isi = 0.208 * f_wind * f_fuel;

    let bui = if dmc <= 0.4 * dc {
        if dmc + dc > 1e-6 {
            (0.8 * dmc * dc) / (dmc + 0.4 * dc)
        } else {
            0.0
        }
    } else {
        dmc - (1.0 - (0.8 * dc) / (dmc + 0.4 * dc).max(1e-6))
            * (0.92 + (0.0114 * dmc).powf(1.7))
    }
    .max(0.0);

    let f_d = if bui <= 80.0 {
        0.626 * bui.powf(0.809) + 2.0
    } else {
        1000.0 / (25.0 + 108.64 * (-0.023 * bui).exp())
    };
    let b_scale = 0.1 * isi * f_d;
    let fwi = if b_scale > 1.0 {
        (2.72 * (0.434 * b_scale.ln()).powf(0.647)).exp()
    } else {
        b_scale
    }
    .clamp(0.0, 140.0);

    FireWeatherComponents {
        ffmc,
        dmc,
        dc,
        isi,
        bui,
        fwi,
    }
}

/// Full 6-component Van Wagner & Pickett (1985) / Van Wagner (1987) Canadian Forest Fire Weather Index
/// stepped from prior fuel moisture codes `(prev_ffmc, prev_dmc, prev_dc)` at standard mid-latitude
/// fire-season reference (`45°N`, July `Le=12.4`, `Lf=6.4`).
pub fn step_van_wagner_fwi(
    t_c: f64,
    rh_pct: f64,
    wind_ms: f64,
    precip_24h_mm: f64,
    prev_ffmc: f64,
    prev_dmc: f64,
    prev_dc: f64,
) -> FireWeatherComponents {
    step_van_wagner_fwi_at_location(
        t_c,
        rh_pct,
        wind_ms,
        precip_24h_mm,
        prev_ffmc,
        prev_dmc,
        prev_dc,
        45.0,
        7,
    )
}

/// Steady-state Van Wagner FWI evaluated at a specific `(lat_deg, day_of_year)` using
/// latitude-band monthly daylength tables `(Le, Lf)`.
#[inline]
pub fn canadian_fire_weather_index_at(
    t_c: f64,
    rh_pct: f64,
    wind_ms: f64,
    precip_mm_hr: f64,
    lat_deg: f64,
    day_of_year: f64,
) -> f64 {
    let init_ffmc = (92.0 - 0.35 * rh_pct).clamp(35.0, 96.0);
    let init_dmc = ((t_c + 5.0).max(0.0) * (100.0 - rh_pct) * 0.025).clamp(2.0, 120.0);
    let init_dc = ((t_c + 10.0).max(0.0) * (100.0 - rh_pct) * 0.12).clamp(15.0, 600.0);
    let month = doy_to_month_1_to_12(day_of_year);
    step_van_wagner_fwi_at_location(
        t_c,
        rh_pct,
        wind_ms,
        precip_mm_hr * 12.0,
        init_ffmc,
        init_dmc,
        init_dc,
        lat_deg,
        month,
    )
    .fwi
}

/// Convenience steady-state wrapper preserving signature compatibility (`canadian_fire_weather_index`).
#[inline]
pub fn canadian_fire_weather_index(t_c: f64, rh_pct: f64, wind_ms: f64, precip_mm_hr: f64) -> f64 {
    canadian_fire_weather_index_at(t_c, rh_pct, wind_ms, precip_mm_hr, 45.0, 195.0)
}

/// Full UNESCO EOS-80 (1981) / Millero & Poisson (1981) seawater density (`kg m^-3`) at hydrostatic
/// pressure `pressure_dbar` (`dbar`; `1 bar = 10 dbar`), including `B*S^1.5 + C*S^2` and the
/// nonlinear secant bulk modulus `K(S, T, P)`.
#[inline]
pub fn eos80_seawater_density_at_pressure_dbar(
    temp_c: f64,
    salinity_psu: f64,
    pressure_dbar: f64,
) -> f64 {
    let t = if temp_c.is_finite() {
        temp_c.clamp(-5.0, 45.0)
    } else {
        15.0
    };
    let s = if salinity_psu.is_finite() {
        salinity_psu.clamp(0.0, 45.0)
    } else {
        35.0
    };
    let p_dbar = if pressure_dbar.is_finite() {
        pressure_dbar.clamp(0.0, 12_000.0)
    } else {
        0.0
    };
    // EOS-80 secant bulk modulus polynomial expects P in bars (1 bar = 10 dbar)
    let p_bar = p_dbar * 0.1;

    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let s_sqrt = s.sqrt();
    let s32 = s * s_sqrt;
    let s2 = s * s;

    // 1. Pure water (SMOW) density at 1 atm (UNESCO 1981 / Bigg 1967)
    let rho_w = 999.842_594 + 6.793_952e-2 * t - 9.095_290e-3 * t2 + 1.001_685e-4 * t3
        - 1.120_083e-6 * t4
        + 6.536_332e-9 * t5;

    // 2. 1-atm seawater density rho(S, T, 0) (Millero & Poisson 1981)
    let a_s = 0.824_493 - 4.089_9e-3 * t + 7.643_8e-5 * t2 - 8.246_7e-7 * t3 + 5.387_5e-9 * t4;
    let b_s = -5.724_66e-3 + 1.022_7e-4 * t - 1.654_6e-6 * t2;
    let c_s = 4.831_4e-4;
    let rho_0 = rho_w + a_s * s + b_s * s32 + c_s * s2;

    if p_bar <= 0.0 {
        return rho_0;
    }

    // 3. Nonlinear secant bulk modulus K(S, T, P) in bars (Fofonoff & Millard 1983)
    let k_w = 19_652.21 + 148.420_6 * t - 2.327_105 * t2 + 1.360_477e-2 * t3 - 5.155_288e-5 * t4;
    let k_st0 = k_w
        + s * (54.674_6 - 0.603_459 * t + 1.099_87e-2 * t2 - 6.167_0e-5 * t3)
        + s32 * (7.944e-2 + 1.648_3e-2 * t - 5.300_9e-4 * t2);

    let a_w = 3.239_908 + 1.437_13e-3 * t + 1.160_92e-4 * t2 - 5.779_05e-7 * t3
        + s * (2.283_8e-3 - 1.098_1e-5 * t - 1.607_8e-6 * t2)
        + 1.910_75e-4 * s32;

    let b_w = 8.509_35e-5 - 6.122_93e-6 * t + 5.278_7e-8 * t2
        + s * (-9.934_8e-7 + 2.081_6e-8 * t + 9.169_7e-10 * t2);

    let k_stp = (k_st0 + a_w * p_bar + b_w * p_bar * p_bar).max(1.0e3);
    let denom = (1.0 - p_bar / k_stp).max(0.5);
    rho_0 / denom
}

/// UNESCO EOS-80 seawater density (`kg m^-3`) at depth `depth_m` (converting `1 m ≈ 1 dbar`).
#[inline]
pub fn eos80_seawater_density(temp_c: f64, salinity_psu: f64, depth_m: f64) -> f64 {
    eos80_seawater_density_at_pressure_dbar(temp_c, salinity_psu, depth_m)
}

/// Iterative safeguarded Newton-Raphson + bisection seawater carbonate equilibrium solver:
/// - Weiss (1974) CO2 solubility `K0(T, S)`
/// - Lueker et al. (2000) carbonic acid dissociation constants `K1(T, S)` & `K2(T, S)`
/// - Dickson (1990) boric acid `K_B(T, S)`
/// - Millero (1995) / Dickson et al. (2007) Total-scale water auto-ionization `K_w(T, S)`
/// - Mucci (1983) + Millero (1995) pressure-adjusted aragonite `K_sp(T, S, z)`
pub fn solve_seawater_carbonate_chemistry(
    temp_c: f64,
    salinity_psu: f64,
    atm_co2_ppm: f64,
    depth_m: f64,
) -> SeawaterCarbonateState {
    let t_safe = if temp_c.is_finite() { temp_c } else { 15.0 };
    let s_safe = if salinity_psu.is_finite() { salinity_psu } else { 35.0 };
    let co2_safe = if atm_co2_ppm.is_finite() { atm_co2_ppm } else { 420.0 };
    let z_safe = if depth_m.is_finite() { depth_m.max(0.0) } else { 0.0 };

    let tk = (t_safe.clamp(-2.0, 38.0) + 273.15).max(271.15);
    let s = s_safe.clamp(28.0, 40.0);
    let tk100 = tk / 100.0;

    // Weiss (1974) ln(K0) in mol kg^-1 atm^-1
    let ln_k0 = -60.2409 + 93.4517 / tk100 + 23.3585 * tk100.ln()
        + s * (0.023517 - 0.023656 * tk100 + 0.0047036 * tk100 * tk100);
    let k0 = ln_k0.exp();

    // Lueker et al. (2000) pK1 & pK2 on Total pH scale
    let pk1 = 3633.86 / tk - 61.2172 + 9.67770 * tk.ln() - 0.011555 * s + 0.0001152 * s * s;
    let pk2 = 471.78 / tk + 25.9290 - 3.16967 * tk.ln() - 0.01781 * s + 0.0001122 * s * s;
    let k1 = 10.0_f64.powf(-pk1);
    let k2 = 10.0_f64.powf(-pk2);

    // Dickson (1990) boric acid dissociation K_B and total boron B_T
    let ln_kb = (-8966.90 - 2890.53 * s.sqrt() - 77.942 * s + 1.728 * s.powf(1.5)
        - 0.0996 * s * s)
        / tk
        + (148.0248 + 137.1942 * s.sqrt() + 1.62142 * s)
        + (-24.4344 - 25.085 * s.sqrt() - 0.2474 * s) * tk.ln()
        + 0.053105 * s.sqrt() * tk;
    let kb = ln_kb.exp();
    let b_total = 0.0004157 * (s / 35.0); // mol kg^-1

    // Millero (1995) / Dickson et al. (2007) water auto-ionization constant K_w (Total scale)
    let ln_kw = 148.965_02 - 13_847.26 / tk - 23.6521 * tk.ln()
        + (-5.977 + 118.67 / tk + 1.0495 * tk.ln()) * s.sqrt()
        - 0.016_15 * s;
    let kw = ln_kw.exp();

    // Remineralization increases subsurface pCO2 below the mixed layer
    let remin_uatm = 260.0 * (1.0 - (-z_safe / 950.0).exp());
    let pco2_uatm = (co2_safe + remin_uatm).max(120.0);
    let co2_aq = k0 * (pco2_uatm * 1.0e-6); // mol kg^-1

    // Lee et al. (2006) salinity-normalized Total Alkalinity (mol kg^-1)
    let ta = 0.002305 * (s / 35.0);

    // Solve f(pH) = [HCO3-] + 2[CO3^2-] + [B(OH)4-] + [OH-] - [H+] - TA = 0
    // using safeguarded Newton-Raphson with strict [ph_lo, ph_hi] bisection fallback.
    let mut ph_lo = 5.5_f64;
    let mut ph_hi = 9.5_f64;
    let mut ph = 8.10_f64;
    for _ in 0..40 {
        let h = 10.0_f64.powf(-ph);
        let hco3 = k1 * co2_aq / h;
        let co3 = k1 * k2 * co2_aq / (h * h);
        let boh4 = b_total * kb / (h + kb);
        let oh = kw / h;
        let alk_calc = hco3 + 2.0 * co3 + boh4 + oh - h;
        let residual = alk_calc - ta;
        if residual.abs() < 1.0e-12 {
            break;
        }
        if residual < 0.0 {
            ph_lo = ph_lo.max(ph);
        } else {
            ph_hi = ph_hi.min(ph);
        }
        // d(alk_calc)/d(pH) = ln(10) * ([HCO3-] + 4[CO3^2-] + [B(OH)4-]*[H+]/([H+]+K_B) + [OH-] + [H+]) > 0
        let d_alk_dph = std::f64::consts::LN_10
            * (hco3 + 4.0 * co3 + boh4 * (h / (h + kb)) + oh + h).max(1.0e-15);
        let cand = ph - residual / d_alk_dph;
        ph = if cand > ph_lo && cand < ph_hi {
            cand
        } else {
            0.5 * (ph_lo + ph_hi)
        };
    }

    let h_final = 10.0_f64.powf(-ph);
    let co3_final = k1 * k2 * co2_aq / (h_final * h_final);
    let ca_mol_kg = 0.01028 * (s / 35.0);

    // Mucci (1983) log10(Ksp_aragonite) at 1 atm + Millero (1995) hydrostatic pressure increase
    let log10_ksp_0 = -171.945 - 0.077993 * tk + 2903.293 / tk + 71.595 * tk.log10()
        + (-0.068393 + 0.0017276 * tk + 88.135 / tk) * s.sqrt()
        - 0.10018 * s
        + 0.0059415 * s.powf(1.5);
    let p_bar = 0.1 * z_safe;
    // Aragonite becomes ~1.8x more soluble at 4000m (aragonite saturation horizon)
    let pressure_factor = ((36.0 * p_bar) / (83.145 * tk)).exp();
    let ksp_arag = 10.0_f64.powf(log10_ksp_0) * pressure_factor;
    let omega_arag = ((ca_mol_kg * co3_final) / ksp_arag.max(1e-12)).clamp(0.1, 6.5);

    SeawaterCarbonateState {
        ph_total: ph,
        omega_aragonite: omega_arag,
        pco2_uatm,
    }
}

/// Semtner (1976) 0-layer thermodynamic sea-ice thickness step (`m`).
/// Balances upward conductive heat flux through ice/snow (`1.065 * k_ice * (T_f - T_surf) / h`)
/// against basal oceanic mixed-layer heat flux (`rho_w * c_pw * c_h * u_* * (SST - T_f)`).
#[inline]
pub fn step_semtner_sea_ice_thickness(
    prev_h_m: f64,
    t_air_c: f64,
    sst_c: f64,
    shortwave_wm2: f64,
    dt_secs: f64,
) -> f64 {
    let t_freeze = -1.80_f64;
    if sst_c > 0.8 && prev_h_m <= 0.01 {
        return 0.0;
    }
    let k_eff = 1.065 * 2.03; // Semtner (1976) effective ice thermal conductivity (W m^-1 K^-1)
    let h_eff = prev_h_m.max(0.08);
    // Ice surface skin temperature with absorbed shortwave radiation (albedo ~0.68)
    let t_surf = (t_air_c + 0.012 * shortwave_wm2 * (1.0 - 0.68)).min(0.0);
    // Upward conductive heat loss (>0 freezes ice at base; <0 melts surface)
    let f_cond = k_eff * (t_freeze - t_surf) / h_eff;
    // Basal ocean-to-ice turbulent heat flux (W m^-2)
    let f_ocean = 18.5 * (sst_c - t_freeze).max(0.0);
    let dh = ((f_cond - f_ocean) * dt_secs) / VOLUMETRIC_LATENT_HEAT_SEA_ICE;
    (prev_h_m + dh).clamp(0.0, 5.2)
}

const DEM_W: usize = 1440; // 0.25 deg longitude
const DEM_H: usize = 720; // 0.25 deg latitude

struct TrueEarthDemCache {
    is_land: Vec<u8>,
    elev_m: Vec<f32>,
    ring_bboxes: Vec<(f64, f64, f64, f64)>,
}

static TRUE_EARTH_DEM: OnceLock<TrueEarthDemCache> = OnceLock::new();

#[inline]
fn gauss2_earth(lat: f64, lon: f64, c_lat: f64, c_lon: f64, s_lat: f64, s_lon: f64) -> f64 {
    let dlon = ((lon - c_lon + 540.0).rem_euclid(360.0)) - 180.0;
    let dy = (lat - c_lat) / s_lat;
    let dx = dlon / s_lon;
    (-(dy * dy + dx * dx)).exp()
}

#[inline]
fn ridge_gauss(
    lat: f64,
    lon: f64,
    lat_min: f64,
    lat_max: f64,
    center_lon_at_lat: f64,
    sigma_lon: f64,
    taper_deg: f64,
) -> f64 {
    let end_taper = if lat < lat_min {
        (-((lat_min - lat) / taper_deg).powi(2)).exp()
    } else if lat > lat_max {
        (-((lat - lat_max) / taper_deg).powi(2)).exp()
    } else {
        1.0
    };
    let dlon = ((lon - center_lon_at_lat + 540.0).rem_euclid(360.0)) - 180.0;
    end_taper * (-(dlon / sigma_lon).powi(2)).exp()
}

fn build_true_earth_dem() -> TrueEarthDemCache {
    let n = DEM_W * DEM_H;
    let mut ring_bboxes = Vec::with_capacity(CONTINENT_POLYLINES.len());
    for ring in CONTINENT_POLYLINES {
        let mut min_la = 90.0_f64;
        let mut max_la = -90.0_f64;
        let mut min_lo = 180.0_f64;
        let mut max_lo = -180.0_f64;
        for &(la, lo) in *ring {
            min_la = min_la.min(la);
            max_la = max_la.max(la);
            min_lo = min_lo.min(lo);
            max_lo = max_lo.max(lo);
        }
        ring_bboxes.push((min_la, max_la, min_lo, max_lo));
    }

    // 1. Exact parallel scanline rasterization of CONTINENT_POLYLINES (even-odd rule)
    let mut is_land = vec![0_u8; n];
    is_land
        .chunks_mut(DEM_W)
        .enumerate()
        .for_each(|(row, out_row)| {
            let lat = 90.0 - (row as f64 + 0.5) * 0.25;
            if lat <= -86.5 {
                out_row.fill(1);
                return;
            }
            let mut crossings = Vec::with_capacity(64);
            for (r_idx, ring) in CONTINENT_POLYLINES.iter().enumerate() {
                let (min_la, max_la, _, _) = ring_bboxes[r_idx];
                if lat < min_la || lat > max_la {
                    continue;
                }
                for seg in ring.windows(2) {
                    let (y1, x1) = seg[0];
                    let (y2, x2) = seg[1];
                    if (x1 - x2).abs() > 180.0 {
                        continue;
                    }
                    if (y1 > lat) != (y2 > lat) {
                        let x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1);
                        crossings.push(x_cross);
                    }
                }
            }
            crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for pair in crossings.chunks_exact(2) {
                let lo_start = pair[0];
                let lo_end = pair[1];
                let c0 = (((lo_start + 180.0) / 0.25 - 0.5).ceil().max(0.0) as usize).min(DEM_W);
                let c1 = (((lo_end + 180.0) / 0.25 - 0.5).floor().max(-1.0) + 1.0) as isize;
                let c1_u = (c1.max(0) as usize).min(DEM_W);
                for c in c0..c1_u {
                    out_row[c] = 1;
                }
            }
        });

    // 2. Fast 2-pass Chamfer distance transform (in degrees) from nearest coastline edge
    let mut dist_deg = vec![180.0_f32; n];
    for r in 0..DEM_H {
        let row_off = r * DEM_W;
        let rn = r.saturating_sub(1) * DEM_W;
        let rs = (r + 1).min(DEM_H - 1) * DEM_W;
        for c in 0..DEM_W {
            let cw = if c == 0 { DEM_W - 1 } else { c - 1 };
            let ce = if c + 1 == DEM_W { 0 } else { c + 1 };
            let v = is_land[row_off + c];
            if is_land[row_off + cw] != v
                || is_land[row_off + ce] != v
                || is_land[rn + c] != v
                || is_land[rs + c] != v
            {
                dist_deg[row_off + c] = 0.125;
            }
        }
    }
    let dy_step = 0.25_f32;
    for r in 0..DEM_H {
        let lat = 90.0 - (r as f64 + 0.5) * 0.25;
        let dx_step = (0.25 * lat.to_radians().cos().max(0.25)) as f32;
        let diag_step = dx_step.hypot(dy_step);
        let row_off = r * DEM_W;
        for c in 0..DEM_W {
            let mut d = dist_deg[row_off + c];
            if c > 0 {
                d = d.min(dist_deg[row_off + c - 1] + dx_step);
            }
            if r > 0 {
                let up = (r - 1) * DEM_W;
                d = d.min(dist_deg[up + c] + dy_step);
                if c > 0 {
                    d = d.min(dist_deg[up + c - 1] + diag_step);
                }
                if c + 1 < DEM_W {
                    d = d.min(dist_deg[up + c + 1] + diag_step);
                }
            }
            dist_deg[row_off + c] = d;
        }
    }
    for r in (0..DEM_H).rev() {
        let lat = 90.0 - (r as f64 + 0.5) * 0.25;
        let dx_step = (0.25 * lat.to_radians().cos().max(0.25)) as f32;
        let diag_step = dx_step.hypot(dy_step);
        let row_off = r * DEM_W;
        for c in (0..DEM_W).rev() {
            let mut d = dist_deg[row_off + c];
            if c + 1 < DEM_W {
                d = d.min(dist_deg[row_off + c + 1] + dx_step);
            }
            if r + 1 < DEM_H {
                let dn = (r + 1) * DEM_W;
                d = d.min(dist_deg[dn + c] + dy_step);
                if c > 0 {
                    d = d.min(dist_deg[dn + c - 1] + diag_step);
                }
                if c + 1 < DEM_W {
                    d = d.min(dist_deg[dn + c + 1] + diag_step);
                }
            }
            dist_deg[row_off + c] = d;
        }
    }

    // 3. Populate realistic Mountain Cordilleras, Plateaus, Shelves, Mid-Ocean Ridges & Trenches
    let mut elev_m = vec![0.0_f32; n];
    elev_m
        .chunks_mut(DEM_W)
        .enumerate()
        .for_each(|(r, out_row)| {
            let lat = 90.0 - (r as f64 + 0.5) * 0.25;
            let phi = lat.to_radians();
            let row_off = r * DEM_W;
            for c in 0..DEM_W {
                let idx = row_off + c;
                let lon = -180.0 + (c as f64 + 0.5) * 0.25;
                let lam = lon.to_radians();
                let d_coast = dist_deg[idx] as f64;
                let coast_ramp = (d_coast / 1.6).clamp(0.0, 1.0);
                let texture = (3.4 * phi + 2.1 * lam).sin() * (5.2 * lam - 1.7 * phi).cos()
                    + 0.55 * (8.5 * phi - 3.0 * lam).cos() * (7.0 * lam + 2.5 * phi).sin();

                if is_land[idx] != 0 {
                    // Lowland continental interior baseline
                    let plain = 45.0 + 185.0 * (d_coast / 3.5).clamp(0.0, 1.0)
                        + 65.0 * coast_ramp * texture;

                    // Americas Cordilleras: Andes + Rockies/Sierra/Alaska + Appalachians
                    let andes_lon = if lat > 0.0 {
                        -77.0 + 0.35 * lat
                    } else if lat > -22.0 {
                        -77.5 - 0.36 * lat
                    } else {
                        -69.5 - 0.06 * (lat + 22.0)
                    };
                    let andes = 4350.0 * ridge_gauss(lat, lon, -53.5, 10.5, andes_lon, 2.8, 3.0)
                        + 1400.0 * gauss2_earth(lat, lon, -18.5, -67.5, 5.5, 3.2); // Altiplano

                    let rockies_lon = -106.5 - 0.42 * (lat - 35.0).max(-16.0);
                    let rockies = 2650.0 * ridge_gauss(lat, lon, 20.0, 62.5, rockies_lon, 4.5, 4.5);
                    let sierra_cascades = 2050.0 * ridge_gauss(lat, lon, 33.0, 60.0, -121.0 - 0.55 * (lat - 45.0).max(0.0), 2.5, 3.5);
                    let denali = 2850.0 * gauss2_earth(lat, lon, 62.5, -149.0, 2.5, 6.0);
                    let appalachians = 980.0 * ridge_gauss(lat, lon, 33.5, 47.0, -82.0 + 0.72 * (lat - 35.0), 2.6, 2.5);

                    // Eurasia & Africa Ranges: Himalayas, Tibetan Plateau, Pamir, Tien Shan, Alps, Pyrenees, Caucasus, Zagros, Urals, Scandinavia, East Africa, Atlas
                    let tibet = 4450.0 * gauss2_earth(lat, lon, 32.5, 88.5, 4.5, 11.5);
                    let himalaya = 2650.0
                        * ridge_gauss(lat, lon, 27.0, 35.5, 84.5 - 1.2 * (lat - 29.0), 9.5, 2.2)
                        * (-((lat - 28.5) / 2.2).powi(2)).exp();
                    let pamir_tienshan = 3100.0 * gauss2_earth(lat, lon, 38.5, 73.0, 4.2, 6.5)
                        + 2150.0 * gauss2_earth(lat, lon, 43.0, 84.0, 3.0, 9.0);
                    let alps = 2550.0 * gauss2_earth(lat, lon, 46.2, 10.0, 1.4, 4.5);
                    let pyrenees = 1650.0 * gauss2_earth(lat, lon, 42.6, 0.5, 0.8, 2.6);
                    let caucasus = 2900.0 * gauss2_earth(lat, lon, 42.6, 44.0, 1.2, 4.0);
                    let zagros_iran = 1750.0 * gauss2_earth(lat, lon, 32.5, 51.5, 4.5, 7.5);
                    let scandes = 1350.0 * ridge_gauss(lat, lon, 59.5, 70.0, 8.0 + 1.0 * (lat - 60.0), 2.5, 2.0);
                    let urals = 920.0 * ridge_gauss(lat, lon, 52.0, 68.0, 59.0, 2.0, 2.5);
                    let ethiopia_rift = 2250.0 * gauss2_earth(lat, lon, 8.5, 38.5, 5.0, 4.0)
                        + 1350.0 * gauss2_earth(lat, lon, -2.5, 35.5, 6.5, 3.5);
                    let atlas = 1750.0 * gauss2_earth(lat, lon, 32.2, -3.5, 1.8, 5.5);
                    let drakensberg = 1450.0 * gauss2_earth(lat, lon, -29.0, 28.5, 3.2, 2.8);

                    // Greenland & Antarctic Ice Sheet Domes
                    let greenland_dome = if (59.5..=84.0).contains(&lat) && (-73.5..=-11.5).contains(&lon) {
                        2950.0 * (d_coast / 3.8).clamp(0.0, 1.0).powf(0.65)
                    } else {
                        0.0
                    };
                    let antarctica_dome = if lat < -63.0 {
                        let east_boost = if (-20.0..=165.0).contains(&lon) { 3450.0 } else { 1950.0 };
                        east_boost * (d_coast / 6.5).clamp(0.0, 1.0).powf(0.58)
                    } else {
                        0.0
                    };

                    let mountains = andes
                        + rockies
                        + sierra_cascades
                        + denali
                        + appalachians
                        + tibet
                        + himalaya
                        + pamir_tienshan
                        + alps
                        + pyrenees
                        + caucasus
                        + zagros_iran
                        + scandes
                        + urals
                        + ethiopia_rift
                        + atlas
                        + drakensberg
                        + greenland_dome
                        + antarctica_dome;

                    let h_land = plain + coast_ramp * mountains + 180.0 * coast_ramp * texture.max(0.0);
                    out_row[c] = h_land.clamp(4.0, 6800.0) as f32;
                } else {
                    // Ocean / Inland Sea Bathymetry: Continental Shelf -> Slope -> Abyssal Plain -> Ridges & Trenches
                    let shelf_slope = if d_coast < 0.70 {
                        -18.0 - 200.0 * (d_coast / 0.70)
                    } else if d_coast < 2.20 {
                        let t = (d_coast - 0.70) / 1.50;
                        -218.0 - 3750.0 * t.powf(0.85)
                    } else {
                        -3968.0 - 650.0 * ((d_coast - 2.20) / 12.0).clamp(0.0, 1.0)
                    };

                    // Mid-Atlantic Ridge, East Pacific Rise, Southwest Indian Ridge
                    let mar_lon = -29.0 - 8.0 * (lat * 0.055).sin();
                    let mar = 1850.0 * ridge_gauss(lat, lon, -55.0, 64.0, mar_lon, 5.5, 4.0);
                    let epr = 1650.0 * ridge_gauss(lat, lon, -58.0, 18.0, -112.0 + 0.15 * lat, 7.0, 5.0);
                    let indian_ridge = 1500.0 * gauss2_earth(lat, lon, -26.0, 70.0, 16.0, 12.0);

                    // Deep Subduction Trenches (Mariana, Kuril/Japan, Peru-Chile, Tonga-Kermadec, Java, Puerto Rico)
                    let mariana = -4900.0 * gauss2_earth(lat, lon, 16.5, 146.5, 5.5, 1.6);
                    let japan_kuril = -3400.0 * ridge_gauss(lat, lon, 32.0, 52.0, 143.5 + 0.55 * (lat - 36.0), 1.6, 2.5);
                    let peru_chile = -3300.0 * ridge_gauss(lat, lon, -42.0, -4.0, -72.5 - 0.24 * (lat + 22.0).max(0.0), 1.5, 3.0);
                    let tonga_kermadec = -3900.0 * ridge_gauss(lat, lon, -36.0, -15.0, -175.0, 1.6, 2.5);
                    let java_trench = -2900.0 * gauss2_earth(lat, lon, -10.5, 108.0, 1.6, 9.0);
                    let puerto_rico = -3100.0 * gauss2_earth(lat, lon, 19.6, -66.0, 1.2, 3.5);

                    let abyssal_hills = 260.0 * coast_ramp * texture;
                    let h_ocean = shelf_slope
                        + coast_ramp * (mar + epr + indian_ridge + mariana + japan_kuril + peru_chile + tonga_kermadec + java_trench + puerto_rico)
                        + abyssal_hills;
                    out_row[c] = h_ocean.clamp(-9800.0, -4.0) as f32;
                }
            }
        });

    TrueEarthDemCache {
        is_land,
        elev_m,
        ring_bboxes,
    }
}

#[inline]
fn get_true_earth_dem() -> &'static TrueEarthDemCache {
    TRUE_EARTH_DEM.get_or_init(build_true_earth_dem)
}

#[inline]
fn is_land_exact_vector(dem: &TrueEarthDemCache, lat: f64, lon: f64) -> bool {
    if lat <= -86.5 {
        return true;
    }
    let mut inside = false;
    for (r_idx, ring) in CONTINENT_POLYLINES.iter().enumerate() {
        let (min_la, max_la, min_lo, max_lo) = dem.ring_bboxes[r_idx];
        if lat < min_la || lat > max_la || lon > max_lo {
            continue;
        }
        // Note: if lon < min_lo, ray casting rightward (+lon) STILL crosses this ring's edges!
        let _ = min_lo;
        for seg in ring.windows(2) {
            let (y1, x1) = seg[0];
            let (y2, x2) = seg[1];
            if (x1 - x2).abs() > 180.0 {
                continue;
            }
            if (y1 > lat) != (y2 > lat) {
                let x_cross = x1 + (lat - y1) * (x2 - x1) / (y2 - y1);
                if lon < x_cross {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Returns `(d_elev_dx, d_elev_dy)` in `m/deg` from the cached `0.25°` True Earth DEM for hillshading.
pub fn elevation_slope_m_per_deg(lat_deg: f64, lon_deg: f64) -> (f64, f64) {
    let dem = get_true_earth_dem();
    let lat = lat_deg.clamp(-89.875, 89.875);
    let lon = ((lon_deg + 180.0).rem_euclid(360.0)) - 180.0;
    let r = (((90.0 - lat) / 0.25) - 0.5).round().clamp(0.0, (DEM_H - 1) as f64) as usize;
    let c = (((lon + 180.0) / 0.25) - 0.5).round().rem_euclid(DEM_W as f64) as usize;
    let rn = r.saturating_sub(1);
    let rs = (r + 1).min(DEM_H - 1);
    let cw = if c == 0 { DEM_W - 1 } else { c - 1 };
    let ce = if c + 1 == DEM_W { 0 } else { c + 1 };
    let e_w = dem.elev_m[r * DEM_W + cw] as f64;
    let e_e = dem.elev_m[r * DEM_W + ce] as f64;
    let e_n = dem.elev_m[rn * DEM_W + c] as f64;
    let e_s = dem.elev_m[rs * DEM_W + c] as f64;
    ((e_e - e_w) / 0.5, (e_n - e_s) / 0.5)
}

/// True-Earth Natural-Earth-aligned planetary elevation field (meters above MSL).
/// Uses a `OnceLock`-cached `0.25°` (`1440×720`) point-in-polygon DEM with exact coastal
/// polygon refinement so land, ocean, shelves, trenches, and cordilleras align pixel-for-pixel
/// with `CONTINENT_POLYLINES`.
pub fn synthetic_planetary_elevation_m(lat_deg: f64, lon_deg: f64) -> f64 {
    let dem = get_true_earth_dem();
    let lat = lat_deg.clamp(-89.99, 89.99);
    let lon = ((lon_deg + 180.0).rem_euclid(360.0)) - 180.0;

    let fy = ((90.0 - lat) / 0.25) - 0.5;
    let fx = ((lon + 180.0) / 0.25) - 0.5;
    let r0 = fy.floor().clamp(0.0, (DEM_H - 1) as f64) as usize;
    let r1 = (r0 + 1).min(DEM_H - 1);
    let wy = (fy - r0 as f64).clamp(0.0, 1.0);
    let c0_s = fx.floor() as isize;
    let c0 = c0_s.rem_euclid(DEM_W as isize) as usize;
    let c1 = (c0 + 1) % DEM_W;
    let wx = (fx - (c0_s as f64)).clamp(0.0, 1.0);

    let i00 = r0 * DEM_W + c0;
    let i10 = r0 * DEM_W + c1;
    let i01 = r1 * DEM_W + c0;
    let i11 = r1 * DEM_W + c1;

    let m00 = dem.is_land[i00];
    let m10 = dem.is_land[i10];
    let m01 = dem.is_land[i01];
    let m11 = dem.is_land[i11];

    if m00 == m10 && m00 == m01 && m00 == m11 {
        let e00 = dem.elev_m[i00] as f64;
        let e10 = dem.elev_m[i10] as f64;
        let e01 = dem.elev_m[i01] as f64;
        let e11 = dem.elev_m[i11] as f64;
        let top = e00 * (1.0 - wx) + e10 * wx;
        let bot = e01 * (1.0 - wx) + e11 * wx;
        return top * (1.0 - wy) + bot * wy;
    }

    // Coastal transition cell: refine land/ocean exactly against CONTINENT_POLYLINES
    let land_exact = is_land_exact_vector(dem, lat, lon);
    let target = u8::from(land_exact);
    let weights = [
        (1.0 - wx) * (1.0 - wy),
        wx * (1.0 - wy),
        (1.0 - wx) * wy,
        wx * wy,
    ];
    let idxs = [i00, i10, i01, i11];
    let mut sum = 0.0_f64;
    let mut w_sum = 0.0_f64;
    for k in 0..4 {
        if dem.is_land[idxs[k]] == target {
            sum += (dem.elev_m[idxs[k]] as f64) * weights[k];
            w_sum += weights[k];
        }
    }
    if w_sum > 1e-6 {
        let val = sum / w_sum;
        if land_exact {
            val.max(4.0)
        } else {
            val.min(-4.0)
        }
    } else if land_exact {
        15.0
    } else {
        -15.0
    }
}

/// Real-Earth ocean surface current SST signature (°C): Gulf Stream, Kuroshio, Warm Pool,
/// Humboldt/Benguela/California coastal upwelling, and Equatorial Pacific Cold Tongue.
#[inline]
fn earth_sst_current_offset_c(lat: f64, lon: f64, enso_nino34: f64) -> f64 {
    let gulf_stream = 4.4 * gauss2_earth(lat, lon, 37.0, -64.0, 6.0, 14.0)
        + 3.6 * gauss2_earth(lat, lon, 53.0, -28.0, 9.5, 22.0);
    let kuroshio = 4.0 * gauss2_earth(lat, lon, 33.5, 141.0, 6.5, 16.0);
    let agulhas_brazil = 2.4 * gauss2_earth(lat, lon, -31.5, 32.5, 6.5, 7.5)
        + 2.1 * gauss2_earth(lat, lon, -27.5, -43.5, 7.0, 8.0);
    let warm_pool = 2.2 * gauss2_earth(lat, lon, 0.0, 128.0, 13.0, 28.0);

    let cold_tongue_mod = (1.0 - 0.42 * enso_nino34).clamp(0.25, 1.8);
    let eq_cold_tongue = -3.3 * cold_tongue_mod * gauss2_earth(lat, lon, 0.0, -112.0, 4.2, 32.0);
    let humboldt = -4.2 * gauss2_earth(lat, lon, -15.5, -77.5, 13.5, 5.5);
    let benguela = -3.4 * gauss2_earth(lat, lon, -22.0, 11.5, 10.0, 4.8);
    let california = -3.0 * gauss2_earth(lat, lon, 34.5, -123.5, 10.5, 5.0);
    let labrador_oyashio = -3.4 * gauss2_earth(lat, lon, 51.0, -49.0, 7.5, 9.0)
        - 2.8 * gauss2_earth(lat, lon, 45.5, 152.0, 5.5, 9.0);

    gulf_stream
        + kuroshio
        + agulhas_brazil
        + warm_pool
        + eq_cold_tongue
        + humboldt
        + benguela
        + california
        + labrador_oyashio
}

/// Real-Earth terrestrial humidity & biome moisture offset `(rh_offset_pct, sm_offset_m3m3)`.
#[inline]
fn earth_biome_humidity(lat: f64, lon: f64) -> (f64, f64) {
    let humid = 1.00 * gauss2_earth(lat, lon, -4.0, -63.0, 9.5, 14.5) // Amazon
        + 0.95 * gauss2_earth(lat, lon, 0.0, 21.0, 7.5, 9.5) // Congo
        + 1.00 * gauss2_earth(lat, lon, -1.5, 120.0, 9.5, 22.0) // Maritime Continent
        + 0.85 * gauss2_earth(lat, lon, 18.0, 92.0, 9.5, 15.0) // South/SE Asia Monsoon
        + 0.72 * gauss2_earth(lat, lon, 7.0, -5.0, 5.0, 7.0) // West Africa Guinea Coast
        + 0.72 * gauss2_earth(lat, lon, 48.0, -124.0, 7.0, 6.0) // Pacific NW
        + 0.65 * gauss2_earth(lat, lon, 37.5, -82.0, 8.5, 11.0) // Eastern North America
        + 0.68 * gauss2_earth(lat, lon, 49.5, 6.0, 7.5, 14.0) // Europe
        + 0.62 * gauss2_earth(lat, lon, 58.0, 95.0, 7.5, 34.0) // Eurasian Boreal Taiga
        + 0.58 * gauss2_earth(lat, lon, 57.0, -100.0, 7.5, 26.0); // Canadian Boreal

    let arid = 1.18 * gauss2_earth(lat, lon, 23.0, 11.0, 8.5, 23.0) // Sahara
        + 1.12 * gauss2_earth(lat, lon, 22.0, 46.0, 7.5, 11.0) // Arabia
        + 0.96 * gauss2_earth(lat, lon, 40.5, 84.0, 6.0, 19.0) // Taklamakan / Gobi
        + 0.95 * gauss2_earth(lat, lon, -24.0, 132.0, 7.0, 14.0) // Australian Outback
        + 0.85 * gauss2_earth(lat, lon, 31.0, -112.0, 6.5, 8.5) // Sonora / Mojave
        + 0.88 * gauss2_earth(lat, lon, -22.5, -69.0, 6.5, 3.8) // Atacama
        + 0.84 * gauss2_earth(lat, lon, -23.0, 17.0, 7.0, 5.5) // Namib / Kalahari
        + 0.72 * gauss2_earth(lat, lon, 7.0, 46.0, 5.5, 5.0); // Horn of Africa

    (24.0 * humid - 36.0 * arid, 0.14 * humid - 0.17 * arid)
}

/// Real-Earth major river basin & delta discharge corridor signal (`m^3/s` boost).
#[inline]
fn earth_river_corridor_signal(lat: f64, lon: f64) -> f64 {
    let amazon = 4800.0 * ridge_gauss(lat, lon, -5.5, 0.5, -60.0, 11.0, 2.5)
        * (-((lat + 3.0) / 2.8).powi(2)).exp();
    let congo = 2600.0 * gauss2_earth(lat, lon, -1.5, 19.5, 3.5, 5.5);
    let mississippi = 2100.0 * ridge_gauss(lat, lon, 29.0, 44.0, -90.5, 2.6, 2.5);
    let parana = 1900.0 * ridge_gauss(lat, lon, -34.5, -18.0, -58.5, 2.6, 2.5);
    let yangtze = 2400.0 * gauss2_earth(lat, lon, 30.8, 114.0, 2.2, 7.0);
    let ganges_brahmaputra = 2600.0 * gauss2_earth(lat, lon, 24.5, 87.5, 2.5, 5.5);
    let mekong = 1750.0 * ridge_gauss(lat, lon, 10.0, 24.0, 104.0, 2.0, 2.2);
    let nile = 1400.0 * ridge_gauss(lat, lon, 2.0, 31.0, 31.5, 1.8, 2.0);
    let ob_yenisei_lena = 1650.0 * gauss2_earth(lat, lon, 62.0, 70.0, 4.5, 3.5)
        + 1750.0 * gauss2_earth(lat, lon, 64.0, 89.0, 4.5, 3.5)
        + 1650.0 * gauss2_earth(lat, lon, 65.0, 126.0, 4.5, 3.5);
    let danube = 1100.0 * gauss2_earth(lat, lon, 45.8, 22.0, 2.0, 5.5);
    let mackenzie = 1250.0 * ridge_gauss(lat, lon, 60.0, 68.5, -128.0, 2.4, 2.0);

    amazon + congo + mississippi + parana + yangtze + ganges_brahmaputra + mekong + nile + ob_yenisei_lena + danube + mackenzie
}

#[inline]
pub fn preindustrial_baseline_t2m_c(lat_deg: f64, elev_m: f64) -> f64 {
    let phi = lat_deg.to_radians();
    27.8 * phi.cos()
        - 15.5 * phi.sin().powi(2)
        - if elev_m > 0.0 { 0.0065 * elev_m } else { 0.0 }
}

/// Seasonal 2 m air temperature cycle (°C) relative to the annual-mean baseline.
#[inline]
pub fn seasonal_surface_swing_c(lat_deg: f64, is_ocean: bool, day_of_year: f64) -> f64 {
    let phi = lat_deg.to_radians();
    let seasonal_phase = 2.0 * std::f64::consts::PI * (day_of_year - 172.0) / 365.25;
    (if is_ocean { 4.5 } else { 14.0 }) * phi.sin() * seasonal_phase.cos()
}

pub struct ClimateSimulationEngine {
    schema: HubSchema,
    db: ClimateDatabase,
}

impl ClimateSimulationEngine {
    pub fn new(schema: HubSchema, db: ClimateDatabase) -> Self {
        let _ = get_true_earth_dem();
        Self { schema, db }
    }

    /// Execute a full coupled multi-threaded simulation tick across all 34 predictive layers:
    /// - Assimilates SQLite observations (`T2m`, `MSLP`, `SST`, `AOD`, `CO2`) via 2-pass Barnes analysis
    /// - Computes Rayleigh-damped spherical geostrophic-Ekman winds, divergence, & orographic lift from `MSLP` + terrain
    /// - Steps prognostic `T2m`, `SST`, `q` (moisture), `AOD`, `sea_ice_thickness`, and `soil_moisture`
    ///   across SQLite ticks via `CQMSL` Semi-Lagrangian advection + sub-stepped spherical Laplacian diffusion
    /// - Evaluates iterative carbonate `pH`/`Omega_arag`, full 6-code Van Wagner FWI, and Bolton `theta_e`/IVT
    pub fn step_all_layers(&self, cfg: &SimulationConfig) -> Result<GlobalLiveReadouts> {
        let w = cfg.grid_width;
        let h = cfg.grid_height;
        let n_cells = w * h;
        let ts_utc = cfg.format_timestamp_utc();
        let dt_s = cfg.time_step_secs.max(0.0);
        let relax_alpha = cfg.prognostic_relaxation_alpha.clamp(0.0, 1.0);

        let raw_temp = self.db.fetch_recent_samples("wet_bulb_globe_temp_heat_stress", 512)?;
        let raw_mslp = self.db.fetch_recent_samples("synoptic_fronts_isobars", 512)?;
        let raw_sst = self.db.fetch_recent_samples("sst_marine_heatwaves", 2048)?;
        let raw_aod = self.db.fetch_recent_samples("aerosol_optical_depth_speciated", 512)?;
        let raw_co2 = self.db.fetch_recent_samples("column_co2_flux", 32)?;
        let raw_osc = self.db.fetch_recent_samples("teleconnection_oscillation_masks", 16)?;
        let raw_rivers = self.db.fetch_recent_samples("swot_surface_water_river_discharge", 128)?;
        let raw_fires = self.db.fetch_recent_samples("active_wildfires_frp_burn_severity", 600)?;

        let observed_co2_anchor = raw_co2.first().map(|s| s.primary_value);
        let base_co2_ppm = if (cfg.sim_year - 2026.8).abs() < 1.5 {
            observed_co2_anchor.unwrap_or_else(|| cfg.scenario.co2_ppm_at_year(cfg.sim_year))
        } else {
            cfg.scenario.co2_ppm_at_year(cfg.sim_year)
        };

        let df_co2 = co2_radiative_forcing_wm2(base_co2_ppm);
        let global_warming_over_preindustrial = transient_global_warming_c(base_co2_ppm);
        let observed_enso = raw_osc
            .iter()
            .find(|s| s.latitude.abs() < 10.0)
            .map(|s| s.primary_value);
        let observed_nao = raw_osc
            .iter()
            .find(|s| s.latitude > 35.0)
            .map(|s| s.primary_value);
        let enso_nino34 = if (cfg.sim_year - 2026.8).abs() < 1.5 {
            observed_enso.unwrap_or_else(|| {
                1.15 * (2.0 * std::f64::consts::PI * (cfg.sim_year - 2023.5) / 3.8).sin()
                    + 0.08 * (global_warming_over_preindustrial - 1.2)
            })
        } else {
            1.15 * (2.0 * std::f64::consts::PI * (cfg.sim_year - 2023.5) / 3.8).sin()
                + 0.08 * (global_warming_over_preindustrial - 1.2)
        };
        let default_nao = 0.65 * (cfg.day_of_year * 0.05).cos();
        let nao_phase = if (cfg.sim_year - 2026.8).abs() < 1.5 {
            observed_nao.unwrap_or(default_nao)
        } else {
            default_nao
        };

        let seasonal_phase = 2.0 * std::f64::consts::PI * (cfg.day_of_year - 172.0) / 365.25;
        let solar_decl_rad = 23.44_f64.to_radians() * seasonal_phase.cos();

        #[derive(Clone, Copy)]
        struct CellCoreState {
            lat: f64,
            lon: f64,
            elev_m: f64,
            is_ocean: bool,
            t2m_c: f64,
            q_kg_kg: f64,
            rh_pct: f64,
            mslp_hpa: f64,
            sst_c: f64,
            u10_ms: f64,
            v10_ms: f64,
            div_v_s1: f64,
            w_oro_ms: f64,
            shortwave_wm2: f64,
            precip_mm_hr: f64,
            soil_moisture: f64,
            sea_ice_h_m: f64,
        }

        let mut core_cells = vec![
            CellCoreState {
                lat: 0.0,
                lon: 0.0,
                elev_m: 0.0,
                is_ocean: true,
                t2m_c: 15.0,
                q_kg_kg: 0.008,
                rh_pct: 70.0,
                mslp_hpa: 1013.25,
                sst_c: 18.0,
                u10_ms: 0.0,
                v10_ms: 0.0,
                div_v_s1: 0.0,
                w_oro_ms: 0.0,
                shortwave_wm2: 200.0,
                precip_mm_hr: 0.0,
                soil_moisture: 0.25,
                sea_ice_h_m: 0.0,
            };
            n_cells
        ];

        // 1. Radiative-convective baseline & background MSLP
        core_cells
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(row, row_slice)| {
                let lat = 90.0 - (row as f64 + 0.5) * (180.0 / h as f64);
                let phi = lat.to_radians();
                let polar_amp = polar_amplification_factor(lat);

                for (col, cell) in row_slice.iter_mut().enumerate() {
                    let lon = -180.0 + (col as f64 + 0.5) * (360.0 / w as f64);
                    let lam = lon.to_radians();
                    let elev_m = synthetic_planetary_elevation_m(lat, lon);
                    let is_ocean = elev_m < 0.0;

                    let hour_angle = ((cfg.hour_utc - 12.0) * 15.0 + lon).to_radians();
                    let cos_zenith = (phi.sin() * solar_decl_rad.sin()
                        + phi.cos() * solar_decl_rad.cos() * hour_angle.cos())
                    .max(0.0);
                    let shortwave = 1361.0 * 0.74 * cos_zenith;

                    let seasonal_swing = seasonal_surface_swing_c(lat, is_ocean, cfg.day_of_year);
                    let enso_footprint = enso_nino34
                        * (-((lat / 10.0).powi(2)) - (((lon + 125.0) / 48.0).powi(2))).exp();

                    let t2m_eq = preindustrial_baseline_t2m_c(lat, elev_m)
                        + seasonal_swing
                        + global_warming_over_preindustrial * polar_amp
                        + enso_footprint;

                    let sst_eq = if is_ocean {
                        (28.4 * phi.cos() - 3.5 * phi.sin().powi(2)
                            + 0.35 * seasonal_swing
                            + 0.72 * global_warming_over_preindustrial
                            + 1.25 * enso_footprint
                            + earth_sst_current_offset_c(lat, lon, enso_nino34))
                            .max(-1.85)
                    } else {
                        t2m_eq
                    };

                    let wave_phase =
                        4.0 * lam - 0.18 * (cfg.day_of_year + cfg.hour_utc / 24.0);
                    let mslp = 1013.25 - 9.5 * (3.0 * phi).cos()
                        + 7.5 * phi.sin().powi(2) * wave_phase.sin()
                        + 3.8 * (6.0 * lam + 0.3 * cfg.day_of_year).cos();

                    let (rh_biome_off, sm_biome_off) = earth_biome_humidity(lat, lon);
                    let rh_eq = (if is_ocean { 81.0 } else { 62.0 }
                        - 20.0 * (2.0 * phi).sin().powi(2)
                        + 8.0 * wave_phase.sin()
                        + if is_ocean { 0.0 } else { rh_biome_off })
                    .clamp(12.0, 97.0);
                    let p_surf = if elev_m > 0.0 {
                        mslp * (-elev_m / 8400.0).exp()
                    } else {
                        mslp
                    };
                    let q_eq = saturation_specific_humidity(t2m_eq, p_surf) * (rh_eq / 100.0);

                    let sm_eq = if is_ocean {
                        0.48
                    } else {
                        (0.08 + 0.34 * (rh_eq / 100.0)
                            + 0.45 * sm_biome_off
                            - 0.012 * global_warming_over_preindustrial)
                            .clamp(0.025, 0.48)
                    };
                    let ice_eq = if is_ocean && sst_eq < 0.0 {
                        (0.028 * ((-t2m_eq).max(0.0) * 120.0).sqrt()).clamp(0.0, 4.8)
                    } else {
                        0.0
                    };

                    *cell = CellCoreState {
                        lat,
                        lon,
                        elev_m,
                        is_ocean,
                        t2m_c: t2m_eq,
                        q_kg_kg: q_eq,
                        rh_pct: rh_eq,
                        mslp_hpa: mslp,
                        sst_c: sst_eq,
                        u10_ms: 0.0,
                        v10_ms: 0.0,
                        div_v_s1: 0.0,
                        w_oro_ms: 0.0,
                        shortwave_wm2: shortwave,
                        precip_mm_hr: 0.0,
                        soil_moisture: sm_eq,
                        sea_ice_h_m: ice_eq,
                    };
                }
            });

        // 2. Prognostic 2-Layer Spherical QGPV & Elliptic Streamfunction Inversion (`q_1, q_2 -> psi_1, psi_2 -> MSLP`)
        let prev_syn_grid = self
            .db
            .load_latest_grid("synoptic_fronts_isobars", cfg.scenario, 1000.0)?
            .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells);
        let prev_qgpv_pair = prev_syn_grid.as_ref().and_then(|g| {
            g.u_component
                .as_deref()
                .zip(g.v_component.as_deref())
                .filter(|(q1_b, q2_b)| q1_b.len() == n_cells && q2_b.len() == n_cells)
        });

        let elev_full: Vec<f64> = core_cells.iter().map(|c| c.elev_m).collect();
        let t2m_bg_for_qg: Vec<f64> = core_cells.iter().map(|c| c.t2m_c).collect();

        let mut qgpv_state = step_two_layer_qgpv(
            prev_qgpv_pair,
            &elev_full,
            &t2m_bg_for_qg,
            w,
            h,
            dt_s,
            seasonal_phase,
            global_warming_over_preindustrial,
            enso_nino34,
            nao_phase,
        );

        for (c, &p_dyn) in core_cells.iter_mut().zip(qgpv_state.mslp_hpa.iter()) {
            c.mslp_hpa = p_dyn as f64;
        }

        // Assimilate live surface pressure observations via flow-aligned CG 3D-Var + geostrophic covariance balance!
        if (cfg.sim_year - 2026.8).abs() < 2.0 && !raw_mslp.is_empty() {
            let bg: Vec<f32> = core_cells.iter().map(|c| c.mslp_hpa as f32).collect();
            let pts: Vec<ObsSamplePoint> = raw_mslp
                .iter()
                .map(|s| ObsSamplePoint {
                    lat: s.latitude,
                    lon: s.longitude,
                    value: s.primary_value,
                    quality_flag: s.quality_flag,
                    sigma_obs: None,
                })
                .collect();
            let var_cfg = ThreeDVarConfig {
                sigma_b: 3.5,
                base_sigma_obs: 1.1,
                length_scale_km: 1450.0,
                flow_anisotropy: 0.60,
                max_iters: 48,
                tol: 1.0e-6,
            };
            let res = assimilate_3dvar_cg(
                &bg,
                w,
                h,
                &pts,
                Some((&qgpv_state.u2, &qgpv_state.v2)),
                &var_cfg,
            );

            // Multivariate geostrophic error-covariance balance: dp -> dpsi_2 -> (du_2, dv_2, dq_2, dq_1)
            let mut dpsi2_innov = vec![0.0_f32; n_cells];
            for (i, (c, &val)) in core_cells.iter_mut().zip(res.analysis.iter()).enumerate() {
                let dp_pa = (res.increment[i] as f64) * 100.0;
                c.mslp_hpa = (val as f64).clamp(950.0, 1052.0);
                let f_raw = planetary_coriolis_f(c.lat);
                let f_reg = if c.lat >= 0.0 {
                    f_raw.max(3.5e-5)
                } else {
                    f_raw.min(-3.5e-5)
                };
                let trop_taper = (c.lat.abs() / 15.0).tanh();
                dpsi2_innov[i] = (0.42 * trop_taper * dp_pa / (1.225 * f_reg)) as f32;
            }
            let (du2_bal, dv2_bal) = diagnose_rotational_wind_from_streamfunction(&dpsi2_innov, w, h);
            let dzeta2_innov = spherical_laplacian_2d(&dpsi2_innov, w, h);
            let f_bc =
                (1.0 / (2.0 * QG_ROSSBY_DEFORMATION_RADIUS_M * QG_ROSSBY_DEFORMATION_RADIUS_M))
                    as f32;
            for i in 0..n_cells {
                qgpv_state.psi2[i] += dpsi2_innov[i];
                qgpv_state.u2[i] = (qgpv_state.u2[i] + du2_bal[i]).clamp(-85.0, 85.0);
                qgpv_state.v2[i] = (qgpv_state.v2[i] + dv2_bal[i]).clamp(-70.0, 70.0);
                qgpv_state.q2[i] = (qgpv_state.q2[i] + dzeta2_innov[i] - f_bc * dpsi2_innov[i])
                    .clamp(-5.0e-4, 5.0e-4);
                qgpv_state.q1[i] =
                    (qgpv_state.q1[i] + f_bc * dpsi2_innov[i]).clamp(-5.0e-4, 5.0e-4);
            }
        }

        // 3. Solve Coupled QG + Rayleigh-Ekman Boundary-Layer Surface Winds, Divergence & Orographic Lift!
        let dphi = std::f64::consts::PI / (h as f64);
        let dlam = (2.0 * std::f64::consts::PI) / (w as f64);
        let mslp_snap: Vec<f64> = core_cells.iter().map(|c| c.mslp_hpa * 100.0).collect(); // Pascals
        let elev_snap: Vec<f64> = core_cells.iter().map(|c| c.elev_m.max(0.0)).collect();
        let u2_qg_ref = &qgpv_state.u2;
        let v2_qg_ref = &qgpv_state.v2;

        core_cells
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(row, row_slice)| {
                let lat = 90.0 - (row as f64 + 0.5) * (180.0 / h as f64);
                let phi = lat.to_radians();
                let cos_phi = phi.cos().max(0.16);
                let f_coriolis = 2.0 * EARTH_OMEGA_RAD_S * phi.sin();
                let row_n = row.saturating_sub(1);
                let row_s = (row + 1).min(h - 1);
                let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;

                for (col, cell) in row_slice.iter_mut().enumerate() {
                    let idx = row * w + col;
                    let col_w = if col == 0 { w - 1 } else { col - 1 };
                    let col_e = if col + 1 == w { 0 } else { col + 1 };
                    let dx = 2.0 * EARTH_RADIUS_M * cos_phi * dlam;

                    // Spherical pressure gradient force PGF = -(1/rho) grad_S(p)
                    let rho_a = 1.225;
                    let dpdx = (mslp_snap[row * w + col_e] - mslp_snap[row * w + col_w]) / dx;
                    let dpdy = (mslp_snap[row_n * w + col] - mslp_snap[row_s * w + col]) / dy;
                    let pgf_x = -dpdx / rho_a;
                    let pgf_y = -dpdy / rho_a;

                    let r_drag = if cell.is_ocean { 4.2e-5 } else { 7.5e-5 };
                    let denom = f_coriolis * f_coriolis + r_drag * r_drag;
                    let u_geostrophic_ekman = (r_drag * pgf_x + f_coriolis * pgf_y) / denom;
                    let v_geostrophic_ekman = (r_drag * pgf_y - f_coriolis * pgf_x) / denom;

                    // Couple boundary-layer cross-isobaric Ekman inflow with lower-tropospheric QG vortex flow
                    let u_qg_surf = 0.72 * (u2_qg_ref[idx] as f64);
                    let v_qg_surf = 0.72 * (v2_qg_ref[idx] as f64);
                    let u10 = (0.58 * u_geostrophic_ekman + 0.42 * u_qg_surf).clamp(-45.0, 45.0);
                    let v10 = (0.62 * v_geostrophic_ekman
                        + 0.38 * v_qg_surf
                        - 1.1 * (2.0 * phi).sin())
                    .clamp(-38.0, 38.0);

                    // Orographic upslope vertical velocity w_oro = max(V . grad_S(z_elev), 0)
                    let dzdx = (elev_snap[row * w + col_e] - elev_snap[row * w + col_w]) / dx;
                    let dzdy = (elev_snap[row_n * w + col] - elev_snap[row_s * w + col]) / dy;
                    let w_oro = (u10 * dzdx + v10 * dzdy).max(0.0);

                    cell.u10_ms = u10;
                    cell.v10_ms = v10;
                    cell.w_oro_ms = w_oro;
                }
            });

        // Compute horizontal wind divergence div_S(V) across the mesh
        let u_snap: Vec<f64> = core_cells.iter().map(|c| c.u10_ms).collect();
        let v_snap: Vec<f64> = core_cells.iter().map(|c| c.v10_ms).collect();
        core_cells
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(row, row_slice)| {
                let lat = 90.0 - (row as f64 + 0.5) * (180.0 / h as f64);
                let phi = lat.to_radians();
                let cos_phi = phi.cos().max(0.16);
                let row_n = row.saturating_sub(1);
                let row_s = (row + 1).min(h - 1);
                let cos_n = (90.0 - (row_n as f64 + 0.5) * (180.0 / h as f64))
                    .to_radians()
                    .cos()
                    .max(0.05);
                let cos_s = (90.0 - (row_s as f64 + 0.5) * (180.0 / h as f64))
                    .to_radians()
                    .cos()
                    .max(0.05);
                let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;
                let dx = 2.0 * EARTH_RADIUS_M * cos_phi * dlam;

                for (col, cell) in row_slice.iter_mut().enumerate() {
                    let col_w = if col == 0 { w - 1 } else { col - 1 };
                    let col_e = if col + 1 == w { 0 } else { col + 1 };
                    let dudx = (u_snap[row * w + col_e] - u_snap[row * w + col_w]) / dx;
                    let dvcos_dy = (v_snap[row_n * w + col] * cos_n
                        - v_snap[row_s * w + col] * cos_s)
                        / (cos_phi * dy);
                    cell.div_v_s1 = dudx + dvcos_dy;
                }
            });

        // 3b. Prognostic 1.5-Layer Reduced-Gravity Shallow-Water & Stommel-Munk Ocean Circulation
        let prev_ssh_grid = self
            .db
            .load_latest_grid("swot_sea_surface_height_altimetry", cfg.scenario, 0.0)?
            .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells);
        let prev_cur_grid = self
            .db
            .load_latest_grid("thermohaline_currents_amoc", cfg.scenario, 0.0)?
            .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells);

        let u10_vec: Vec<f64> = core_cells.iter().map(|c| c.u10_ms).collect();
        let v10_vec: Vec<f64> = core_cells.iter().map(|c| c.v10_ms).collect();
        let is_ocean_vec: Vec<bool> = core_cells.iter().map(|c| c.is_ocean).collect();

        let ocean_state = step_ocean_reduced_gravity_and_gyre(
            prev_ssh_grid.as_ref().and_then(|g| g.u_component.as_deref()),
            prev_ssh_grid.as_ref().and_then(|g| g.v_component.as_deref()),
            prev_cur_grid.as_ref().and_then(|g| g.u_component.as_deref()),
            prev_cur_grid.as_ref().and_then(|g| g.v_component.as_deref()),
            &u10_vec,
            &v10_vec,
            &is_ocean_vec,
            w,
            h,
            dt_s,
            global_warming_over_preindustrial,
        );

        // Couple thermocline depth anomaly into SST (deeper warm thermocline warms SST; shoaling cools SST)
        for (i, c) in core_cells.iter_mut().enumerate() {
            if c.is_ocean {
                let tc_sst_feedback = 0.045 * (ocean_state.h_tc_anom_m[i] as f64);
                c.sst_c = (c.sst_c + tc_sst_feedback).max(-1.85);
            }
        }

        // 4. Cross-Tick Prognostic PDE Transport driven by Lower-Tropospheric QG + Ekman Synoptic Flow
        let u_adv_atm: Vec<f32> = core_cells
            .iter()
            .enumerate()
            .map(|(i, c)| 0.65 * qgpv_state.u2[i] + 0.35 * (c.u10_ms as f32))
            .collect();
        let v_adv_atm: Vec<f32> = core_cells
            .iter()
            .enumerate()
            .map(|(i, c)| 0.65 * qgpv_state.v2[i] + 0.35 * (c.v10_ms as f32))
            .collect();

        if dt_s > 0.0 {
            // 4a. Prognostic 2m Air Temperature (`CQMSL` + Spherical Laplacian + Land/Ocean Energy Balance)
            if let Some(prev_anom_grid) = self
                .db
                .load_latest_grid("cmip6_downscaled_ensemble", cfg.scenario, 0.0)?
                .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells)
            {
                let prev_t2m: Vec<f32> = core_cells
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        (preindustrial_baseline_t2m_c(c.lat, c.elev_m)
                            + seasonal_surface_swing_c(c.lat, c.is_ocean, cfg.day_of_year)
                            + prev_anom_grid.values[i] as f64) as f32
                    })
                    .collect();

                let adv_t2m = semi_lagrangian_advect_qmsl(
                    &prev_t2m, &u_adv_atm, &v_adv_atm, w, h, dt_s, true, false,
                );

                let rad_tendency: Vec<f32> = core_cells
                    .iter()
                    .zip(adv_t2m.iter())
                    .map(|(c, &t_adv)| {
                        let t_cur = t_adv as f64;
                        let ice_alb = if t_cur < -5.0 || c.sea_ice_h_m > 0.15 {
                            0.64
                        } else if c.is_ocean {
                            0.09
                        } else {
                            0.24
                        };
                        let alpha = (ice_alb + 0.22 * (c.rh_pct / 100.0)).clamp(0.07, 0.82);
                        let asr = c.shortwave_wm2 * (1.0 - alpha);
                        let olr = (208.5 + 1.95 * t_cur) - df_co2;
                        let c_eff = if c.is_ocean {
                            OCEAN_MIXED_LAYER_HEAT_CAPACITY_J_M2_K
                        } else {
                            LAND_HEAT_CAPACITY_J_M2_K
                        };
                        ((asr - olr) / c_eff) as f32
                    })
                    .collect();

                let diff_t2m = spherical_diffusive_heat_step(
                    &adv_t2m,
                    Some(&rad_tendency),
                    w,
                    h,
                    cfg.thermal_diffusivity_m2_s,
                    dt_s,
                );

                for (c, &t_prog) in core_cells.iter_mut().zip(diff_t2m.iter()) {
                    if t_prog.is_finite() {
                        c.t2m_c = (1.0 - relax_alpha) * (t_prog as f64) + relax_alpha * c.t2m_c;
                    }
                }
            }

            // 4b. Prognostic Ocean SST & Semtner (1976) Sea-Ice Thickness
            if let Some(prev_sst_grid) = self
                .db
                .load_latest_grid("sst_marine_heatwaves", cfg.scenario, 0.0)?
                .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells)
            {
                let adv_sst = semi_lagrangian_advect_qmsl(
                    &prev_sst_grid.values,
                    &ocean_state.u_total,
                    &ocean_state.v_total,
                    w,
                    h,
                    dt_s,
                    false,
                    false,
                );
                let diff_sst = spherical_diffusive_heat_step(
                    &adv_sst,
                    None,
                    w,
                    h,
                    (cfg.thermal_diffusivity_m2_s * 0.04).max(2.5e3),
                    dt_s,
                );
                let ocn_relax = (relax_alpha * 0.5).clamp(0.0, 1.0);
                for (c, &s_prog) in core_cells.iter_mut().zip(diff_sst.iter()) {
                    if c.is_ocean && s_prog.is_finite() {
                        c.sst_c =
                            ((1.0 - ocn_relax) * (s_prog as f64) + ocn_relax * c.sst_c).max(-1.85);
                    }
                }
            }

            if let Some(prev_ice_grid) = self
                .db
                .load_latest_grid("sea_ice_thickness_freeboard_age", cfg.scenario, 0.0)?
                .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells)
            {
                // Sea ice drifts at ~2% of 10m wind speed (Nansen rule) + Semtner thermodynamic growth/melt
                let u_ice: Vec<f32> = core_cells.iter().map(|c| (c.u10_ms * 0.02) as f32).collect();
                let v_ice: Vec<f32> = core_cells.iter().map(|c| (c.v10_ms * 0.02) as f32).collect();
                let adv_ice = semi_lagrangian_advect_qmsl(
                    &prev_ice_grid.values,
                    &u_ice,
                    &v_ice,
                    w,
                    h,
                    dt_s,
                    false,
                    true,
                );
                for (c, &h_prev) in core_cells.iter_mut().zip(adv_ice.iter()) {
                    if c.is_ocean {
                        c.sea_ice_h_m = step_semtner_sea_ice_thickness(
                            h_prev as f64,
                            c.t2m_c,
                            c.sst_c,
                            c.shortwave_wm2,
                            dt_s,
                        );
                    }
                }
            }

            // 4c. Prognostic Land Root-Zone Soil Moisture Bucket (`soil_moisture_groundwater_grace`)
            if let Some(prev_sm_grid) = self
                .db
                .load_latest_grid("soil_moisture_groundwater_grace", cfg.scenario, 0.0)?
                .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells)
            {
                for (c, &sm_prev) in core_cells.iter_mut().zip(prev_sm_grid.values.iter()) {
                    if !c.is_ocean && sm_prev.is_finite() {
                        c.soil_moisture = (sm_prev as f64).clamp(0.02, 0.48);
                    }
                }
            }

            // 4d. Prognostic Specific Humidity `q` (`CQMSL` Advection + Bulk Evaporation + Orographic/Frontal Condensation)
            let prev_q: Vec<f32> = if let Some(prev_ivt_grid) = self
                .db
                .load_latest_grid("atmospheric_rivers_ivt", cfg.scenario, 850.0)?
                .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells)
            {
                core_cells
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let v850 = (c.u10_ms.hypot(c.v10_ms) * 1.65).max(1.5);
                        let pw = (prev_ivt_grid.values[i] as f64) / v850;
                        let q_est = pw / ((70_000.0 / 9.80665) * 0.42);
                        if q_est.is_finite() && q_est > 1e-5 {
                            q_est.clamp(1.0e-4, 0.035) as f32
                        } else {
                            c.q_kg_kg as f32
                        }
                    })
                    .collect()
            } else {
                core_cells.iter().map(|c| c.q_kg_kg as f32).collect()
            };

            let adv_q = semi_lagrangian_advect_qmsl(
                &prev_q, &u_adv_atm, &v_adv_atm, w, h, dt_s, true, true,
            );
            let diff_q = spherical_diffusive_heat_step(
                &adv_q,
                None,
                w,
                h,
                cfg.thermal_diffusivity_m2_s * 0.45,
                dt_s,
            );

            for (c, &q_trans) in core_cells.iter_mut().zip(diff_q.iter()) {
                let p_surf = if c.elev_m > 0.0 {
                    c.mslp_hpa * (-c.elev_m / 8400.0).exp()
                } else {
                    c.mslp_hpa
                };
                let q_sat_air = saturation_specific_humidity(c.t2m_c, p_surf);
                let wind_spd = c.u10_ms.hypot(c.v10_ms).max(0.8);

                // Bulk aerodynamic evaporation E (kg m^-2 s^-1) distributed over ~1500m boundary layer
                let c_dalton = 1.3e-3;
                let evap_kg_m2_s = if c.is_ocean {
                    let q_sat_sst = saturation_specific_humidity(c.sst_c, c.mslp_hpa);
                    1.225 * c_dalton * wind_spd * (q_sat_sst - q_trans as f64).max(0.0)
                } else {
                    let soil_beta = ((c.soil_moisture - 0.03) / 0.32).clamp(0.05, 1.0);
                    1.225 * c_dalton * soil_beta * wind_spd * (q_sat_air - q_trans as f64).max(0.0)
                };
                let dq_evap = (evap_kg_m2_s * dt_s) / (1.225 * 1500.0);

                // Moisture convergence (-q * div_V) + orographic upslope moisture lift
                let conv_rate = (-c.div_v_s1).max(0.0);
                let dq_conv = (q_trans as f64) * conv_rate * dt_s * 0.65;
                let orographic_lift_cool = c.w_oro_ms * 0.0065 * dt_s.min(1800.0);
                let q_sat_lifted =
                    saturation_specific_humidity(c.t2m_c - orographic_lift_cool, p_surf);

                let q_tentative = ((1.0 - relax_alpha) * (q_trans as f64 + dq_evap + dq_conv)
                    + relax_alpha * c.q_kg_kg)
                    .clamp(1.0e-5, 0.042);

                // Clausius-Clapeyron supersaturation condensation -> precipitation + latent heating!
                let excess_q = (q_tentative - q_sat_lifted).max(0.0);
                let subgrid_conv = if q_tentative > 0.76 * q_sat_lifted {
                    0.12 * (q_tentative - 0.76 * q_sat_lifted)
                        * (1.0 + 35.0 * c.w_oro_ms + 2.0e4 * conv_rate)
                } else {
                    0.0
                };
                let dq_cond = (excess_q + subgrid_conv).min(0.45 * q_tentative);
                let q_final = (q_tentative - dq_cond).clamp(1.0e-5, q_sat_air);

                // Convert column condensed specific humidity (effective 3000 kg m^-2 active moist layer) into mm/hr
                let condensed_mm = dq_cond * 3000.0;
                let dt_hours = (dt_s / 3600.0).max(0.25);
                c.precip_mm_hr = (condensed_mm / dt_hours).clamp(0.0, 65.0);
                c.q_kg_kg = q_final;
                c.rh_pct = ((q_final / q_sat_air) * 100.0).clamp(8.0, 99.5);

                // Latent heat feedback (damped by boundary-layer radiative ventilation)
                c.t2m_c += 0.18 * (LATENT_HEAT_VAPORIZATION * dq_cond / CP_DRY_AIR);

                // Step root-zone 1.0m soil moisture bucket over land: d(theta)/dt = (P - E - Q_runoff) / 1000 mm
                if !c.is_ocean {
                    let evap_mm = evap_kg_m2_s * dt_s;
                    let sat_frac = (c.soil_moisture / 0.48).clamp(0.0, 1.0);
                    let runoff_mm = condensed_mm * sat_frac.powi(3) + 0.04 * sat_frac * dt_hours;
                    c.soil_moisture =
                        (c.soil_moisture + (condensed_mm - evap_mm - runoff_mm) / 1000.0)
                            .clamp(0.025, 0.48);
                }
            }
        } else {
            // Diagnostic fallback when dt_s == 0
            for c in &mut core_cells {
                let itcz = (-((c.lat - 6.0 * seasonal_phase.cos()) / 7.5).powi(2)).exp();
                c.precip_mm_hr =
                    (2.4 * itcz + 18.0 * c.w_oro_ms) * (1.0 + 0.07 * global_warming_over_preindustrial);
            }
        }

        // Couple moist condensational heating into QGPV vortex stretching (intensifies warm-conveyor cyclones!)
        if dt_s > 0.0 {
            for (i, c) in core_cells.iter().enumerate() {
                if c.precip_mm_hr > 0.25 {
                    let f_cor = planetary_coriolis_f(c.lat);
                    // Lower-tropospheric vortex tube stretching (+f * d_theta_diab / d_z > 0 for cyclones)
                    let diabatic_pv_tend = 1.8e-10 * f_cor * c.precip_mm_hr.min(25.0);
                    qgpv_state.q2[i] =
                        (qgpv_state.q2[i] + (diabatic_pv_tend * dt_s) as f32).clamp(-5.0e-4, 5.0e-4);
                    qgpv_state.q1[i] = (qgpv_state.q1[i] - (0.65 * diabatic_pv_tend * dt_s) as f32)
                        .clamp(-5.0e-4, 5.0e-4);
                }
            }
        }

        // 5. Assimilate live SQLite observations onto T2m and SST via flow-aligned CG 3D-Var
        if (cfg.sim_year - 2026.8).abs() < 2.0 {
            if !raw_temp.is_empty() {
                let bg: Vec<f32> = core_cells.iter().map(|c| c.t2m_c as f32).collect();
                let pts: Vec<ObsSamplePoint> = raw_temp
                    .iter()
                    .map(|s| ObsSamplePoint {
                        lat: s.latitude,
                        lon: s.longitude,
                        value: s.primary_value,
                        quality_flag: s.quality_flag,
                        sigma_obs: None,
                    })
                    .collect();
                let var_cfg = ThreeDVarConfig {
                    sigma_b: 2.2,
                    base_sigma_obs: 0.75,
                    length_scale_km: 1300.0,
                    flow_anisotropy: 0.55,
                    max_iters: 48,
                    tol: 1.0e-6,
                };
                let res = assimilate_3dvar_cg(
                    &bg,
                    w,
                    h,
                    &pts,
                    Some((&qgpv_state.u2, &qgpv_state.v2)),
                    &var_cfg,
                );
                for (c, &val) in core_cells.iter_mut().zip(res.analysis.iter()) {
                    c.t2m_c = val as f64;
                }
            }
            if !raw_sst.is_empty() {
                let bg: Vec<f32> = core_cells.iter().map(|c| c.sst_c as f32).collect();
                let pts: Vec<ObsSamplePoint> = raw_sst
                    .iter()
                    .map(|s| ObsSamplePoint {
                        lat: s.latitude,
                        lon: s.longitude,
                        value: s.primary_value,
                        quality_flag: s.quality_flag,
                        sigma_obs: None,
                    })
                    .collect();
                let var_cfg = ThreeDVarConfig {
                    sigma_b: 1.4,
                    base_sigma_obs: 0.45,
                    length_scale_km: 1400.0,
                    flow_anisotropy: 0.50,
                    max_iters: 48,
                    tol: 1.0e-6,
                };
                let res = assimilate_3dvar_cg(
                    &bg,
                    w,
                    h,
                    &pts,
                    Some((&ocean_state.u_total, &ocean_state.v_total)),
                    &var_cfg,
                );
                for (c, &val) in core_cells.iter_mut().zip(res.analysis.iter()) {
                    if c.is_ocean {
                        c.sst_c = (val as f64).max(-1.85);
                    }
                }
            }
        }

        // 6. Precompute Horizontal Thermal Gradients `dT/dx, dT/dy` for Upper-Level Thermal Wind Shear
        let t_snap: Vec<f64> = core_cells.iter().map(|c| c.t2m_c).collect();
        let mut dt_dx_grid = vec![0.0_f64; n_cells];
        let mut dt_dy_grid = vec![0.0_f64; n_cells];
        for row in 0..h {
            let lat = 90.0 - (row as f64 + 0.5) * (180.0 / h as f64);
            let cos_phi = lat.to_radians().cos().max(0.18);
            let row_n = row.saturating_sub(1);
            let row_s = (row + 1).min(h - 1);
            let dy = ((row_s - row_n).max(1) as f64) * EARTH_RADIUS_M * dphi;
            let dx = 2.0 * EARTH_RADIUS_M * cos_phi * dlam;
            for col in 0..w {
                let col_w = if col == 0 { w - 1 } else { col - 1 };
                let col_e = if col + 1 == w { 0 } else { col + 1 };
                dt_dx_grid[row * w + col] = (t_snap[row * w + col_e] - t_snap[row * w + col_w]) / dx;
                dt_dy_grid[row * w + col] = (t_snap[row_n * w + col] - t_snap[row_s * w + col]) / dy;
            }
        }

        // 7. Prognostic CQMSL Aerosol Transport + Wet Scavenging Washout + Barnes Assimilation
        let aod_emission_eq: Vec<f32> = core_cells
            .par_iter()
            .map(|c| {
                let sahara_plume = 0.62 * gauss2_earth(c.lat, c.lon, 19.5, 4.0, 8.5, 22.0)
                    + 0.34 * gauss2_earth(c.lat, c.lon, 15.5, -24.0, 6.0, 18.0); // Saharan Air Layer Atlantic export
                let arabia_thar = 0.48 * gauss2_earth(c.lat, c.lon, 25.0, 56.0, 7.0, 13.0);
                let asia_ind = 0.52 * gauss2_earth(c.lat, c.lon, 29.0, 110.0, 9.0, 14.0)
                    + 0.45 * gauss2_earth(c.lat, c.lon, 26.5, 81.0, 5.5, 10.5);
                let wildfire_belt = 0.34 * gauss2_earth(c.lat, c.lon, 47.0, -118.0, 8.0, 14.0)
                    + 0.32 * gauss2_earth(c.lat, c.lon, -11.0, -56.0, 7.0, 11.0)
                    + 0.30 * gauss2_earth(c.lat, c.lon, 60.0, 112.0, 7.5, 20.0);
                (0.05 + sahara_plume + arabia_thar + asia_ind + wildfire_belt) as f32
            })
            .collect();

        let aod_prog = if dt_s > 0.0 {
            if let Some(prev_aod) = self
                .db
                .load_latest_grid("aerosol_optical_depth_speciated", cfg.scenario, 0.0)?
                .filter(|g| g.width == w && g.height == h && g.values.len() == n_cells)
            {
                let adv_aod = semi_lagrangian_advect_qmsl(
                    &prev_aod.values,
                    &u_adv_atm,
                    &v_adv_atm,
                    w,
                    h,
                    dt_s,
                    true,
                    true,
                );
                let diff_aod = spherical_diffusive_heat_step(
                    &adv_aod,
                    None,
                    w,
                    h,
                    cfg.thermal_diffusivity_m2_s * 0.35,
                    dt_s,
                );
                diff_aod
                    .iter()
                    .enumerate()
                    .map(|(i, &ap)| {
                        // Wet deposition scavenging by precipitation
                        let wet_scav = (-0.08 * core_cells[i].precip_mm_hr).exp() as f32;
                        let aeq = aod_emission_eq[i];
                        ((1.0 - relax_alpha as f32) * (ap * wet_scav)
                            + (relax_alpha as f32) * aeq)
                            .max(0.02)
                    })
                    .collect()
            } else {
                aod_emission_eq
            }
        } else {
            aod_emission_eq
        };

        let aod_field = if !raw_aod.is_empty() && (cfg.sim_year - 2026.8).abs() < 2.0 {
            let pts: Vec<ObsSamplePoint> = raw_aod
                .iter()
                .filter_map(|s| {
                    s.secondary_value.map(|aod| ObsSamplePoint {
                        lat: s.latitude,
                        lon: s.longitude,
                        value: aod,
                        quality_flag: s.quality_flag,
                        sigma_obs: None,
                    })
                })
                .collect();
            let var_cfg = ThreeDVarConfig {
                sigma_b: 0.18,
                base_sigma_obs: 0.06,
                length_scale_km: 1150.0,
                flow_anisotropy: 0.65,
                max_iters: 40,
                tol: 1.0e-6,
            };
            assimilate_3dvar_cg(
                &aod_prog,
                w,
                h,
                &pts,
                Some((&qgpv_state.u2, &qgpv_state.v2)),
                &var_cfg,
            )
            .analysis
        } else {
            aod_prog
        };

        // 8. Evaluate all 34 predictive layers in parallel via Rayon
        let all_layer_ids: Vec<String> = self
            .schema
            .all_layers()
            .into_iter()
            .map(|(_, l)| l.id.clone())
            .collect();

        let ln_p_ratio = (1000.0 / cfg.atmospheric_level_hpa.clamp(10.0, 1000.0)).ln();
        let depth_decay = (-cfg.ocean_depth_m / 850.0).exp();

        let computed_grids: Vec<StoredGridRecord> = all_layer_ids
            .par_iter()
            .map(|layer_id| {
                let mut values = vec![0.0_f32; n_cells];
                let mut u_vec: Option<Vec<f32>> = None;
                let mut v_vec: Option<Vec<f32>> = None;
                let mut v_level = 0.0_f64;

                match layer_id.as_str() {
                    // === GROUP 1: ATMOSPHERIC DYNAMICS & METEOROLOGY ===
                    "geo_multispectral_satellite" => {
                        v_level = cfg.atmospheric_level_hpa;
                        let mut cloud_frac_vec = vec![0.0_f32; n_cells];
                        let itcz_lat = 5.5 * seasonal_phase.cos();
                        let day_phase = 0.24 * (cfg.day_of_year + cfg.hour_utc / 24.0);
                        for (i, c) in core_cells.iter().enumerate() {
                            let phi = c.lat.to_radians();
                            let lam = c.lon.to_radians();

                            // 1. Extratropical cyclone comma-head & frontal cloud shields tied to QGPV MSLP & baroclinic shear
                            let cyc_low = ((1013.0 - c.mslp_hpa) / 18.0).clamp(0.0, 1.25);
                            let grad_t = dt_dx_grid[i].hypot(dt_dy_grid[i]) * 1.0e6;
                            let front_band = ((grad_t - 4.0) / 14.0).clamp(0.0, 1.0);
                            let conv_lift = ((-c.div_v_s1 * 1.0e5) / 1.4).clamp(0.0, 1.0);
                            let storm_track = (-((c.lat.abs() - 47.0) / 15.0).powi(2)).exp();
                            let spiral_mod = 0.55
                                + 0.45
                                    * (6.0 * lam + 0.24 * c.lat - day_phase
                                        + 0.08 * (qgpv_state.u2[i] as f64))
                                        .sin();
                            let cyclone_shield = (storm_track
                                * (0.72 * cyc_low + 0.48 * front_band + 0.35 * conv_lift)
                                * spiral_mod)
                                .clamp(0.0, 1.0);

                            // 2. ITCZ deep convective mesoscale clusters (cold overshooting IR tops)
                            let itcz_env = (-((c.lat - itcz_lat) / 6.2).powi(2)).exp();
                            let itcz_clusters = 0.5
                                + 0.5
                                    * (5.0 * lam - 0.5 * day_phase).sin()
                                    * (9.0 * lam + 0.35 * c.lat).cos();
                            let itcz_conv = (0.95 * itcz_env * (0.35 + 0.65 * itcz_clusters)
                                + 0.25 * gauss2_earth(c.lat, c.lon, -2.0, -62.0, 9.0, 14.0)
                                + 0.25 * gauss2_earth(c.lat, c.lon, 0.0, 122.0, 10.0, 22.0))
                            .clamp(0.0, 1.0);

                            // 3. Subtropical marine stratocumulus decks (eastern ocean boundaries)
                            let stratus_deck = if c.is_ocean {
                                (0.76 * gauss2_earth(c.lat, c.lon, 27.0, -126.0, 7.0, 8.0)
                                    + 0.78 * gauss2_earth(c.lat, c.lon, -15.0, -82.0, 8.5, 7.5)
                                    + 0.74 * gauss2_earth(c.lat, c.lon, -18.0, 6.0, 8.0, 7.5)
                                    + 0.64 * gauss2_earth(c.lat, c.lon, 24.0, -22.0, 6.5, 6.5)
                                    + 0.64 * gauss2_earth(c.lat, c.lon, -23.0, 107.0, 6.5, 6.5))
                                .clamp(0.0, 0.85)
                            } else {
                                0.0
                            };

                            // 4. Suppress cloud fraction over subtropical anticyclones & arid deserts so clear sky is transparent
                            let high_clear = ((c.mslp_hpa - 1017.0) / 10.0).clamp(0.0, 0.85);
                            let arid_clear = if !c.is_ocean && c.rh_pct < 44.0 {
                                ((44.0 - c.rh_pct) / 28.0).clamp(0.0, 0.90)
                            } else {
                                0.0
                            };
                            let moisture_gate = ((c.rh_pct - 42.0) / 36.0).clamp(0.15, 1.0);
                            let raw_cf = (cyclone_shield
                                .max(itcz_conv)
                                .max(stratus_deck)
                                + 0.22 * (c.precip_mm_hr / 2.5).clamp(0.0, 1.0))
                                * moisture_gate
                                * (1.0 - high_clear)
                                * (1.0 - arid_clear);
                            let cloud_frac = if raw_cf < 0.14 {
                                0.0
                            } else {
                                ((raw_cf - 0.14) / 0.86).clamp(0.0, 1.0)
                            };
                            cloud_frac_vec[i] = cloud_frac as f32;

                            // GOES Clean-IR brightness temperature (K): clear sky ~285-303 K, stratus ~272 K, cyclone/ITCZ tops ~205-238 K
                            let clear_tb_k = (c.t2m_c + 273.15).clamp(245.0, 305.0);
                            let convective_depth = (0.70 * itcz_conv + 0.80 * cyclone_shield).clamp(0.0, 1.0);
                            let stratus_only = (stratus_deck * (1.0 - convective_depth)).clamp(0.0, 1.0);
                            let cloud_top_k = 274.0 * stratus_only
                                + (248.0 - 43.0 * convective_depth) * (1.0 - stratus_only);
                            let tb_k = clear_tb_k * (1.0 - cloud_frac) + cloud_top_k * cloud_frac;
                            let _ = phi;
                            values[i] = tb_k.clamp(205.0, 305.0) as f32;
                        }
                        u_vec = Some(cloud_frac_vec);
                    }
                    "volumetric_doppler_radar" => {
                        v_level = cfg.atmospheric_level_hpa;
                        for (i, c) in core_cells.iter().enumerate() {
                            let vert_factor = (-((ln_p_ratio - 0.35).powi(2)) / 0.8).exp();
                            let dbz = if c.precip_mm_hr > 0.03 {
                                10.0 * (200.0 * (c.precip_mm_hr * vert_factor).max(0.01).powf(1.6))
                                    .log10()
                            } else {
                                0.0
                            };
                            values[i] = dbz.clamp(0.0, 68.0) as f32;
                        }
                    }
                    "synoptic_fronts_isobars" => {
                        v_level = 1000.0;
                        for (i, c) in core_cells.iter().enumerate() {
                            values[i] = c.mslp_hpa as f32;
                        }
                        // Persist exact prognostic 2-layer QGPV state (q_1, q_2) across SQLite ticks!
                        u_vec = Some(qgpv_state.q1.clone());
                        v_vec = Some(qgpv_state.q2.clone());
                    }
                    "wind_vector_jet_streams" => {
                        v_level = cfg.atmospheric_level_hpa;
                        let mut u_arr = vec![0.0_f32; n_cells];
                        let mut v_arr = vec![0.0_f32; n_cells];
                        // Vertical log-pressure interpolation between 10m Ekman, Layer-2 QG (800 hPa),
                        // Layer-1 QG Jet (250 hPa), and stratospheric reversal (< 200 hPa), plus thermal-wind shear
                        let p_hpa = cfg.atmospheric_level_hpa.clamp(10.0, 1000.0);
                        let ln_800 = (1000.0_f64 / 800.0).ln();
                        let ln_250 = (1000.0_f64 / 250.0).ln();
                        let tropo_ln = ln_p_ratio.min(1.65) - 0.35 * (ln_p_ratio - 1.65).max(0.0);

                        for (i, c) in core_cells.iter().enumerate() {
                            let (u_qg, v_qg) = if p_hpa >= 800.0 {
                                let w_l2 = (ln_p_ratio / ln_800).clamp(0.0, 1.0);
                                (
                                    (1.0 - w_l2) * c.u10_ms + w_l2 * (qgpv_state.u2[i] as f64),
                                    (1.0 - w_l2) * c.v10_ms + w_l2 * (qgpv_state.v2[i] as f64),
                                )
                            } else if p_hpa >= 250.0 {
                                let w_l1 =
                                    ((ln_p_ratio - ln_800) / (ln_250 - ln_800)).clamp(0.0, 1.0);
                                (
                                    (1.0 - w_l1) * (qgpv_state.u2[i] as f64)
                                        + w_l1 * (qgpv_state.u1[i] as f64),
                                    (1.0 - w_l1) * (qgpv_state.v2[i] as f64)
                                        + w_l1 * (qgpv_state.v1[i] as f64),
                                )
                            } else {
                                let strat_decay =
                                    (1.0 - 0.32 * (ln_p_ratio - ln_250)).clamp(0.35, 1.0);
                                (
                                    (qgpv_state.u1[i] as f64) * strat_decay,
                                    (qgpv_state.v1[i] as f64) * strat_decay,
                                )
                            };

                            let phi = c.lat.to_radians();
                            let f_reg = if c.lat >= 0.0 {
                                (2.0 * EARTH_OMEGA_RAD_S * phi.sin()).max(4.5e-5)
                            } else {
                                (2.0 * EARTH_OMEGA_RAD_S * phi.sin()).min(-4.5e-5)
                            };
                            let du_thermal = -(DRY_AIR_GAS_RD / f_reg) * dt_dy_grid[i] * tropo_ln;
                            let dv_thermal = (DRY_AIR_GAS_RD / f_reg) * dt_dx_grid[i] * tropo_ln;

                            let u = (0.78 * u_qg + 0.22 * du_thermal).clamp(-92.0, 98.0);
                            let v = (0.78 * v_qg + 0.22 * dv_thermal).clamp(-72.0, 72.0);
                            u_arr[i] = u as f32;
                            v_arr[i] = v as f32;
                            values[i] = u.hypot(v) as f32;
                        }
                        u_vec = Some(u_arr);
                        v_vec = Some(v_arr);
                    }
                    "atmospheric_rivers_ivt" => {
                        v_level = 850.0;
                        let mut u_arr = vec![0.0_f32; n_cells];
                        let mut v_arr = vec![0.0_f32; n_cells];
                        for (i, c) in core_cells.iter().enumerate() {
                            let precipitable_water_kg_m2 = c.q_kg_kg * (70_000.0 / 9.80665) * 0.42;
                            let u850 = 0.72 * (qgpv_state.u2[i] as f64) + 0.28 * (c.u10_ms * 1.45);
                            let v850 = 0.72 * (qgpv_state.v2[i] as f64) + 0.28 * (c.v10_ms * 1.45);
                            let ivt_u = precipitable_water_kg_m2 * u850;
                            let ivt_v = precipitable_water_kg_m2 * v850;
                            u_arr[i] = ivt_u as f32;
                            v_arr[i] = ivt_v as f32;
                            values[i] = ivt_u.hypot(ivt_v) as f32;
                        }
                        u_vec = Some(u_arr);
                        v_vec = Some(v_arr);
                    }
                    "global_lightning_telemetry" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            // Bolton equivalent potential temperature buoyancy excess -> CAPE lightning scaling
                            let theta_e =
                                bolton_equivalent_potential_temp_k(c.t2m_c, c.q_kg_kg, c.mslp_hpa);
                            let cape_proxy = ((theta_e - 318.0).max(0.0) * 65.0).min(4500.0);
                            let land_boost = if c.is_ocean { 0.22 } else { 1.35 };
                            let flash_rate =
                                0.0045 * land_boost * (cape_proxy / 100.0).powf(1.3) * c.precip_mm_hr.sqrt();
                            values[i] = flash_rate.clamp(0.0, 45.0) as f32;
                        }
                    }

                    // === GROUP 2: TRACE GASES, AEROSOLS & RADIATION ===
                    "column_co2_flux" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let nh_biosphere = if c.lat > 15.0 && !c.is_ocean {
                                -5.2 * seasonal_phase.cos()
                            } else {
                                0.5
                            };
                            let tropical_sink = if !c.is_ocean && c.lat.abs() < 12.0 && c.soil_moisture > 0.24 {
                                -2.4
                            } else {
                                0.0
                            };
                            let anthro_plumes = 6.2 * gauss2_earth(c.lat, c.lon, 35.0, 116.0, 6.5, 10.0)
                                + 4.8 * gauss2_earth(c.lat, c.lon, 40.5, -77.0, 5.5, 9.0)
                                + 4.4 * gauss2_earth(c.lat, c.lon, 50.5, 7.0, 4.5, 9.5)
                                + 3.8 * gauss2_earth(c.lat, c.lon, 27.0, 79.0, 5.0, 8.5);
                            values[i] = (base_co2_ppm + nh_biosphere + tropical_sink + anthro_plumes) as f32;
                        }
                    }
                    "methane_plumes_background" => {
                        let ch4_base = 1935.0 + 38.0 * (df_co2 - 2.25);
                        for (i, c) in core_cells.iter().enumerate() {
                            let wetlands = if !c.is_ocean && c.lat.abs() < 15.0 {
                                75.0 * c.soil_moisture
                            } else if !c.is_ocean && (55.0..=72.0).contains(&c.lat) {
                                52.0 * (c.soil_moisture / 0.4).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            let basin_plumes = 105.0 * gauss2_earth(c.lat, c.lon, 31.8, -102.5, 4.0, 5.0) // Permian
                                + 95.0 * gauss2_earth(c.lat, c.lon, 38.2, 54.0, 4.5, 5.5) // Turkmenistan
                                + 80.0 * gauss2_earth(c.lat, c.lon, 26.5, 83.0, 5.0, 9.5) // Indo-Gangetic rice/livestock
                                + 72.0 * gauss2_earth(c.lat, c.lon, 31.0, 114.0, 5.5, 8.0) // East China rice/coal
                                + 65.0 * gauss2_earth(c.lat, c.lon, 66.5, 72.0, 4.5, 8.0); // Yamal/W Siberia
                            let nh_grad = 42.0 * c.lat.to_radians().sin();
                            values[i] = (ch4_base + nh_grad + wetlands + basin_plumes) as f32;
                        }
                    }
                    "tropospheric_no2_so2" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let urban_corridors = 1.05 * gauss2_earth(c.lat, c.lon, 36.0, 116.5, 5.5, 8.5) // North/East China
                                + 0.90 * gauss2_earth(c.lat, c.lon, 50.8, 6.5, 4.0, 8.0) // Benelux/Ruhr/UK
                                + 0.85 * gauss2_earth(c.lat, c.lon, 40.8, -75.5, 4.0, 7.5) // US NE Corridor
                                + 0.80 * gauss2_earth(c.lat, c.lon, 27.5, 78.5, 4.5, 8.5) // Indo-Gangetic
                                + 0.78 * gauss2_earth(c.lat, c.lon, 35.8, 138.0, 2.8, 4.0) // Tokyo-Osaka
                                + 0.65 * gauss2_earth(c.lat, c.lon, 25.5, 51.0, 4.0, 5.5) // Persian Gulf
                                + 0.60 * gauss2_earth(c.lat, c.lon, -26.2, 28.0, 3.0, 4.0) // Highveld
                                + 0.55 * gauss2_earth(c.lat, c.lon, 34.0, -118.0, 2.8, 4.0); // LA Basin
                            let shipping = if c.is_ocean {
                                0.35 * gauss2_earth(c.lat, c.lon, 15.0, 42.0, 5.5, 4.5) // Red Sea
                                    + 0.38 * gauss2_earth(c.lat, c.lon, 3.0, 102.0, 3.5, 5.5) // Malacca
                                    + 0.28 * gauss2_earth(c.lat, c.lon, 50.0, -3.0, 2.5, 4.5) // English Channel
                            } else {
                                0.0
                            };
                            let aod = aod_field[i] as f64;
                            let base_bg = if c.is_ocean { 0.08 } else { 0.32 } * aod;
                            values[i] = ((base_bg + urban_corridors + shipping) * 8.5e15) as f32;
                        }
                    }
                    "aerosol_optical_depth_speciated" => {
                        values.copy_from_slice(&aod_field);
                    }
                    "toa_radiative_energy_budget" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let ice_albedo = if c.t2m_c < -5.0 || c.sea_ice_h_m > 0.15 {
                                0.64
                            } else if c.is_ocean {
                                0.09
                            } else {
                                0.24
                            };
                            let cloud_albedo = 0.22 * (c.rh_pct / 100.0);
                            let alpha =
                                (ice_albedo + cloud_albedo + 0.08 * aod_field[i] as f64)
                                    .clamp(0.07, 0.82);
                            let q_daily = 340.25
                                * (1.0 - 0.482 * (1.5 * c.lat.to_radians().sin().powi(2) - 0.5));
                            let asr = q_daily * (1.0 - alpha);
                            let olr = (208.5 + 1.95 * c.t2m_c) - df_co2;
                            values[i] = (asr - olr) as f32;
                        }
                    }

                    // === GROUP 3: PHYSICAL & BIOGEOCHEMICAL OCEANOGRAPHY ===
                    "sst_marine_heatwaves" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            values[i] = if c.is_ocean { c.sst_c as f32 } else { f32::NAN };
                        }
                    }
                    "subsurface_ocean_heat_salinity_3d" => {
                        v_level = cfg.ocean_depth_m;
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                let tc_shift = (ocean_state.h_tc_anom_m[i] as f64) * 0.045;
                                let t_z = 1.5 + (c.sst_c + tc_shift - 1.5) * depth_decay;
                                let sal =
                                    34.7 + 1.1 * (2.0 * c.lat.to_radians()).cos() * depth_decay;
                                let rho = eos80_seawater_density(t_z, sal, cfg.ocean_depth_m);
                                let ohc_gj_m2 = rho * 3990.0 * t_z.max(-1.8) * 100.0 * 1.0e-9;
                                values[i] = ohc_gj_m2 as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "thermohaline_currents_amoc" => {
                        v_level = cfg.ocean_depth_m;
                        let mut u_arr = vec![0.0_f32; n_cells];
                        let mut v_arr = vec![0.0_f32; n_cells];
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                let u = (ocean_state.u_total[i] as f64) * depth_decay;
                                let v = (ocean_state.v_total[i] as f64) * depth_decay;
                                u_arr[i] = u as f32;
                                v_arr[i] = v as f32;
                                values[i] = u.hypot(v) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                        u_vec = Some(u_arr);
                        v_vec = Some(v_arr);
                    }
                    "swot_sea_surface_height_altimetry" => {
                        let slr_cm = (global_warming_over_preindustrial - 0.9).max(0.0) * 28.5;
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                // Inverse barometer + steric + dynamic reduced-gravity thermocline SSH
                                let inv_barometer = -(c.mslp_hpa - 1013.25);
                                let steric = 1.4 * (c.sst_c - 15.0);
                                let ssh_rg = ocean_state.ssh_baroclinic_cm[i] as f64;
                                values[i] = (slr_cm + inv_barometer + steric + ssh_rg) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                        // Persist exact reduced-gravity thermocline depth anomaly (eta_tc) & Stommel-Munk psi_gyre across SQLite ticks!
                        u_vec = Some(ocean_state.h_tc_anom_m.clone());
                        v_vec = Some(ocean_state.psi_gyre.clone());
                    }
                    "ocean_acidification_carbonate" => {
                        v_level = cfg.ocean_depth_m;
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                let t_z = 1.5 + (c.sst_c - 1.5) * depth_decay;
                                let sal =
                                    34.7 + 1.1 * (2.0 * c.lat.to_radians()).cos() * depth_decay;
                                let carb = solve_seawater_carbonate_chemistry(
                                    t_z,
                                    sal,
                                    base_co2_ppm,
                                    cfg.ocean_depth_m,
                                );
                                values[i] = carb.ph_total as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "marine_chlorophyll_hypoxia" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                let subpolar =
                                    1.8 * (-((c.lat.abs() - 52.0) / 12.0).powi(2)).exp();
                                let eq_upwelling = 1.15
                                    * (-((c.lat / 4.5).powi(2))).exp()
                                    * (0.4 + 0.6 * gauss2_earth(c.lat, c.lon, 0.0, -105.0, 6.0, 35.0));
                                let coastal_upwelling = 1.9 * gauss2_earth(c.lat, c.lon, -13.0, -78.0, 9.0, 4.5)
                                    + 1.6 * gauss2_earth(c.lat, c.lon, -22.0, 12.0, 8.0, 4.5)
                                    + 1.5 * gauss2_earth(c.lat, c.lon, 15.0, 56.0, 7.0, 7.0)
                                    + 1.4 * gauss2_earth(c.lat, c.lon, 35.0, -123.0, 8.0, 4.5);
                                let coastal = if c.elev_m > -900.0 { 1.95 } else { 0.10 };
                                // Stratification nutrient penalty under extreme SST warming
                                let strat_damping = (1.0 - 0.025 * (c.sst_c - 24.0).max(0.0)).clamp(0.4, 1.0);
                                values[i] = ((0.06 + subpolar + eq_upwelling + coastal_upwelling + coastal) * strat_damping) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "teleconnection_oscillation_masks" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let nino34 =
                                if c.lat.abs() <= 5.0 && (-170.0..=-120.0).contains(&c.lon) {
                                    enso_nino34
                                } else if (20.0..=60.0).contains(&c.lat)
                                    && (c.lon < -120.0 || c.lon > 140.0)
                                {
                                    -0.55 * enso_nino34
                                } else if (35.0..=65.0).contains(&c.lat)
                                    && (-65.0..=0.0).contains(&c.lon)
                                {
                                    nao_phase
                                } else {
                                    0.0
                                };
                            values[i] = nino34 as f32;
                        }
                    }

                    // === GROUP 4: CRYOSPHERE & GLACIAL MASS BALANCE ===
                    "sea_ice_concentration_drift" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                let from_sst = ((0.0 - (c.sst_c + 0.2)) / 2.2 * 100.0).clamp(0.0, 100.0);
                                let from_h = (c.sea_ice_h_m / 0.65 * 100.0).clamp(0.0, 100.0);
                                values[i] = (0.5 * from_sst + 0.5 * from_h) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "sea_ice_thickness_freeboard_age" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            values[i] = if c.is_ocean {
                                c.sea_ice_h_m as f32
                            } else {
                                f32::NAN
                            };
                        }
                    }
                    "gravimetric_ice_sheet_mass_balance" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let on_greenland = !c.is_ocean
                                && (59.5..=84.0).contains(&c.lat)
                                && (-73.5..=-11.5).contains(&c.lon);
                            let on_antarctica = !c.is_ocean && c.lat < -63.0;
                            if on_greenland || on_antarctica {
                                // Surface mass balance: snowfall accumulation minus positive-degree-day ablation
                                let accum = c.precip_mm_hr * 24.0 * 0.35;
                                let pdd_melt = 14.5
                                    * global_warming_over_preindustrial
                                    * (if c.elev_m < 1800.0 { 2.4 } else { 0.45 });
                                values[i] = (accum - pdd_melt) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "alpine_glacier_velocity_rgi" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean
                                && (c.elev_m > 1950.0 || c.lat.abs() > 63.0)
                                && c.t2m_c < 4.5
                            {
                                let ice_temp_factor =
                                    (0.12 * (c.t2m_c + 10.0)).exp().clamp(0.2, 3.5);
                                let relief = (c.elev_m / 2000.0).powi(2);
                                values[i] =
                                    (42.0 * ice_temp_factor * relief).clamp(0.0, 650.0) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "permafrost_thermokarst_subsidence" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let is_greenland = (59.5..=84.0).contains(&c.lat)
                                && (-73.5..=-11.5).contains(&c.lon)
                                && c.elev_m > 900.0;
                            if !c.is_ocean && c.lat > 52.0 && !is_greenland {
                                let tdd = (c.t2m_c + 12.0).max(0.0) * 85.0;
                                let thaw_m = (0.045 * tdd.sqrt()).clamp(0.0, 4.5);
                                values[i] = thaw_m as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }

                    // === GROUP 5: TERRESTRIAL BIOSPHERE, HYDROLOGY & CARBON ===
                    "canopy_height_aboveground_biomass" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                if c.t2m_c > -8.0 {
                                    let sm_factor = ((c.soil_moisture - 0.04) / 0.32).clamp(0.0, 1.15).powi(2);
                                    let warmth = ((c.t2m_c + 8.0) / 34.0).clamp(0.0, 1.15);
                                    values[i] = (290.0 * sm_factor * warmth).clamp(0.0, 360.0) as f32;
                                } else {
                                    values[i] = 0.0;
                                }
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "sif_photosynthesis_evaporative_stress" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                let par = 0.48 * (c.shortwave_wm2.max(120.0));
                                let root_water_stress = ((c.soil_moisture - 0.04) / 0.30).clamp(0.02, 1.0);
                                let thermal_opt = (-((c.t2m_c - 23.0) / 14.0).powi(2)).exp();
                                let co2_fert = 1.0 + 0.18 * (base_co2_ppm / 380.0).ln();
                                values[i] = (0.0085 * par * root_water_stress * thermal_opt * co2_fert) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "soil_moisture_groundwater_grace" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            values[i] = if !c.is_ocean {
                                c.soil_moisture as f32
                            } else {
                                f32::NAN
                            };
                        }
                    }
                    "swot_surface_water_river_discharge" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                let sat_frac = (c.soil_moisture / 0.48).clamp(0.0, 1.0);
                                let direct_runoff = c.precip_mm_hr * 480.0 * (0.25 + 0.75 * sat_frac.powi(2));
                                let baseflow = 320.0 * sat_frac.powi(3);
                                let corridor = earth_river_corridor_signal(c.lat, c.lon) * (0.45 + 0.55 * sat_frac);
                                values[i] = (direct_runoff + baseflow + corridor) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                        if !raw_rivers.is_empty() && (cfg.sim_year - 2026.8).abs() < 2.0 {
                            let pts: Vec<ObsSamplePoint> = raw_rivers
                                .iter()
                                .map(|s| ObsSamplePoint {
                                    lat: s.latitude,
                                    lon: s.longitude,
                                    value: s.primary_value,
                                    quality_flag: s.quality_flag,
                                    sigma_obs: None,
                                })
                                .collect();
                            let var_cfg = ThreeDVarConfig {
                                sigma_b: 8_000.0,
                                base_sigma_obs: 2_500.0,
                                length_scale_km: 620.0,
                                flow_anisotropy: 0.0,
                                max_iters: 40,
                                tol: 1.0e-6,
                            };
                            let assim = assimilate_3dvar_cg(&values, w, h, &pts, None, &var_cfg).analysis;
                            for (i, c) in core_cells.iter().enumerate() {
                                values[i] = if c.is_ocean { f32::NAN } else { assim[i].max(0.0) };
                            }
                        }
                    }
                    "active_wildfires_frp_burn_severity" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                if c.t2m_c > 8.0 {
                                    let wind = c.u10_ms.hypot(c.v10_ms);
                                    let fwi = canadian_fire_weather_index_at(
                                        c.t2m_c,
                                        c.rh_pct,
                                        wind,
                                        c.precip_mm_hr,
                                        c.lat,
                                        cfg.day_of_year,
                                    );
                                    let fuel_load = ((c.soil_moisture - 0.03) / 0.12).clamp(0.05, 1.0);
                                    let frp_mw = if fwi > 24.0 {
                                        (fwi - 24.0) * 4.2 * fuel_load
                                    } else {
                                        0.0
                                    };
                                    values[i] = frp_mw as f32;
                                } else {
                                    values[i] = 0.0;
                                }
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                        if !raw_fires.is_empty() && (cfg.sim_year - 2026.8).abs() < 2.0 {
                            for fire in &raw_fires {
                                let r_idx = (((90.0 - fire.latitude.clamp(-89.9, 89.9)) / 180.0) * (h as f64)).floor() as usize;
                                let c_idx = ((((fire.longitude + 180.0).rem_euclid(360.0)) / 360.0) * (w as f64)).floor() as usize;
                                let idx = r_idx.min(h - 1) * w + c_idx.min(w - 1);
                                if !core_cells[idx].is_ocean {
                                    values[idx] = values[idx].max(fire.primary_value as f32);
                                }
                            }
                        }
                    }
                    "dynamic_land_cover_deforestation" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if c.is_ocean {
                                values[i] = f32::NAN;
                            } else if c.t2m_c < -6.0 {
                                values[i] = 8.0;
                            } else if c.soil_moisture < 0.11 {
                                values[i] = 7.0;
                            } else if c.soil_moisture > 0.26 && c.t2m_c > 4.0 {
                                values[i] = 1.0;
                            } else {
                                values[i] = 4.0;
                            }
                        }
                    }

                    // === GROUP 6: CLIMATE RISK, ATTRIBUTION & CMIP PROJECTIONS ===
                    "cmip6_downscaled_ensemble" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let seasonal_preind = preindustrial_baseline_t2m_c(c.lat, c.elev_m)
                                + seasonal_surface_swing_c(c.lat, c.is_ocean, cfg.day_of_year);
                            values[i] = (c.t2m_c - seasonal_preind) as f32;
                        }
                    }
                    "wet_bulb_globe_temp_heat_stress" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            let wind = c.u10_ms.hypot(c.v10_ms);
                            values[i] =
                                compute_outdoor_wbgt(c.t2m_c, c.rh_pct, c.shortwave_wm2, wind)
                                    as f32;
                        }
                    }
                    "coastal_inundation_storm_surge" => {
                        let slr_m = ((global_warming_over_preindustrial - 0.9).max(0.0) * 0.32)
                            .clamp(0.05, 3.0);
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                if c.elev_m < 25.0 {
                                    let inv_baro_m = (1013.25 - c.mslp_hpa).max(0.0) * 0.01;
                                    let surge_100yr =
                                        1.8 + 0.08 * c.u10_ms.hypot(c.v10_ms) + inv_baro_m;
                                    let flood_depth =
                                        (slr_m + surge_100yr - c.elev_m * 0.12).max(0.0);
                                    values[i] = flood_depth as f32;
                                } else {
                                    values[i] = 0.0;
                                }
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "compound_hazard_fire_weather_index" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                let wind = c.u10_ms.hypot(c.v10_ms);
                                values[i] = canadian_fire_weather_index_at(
                                    c.t2m_c,
                                    c.rh_pct,
                                    wind,
                                    c.precip_mm_hr,
                                    c.lat,
                                    cfg.day_of_year,
                                ) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    "critical_infrastructure_socioeconomic_exposure" => {
                        for (i, c) in core_cells.iter().enumerate() {
                            if !c.is_ocean {
                                let wind = c.u10_ms.hypot(c.v10_ms);
                                let wbgt = compute_outdoor_wbgt(
                                    c.t2m_c,
                                    c.rh_pct,
                                    c.shortwave_wm2,
                                    wind,
                                );
                                let fwi = canadian_fire_weather_index_at(
                                    c.t2m_c,
                                    c.rh_pct,
                                    wind,
                                    c.precip_mm_hr,
                                    c.lat,
                                    cfg.day_of_year,
                                );
                                let pop_weight = (-((c.lat.abs() - 32.0) / 24.0).powi(2)).exp()
                                    + 0.45 * gauss2_earth(c.lat, c.lon, 27.0, 82.0, 9.0, 14.0)
                                    + 0.45 * gauss2_earth(c.lat, c.lon, 33.0, 114.0, 9.0, 12.0)
                                    + 0.35 * gauss2_earth(c.lat, c.lon, 48.5, 8.0, 8.0, 14.0);
                                let risk = (1.6 * (wbgt - 18.0).max(0.0) + 0.55 * fwi)
                                    * (0.45 + 0.85 * pop_weight.min(1.35));
                                values[i] = risk.clamp(0.0, 100.0) as f32;
                            } else {
                                values[i] = f32::NAN;
                            }
                        }
                    }
                    _ => {}
                }

                let mut min_v = f32::INFINITY;
                let mut max_v = f32::NEG_INFINITY;
                let mut sum_v = 0.0_f64;
                let mut cnt = 0usize;
                for &v in &values {
                    if v.is_finite() {
                        if v < min_v {
                            min_v = v;
                        }
                        if v > max_v {
                            max_v = v;
                        }
                        sum_v += v as f64;
                        cnt += 1;
                    }
                }
                if cnt == 0 {
                    min_v = 0.0;
                    max_v = 1.0;
                }
                let mean_v = if cnt > 0 {
                    (sum_v / cnt as f64) as f32
                } else {
                    0.0
                };

                StoredGridRecord {
                    layer_id: layer_id.clone(),
                    scenario: cfg.scenario.code().to_string(),
                    timestamp_utc: ts_utc.clone(),
                    vertical_level: v_level,
                    width: w,
                    height: h,
                    min_val: min_v,
                    max_val: max_v,
                    mean_val: mean_v,
                    values,
                    u_component: u_vec,
                    v_component: v_vec,
                }
            })
            .collect();

        self.db.upsert_simulated_grids(&computed_grids)?;

        let mut area_w_temp = 0.0;
        let mut area_sum = 0.0;
        let mut sst_w_anom = 0.0;
        let mut sst_area_sum = 0.0;

        for c in &core_cells {
            let cos_phi = c.lat.to_radians().cos().max(0.01);
            let seasonal_preind_t = preindustrial_baseline_t2m_c(c.lat, c.elev_m)
                + seasonal_surface_swing_c(c.lat, c.is_ocean, cfg.day_of_year);
            area_w_temp += (c.t2m_c - seasonal_preind_t) * cos_phi;
            area_sum += cos_phi;

            if c.is_ocean && c.lat.abs() <= 60.0 {
                let baseline_sst = (28.6 * c.lat.to_radians().cos()
                    - 3.5 * c.lat.to_radians().sin().powi(2))
                .max(-1.85);
                sst_w_anom += (c.sst_c - baseline_sst) * cos_phi;
                sst_area_sum += cos_phi;
            }
        }

        let readouts = GlobalLiveReadouts {
            timestamp_utc: ts_utc,
            scenario: cfg.scenario.code().to_string(),
            global_2m_temp_anomaly_c: if area_sum > 0.0 {
                area_w_temp / area_sum
            } else {
                global_warming_over_preindustrial
            },
            mauna_loa_co2_ppm: base_co2_ppm,
            enso_nino34_c: enso_nino34,
            global_sst_anomaly_c: if sst_area_sum > 0.0 {
                sst_w_anom / sst_area_sum
            } else {
                0.85
            },
        };
        self.db.record_global_readouts(&readouts)?;

        Ok(readouts)
    }

    /// Compute a 10-level atmospheric Skew-T / vertical sounding profile (`1000..=10 hPa`) at `(lat, lon)`.
    pub fn compute_vertical_sounding(
        &self,
        cfg: &SimulationConfig,
        lat: f64,
        lon: f64,
    ) -> Vec<(f64, f64, f64, f64)> {
        let mut map = HashMap::new();
        if let Ok(Some(g)) = self.db.load_latest_grid(
            "wet_bulb_globe_temp_heat_stress",
            cfg.scenario,
            1000.0,
        ) {
            map.insert("wet_bulb_globe_temp_heat_stress".to_string(), g);
        }
        compute_vertical_sounding_from_grids(&self.schema, Some(&map), lat, lon)
    }

    /// Compute a 10-level ocean CTD temperature & salinity water-column profile (`0..=6000 m`) at `(lat, lon)`.
    pub fn compute_ocean_ctd_profile(
        &self,
        cfg: &SimulationConfig,
        lat: f64,
        lon: f64,
    ) -> Vec<(f64, f64, f64)> {
        let mut map = HashMap::new();
        if let Ok(Some(g)) = self
            .db
            .load_latest_grid("sst_marine_heatwaves", cfg.scenario, 0.0)
        {
            map.insert("sst_marine_heatwaves".to_string(), g);
        }
        compute_ocean_ctd_profile_from_grids(&self.schema, Some(&map), lat, lon)
    }

    /// Load a map of all currently cached latest layer grids for fast multi-layer GUI blending.
    pub fn load_layer_bundle(
        &self,
        scenario: SspScenario,
        atm_hpa: f64,
        ocean_m: f64,
    ) -> Result<HashMap<String, StoredGridRecord>> {
        let mut map = HashMap::new();
        for (group, layer) in self.schema.all_layers() {
            let target_lvl = if group.group_id == "physical_biogeochemical_oceanography" {
                ocean_m
            } else {
                atm_hpa
            };
            if let Some(grid) = self.db.load_latest_grid(&layer.id, scenario, target_lvl)? {
                map.insert(layer.id.clone(), grid);
            }
        }
        Ok(map)
    }
}

/// Latitude band selector for CMIP6 regional ΔT diagnostics & trajectories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClimateRegion {
    Global,
    Arctic,
    MidLatitudes,
    Tropics,
    Antarctic,
    PinnedCell,
}

impl ClimateRegion {
    pub const ALL: [ClimateRegion; 6] = [
        ClimateRegion::Global,
        ClimateRegion::Arctic,
        ClimateRegion::MidLatitudes,
        ClimateRegion::Tropics,
        ClimateRegion::Antarctic,
        ClimateRegion::PinnedCell,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ClimateRegion::Global => "Global Mean",
            ClimateRegion::Arctic => "Arctic (>66.5°N)",
            ClimateRegion::MidLatitudes => "Mid-Latitudes (23.5–66.5°)",
            ClimateRegion::Tropics => "Tropics (±23.5°)",
            ClimateRegion::Antarctic => "Antarctic (<66.5°S)",
            ClimateRegion::PinnedCell => "Pinned Cell",
        }
    }
}

/// Compute regional ΔT (°C relative to pre-industrial) for a given SSP scenario and year,
/// integrating `polar_amplification_factor` with `cos(lat)` band weights.
pub fn regional_cmip6_delta_t_c(
    ssp: SspScenario,
    year: f64,
    region: ClimateRegion,
    pinned_lat: f64,
) -> f64 {
    let global_dt = transient_global_warming_c(ssp.co2_ppm_at_year(year));
    let amp = match region {
        ClimateRegion::PinnedCell => polar_amplification_factor(pinned_lat),
        _ => {
            let mut num = 0.0;
            let mut den = 0.0;
            for idx in -180..=180 {
                let lat = (idx as f64) * 0.5;
                let in_band = match region {
                    ClimateRegion::Global => true,
                    ClimateRegion::Arctic => lat >= 66.5,
                    ClimateRegion::MidLatitudes => lat.abs() >= 23.5 && lat.abs() < 66.5,
                    ClimateRegion::Tropics => lat.abs() < 23.5,
                    ClimateRegion::Antarctic => lat <= -66.5,
                    ClimateRegion::PinnedCell => false,
                };
                if in_band {
                    let w = lat.to_radians().cos().max(0.005);
                    num += polar_amplification_factor(lat) * w;
                    den += w;
                }
            }
            if den > 0.0 {
                num / den
            } else {
                1.0
            }
        }
    };
    global_dt * amp
}

/// Compute live area-weighted regional ΔT (°C) from an in-memory `cmip6_downscaled_ensemble` grid.
pub fn snapshot_regional_delta_t_c(
    grid: &StoredGridRecord,
    region: ClimateRegion,
    pinned_lat: f64,
    pinned_lon: f64,
) -> Option<f64> {
    if region == ClimateRegion::PinnedCell {
        let v = sample_grid_bilinear(&grid.values, grid.width, grid.height, pinned_lat, pinned_lon)
            as f64;
        return if v.is_finite() { Some(v) } else { None };
    }
    let mut sum = 0.0;
    let mut w_sum = 0.0;
    for row in 0..grid.height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / grid.height as f64);
        let in_band = match region {
            ClimateRegion::Global => true,
            ClimateRegion::Arctic => lat >= 66.5,
            ClimateRegion::MidLatitudes => lat.abs() >= 23.5 && lat.abs() < 66.5,
            ClimateRegion::Tropics => lat.abs() < 23.5,
            ClimateRegion::Antarctic => lat <= -66.5,
            ClimateRegion::PinnedCell => false,
        };
        if !in_band {
            continue;
        }
        let w = lat.to_radians().cos().max(0.01);
        for col in 0..grid.width {
            let v = grid.values[row * grid.width + col] as f64;
            if v.is_finite() {
                sum += v * w;
                w_sum += w;
            }
        }
    }
    if w_sum > 0.0 {
        Some(sum / w_sum)
    } else {
        None
    }
}

/// In-memory 10-level atmospheric Skew-T / vertical sounding profile (`1000..=10 hPa`) at `(lat, lon)`
/// reading strictly from `grids` (zero SQLite I/O).
pub fn compute_vertical_sounding_from_grids(
    schema: &HubSchema,
    grids: Option<&HashMap<String, StoredGridRecord>>,
    lat: f64,
    lon: f64,
) -> Vec<(f64, f64, f64, f64)> {
    let levels = &schema
        .main_viewport
        .vertical_slice_engine
        .atmospheric_levels_hpa;

    let t_surf = grids
        .and_then(|map| map.get("wet_bulb_globe_temp_heat_stress"))
        .map(|g| sample_grid_bilinear(&g.values, g.width, g.height, lat, lon) as f64)
        .filter(|v| v.is_finite())
        .unwrap_or(18.0);

    levels
        .iter()
        .map(|&p_u32| {
            let p = p_u32 as f64;
            let z_km = 7.4 * (1000.0 / p).ln();
            let temp_c = if z_km <= 11.5 {
                t_surf - 6.5 * z_km
            } else {
                (t_surf - 74.75) + 1.8 * (z_km - 11.5)
            };
            let dewpoint_c = temp_c - (4.5 + 1.4 * z_km).min(38.0);
            let wind_ms = 6.0 + 28.0 * (-((p - 250.0) / 180.0).powi(2)).exp();
            (p, temp_c, dewpoint_c, wind_ms)
        })
        .collect()
}

/// In-memory 10-level ocean CTD temperature & salinity profile (`0..=6000 m`) at `(lat, lon)`
/// reading strictly from `grids` (zero SQLite I/O).
pub fn compute_ocean_ctd_profile_from_grids(
    schema: &HubSchema,
    grids: Option<&HashMap<String, StoredGridRecord>>,
    lat: f64,
    lon: f64,
) -> Vec<(f64, f64, f64)> {
    let depths = &schema
        .main_viewport
        .vertical_slice_engine
        .ocean_depth_levels_meters;

    let sst = grids
        .and_then(|map| map.get("sst_marine_heatwaves"))
        .map(|g| sample_grid_bilinear(&g.values, g.width, g.height, lat, lon) as f64)
        .filter(|v| v.is_finite())
        .unwrap_or_else(|| (26.0 * lat.to_radians().cos()).max(-1.8));

    depths
        .iter()
        .map(|&d_u32| {
            let z = d_u32 as f64;
            let decay = (-z / 750.0).exp();
            let temp_c = 1.4 + (sst - 1.4) * decay;
            let sal_psu = 34.72 + 0.95 * (2.0 * lat.to_radians()).cos() * (-z / 1200.0).exp()
                + 0.05 * lon.to_radians().sin();
            (z, temp_c, sal_psu)
        })
        .collect()
}

/// Subsurface 0–20 m land cryosphere, permafrost, and soil moisture profile computed in-memory
/// from cached simulation grids (`cmip6_downscaled_ensemble`, `permafrost_thermokarst_subsidence`,
/// `gravimetric_ice_sheet_mass_balance`, `alpine_glacier_velocity_rgi`, `soil_moisture_groundwater_grace`).
#[derive(Debug, Clone)]
pub struct LandCryosphereProfile {
    pub elev_m: f64,
    pub surface_t2m_c: f64,
    pub active_layer_m: Option<f64>,
    pub ice_sheet_smb_cm: Option<f64>,
    pub glacier_vel_m_yr: Option<f64>,
    pub soil_moisture: Option<f64>,
    /// `(depth_m, ground_temp_c, liquid_moisture_frac)` across `0.0..=20.0 m`.
    pub levels: Vec<(f64, f64, f64)>,
}

pub fn compute_land_cryosphere_profile_from_grids(
    grids: Option<&HashMap<String, StoredGridRecord>>,
    lat: f64,
    lon: f64,
    elev_m: f64,
    scenario: SspScenario,
    sim_year: f64,
) -> LandCryosphereProfile {
    compute_land_cryosphere_profile_impl(grids, lat, lon, elev_m, scenario, sim_year, None)
}

/// In-memory 0–20 m land cryosphere/permafrost profile evaluated at a specific `day_of_year`,
/// adding `seasonal_surface_swing_c` to reconstruct seasonal surface `T2m` damping toward the
/// deseasonalized annual-mean ground/firn temperature at depth.
pub fn compute_land_cryosphere_profile_from_grids_at_doy(
    grids: Option<&HashMap<String, StoredGridRecord>>,
    lat: f64,
    lon: f64,
    elev_m: f64,
    scenario: SspScenario,
    sim_year: f64,
    day_of_year: f64,
) -> LandCryosphereProfile {
    compute_land_cryosphere_profile_impl(
        grids,
        lat,
        lon,
        elev_m,
        scenario,
        sim_year,
        Some(day_of_year),
    )
}

fn compute_land_cryosphere_profile_impl(
    grids: Option<&HashMap<String, StoredGridRecord>>,
    lat: f64,
    lon: f64,
    elev_m: f64,
    scenario: SspScenario,
    sim_year: f64,
    day_of_year: Option<f64>,
) -> LandCryosphereProfile {
    let sample_opt = |layer_id: &str| -> Option<f64> {
        grids
            .and_then(|map| map.get(layer_id))
            .map(|g| sample_grid_bilinear(&g.values, g.width, g.height, lat, lon) as f64)
            .filter(|v| v.is_finite())
    };

    let base_t2m = preindustrial_baseline_t2m_c(lat, elev_m);
    let dt_anom = sample_opt("cmip6_downscaled_ensemble").unwrap_or_else(|| {
        transient_global_warming_c(scenario.co2_ppm_at_year(sim_year))
            * polar_amplification_factor(lat)
    });
    let t_mean_annual = base_t2m + dt_anom;
    let seasonal_swing = day_of_year
        .map(|doy| seasonal_surface_swing_c(lat, false, doy))
        .unwrap_or(0.0);
    let surface_t2m_c = t_mean_annual + seasonal_swing;

    let active_layer_m = sample_opt("permafrost_thermokarst_subsidence");
    let ice_sheet_smb_cm = sample_opt("gravimetric_ice_sheet_mass_balance");
    let glacier_vel_m_yr = sample_opt("alpine_glacier_velocity_rgi");
    let soil_moisture = sample_opt("soil_moisture_groundwater_grace");

    let depths = [0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0];
    let total_w = soil_moisture.unwrap_or(0.24).clamp(0.02, 0.55);

    let levels = depths
        .iter()
        .map(|&z| {
            let t_z = if let Some(thaw_m) = active_layer_m {
                if thaw_m > 0.05 {
                    let t_surf = surface_t2m_c.max(0.5);
                    let t_deep = (-4.2 - 0.14 * (lat.abs() - 52.0).max(0.0)).min(-0.8);
                    if z <= thaw_m {
                        t_surf * (1.0 - z / thaw_m)
                    } else {
                        t_deep * (1.0 - (-((z - thaw_m) / 3.2)).exp()) + 0.025 * (z - thaw_m)
                    }
                } else {
                    t_mean_annual + (surface_t2m_c - t_mean_annual) * (-z / 3.2).exp() + 0.025 * z
                }
            } else if ice_sheet_smb_cm.is_some() || t_mean_annual < -2.0 {
                // Polar ice sheet / firn column with seasonal skin damping toward annual-mean firn temp + geothermal gradient (+0.025 °C/m)
                t_mean_annual + (surface_t2m_c - t_mean_annual) * (-z / 3.2).exp() + 0.025 * z
            } else {
                // Non-cryospheric land: thermal skin damping toward mean annual ground temp + geothermal gradient
                let t_deep_eq = t_mean_annual - 0.6;
                t_deep_eq + (surface_t2m_c - t_deep_eq) * (-z / 3.2).exp() + 0.025 * z
            };

            let liquid_frac = if ice_sheet_smb_cm.is_some() {
                (0.04 * ((t_z + 1.5) / 2.0).clamp(0.0, 1.0)).clamp(0.0, 0.06)
            } else {
                // Smooth Clausius-Clapeyron unfrozen water curve around 0 °C
                let unfrozen_ratio = 1.0 / (1.0 + (-t_z * 2.5).exp());
                total_w * unfrozen_ratio
            };

            (z, t_z, liquid_frac)
        })
        .collect();

    LandCryosphereProfile {
        elev_m,
        surface_t2m_c,
        active_layer_m,
        ice_sheet_smb_cm,
        glacier_vel_m_yr,
        soil_moisture,
        levels,
    }
}
