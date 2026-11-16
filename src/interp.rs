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

//! Multi-threaded spatial assimilation, quasi-monotone bicubic interpolation, mass conservation,
//! and spherical map projection engine.
//! Implements:
//! - Great-circle Haversine distance & Spherical Inverse-Distance Weighting (IDW)
//! - 2-pass Barnes (1964) Objective Analysis Gaussian spectral correction
//! - Preconditioned Conjugate-Gradient 3D-Var Data Assimilation (`PSAS`) with `quality_flag`
//!   observation error variances `R` and anisotropic flow-aligned Matérn-3/2 (`SOAR`) covariance `B`
//! - Bilinear & Quasi-Monotone Bicubic Catmull-Rom interpolation (Bermejo & Staniforth 1992)
//! - Cosine-latitude spherical area-weighted integral & mass conservation fixer
//! - Log-pressure (`ln P`) vertical atmospheric interpolation
//! - Forward & inverse map projections for `EPSG:4978` (3D WGS84 Globe), `EPSG:4326` (Equirectangular),
//!   `EPSG:3995` (Arctic Polar Stereographic), and `EPSG:3031` (Antarctic Polar Stereographic).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const EARTH_RADIUS_KM: f64 = 6_371.0088;
pub const WGS84_A_METERS: f64 = 6_378_137.0;
pub const WGS84_E2: f64 = 0.006_694_379_990_14;

#[derive(Debug, Clone, Copy)]
pub struct GeoSamplePoint {
    pub lat: f64,
    pub lon: f64,
    pub value: f64,
}

/// Quality-weighted observation point for Conjugate-Gradient 3D-Var assimilation.
#[derive(Debug, Clone, Copy)]
pub struct ObsSamplePoint {
    pub lat: f64,
    pub lon: f64,
    pub value: f64,
    /// `0` = unverified/coarse, `1` = standard WMO/NOAA/satellite, `>= 2` = high-precision observatory.
    pub quality_flag: i32,
    /// Optional explicit observation error standard deviation `sigma_o` (overrides quality heuristic).
    pub sigma_obs: Option<f64>,
}

impl ObsSamplePoint {
    /// Convert `quality_flag` (or explicit `sigma_obs`) into observation error variance `R_ii = sigma_o^2`.
    #[inline]
    pub fn obs_error_variance(&self, base_sigma_obs: f64) -> f64 {
        let sig = if let Some(s) = self.sigma_obs {
            s.max(1.0e-4)
        } else {
            let mult = match self.quality_flag {
                q if q >= 2 => 0.55,
                1 => 1.0,
                0 => 1.85,
                _ => 3.2,
            };
            (base_sigma_obs * mult).max(1.0e-4)
        };
        sig * sig
    }
}

/// Configuration for Preconditioned Conjugate-Gradient 3D-Var Data Assimilation.
#[derive(Debug, Clone, Copy)]
pub struct ThreeDVarConfig {
    /// Background error standard deviation `sigma_b` (units of the assimilated field).
    pub sigma_b: f64,
    /// Baseline observation error standard deviation `sigma_o` (scaled by `quality_flag`).
    pub base_sigma_obs: f64,
    /// Isotropic background correlation length scale `L_b` (km) for Matérn-3/2 (`SOAR`) kernel.
    pub length_scale_km: f64,
    /// Flow-aligned anisotropy strength (`0.0` = isotropic, `0.65` = stretched along jet/wind).
    pub flow_anisotropy: f64,
    /// Maximum PCG iterations.
    pub max_iters: usize,
    /// Relative residual convergence tolerance.
    pub tol: f64,
}

impl Default for ThreeDVarConfig {
    fn default() -> Self {
        Self {
            sigma_b: 1.8,
            base_sigma_obs: 0.75,
            length_scale_km: 1200.0,
            flow_anisotropy: 0.55,
            max_iters: 48,
            tol: 1.0e-6,
        }
    }
}

/// Output of the Conjugate-Gradient 3D-Var solver (`x_a = x_b + B H^T w`).
#[derive(Debug, Clone)]
pub struct ThreeDVarResult {
    pub analysis: Vec<f32>,
    pub increment: Vec<f32>,
    pub iterations: usize,
    pub rel_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MapProjectionKind {
    Globe3dWgs84,         // EPSG:4978
    Equirectangular2d,    // EPSG:4326
    ArcticPolarStereo,    // EPSG:3995
    AntarcticPolarStereo, // EPSG:3031
}

impl MapProjectionKind {
    pub const ALL: [MapProjectionKind; 4] = [
        MapProjectionKind::Globe3dWgs84,
        MapProjectionKind::Equirectangular2d,
        MapProjectionKind::ArcticPolarStereo,
        MapProjectionKind::AntarcticPolarStereo,
    ];

    pub fn id(&self) -> &'static str {
        match self {
            MapProjectionKind::Globe3dWgs84 => "globe_3d_wgs84",
            MapProjectionKind::Equirectangular2d => "equirectangular_2d",
            MapProjectionKind::ArcticPolarStereo => "arctic_polar_stereo",
            MapProjectionKind::AntarcticPolarStereo => "antarctic_polar_stereo",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            MapProjectionKind::Globe3dWgs84 => "3D Planetary Globe (WGS84 / EPSG:4978)",
            MapProjectionKind::Equirectangular2d => "2D Equirectangular (EPSG:4326)",
            MapProjectionKind::ArcticPolarStereo => "Arctic Polar Stereo (EPSG:3995)",
            MapProjectionKind::AntarcticPolarStereo => "Antarctic Polar Stereo (EPSG:3031)",
        }
    }

    pub fn from_id(id: &str) -> Self {
        match id {
            "equirectangular_2d" | "EPSG:4326" => MapProjectionKind::Equirectangular2d,
            "arctic_polar_stereo" | "EPSG:3995" => MapProjectionKind::ArcticPolarStereo,
            "antarctic_polar_stereo" | "EPSG:3031" => MapProjectionKind::AntarcticPolarStereo,
            _ => MapProjectionKind::Globe3dWgs84,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CameraState {
    pub center_lon: f64,
    pub center_lat: f64,
    pub zoom: f64,
    pub pitch_deg: f64,
    pub bearing_deg: f64,
    pub vertical_exaggeration: f64,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            center_lon: -20.0, // Global Atlantic-Americas default from schema
            center_lat: 20.0,
            zoom: 1.0,
            pitch_deg: 0.0,
            bearing_deg: 0.0,
            vertical_exaggeration: 15.0,
        }
    }
}

/// Great-circle Haversine distance in kilometers between two `(lat, lon)` points in degrees.
#[inline]
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let phi1 = lat1.to_radians();
    let phi2 = lat2.to_radians();
    let a = (0.5 * dlat).sin().powi(2) + phi1.cos() * phi2.cos() * (0.5 * dlon).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * a.clamp(0.0, 1.0).sqrt().asin()
}

/// Generate a uniform Fibonacci geodesic lattice of `n` points across the globe.
pub fn fibonacci_sphere_points(n: usize) -> Vec<(f64, f64)> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * (i as f64 + 0.5)) / (n as f64);
            let lat = y.clamp(-1.0, 1.0).asin().to_degrees();
            let theta = golden_angle * (i as f64);
            let lon = ((theta.to_degrees() + 180.0).rem_euclid(360.0)) - 180.0;
            (lat, lon)
        })
        .collect()
}

/// Multi-threaded 2-pass Barnes (1964) Objective Analysis + background blend.
pub fn barnes_assimilate_grid(
    background: &[f32],
    width: usize,
    height: usize,
    samples: &[GeoSamplePoint],
    length_scale_km: f64,
    gamma_second_pass: f64,
) -> Vec<f32> {
    assert_eq!(background.len(), width * height);
    if samples.is_empty() {
        return background.to_vec();
    }

    let c1 = 2.0 * length_scale_km * length_scale_km;
    let c2 = c1 * gamma_second_pass.clamp(0.15, 0.85);

    let obs_increments: Vec<(f64, f64, f64)> = samples
        .iter()
        .map(|s| {
            let bg_at_obs = sample_grid_bilinear(background, width, height, s.lat, s.lon) as f64;
            (s.lat, s.lon, s.value - bg_at_obs)
        })
        .collect();

    let mut pass1 = vec![0.0_f32; width * height];
    pass1
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, row_slice)| {
            let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
            for (col, out_cell) in row_slice.iter_mut().enumerate() {
                let lon = -180.0 + (col as f64 + 0.5) * (360.0 / width as f64);
                let mut w_sum = 0.0;
                let mut inc_sum = 0.0;
                for &(olat, olon, oinc) in &obs_increments {
                    let d = haversine_km(lat, lon, olat, olon);
                    if d < length_scale_km * 3.5 {
                        let w = (-d * d / c1).exp();
                        w_sum += w;
                        inc_sum += w * oinc;
                    }
                }
                let bg = background[row * width + col] as f64;
                let blended_inc = if w_sum > 1e-6 {
                    let confidence = (w_sum / (w_sum + 0.35)).clamp(0.0, 1.0);
                    confidence * (inc_sum / w_sum)
                } else {
                    0.0
                };
                *out_cell = (bg + blended_inc) as f32;
            }
        });

    let residuals: Vec<(f64, f64, f64)> = samples
        .iter()
        .map(|s| {
            let p1_at_obs = sample_grid_bilinear(&pass1, width, height, s.lat, s.lon) as f64;
            (s.lat, s.lon, s.value - p1_at_obs)
        })
        .collect();

    let mut pass2 = pass1.clone();
    pass2
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, row_slice)| {
            let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
            for (col, out_cell) in row_slice.iter_mut().enumerate() {
                let lon = -180.0 + (col as f64 + 0.5) * (360.0 / width as f64);
                let mut w_sum = 0.0;
                let mut res_sum = 0.0;
                for &(olat, olon, ores) in &residuals {
                    let d = haversine_km(lat, lon, olat, olon);
                    if d < length_scale_km * 2.2 {
                        let w = (-d * d / c2).exp();
                        w_sum += w;
                        res_sum += w * ores;
                    }
                }
                if w_sum > 1e-5 {
                    let confidence = (w_sum / (w_sum + 0.5)).clamp(0.0, 1.0);
                    *out_cell += (confidence * (res_sum / w_sum)) as f32;
                }
            }
        });

    pass2
}

/// Flow-aligned anisotropic Matérn-3/2 (`SOAR`) background error covariance `B(r_1, r_2)`:
/// `B = sigma_b^2 * (1 + s_eff) * exp(-s_eff)` where `s_eff` stretches correlation along `(u_flow, v_flow)`.
#[inline]
fn anisotropic_soar_covariance(
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
    u_flow: f64,
    v_flow: f64,
    cfg: &ThreeDVarConfig,
) -> f64 {
    let d_iso = haversine_km(lat1, lon1, lat2, lon2);
    if d_iso > cfg.length_scale_km * 5.0 {
        return 0.0;
    }
    let s_eff = if cfg.flow_anisotropy > 1e-4 {
        let spd = u_flow.hypot(v_flow);
        if spd > 1.5 {
            let mid_lat = 0.5 * (lat1 + lat2);
            let cos_phi = mid_lat.to_radians().cos().max(0.18);
            let mut dlon = lon2 - lon1;
            if dlon > 180.0 {
                dlon -= 360.0;
            } else if dlon < -180.0 {
                dlon += 360.0;
            }
            let dx_km = dlon.to_radians() * EARTH_RADIUS_KM * cos_phi;
            let dy_km = (lat2 - lat1).to_radians() * EARTH_RADIUS_KM;

            let tx = u_flow / spd;
            let ty = v_flow / spd;
            let s_par = dx_km * tx + dy_km * ty;
            let s_perp = -dx_km * ty + dy_km * tx;

            let stretch = 1.0 + cfg.flow_anisotropy * (spd / 25.0).min(1.0);
            let l_par = cfg.length_scale_km * stretch;
            let l_perp = cfg.length_scale_km / (1.0 + 0.5 * cfg.flow_anisotropy * (spd / 25.0).min(1.0));
            ((s_par / l_par).powi(2) + (s_perp / l_perp).powi(2)).sqrt()
        } else {
            d_iso / cfg.length_scale_km
        }
    } else {
        d_iso / cfg.length_scale_km
    };

    let sb2 = cfg.sigma_b * cfg.sigma_b;
    sb2 * (1.0 + s_eff) * (-s_eff).exp()
}

/// Preconditioned Conjugate-Gradient 3D-Var Data Assimilation (`PSAS` dual solver):
/// Minimizes `J(dx) = 0.5 dx^T B^-1 dx + 0.5 (H dx - d)^T R^-1 (H dx - d)` by solving
/// `(H B H^T + R) w = d` via Jacobi-PCG and projecting `dx = B H^T w` in parallel via `rayon`.
pub fn assimilate_3dvar_cg(
    background: &[f32],
    width: usize,
    height: usize,
    samples: &[ObsSamplePoint],
    flow_uv: Option<(&[f32], &[f32])>,
    cfg: &ThreeDVarConfig,
) -> ThreeDVarResult {
    let n = width * height;
    assert_eq!(background.len(), n);
    if samples.is_empty() {
        return ThreeDVarResult {
            analysis: background.to_vec(),
            increment: vec![0.0_f32; n],
            iterations: 0,
            rel_residual: 0.0,
        };
    }

    // Subsample/cap to at most 768 observations for deterministic sub-millisecond PCG solve
    let obs: Vec<ObsSamplePoint> = if samples.len() > 768 {
        let step = (samples.len() as f64 / 768.0).ceil() as usize;
        samples.iter().step_by(step.max(1)).copied().collect()
    } else {
        samples.to_vec()
    };
    let m = obs.len();

    // 1. Innovation vector d = y - H(x_b), observation error variance R, and local flow at obs
    let mut d_vec = vec![0.0_f64; m];
    let mut r_diag = vec![0.0_f64; m];
    let mut obs_uv = vec![(0.0_f64, 0.0_f64); m];
    for (i, s) in obs.iter().enumerate() {
        let xb = sample_grid_bilinear(background, width, height, s.lat, s.lon) as f64;
        d_vec[i] = if xb.is_finite() { s.value - xb } else { 0.0 };
        r_diag[i] = s.obs_error_variance(cfg.base_sigma_obs);
        if let Some((u_arr, v_arr)) = flow_uv {
            let u = sample_grid_bilinear(u_arr, width, height, s.lat, s.lon) as f64;
            let v = sample_grid_bilinear(v_arr, width, height, s.lat, s.lon) as f64;
            obs_uv[i] = (
                if u.is_finite() { u } else { 0.0 },
                if v.is_finite() { v } else { 0.0 },
            );
        }
    }

    // 2. Assemble symmetric positive-definite dual matrix A = H B H^T + R (m x m)
    let mut a_mat = vec![0.0_f64; m * m];
    a_mat
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(i, row_slice)| {
            let si = obs[i];
            let (ui, vi) = obs_uv[i];
            for j in 0..m {
                if i == j {
                    row_slice[j] = cfg.sigma_b * cfg.sigma_b + r_diag[i];
                } else {
                    let sj = obs[j];
                    let (uj, vj) = obs_uv[j];
                    row_slice[j] = anisotropic_soar_covariance(
                        si.lat,
                        si.lon,
                        sj.lat,
                        sj.lon,
                        0.5 * (ui + uj),
                        0.5 * (vi + vj),
                        cfg,
                    );
                }
            }
        });

    // 3. Solve A w = d via Jacobi-Preconditioned Conjugate Gradient (PCG)
    let mut w = vec![0.0_f64; m];
    let mut r = d_vec.clone();
    let mut z: Vec<f64> = (0..m).map(|i| r[i] / a_mat[i * m + i].max(1e-12)).collect();
    let mut p = z.clone();

    let d_norm = d_vec.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
    let mut rz_old: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();
    let mut iters = 0usize;
    let mut rel_res = 1.0_f64;

    for k in 0..cfg.max_iters {
        iters = k + 1;
        let mut ap = vec![0.0_f64; m];
        for i in 0..m {
            let row_off = i * m;
            let mut sum = 0.0_f64;
            for j in 0..m {
                sum += a_mat[row_off + j] * p[j];
            }
            ap[i] = sum;
        }
        let denom: f64 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
        if denom.abs() < 1e-24 {
            break;
        }
        let alpha = rz_old / denom;
        for i in 0..m {
            w[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let r_norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        rel_res = r_norm / d_norm;
        if rel_res <= cfg.tol {
            break;
        }
        for i in 0..m {
            z[i] = r[i] / a_mat[i * m + i].max(1e-12);
        }
        let rz_new: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();
        let beta = rz_new / rz_old.max(1e-24);
        for i in 0..m {
            p[i] = z[i] + beta * p[i];
        }
        rz_old = rz_new;
    }

    // 4. Project dual weights onto global grid in parallel: dx = B H^T w, x_a = x_b + dx
    let mut increment = vec![0.0_f32; n];
    let mut analysis = background.to_vec();

    increment
        .par_chunks_mut(width)
        .zip(analysis.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, (inc_row, ana_row))| {
            let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
            let row_off = row * width;
            for col in 0..width {
                let idx = row_off + col;
                let bg = background[idx];
                if !bg.is_finite() {
                    inc_row[col] = 0.0;
                    ana_row[col] = bg;
                    continue;
                }
                let lon = -180.0 + (col as f64 + 0.5) * (360.0 / width as f64);
                let (ug, vg) = if let Some((u_arr, v_arr)) = flow_uv {
                    (
                        if u_arr[idx].is_finite() { u_arr[idx] as f64 } else { 0.0 },
                        if v_arr[idx].is_finite() { v_arr[idx] as f64 } else { 0.0 },
                    )
                } else {
                    (0.0, 0.0)
                };

                let mut dx = 0.0_f64;
                for (j, s) in obs.iter().enumerate() {
                    let (uj, vj) = obs_uv[j];
                    let b_val = anisotropic_soar_covariance(
                        lat,
                        lon,
                        s.lat,
                        s.lon,
                        0.5 * (ug + uj),
                        0.5 * (vg + vj),
                        cfg,
                    );
                    if b_val != 0.0 {
                        dx += b_val * w[j];
                    }
                }
                inc_row[col] = dx as f32;
                ana_row[col] = (bg as f64 + dx) as f32;
            }
        });

    ThreeDVarResult {
        analysis,
        increment,
        iterations: iters,
        rel_residual: rel_res,
    }
}

/// Bilinear interpolation on a global `(height x width)` equirectangular grid with periodic longitude wrapping.
pub fn sample_grid_bilinear(
    grid: &[f32],
    width: usize,
    height: usize,
    lat: f64,
    lon: f64,
) -> f32 {
    if grid.is_empty() || width == 0 || height == 0 {
        return 0.0;
    }
    let clamped_lat = lat.clamp(-89.999, 89.999);
    let wrapped_lon = ((lon + 180.0).rem_euclid(360.0)) - 180.0;

    let fy = ((90.0 - clamped_lat) / 180.0) * (height as f64) - 0.5;
    let fx = ((wrapped_lon + 180.0) / 360.0) * (width as f64) - 0.5;

    let y0 = fy.floor().clamp(0.0, (height - 1) as f64) as usize;
    let y1 = (y0 + 1).min(height - 1);
    let wy = (fy - fy.floor()).clamp(0.0, 1.0) as f32;

    let x0_signed = fx.floor() as isize;
    let x0 = x0_signed.rem_euclid(width as isize) as usize;
    let x1 = (x0 + 1) % width;
    let wx = (fx - fx.floor()).clamp(0.0, 1.0) as f32;

    let v00 = grid[y0 * width + x0];
    let v10 = grid[y0 * width + x1];
    let v01 = grid[y1 * width + x0];
    let v11 = grid[y1 * width + x1];

    let top = v00 * (1.0 - wx) + v10 * wx;
    let bot = v01 * (1.0 - wx) + v11 * wx;
    top * (1.0 - wy) + bot * wy
}

#[inline]
fn catmull_rom_1d(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Quasi-Monotone Bicubic Hermite (Catmull-Rom) spherical grid interpolation
/// (Bermejo & Staniforth 1992, Mon. Wea. Rev., 120, 2622-2632).
///
/// Evaluates a `4x4` stencil with periodic longitude wrapping for third-order accuracy on smooth
/// plumes/fronts while clamping the interpolated value strictly within the bounding `2x2` cell
/// `[min_local, max_local]` interval to eliminate Gibbs ringing and guarantee positivity.
/// Automatically falls back to bilinear or nearest valid neighbor near `NaN` coastline masks.
pub fn sample_grid_bicubic_monotone(
    grid: &[f32],
    width: usize,
    height: usize,
    lat: f64,
    lon: f64,
) -> f32 {
    if grid.is_empty() || width < 4 || height < 4 {
        return sample_grid_bilinear(grid, width, height, lat, lon);
    }
    let clamped_lat = lat.clamp(-89.999, 89.999);
    let wrapped_lon = ((lon + 180.0).rem_euclid(360.0)) - 180.0;

    let fy = ((90.0 - clamped_lat) / 180.0) * (height as f64) - 0.5;
    let fx = ((wrapped_lon + 180.0) / 360.0) * (width as f64) - 0.5;

    let iy = fy.floor() as isize;
    let ix = fx.floor() as isize;
    let ty = (fy - (iy as f64)).clamp(0.0, 1.0);
    let tx = (fx - (ix as f64)).clamp(0.0, 1.0);

    let y_idx = [
        (iy - 1).clamp(0, height as isize - 1) as usize,
        iy.clamp(0, height as isize - 1) as usize,
        (iy + 1).clamp(0, height as isize - 1) as usize,
        (iy + 2).clamp(0, height as isize - 1) as usize,
    ];
    let x_idx = [
        (ix - 1).rem_euclid(width as isize) as usize,
        ix.rem_euclid(width as isize) as usize,
        (ix + 1).rem_euclid(width as isize) as usize,
        (ix + 2).rem_euclid(width as isize) as usize,
    ];

    // Verify inner 2x2 cell finiteness first
    let c00 = grid[y_idx[1] * width + x_idx[1]];
    let c10 = grid[y_idx[1] * width + x_idx[2]];
    let c01 = grid[y_idx[2] * width + x_idx[1]];
    let c11 = grid[y_idx[2] * width + x_idx[2]];
    if !c00.is_finite() || !c10.is_finite() || !c01.is_finite() || !c11.is_finite() {
        return c00;
    }

    let local_min = c00.min(c10).min(c01).min(c11) as f64;
    let local_max = c00.max(c10).max(c01).max(c11) as f64;

    let mut row_vals = [0.0_f64; 4];
    for r in 0..4 {
        let mut p = [0.0_f64; 4];
        for c in 0..4 {
            let v = grid[y_idx[r] * width + x_idx[c]];
            if !v.is_finite() {
                // Outer halo crossed a coastline mask; fall back to monotone bilinear
                return sample_grid_bilinear(grid, width, height, lat, lon);
            }
            p[c] = v as f64;
        }
        row_vals[r] = catmull_rom_1d(p[0], p[1], p[2], p[3], tx);
    }

    let raw_cubic = catmull_rom_1d(row_vals[0], row_vals[1], row_vals[2], row_vals[3], ty);
    raw_cubic.clamp(local_min, local_max) as f32
}

/// Compute the `cos(latitude)` area-weighted mean over all finite cells in `grid`.
pub fn spherical_area_weighted_mean(grid: &[f32], width: usize, height: usize) -> f64 {
    assert_eq!(grid.len(), width * height);
    let mut weighted_sum = 0.0_f64;
    let mut area_sum = 0.0_f64;
    for row in 0..height {
        let lat = 90.0 - (row as f64 + 0.5) * (180.0 / height as f64);
        let cos_w = lat.to_radians().cos().max(1e-4);
        for col in 0..width {
            let v = grid[row * width + col];
            if v.is_finite() {
                weighted_sum += (v as f64) * cos_w;
                area_sum += cos_w;
            }
        }
    }
    if area_sum > 0.0 {
        weighted_sum / area_sum
    } else {
        0.0
    }
}

///Multiplicative/additive a-posteriori mass fixer (Priestley 1993 / Bermejo & Conde 2002)
/// restoring the exact spherical area-weighted mean `target_mean` after semi-Lagrangian advection.
pub fn conserve_spherical_mass(
    grid: &mut [f32],
    width: usize,
    height: usize,
    target_mean: f64,
    non_negative: bool,
) {
    let cur_mean = spherical_area_weighted_mean(grid, width, height);
    if !cur_mean.is_finite() || !target_mean.is_finite() {
        return;
    }
    if non_negative && cur_mean > 1e-9 && target_mean >= 0.0 {
        let scale = (target_mean / cur_mean).clamp(0.5, 2.0) as f32;
        for v in grid.iter_mut() {
            if v.is_finite() {
                *v = (*v * scale).max(0.0);
            }
        }
    } else {
        let delta = (target_mean - cur_mean) as f32;
        for v in grid.iter_mut() {
            if v.is_finite() {
                *v += delta;
                if non_negative {
                    *v = v.max(0.0);
                }
            }
        }
    }
}

/// Vertical atmospheric interpolation in log-pressure coordinates (`ln P`) between discrete pressure levels.
pub fn interpolate_log_pressure(levels_hpa: &[f64], values: &[f64], target_hpa: f64) -> f64 {
    assert_eq!(levels_hpa.len(), values.len());
    if levels_hpa.is_empty() {
        return 0.0;
    }
    let ln_t = target_hpa.max(1.0).ln();
    if target_hpa >= levels_hpa[0] {
        return values[0];
    }
    let last = levels_hpa.len() - 1;
    if target_hpa <= levels_hpa[last] {
        return values[last];
    }
    for i in 0..last {
        let p_hi = levels_hpa[i];
        let p_lo = levels_hpa[i + 1];
        if target_hpa <= p_hi && target_hpa >= p_lo {
            let ln_hi = p_hi.ln();
            let ln_lo = p_lo.ln();
            let frac = ((ln_hi - ln_t) / (ln_hi - ln_lo)).clamp(0.0, 1.0);
            return values[i] * (1.0 - frac) + values[i + 1] * frac;
        }
    }
    values[0]
}

/// Forward map projection from `(lat_deg, lon_deg, elevation_m)` to normalized viewport coordinates
/// `[-1.0, 1.0] x [-1.0, 1.0]`.
pub fn project_latlon_to_ndc(
    proj: MapProjectionKind,
    cam: &CameraState,
    lat_deg: f64,
    lon_deg: f64,
    elevation_m: f64,
    aspect_w_over_h: f64,
) -> Option<(f32, f32)> {
    let aspect = aspect_w_over_h.max(0.2);
    match proj {
        MapProjectionKind::Equirectangular2d => {
            let dlon = ((lon_deg - cam.center_lon + 540.0).rem_euclid(360.0)) - 180.0;
            let dlat = lat_deg - cam.center_lat;
            let x = (dlon / 180.0) * cam.zoom;
            let y = (dlat / 90.0) * cam.zoom;
            if x.abs() <= 1.25 && y.abs() <= 1.25 {
                Some((x as f32, y as f32))
            } else {
                None
            }
        }
        MapProjectionKind::Globe3dWgs84 => {
            let phi = lat_deg.to_radians();
            let lam = (lon_deg - cam.center_lon).to_radians();
            let phi0 = cam.center_lat.to_radians();

            let r_scale = 1.0 + (elevation_m * cam.vertical_exaggeration) / WGS84_A_METERS;
            let cos_c = phi0.sin() * phi.sin() + phi0.cos() * phi.cos() * lam.cos();
            if cos_c < 0.0 {
                return None;
            }
            let raw_x = r_scale * phi.cos() * lam.sin();
            let raw_y = r_scale * (phi0.cos() * phi.sin() - phi0.sin() * phi.cos() * lam.cos());

            let b = cam.bearing_deg.to_radians();
            let rot_x = raw_x * b.cos() - raw_y * b.sin();
            let rot_y = raw_x * b.sin() + raw_y * b.cos();

            let radius = 0.86 * cam.zoom;
            Some(((rot_x * radius / aspect) as f32, (rot_y * radius) as f32))
        }
        MapProjectionKind::ArcticPolarStereo => {
            if lat_deg < -5.0 {
                return None;
            }
            let phi = lat_deg.to_radians();
            let lam = (lon_deg - cam.center_lon).to_radians();
            let rho = (std::f64::consts::FRAC_PI_4 - 0.5 * phi).tan();
            let scale = 0.78 * cam.zoom;
            let x = rho * lam.sin() * scale / aspect;
            let y = -rho * lam.cos() * scale;
            Some((x as f32, y as f32))
        }
        MapProjectionKind::AntarcticPolarStereo => {
            if lat_deg > 5.0 {
                return None;
            }
            let phi = (-lat_deg).to_radians();
            let lam = (lon_deg - cam.center_lon).to_radians();
            let rho = (std::f64::consts::FRAC_PI_4 - 0.5 * phi).tan();
            let scale = 0.78 * cam.zoom;
            let x = rho * lam.sin() * scale / aspect;
            let y = rho * lam.cos() * scale;
            Some((x as f32, y as f32))
        }
    }
}

/// Inverse map projection from normalized viewport coordinates `[-1.0, 1.0] x [-1.0, 1.0]`
/// to `(lat_deg, lon_deg)`.
pub fn unproject_ndc_to_latlon(
    proj: MapProjectionKind,
    cam: &CameraState,
    ndc_x: f64,
    ndc_y: f64,
    aspect_w_over_h: f64,
) -> Option<(f64, f64)> {
    let aspect = aspect_w_over_h.max(0.2);
    match proj {
        MapProjectionKind::Equirectangular2d => {
            let lat = cam.center_lat + (ndc_y / cam.zoom) * 90.0;
            if !(-90.0..=90.0).contains(&lat) {
                return None;
            }
            let lon =
                ((cam.center_lon + (ndc_x / cam.zoom) * 180.0 + 540.0).rem_euclid(360.0)) - 180.0;
            Some((lat, lon))
        }
        MapProjectionKind::Globe3dWgs84 => {
            let radius = 0.86 * cam.zoom;
            let sx = (ndc_x * aspect) / radius;
            let sy = ndc_y / radius;
            let rho2 = sx * sx + sy * sy;
            if rho2 > 1.0 {
                return None;
            }
            let b = -cam.bearing_deg.to_radians();
            let rx = sx * b.cos() - sy * b.sin();
            let ry = sx * b.sin() + sy * b.cos();

            let rho = rho2.sqrt();
            if rho < 1e-9 {
                return Some((cam.center_lat, cam.center_lon));
            }
            let c = rho.asin();
            let phi0 = cam.center_lat.to_radians();
            let sin_c = c.sin();
            let cos_c = c.cos();

            let phi = (cos_c * phi0.sin() + (ry * sin_c * phi0.cos() / rho))
                .clamp(-1.0, 1.0)
                .asin();
            let lam = cam.center_lon.to_radians()
                + (rx * sin_c).atan2(rho * phi0.cos() * cos_c - ry * phi0.sin() * sin_c);
            let lon = ((lam.to_degrees() + 540.0).rem_euclid(360.0)) - 180.0;
            Some((phi.to_degrees(), lon))
        }
        MapProjectionKind::ArcticPolarStereo => {
            let scale = 0.78 * cam.zoom;
            let x = (ndc_x * aspect) / scale;
            let y = -ndc_y / scale;
            let rho = (x * x + y * y).sqrt();
            if rho > 1.35 {
                return None;
            }
            let phi = std::f64::consts::FRAC_PI_2 - 2.0 * rho.atan();
            let lam = cam.center_lon.to_radians() + x.atan2(y);
            let lon = ((lam.to_degrees() + 540.0).rem_euclid(360.0)) - 180.0;
            Some((phi.to_degrees().clamp(-90.0, 90.0), lon))
        }
        MapProjectionKind::AntarcticPolarStereo => {
            let scale = 0.78 * cam.zoom;
            let x = (ndc_x * aspect) / scale;
            let y = ndc_y / scale;
            let rho = (x * x + y * y).sqrt();
            if rho > 1.35 {
                return None;
            }
            let phi = -(std::f64::consts::FRAC_PI_2 - 2.0 * rho.atan());
            let lam = cam.center_lon.to_radians() + x.atan2(y);
            let lon = ((lam.to_degrees() + 540.0).rem_euclid(360.0)) - 180.0;
            Some((phi.to_degrees().clamp(-90.0, 90.0), lon))
        }
    }
}

/// Spherical line segment on a `level` isoline extracted by Marching Squares.
#[derive(Debug, Clone, Copy)]
pub struct ContourSegment {
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
    pub level: f32,
}

/// Detected synoptic cyclone (`is_low = true`) or anticyclone (`is_low = false`) pressure center.
#[derive(Debug, Clone, Copy)]
pub struct PressureExtrema {
    pub lat: f64,
    pub lon: f64,
    pub pressure_hpa: f32,
    pub is_low: bool,
}

/// Detected baroclinic thermal front segment (`is_cold_front = true` => cold advection, `false` => warm advection).
#[derive(Debug, Clone, Copy)]
pub struct FrontSegment {
    pub lat1: f64,
    pub lon1: f64,
    pub lat2: f64,
    pub lon2: f64,
    pub is_cold_front: bool,
}

/// Extract closed spherical isoline segments (e.g. `4 hPa` MSLP isobars) via 16-case Marching Squares
/// with periodic longitude wrapping and saddle-point mean disambiguation.
pub fn extract_marching_squares_contours(
    grid: &[f32],
    width: usize,
    height: usize,
    interval: f32,
    min_level: f32,
    max_level: f32,
) -> Vec<ContourSegment> {
    if grid.len() != width * height || width < 4 || height < 4 || interval <= 0.0 {
        return Vec::new();
    }

    let mut levels = Vec::new();
    let mut lv = min_level;
    while lv <= max_level + 1e-4 {
        levels.push(lv);
        lv += interval;
    }

    let dlat = 180.0 / (height as f64);
    let dlon = 360.0 / (width as f64);
    let mut segments = Vec::with_capacity(levels.len() * width);

    let interp_edge = |p1: (f64, f64), p2: (f64, f64), v1: f32, v2: f32, iso: f32| -> (f64, f64) {
        let denom = v2 - v1;
        let t = if denom.abs() < 1.0e-6 {
            0.5
        } else {
            ((iso - v1) / denom).clamp(0.0, 1.0) as f64
        };
        (p1.0 + t * (p2.0 - p1.0), p1.1 + t * (p2.1 - p1.1))
    };

    for &iso in &levels {
        for r in 0..(height - 1) {
            let lat_top = 90.0 - (r as f64 + 0.5) * dlat;
            let lat_bot = 90.0 - ((r + 1) as f64 + 0.5) * dlat;
            for c in 0..width {
                let c_next = (c + 1) % width;
                let lon_l = -180.0 + (c as f64 + 0.5) * dlon;
                let lon_r = lon_l + dlon;

                // Corners: TL (3), TR (2), BR (1), BL (0)
                let v_tl = grid[r * width + c];
                let v_tr = grid[r * width + c_next];
                let v_br = grid[(r + 1) * width + c_next];
                let v_bl = grid[(r + 1) * width + c];

                if !v_tl.is_finite() || !v_tr.is_finite() || !v_br.is_finite() || !v_bl.is_finite()
                {
                    continue;
                }

                let mut case_idx = 0u8;
                if v_bl >= iso {
                    case_idx |= 1;
                }
                if v_br >= iso {
                    case_idx |= 2;
                }
                if v_tr >= iso {
                    case_idx |= 4;
                }
                if v_tl >= iso {
                    case_idx |= 8;
                }
                if case_idx == 0 || case_idx == 15 {
                    continue;
                }

                let p_tl = (lat_top, lon_l);
                let p_tr = (lat_top, lon_r);
                let p_br = (lat_bot, lon_r);
                let p_bl = (lat_bot, lon_l);

                // Edge midpoints: top(0), right(1), bottom(2), left(3)
                let e_top = || interp_edge(p_tl, p_tr, v_tl, v_tr, iso);
                let e_right = || interp_edge(p_tr, p_br, v_tr, v_br, iso);
                let e_bot = || interp_edge(p_bl, p_br, v_bl, v_br, iso);
                let e_left = || interp_edge(p_tl, p_bl, v_tl, v_bl, iso);

                let push_seg = |segs: &mut Vec<ContourSegment>, a: (f64, f64), b: (f64, f64)| {
                    segs.push(ContourSegment {
                        lat1: a.0,
                        lon1: ((a.1 + 180.0).rem_euclid(360.0)) - 180.0,
                        lat2: b.0,
                        lon2: ((b.1 + 180.0).rem_euclid(360.0)) - 180.0,
                        level: iso,
                    });
                };

                match case_idx {
                    1 | 14 => push_seg(&mut segments, e_left(), e_bot()),
                    2 | 13 => push_seg(&mut segments, e_bot(), e_right()),
                    3 | 12 => push_seg(&mut segments, e_left(), e_right()),
                    4 | 11 => push_seg(&mut segments, e_top(), e_right()),
                    6 | 9 => push_seg(&mut segments, e_top(), e_bot()),
                    7 | 8 => push_seg(&mut segments, e_top(), e_left()),
                    5 => {
                        let center = 0.25 * (v_tl + v_tr + v_br + v_bl);
                        if center >= iso {
                            push_seg(&mut segments, e_top(), e_left());
                            push_seg(&mut segments, e_bot(), e_right());
                        } else {
                            push_seg(&mut segments, e_top(), e_right());
                            push_seg(&mut segments, e_left(), e_bot());
                        }
                    }
                    10 => {
                        let center = 0.25 * (v_tl + v_tr + v_br + v_bl);
                        if center >= iso {
                            push_seg(&mut segments, e_top(), e_right());
                            push_seg(&mut segments, e_left(), e_bot());
                        } else {
                            push_seg(&mut segments, e_top(), e_left());
                            push_seg(&mut segments, e_bot(), e_right());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    segments
}

/// Detect synoptic `L` (cyclone) and `H` (anticyclone) pressure extrema with 3x3 neighborhood check
/// and great-circle non-maximum suppression (`>= 1800 km` separation).
pub fn detect_mslp_extrema(
    mslp: &[f32],
    width: usize,
    height: usize,
    min_prominence_hpa: f32,
) -> Vec<PressureExtrema> {
    if mslp.len() != width * height || width < 6 || height < 6 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for r in 2..(height - 2) {
        let lat = 90.0 - (r as f64 + 0.5) * (180.0 / height as f64);
        if lat.abs() > 82.0 {
            continue;
        }
        for c in 0..width {
            let v = mslp[r * width + c];
            if !v.is_finite() {
                continue;
            }
            let mut is_min = true;
            let mut is_max = true;
            let mut ring_sum = 0.0_f32;
            let mut ring_cnt = 0usize;

            for dr in -1_isize..=1 {
                for dc in -1_isize..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let rr = (r as isize + dr) as usize;
                    let cc = (c as isize + dc).rem_euclid(width as isize) as usize;
                    let nb = mslp[rr * width + cc];
                    if !nb.is_finite() {
                        continue;
                    }
                    if nb < v {
                        is_min = false;
                    }
                    if nb > v {
                        is_max = false;
                    }
                    ring_sum += nb;
                    ring_cnt += 1;
                }
            }

            if ring_cnt == 8 && (is_min || is_max) {
                let ring_mean = ring_sum / 8.0;
                let prom = (v - ring_mean).abs();
                if prom >= min_prominence_hpa.max(1e-5) {
                    let lon = -180.0 + (c as f64 + 0.5) * (360.0 / width as f64);
                    candidates.push((
                        prom,
                        PressureExtrema {
                            lat,
                            lon,
                            pressure_hpa: v,
                            is_low: is_min,
                        },
                    ));
                }
            }
        }
    }

    // Sort strongest extrema first and enforce >= 1800 km spatial separation
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<PressureExtrema> = Vec::new();
    for (_, cand) in candidates {
        let too_close = kept
            .iter()
            .any(|k| k.is_low == cand.is_low && haversine_km(k.lat, k.lon, cand.lat, cand.lon) < 1800.0);
        if !too_close {
            kept.push(cand);
            if kept.len() >= 18 {
                break;
            }
        }
    }
    kept
}

/// Detect baroclinic cold and warm fronts from horizontal temperature gradient `|grad_S(T)|`
/// and thermal advection `-V . grad_S(T)` (`< 0` = cold front, `> 0` = warm front).
pub fn detect_thermal_fronts(
    temp_field: &[f32],
    u_wind: &[f32],
    v_wind: &[f32],
    width: usize,
    height: usize,
    grad_threshold_k_per_1000km: f64,
) -> Vec<FrontSegment> {
    let n = width * height;
    if temp_field.len() != n || u_wind.len() != n || v_wind.len() != n || width < 6 || height < 6 {
        return Vec::new();
    }

    let dphi = std::f64::consts::PI / (height as f64);
    let dlam = (2.0 * std::f64::consts::PI) / (width as f64);
    let r_km = EARTH_RADIUS_KM;
    let dlat_deg = 180.0 / (height as f64);
    let dlon_deg = 360.0 / (width as f64);
    let mut fronts = Vec::new();

    for r in 2..(height - 2) {
        let lat = 90.0 - (r as f64 + 0.5) * dlat_deg;
        if lat.abs() < 18.0 || lat.abs() > 75.0 {
            continue;
        }
        let cos_phi = lat.to_radians().cos().max(0.2);
        let dy_1000km = (2.0 * r_km * dphi) / 1000.0;
        let dx_1000km = (2.0 * r_km * cos_phi * dlam) / 1000.0;

        for c in 0..width {
            let idx = r * width + c;
            let tc = temp_field[idx];
            if !tc.is_finite() {
                continue;
            }
            let cw = if c == 0 { width - 1 } else { c - 1 };
            let ce = if c + 1 == width { 0 } else { c + 1 };

            let dtdx = (temp_field[r * width + ce] - temp_field[r * width + cw]) as f64 / dx_1000km;
            let dtdy = (temp_field[(r - 1) * width + c] - temp_field[(r + 1) * width + c]) as f64
                / dy_1000km;
            let grad_mag = dtdx.hypot(dtdy);
            if grad_mag < grad_threshold_k_per_1000km {
                continue;
            }

            let u = u_wind[idx] as f64;
            let v = v_wind[idx] as f64;
            let adv = -(u * dtdx + v * dtdy);
            if adv.abs() < 4.0 {
                continue;
            }

            // Orient front tangent perpendicular to grad_S(T)
            let tx = -dtdy / grad_mag;
            let ty = dtdx / grad_mag;
            let lon = -180.0 + (c as f64 + 0.5) * dlon_deg;
            let half_lat = 0.42 * dlat_deg * ty;
            let half_lon = 0.42 * dlon_deg * tx / cos_phi.max(0.35);

            fronts.push(FrontSegment {
                lat1: (lat - half_lat).clamp(-85.0, 85.0),
                lon1: ((lon - half_lon + 180.0).rem_euclid(360.0)) - 180.0,
                lat2: (lat + half_lat).clamp(-85.0, 85.0),
                lon2: ((lon + half_lon + 180.0).rem_euclid(360.0)) - 180.0,
                is_cold_front: adv < 0.0,
            });
        }
    }

    fronts
}
