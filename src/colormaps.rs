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

//! Perceptually uniform scientific colormaps and basemap shading for all 34 predictive layers.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColormapPalette {
    Turbo,
    Viridis,
    Plasma,
    CoolWarmDiverging,
    CryoIce,
    ChlorophyllBio,
    ThermalIr,
}

impl ColormapPalette {
    pub const ALL: [ColormapPalette; 7] = [
        ColormapPalette::Turbo,
        ColormapPalette::Viridis,
        ColormapPalette::Plasma,
        ColormapPalette::CoolWarmDiverging,
        ColormapPalette::CryoIce,
        ColormapPalette::ChlorophyllBio,
        ColormapPalette::ThermalIr,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ColormapPalette::Turbo => "Turbo (Rainbow Perceptual)",
            ColormapPalette::Viridis => "Viridis (Sequential)",
            ColormapPalette::Plasma => "Plasma (High-Energy)",
            ColormapPalette::CoolWarmDiverging => "CoolWarm (Diverging Anomaly)",
            ColormapPalette::CryoIce => "CryoIce (Polar Albedo)",
            ColormapPalette::ChlorophyllBio => "Chlorophyll / Biosphere",
            ColormapPalette::ThermalIr => "Clean IR / Satellite",
        }
    }

    /// Map normalized value `t` in `[0.0, 1.0]` to `[R, G, B]` bytes.
    pub fn sample_rgb(&self, t: f32) -> [u8; 3] {
        let x = t.clamp(0.0, 1.0);
        let stops: &[[u8; 3]] = match self {
            ColormapPalette::Turbo => &[
                [48, 18, 59],
                [69, 91, 205],
                [40, 188, 235],
                [109, 254, 98],
                [238, 207, 58],
                [246, 108, 25],
                [122, 4, 3],
            ],
            ColormapPalette::Viridis => &[
                [68, 1, 84],
                [59, 82, 139],
                [33, 145, 140],
                [94, 201, 98],
                [253, 231, 37],
            ],
            ColormapPalette::Plasma => &[
                [13, 8, 135],
                [126, 3, 168],
                [204, 71, 120],
                [248, 149, 64],
                [240, 249, 33],
            ],
            ColormapPalette::CoolWarmDiverging => &[
                [33, 102, 172],
                [103, 169, 207],
                [247, 247, 247],
                [239, 138, 98],
                [178, 24, 43],
            ],
            ColormapPalette::CryoIce => &[
                [8, 29, 88],
                [34, 94, 168],
                [65, 182, 196],
                [199, 233, 180],
                [255, 255, 255],
            ],
            ColormapPalette::ChlorophyllBio => &[
                [14, 34, 56],
                [22, 105, 122],
                [64, 168, 96],
                [158, 218, 68],
                [244, 248, 118],
            ],
            ColormapPalette::ThermalIr => &[
                [255, 255, 255],
                [172, 136, 255],
                [68, 192, 252],
                [205, 226, 246],
                [112, 136, 166],
                [42, 48, 58],
                [22, 26, 34],
            ],
        };

        let segs = (stops.len() - 1) as f32;
        let pos = x * segs;
        let idx = (pos.floor() as usize).min(stops.len() - 2);
        let frac = pos - idx as f32;
        let a = stops[idx];
        let b = stops[idx + 1];
        [
            (a[0] as f32 + (b[0] as f32 - a[0] as f32) * frac) as u8,
            (a[1] as f32 + (b[1] as f32 - a[1] as f32) * frac) as u8,
            (a[2] as f32 + (b[2] as f32 - a[2] as f32) * frac) as u8,
        ]
    }
}

/// Recommended scientific palette and `[min, max]` physical range for each of the 34 layers.
pub fn default_layer_style(layer_id: &str) -> (ColormapPalette, f32, f32) {
    match layer_id {
        "geo_multispectral_satellite" => (ColormapPalette::ThermalIr, 205.0, 305.0),
        "volumetric_doppler_radar" => (ColormapPalette::Turbo, 2.0, 62.0),
        "synoptic_fronts_isobars" => (ColormapPalette::CoolWarmDiverging, 985.0, 1038.0),
        "wind_vector_jet_streams" => (ColormapPalette::Plasma, 0.0, 58.0),
        "atmospheric_rivers_ivt" => (ColormapPalette::Viridis, 150.0, 1100.0),
        "global_lightning_telemetry" => (ColormapPalette::Plasma, 0.2, 18.0),

        "column_co2_flux" => (ColormapPalette::Turbo, 390.0, 460.0),
        "methane_plumes_background" => (ColormapPalette::Plasma, 1860.0, 2080.0),
        "tropospheric_no2_so2" => (ColormapPalette::Plasma, 0.5e15, 1.2e16),
        "aerosol_optical_depth_speciated" => (ColormapPalette::Turbo, 0.04, 0.85),
        "toa_radiative_energy_budget" => (ColormapPalette::CoolWarmDiverging, -95.0, 95.0),

        "sst_marine_heatwaves" => (ColormapPalette::Turbo, -1.8, 31.5),
        "subsurface_ocean_heat_salinity_3d" => (ColormapPalette::Plasma, 0.0, 12.5),
        "thermohaline_currents_amoc" => (ColormapPalette::Viridis, 0.0, 1.4),
        "swot_sea_surface_height_altimetry" => (ColormapPalette::CoolWarmDiverging, -35.0, 55.0),
        "ocean_acidification_carbonate" => (ColormapPalette::Viridis, 7.65, 8.20),
        "marine_chlorophyll_hypoxia" => (ColormapPalette::ChlorophyllBio, 0.05, 4.0),
        "teleconnection_oscillation_masks" => (ColormapPalette::CoolWarmDiverging, -2.5, 2.5),

        "sea_ice_concentration_drift" => (ColormapPalette::CryoIce, 5.0, 100.0),
        "sea_ice_thickness_freeboard_age" => (ColormapPalette::CryoIce, 0.1, 4.5),
        "gravimetric_ice_sheet_mass_balance" => (ColormapPalette::CoolWarmDiverging, -65.0, 15.0),
        "alpine_glacier_velocity_rgi" => (ColormapPalette::CryoIce, 5.0, 350.0),
        "permafrost_thermokarst_subsidence" => (ColormapPalette::Viridis, 0.1, 3.8),

        "canopy_height_aboveground_biomass" => (ColormapPalette::ChlorophyllBio, 5.0, 310.0),
        "sif_photosynthesis_evaporative_stress" => (ColormapPalette::ChlorophyllBio, 0.05, 3.2),
        "soil_moisture_groundwater_grace" => (ColormapPalette::Viridis, 0.05, 0.45),
        "swot_surface_water_river_discharge" => (ColormapPalette::CryoIce, 10.0, 2200.0),
        "active_wildfires_frp_burn_severity" => (ColormapPalette::Plasma, 2.0, 180.0),
        "dynamic_land_cover_deforestation" => (ColormapPalette::ChlorophyllBio, 0.0, 8.0),

        "cmip6_downscaled_ensemble" => (ColormapPalette::CoolWarmDiverging, -2.0, 6.5),
        "wet_bulb_globe_temp_heat_stress" => (ColormapPalette::Turbo, -15.0, 36.5),
        "coastal_inundation_storm_surge" => (ColormapPalette::Plasma, 0.1, 3.5),
        "compound_hazard_fire_weather_index" => (ColormapPalette::Turbo, 0.0, 85.0),
        "critical_infrastructure_socioeconomic_exposure" => (ColormapPalette::Plasma, 0.0, 90.0),
        _ => (ColormapPalette::Viridis, 0.0, 100.0),
    }
}

#[inline]
fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let x = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * x,
        a[1] + (b[1] - a[1]) * x,
        a[2] + (b[2] - a[2]) * x,
    ]
}

#[inline]
fn gauss2(lat: f64, lon: f64, c_lat: f64, c_lon: f64, s_lat: f64, s_lon: f64) -> f64 {
    let mut dlon = lon - c_lon;
    if dlon > 180.0 {
        dlon -= 360.0;
    } else if dlon < -180.0 {
        dlon += 360.0;
    }
    let dy = (lat - c_lat) / s_lat;
    let dx = dlon / s_lon;
    (-(dy * dy + dx * dx)).exp()
}

/// Real-Earth desert/aridity weight (`0.0` = humid forest/grassland, `1.0` = hyper-arid desert).
#[inline]
fn earth_desert_weight(lat: f64, lon: f64) -> f32 {
    let w = (1.15 * gauss2(lat, lon, 23.0, 11.0, 8.5, 23.0) // Sahara
        + 1.10 * gauss2(lat, lon, 22.0, 46.0, 7.5, 11.0) // Arabian
        + 0.95 * gauss2(lat, lon, 40.5, 84.0, 6.0, 19.0) // Taklamakan / Gobi
        + 0.95 * gauss2(lat, lon, -24.0, 132.0, 7.0, 14.0) // Australian Outback
        + 0.80 * gauss2(lat, lon, 31.0, -112.0, 6.5, 8.5) // Sonora / Mojave
        + 0.85 * gauss2(lat, lon, -22.5, -69.0, 6.5, 3.8) // Atacama
        + 0.82 * gauss2(lat, lon, -23.0, 17.0, 7.0, 5.5) // Namib / Kalahari
        + 0.72 * gauss2(lat, lon, 27.0, 64.0, 5.5, 7.5)) // Thar / Lut
    .clamp(0.0, 1.0);
    w as f32
}

/// Shade underlying basemap (`bathymetry_topography_shaded`, `dark_telemetry_vector`, or `viirs_nightlights_baseline`)
/// from synthetic planetary elevation (meters) and `(lat, lon)`.
pub fn shade_basemap_rgb(basemap_id: &str, lat: f64, lon: f64, elev_m: f64) -> [u8; 3] {
    shade_basemap_with_slope_rgb(basemap_id, lat, lon, elev_m, 0.0, 0.0)
}

/// Shade underlying basemap with NW-illuminated terrain and bathymetric slope relief `(d_elev_dx, d_elev_dy)` in `m/deg`.
pub fn shade_basemap_with_slope_rgb(
    basemap_id: &str,
    lat: f64,
    lon: f64,
    elev_m: f64,
    d_elev_dx: f64,
    d_elev_dy: f64,
) -> [u8; 3] {
    match basemap_id {
        "dark_telemetry_vector" => {
            if elev_m < 0.0 {
                [10, 16, 27]
            } else {
                [24, 32, 45]
            }
        }
        "viirs_nightlights_baseline" => {
            if elev_m < 0.0 {
                [6, 10, 18]
            } else {
                // Realistic major urban/metropolitan corridors
                let urban = (1.00 * gauss2(lat, lon, 40.7, -76.0, 5.0, 9.0) // US Northeast / Midwest
                    + 0.75 * gauss2(lat, lon, 34.5, -118.0, 3.5, 4.5) // California
                    + 0.95 * gauss2(lat, lon, 50.5, 6.0, 5.5, 11.0) // Western Europe
                    + 0.85 * gauss2(lat, lon, 44.5, 11.5, 3.5, 7.0) // Northern Italy / Central Europe
                    + 0.90 * gauss2(lat, lon, 27.5, 78.5, 6.0, 10.5) // Indo-Gangetic Plain
                    + 1.05 * gauss2(lat, lon, 34.0, 116.5, 7.5, 9.0) // Eastern China
                    + 0.95 * gauss2(lat, lon, 35.8, 138.0, 3.0, 4.5) // Tokyo / Honshu
                    + 0.85 * gauss2(lat, lon, 30.2, 31.2, 2.5, 2.2) // Nile Delta
                    + 0.75 * gauss2(lat, lon, -23.5, -46.8, 3.2, 4.0) // Sao Paulo / Rio
                    + 0.72 * gauss2(lat, lon, 13.8, 100.8, 4.5, 5.0)) // SE Asia
                .clamp(0.0, 1.0);
                let g = (urban * 210.0) as u8;
                [
                    12_u8.saturating_add(g),
                    15_u8.saturating_add((g as u16 * 4 / 5) as u8),
                    24_u8.saturating_add((g as u16 / 3) as u8),
                ]
            }
        }
        _ => {
            // NW directional hillshade illumination (subtle slope exaggeration for planetary scale)
            let sx = (d_elev_dx / 2400.0).clamp(-0.75, 0.75) as f32;
            let sy = (d_elev_dy / 2400.0).clamp(-0.75, 0.75) as f32;
            let shade = (1.0 + 0.34 * (-0.707 * sx + 0.707 * sy)).clamp(0.68, 1.32);

            if elev_m < 0.0 {
                // GEBCO bathymetry: shelf -> slope -> abyssal plain -> ocean trench
                let depth = (-elev_m) as f32;
                let rgb = if depth <= 260.0 {
                    lerp_rgb([30.0, 92.0, 134.0], [18.0, 62.0, 106.0], depth / 260.0)
                } else if depth <= 2600.0 {
                    lerp_rgb(
                        [18.0, 62.0, 106.0],
                        [12.0, 39.0, 76.0],
                        (depth - 260.0) / 2340.0,
                    )
                } else if depth <= 5600.0 {
                    lerp_rgb(
                        [12.0, 39.0, 76.0],
                        [7.0, 21.0, 48.0],
                        (depth - 2600.0) / 3000.0,
                    )
                } else {
                    lerp_rgb(
                        [7.0, 21.0, 48.0],
                        [4.0, 12.0, 32.0],
                        (depth - 5600.0) / 4000.0,
                    )
                };
                let bathy_shade = 0.85 + 0.15 * shade;
                [
                    (rgb[0] * bathy_shade).clamp(0.0, 255.0) as u8,
                    (rgb[1] * bathy_shade).clamp(0.0, 255.0) as u8,
                    (rgb[2] * bathy_shade).clamp(0.0, 255.0) as u8,
                ]
            } else {
                let on_greenland =
                    (59.5..=84.0).contains(&lat) && (-73.5..=-11.5).contains(&lon) && elev_m > 340.0;
                let on_antarctica = lat < -63.0;
                let h = elev_m as f32;

                let rgb = if on_antarctica || on_greenland || h > 5200.0 {
                    lerp_rgb([188.0, 200.0, 214.0], [224.0, 232.0, 242.0], (h / 4200.0).clamp(0.0, 1.0))
                } else {
                    let desert = earth_desert_weight(lat, lon);
                    let boreal_tundra = ((lat.abs() - 61.0) / 11.0).clamp(0.0, 1.0) as f32;
                    let low_humid = [34.0, 76.0, 48.0];
                    let low_arid = [146.0, 124.0, 86.0];
                    let mut base = lerp_rgb(low_humid, low_arid, desert);
                    base = lerp_rgb(base, [92.0, 102.0, 84.0], 0.55 * boreal_tundra);

                    if h <= 1400.0 {
                        lerp_rgb(base, [88.0, 102.0, 68.0], (h / 1400.0) * 0.45)
                    } else if h <= 3600.0 {
                        let mid = lerp_rgb(base, [116.0, 104.0, 82.0], 0.70);
                        lerp_rgb(mid, [132.0, 126.0, 118.0], (h - 1400.0) / 2200.0)
                    } else {
                        lerp_rgb(
                            [132.0, 126.0, 118.0],
                            [194.0, 204.0, 216.0],
                            ((h - 3600.0) / 1600.0).clamp(0.0, 0.78),
                        )
                    }
                };

                [
                    (rgb[0] * shade).clamp(0.0, 255.0) as u8,
                    (rgb[1] * shade).clamp(0.0, 255.0) as u8,
                    (rgb[2] * shade).clamp(0.0, 255.0) as u8,
                ]
            }
        }
    }
}
