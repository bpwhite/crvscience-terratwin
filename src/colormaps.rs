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
                [245, 248, 255],
                [160, 210, 240],
                [70, 110, 160],
                [45, 45, 55],
                [195, 55, 28],
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

/// Shade underlying basemap (`bathymetry_topography_shaded`, `dark_telemetry_vector`, or `viirs_nightlights_baseline`)
/// from synthetic planetary elevation (meters) and `(lat, lon)`.
pub fn shade_basemap_rgb(basemap_id: &str, lat: f64, lon: f64, elev_m: f64) -> [u8; 3] {
    match basemap_id {
        "dark_telemetry_vector" => {
            if elev_m < 0.0 {
                [11, 17, 28]
            } else {
                [24, 31, 44]
            }
        }
        "viirs_nightlights_baseline" => {
            if elev_m < 0.0 {
                [6, 10, 18]
            } else {
                // Urban corridor nightlight glow proxy
                let city_glow = (((lon * 0.35).sin() * (lat * 0.42).cos()).abs())
                    * (-((lat.abs() - 34.0) / 22.0).powi(2)).exp();
                let g = (city_glow * 195.0).clamp(0.0, 220.0) as u8;
                [14_u8.saturating_add(g), 16_u8.saturating_add(g * 4 / 5), 24_u8.saturating_add(g / 3)]
            }
        }
        _ => {
            // GEBCO Bathymetry & FABDEM Terrain Relief
            if elev_m < 0.0 {
                let depth_frac = ((-elev_m) / 6000.0).clamp(0.0, 1.0) as f32;
                let r = (24.0 * (1.0 - depth_frac) + 7.0 * depth_frac) as u8;
                let g = (64.0 * (1.0 - depth_frac) + 21.0 * depth_frac) as u8;
                let b = (108.0 * (1.0 - depth_frac) + 46.0 * depth_frac) as u8;
                [r, g, b]
            } else {
                let h_frac = (elev_m / 4500.0).clamp(0.0, 1.0) as f32;
                if lat.abs() > 70.0 || elev_m > 3800.0 {
                    [198, 208, 218]
                } else {
                    let r = (34.0 + 96.0 * h_frac) as u8;
                    let g = (58.0 + 62.0 * h_frac) as u8;
                    let b = (42.0 + 48.0 * h_frac) as u8;
                    [r, g, b]
                }
            }
        }
    }
}
