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

//! Strongly-typed schema model matching `assets/climate_modeling_hub_schema.json`.
//! Parses all 6 layer groups, 34 predictive layers, 4 projections, vertical slice levels,
//! basemaps, and UI dock definitions at startup and registers them in SQLite.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Compile-time embedded canonical schema JSON.
pub const EMBEDDED_SCHEMA_JSON: &str = include_str!("../assets/climate_modeling_hub_schema.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HubSchema {
    pub application: ApplicationMeta,
    pub main_viewport: MainViewportSpec,
    pub ui_layout: UiLayoutSpec,
    pub layer_groups: Vec<LayerGroupSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApplicationMeta {
    pub id: String,
    pub name: String,
    pub version: String,
    pub architecture: String,
    pub compute_backend: ComputeBackendSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputeBackendSpec {
    pub ingestion_engine: String,
    pub tile_server: String,
    pub reprojection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MainViewportSpec {
    #[serde(rename = "type")]
    pub viewport_type: String,
    pub default_projection: String,
    pub supported_projections: Vec<ProjectionSpec>,
    pub camera_defaults: CameraDefaults,
    pub vertical_slice_engine: VerticalSliceEngineSpec,
    pub base_maps: Vec<BaseMapSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectionSpec {
    pub id: String,
    pub label: String,
    pub epsg: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraDefaults {
    pub longitude: f64,
    pub latitude: f64,
    pub altitude_meters: f64,
    pub pitch_degrees: f64,
    pub bearing_degrees: f64,
    pub vertical_exaggeration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerticalSliceEngineSpec {
    pub enabled: bool,
    pub atmospheric_levels_hpa: Vec<u32>,
    pub ocean_depth_levels_meters: Vec<u32>,
    pub cross_section_transect_tool: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseMapSpec {
    pub id: String,
    pub name: String,
    pub source: String,
    pub format: String,
    pub default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiLayoutSpec {
    pub top_bar: TopBarSpec,
    pub left_sidebar: LeftSidebarSpec,
    pub right_drawer: RightDrawerSpec,
    pub bottom_dock: BottomDockSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TopBarSpec {
    pub components: Vec<String>,
    pub live_readouts: Vec<LiveReadoutSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveReadoutSpec {
    pub id: String,
    pub label: String,
    pub unit: String,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeftSidebarSpec {
    pub id: String,
    pub collapsible: bool,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RightDrawerSpec {
    pub id: String,
    pub collapsible: bool,
    pub panels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BottomDockSpec {
    pub id: String,
    pub range_start: String,
    pub range_end: String,
    pub modes: Vec<TemporalModeSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalModeSpec {
    pub id: String,
    pub label: String,
    pub step: String,
    #[serde(default)]
    pub scenarios: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerGroupSpec {
    pub group_id: String,
    pub group_name: String,
    pub icon: String,
    pub default_expanded: bool,
    pub layers: Vec<LayerSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerSpec {
    pub id: String,
    pub name: String,
    pub description: String,
    pub public_sources: Vec<String>,
    pub render_type: String,
    pub data_format: String,
    pub spatial_resolution: String,
    pub temporal_resolution: String,
    pub vertical_dimension: String,
    pub units: String,
    #[serde(default)]
    pub sub_bands: Option<Vec<String>>,
    #[serde(default)]
    pub selectable_pressure_levels_hpa: Option<Vec<u32>>,
    #[serde(default)]
    pub sub_variables: Option<Vec<String>>,
    pub default_visible: bool,
    pub default_opacity: f32,
}

/// Shared Socioeconomic Pathway (SSP) forcing scenarios from IPCC AR6 / CMIP6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SspScenario {
    Ssp1_19,
    Ssp1_26,
    Ssp2_45,
    Ssp3_70,
    Ssp5_85,
}

impl SspScenario {
    pub const ALL: [SspScenario; 5] = [
        SspScenario::Ssp1_19,
        SspScenario::Ssp1_26,
        SspScenario::Ssp2_45,
        SspScenario::Ssp3_70,
        SspScenario::Ssp5_85,
    ];

    pub fn code(&self) -> &'static str {
        match self {
            SspScenario::Ssp1_19 => "SSP1-1.9",
            SspScenario::Ssp1_26 => "SSP1-2.6",
            SspScenario::Ssp2_45 => "SSP2-4.5",
            SspScenario::Ssp3_70 => "SSP3-7.0",
            SspScenario::Ssp5_85 => "SSP5-8.5",
        }
    }

    pub fn from_code(s: &str) -> Self {
        match s.trim().to_uppercase().as_str() {
            "SSP1-1.9" | "SSP1_19" | "SSP119" => SspScenario::Ssp1_19,
            "SSP1-2.6" | "SSP1_26" | "SSP126" => SspScenario::Ssp1_26,
            "SSP3-7.0" | "SSP3_70" | "SSP370" => SspScenario::Ssp3_70,
            "SSP5-8.5" | "SSP5_85" | "SSP585" => SspScenario::Ssp5_85,
            _ => SspScenario::Ssp2_45,
        }
    }

    /// Radiative forcing target at 2100 (W m^-2) and projected 2100 CO2 concentration (ppm)
    /// aligned with Meinshausen et al. (2020) CMIP6 GHG concentrations.
    pub fn forcing_2100_wm2(&self) -> f64 {
        match self {
            SspScenario::Ssp1_19 => 1.9,
            SspScenario::Ssp1_26 => 2.6,
            SspScenario::Ssp2_45 => 4.5,
            SspScenario::Ssp3_70 => 7.0,
            SspScenario::Ssp5_85 => 8.5,
        }
    }

    pub fn co2_ppm_at_year(&self, year: f64) -> f64 {
        if year <= 1950.0 {
            return 311.0;
        }
        if year <= 2026.0 {
            // Historical Keeling Curve fit (1950: ~311 ppm -> 2026: ~427.5 ppm)
            let frac = (year - 1950.0) / (2026.0 - 1950.0);
            return 311.0 + 116.5 * frac.powf(1.38);
        }
        let t = ((year - 2026.0) / (2100.0 - 2026.0)).clamp(0.0, 1.0);
        match self {
            // Peak around 2040 (~440 ppm) then net-negative drawdown to ~393 ppm
            SspScenario::Ssp1_19 => 427.5 + 32.0 * (std::f64::consts::PI * t).sin() - 34.5 * t,
            // Peak mid-century (~468 ppm) then decline to ~446 ppm
            SspScenario::Ssp1_26 => 427.5 + 48.0 * (std::f64::consts::PI * 0.85 * t).sin(),
            // Intermediate stabilization toward ~603 ppm by 2100
            SspScenario::Ssp2_45 => 427.5 + 175.5 * t.powf(1.05),
            // High regional rivalry pathway toward ~867 ppm by 2100
            SspScenario::Ssp3_70 => 427.5 + 439.5 * t.powf(1.32),
            // Fossil-fueled intensive growth toward ~1135 ppm by 2100
            SspScenario::Ssp5_85 => 427.5 + 707.5 * t.powf(1.48),
        }
    }
}

impl HubSchema {
    /// Load from an external JSON file path if provided, otherwise parse the embedded schema.
    pub fn load(custom_path: Option<&std::path::Path>) -> Result<Self> {
        let raw = if let Some(path) = custom_path {
            std::fs::read_to_string(path)
                .with_context(|| format!("Failed reading schema JSON at {}", path.display()))?
        } else {
            EMBEDDED_SCHEMA_JSON.to_string()
        };
        let schema: HubSchema =
            serde_json::from_str(&raw).context("Failed parsing climate_modeling_hub_schema.json")?;
        Ok(schema)
    }

    /// Flatten all predictive layers across all groups in canonical order.
    pub fn all_layers(&self) -> Vec<(&LayerGroupSpec, &LayerSpec)> {
        let mut out = Vec::new();
        for g in &self.layer_groups {
            for l in &g.layers {
                out.push((g, l));
            }
        }
        out
    }

    pub fn find_layer(&self, layer_id: &str) -> Option<&LayerSpec> {
        self.layer_groups
            .iter()
            .flat_map(|g| g.layers.iter())
            .find(|l| l.id == layer_id)
    }
}
