# TerraTwin User Guide (`v2.5.0`)

TerraTwin (`terratwin`) is a native Rust desktop climate modeling workstation and headless HPC simulation daemon powered by `eframe`/`egui`, multi-threaded `rayon` physics solvers, and `rusqlite` WAL persistence. It implements the complete `assets/climate_modeling_hub_schema.json` specification across **6 domain groups and 34 predictive layers** spanning `1950`–`2100`, alongside an optional single-file HTML5 Canvas browser preview (`web/terratwin_live_preview.html`).

---

## 1. Installation & Build Options

### Prerequisites
- **Rust Toolchain**: `rustc` / `cargo` `1.80+`
- **System Libraries**:
  - **Windows / macOS**: Zero external C/OpenSSL dependencies (`rusqlite` bundles SQLite and `reqwest` uses `rustls-tls`).
  - **Linux (GUI build)**: `libxkbcommon-dev`, `libwayland-dev`, `libx11-dev`, `libgl1-mesa-dev`, `libgtk-3-dev`.

### Build Commands
```bash
# Full desktop GUI + headless daemon (default features)
cargo build --release

# Headless server-only binary (zero windowing/GPU dependencies)
cargo build --release --no-default-features

# Launch the interactive GUI
cargo run --release -- --db terratwin_climate.db --max-history-ticks 96
```

To inspect the companion browser preview without compiling Rust, open [`web/terratwin_live_preview.html`](../web/terratwin_live_preview.html) directly in a modern browser.

---

## 2. Desktop GUI Workflow

The native workstation is divided into five integrated regions:

### 2.1 Top Bar
- **Drawer Toggles**: `Layers` (`L`) and `Analytics` (`A`) collapse or expand the left and right side panels.
- **Live Readout Badges**: Hovering over any badge displays its physical definition, spatial domain, and upstream observational/modeling source:
  - **2m Temp Anom**: Global cosine-area-weighted 2 m air temperature anomaly vs. 1850–1900 pre-industrial baseline (ECMWF ERA5 / GISTEMP + TerraTwin coupled QGPV/EBM solver).
  - **Mauna Loa CO2**: Atmospheric $\text{CO}_2$ dry-air mole fraction (`ppm`) anchored to NOAA GML Mauna Loa & CMIP6 SSP pathways.
  - **ENSO Niño 3.4**: Equatorial Pacific (`5°S–5°N, 170°W–120°W`) SST anomaly index (`°C`) from NOAA CPC & 1.5-layer reduced-gravity ocean dynamics.
  - **Global SST Anom**: Global `60°S–60°N` Sea Surface Temperature anomaly (`°C`) from NOAA OISST v2.1 / CoastWatch ERDDAP.
- **Projection Switcher**: Switch dynamically between `3D Planetary Globe (WGS84 / EPSG:4978)`, `2D Equirectangular (EPSG:4326)`, `Arctic Polar Stereo (EPSG:3995)`, and `Antarctic Polar Stereo (EPSG:3031)` (hotkeys `1`–`4`).
- **SSP Scenario Selector**: Choose among IPCC AR6 pathways `SSP1-1.9`, `SSP1-2.6`, `SSP2-4.5`, `SSP3-7.0`, and `SSP5-8.5`.
- **Coordinate Search / Jump (`Coord:`)**: Enter `lat, lon` or `lat lon` (e.g., `36.0, -56.0`) and press `Enter` or click `Go` to center the camera and pin the vertical sounding/CTD target.
- **Tools**: `Crosshair` toggles the target pin reticle; `Transect A-B` lets you click two map points to plot a 64-segment great-circle SLERP cross-section; `Reset View` (`R`) resets center, zoom, and bearing; `About` (`F1`) shows version and citation metadata.

### 2.2 Left Sidebar (`layer_tree_controller`)
- **Anomaly Mode (vs Mean)**: Renders signed spatial deviations around each layer's global mean using the diverging `CoolWarm` palette and an explicit highlighted `0` marker on the viewport colorbar.
- **Basemap Selector**: Choose between *GEBCO Bathymetry & FABDEM Terrain Relief*, *Dark Scientific Telemetry Basemap*, and *VIIRS Black Marble Nighttime Lights*.
- **Quick Scenario Presets**:
  - **Baseline (Now)**: `SSP2-4.5`, present UTC date/time, `Live Telemetry & Nowcast` mode, default meteorological layers.
  - **Moderate SSP2-4.5**: `SSP2-4.5`, `2050.0`, `CMIP6/CMIP7 Projections` mode with CMIP6 ensemble ΔT, SST marine heatwaves, and synoptic fronts.
  - **High SSP5-8.5**: `SSP5-8.5`, `2100.0`, `CMIP6/CMIP7 Projections` mode with CMIP6 ensemble ΔT, SST marine heatwaves, WBGT heat stress, coastal surge, and sea-ice concentration.
- **Camera & Bearing**: Adjust `Bearing` (`-180°` to `+180°`) via slider or `0°` reset button, or `Shift+Drag` directly on the viewport.
- **Vertical Slice Engine**:
  - `Atm (hPa)`: Logarithmic isobaric slider from `1000 hPa` (surface) to `10 hPa` (stratosphere).
  - `Ocean (m)`: Subsurface ocean depth slider from `0 m` (sea surface) to `6000 m` (abyssal).
- **Live Layer Search (`Filter 34 layers...`)**: Case-insensitive real-time filter matching layer names, IDs, descriptions, units, domain groups, and public data sources (`Esc` or `✕` clears the query).
- **Per-Layer Controls**: Toggle visibility, click `👁 Solo` to isolate a single layer (click again to restore prior visibility), open `Info` for resolution/format/source citations, and adjust `Opacity` (`0.08–1.0`) and `Palette` (`Turbo`, `Viridis`, `Plasma`, `CoolWarm`, `CryoIce`, `Chlorophyll`, `Clean IR`).

### 2.3 Central Map Viewport (`global_map_canvas`)
- **3D Orthographic Globe (`EPSG:4978`)**: Features Lambertian solar shading, Eddington/Minnaert limb darkening, Fresnel Rayleigh atmospheric limb scattering, and HiDPI-responsive composite rasterization.
- **Graticules & Synoptic Overlays**:
  - `30°` parallels and meridians with degree labels (`60°N`, `30°S`, `0°`, `60°E`, `120°W`, etc.).
  - Marching-squares `4 hPa` MSLP isobars with deconflicted numeric `hPa` labels and `L`/`H` pressure extrema centers.
  - Baroclinic thermal fronts rendered with WMO blue triangular pips on cold fronts (pointing toward warm air) and red semicircular pips on warm fronts (bulging toward cold air).
  - Animated Lagrangian particle streamlines (`wind_vector_jet_streams`, `atmospheric_rivers_ivt`, `thermohaline_currents_amoc`) and teleconnection oscillation boxes (`ENSO Niño 3.4`, `PDO`, `NAO`, `IOD`, `AMO`).
- **Cursor HUD & 5-Tick Colorbars**:
  - Moving the pointer over the map displays `Cursor: lat°, lon° | Elev: m` alongside a real-time bilinear readout (`sample_grid_bilinear`) of the topmost finite visible scalar layer.
  - Bottom-right colorbar cards display 5 scale ticks (`0%`, `25%`, `50%`, `75%`, `100%`) with an explicit white `0` tick marker in Anomaly Mode.

### 2.4 Right Analytics Drawer (`telemetry_analytics_dock`)
- **Observatory Pins**: 20 one-click reference stations spanning all 7 continents and polar caps.
- **Active Layer Point Telemetry**: Bilinear readouts at the pinned `(lat, lon)` for every visible layer.
- **Vertical Profiles**:
  - *Atmospheric Vertical Sounding (`1000–10 hPa`)*: Temperature and dewpoint Skew-T / Log-P profile.
  - *Ocean Water-Column CTD (`0–6000 m`)* (when `elev < 0 m`): Temperature and salinity profile.
  - *Land Cryosphere & Permafrost (`0–20 m`)* (when `elev >= 0 m`): Ground/firn temperature, unfrozen liquid water curve, `0 °C` isotherm, and permafrost active-layer thaw depth.
- **CMIP6 1950–2100 Trajectory & Ensemble Spread**: Switch between `CO₂ (ppm)` and `Regional ΔT (°C)` (`Global Mean`, `Arctic >66.5°N`, `Mid-Latitudes`, `Tropics ±23.5°`, `Antarctic <66.5°S`, `Pinned Cell`) with a 2100 percentile boxplot.
- **Cross-Section Transect A-B**: Plots the active visible scalar along the 64-segment great-circle arc between points `A` and `B`.
- **Zarr v3 / JSON Subset Exporter**: Export regional bounding-box subsets (`±20° lat`, `±25° lon` around the pinned coordinate) directly from SQLite. Export and validation status messages remain visible in the bottom dock regardless of whether the right drawer is open or collapsed.

### 2.5 Bottom Temporal Controller (`temporal_dimension_controller`)
- **Play / Pause** (`Space`) and **Step Backward / Forward** (`ArrowLeft` / `ArrowRight`).
- **Temporal Modes**:
  - `Live Telemetry & Nowcast (-7d to +10d)` (`10m` step)
  - `Historical Reanalysis (1950-Present)` (`1d` step)
  - `CMIP6/CMIP7 Projections (2026-2100)` (`1M` step)
- **Live vs 4D Replay**: Toggle `4D Replay` to scrub through persisted SQLite simulation snapshots (`Tick i/N` slider and `◀`/`▶` buttons).

---

## 3. Controls & Keyboard Shortcuts

All map/navigation hotkeys are suppressed automatically while typing in text inputs (`Coord:` or layer `Search:`) via `!ctx.wants_keyboard_input()`.

| Input | Action |
| :--- | :--- |
| **Left Drag** | Pan map / rotate 3D globe center (bearing-aware in `EPSG:4978`) |
| **Shift + Left Drag** | Rotate camera `bearing_deg` (`-180°..=+180°`) |
| **Mouse Wheel** | Cursor-centered zoom (`0.55x–8.5x`) anchored under pointer coordinates |
| **Left Click** | Pin coordinate for Sounding/CTD/Cryosphere profile (or place `A`/`B` when Transect mode is active) |
| **`R`** | Reset camera center (`20°N, -20°E`), zoom (`1.0x`), and bearing (`0°`) |
| **`1` / `2` / `3` / `4`** | Switch projection: `1` 3D Globe (`EPSG:4978`), `2` 2D Equirectangular (`EPSG:4326`), `3` Arctic Polar (`EPSG:3995`), `4` Antarctic Polar (`EPSG:3031`) |
| **`L`** | Toggle Left Predictive Layer Tree sidebar |
| **`A`** | Toggle Right Telemetry Analytics drawer |
| **`Left` / `Right` Arrows** | Step 1 tick backward/forward in 4D Replay or ±1 step in Live/Historical/CMIP timeline |
| **`Space`** | Play / Pause timeline animation |
| **`F1`** | Toggle About TerraTwin window |
| **`Esc`** | Close About window -> exit/clear Transect A-B -> clear Layer Search filter |

---

## 4. Complete Catalog of All 34 Predictive Layers

### Group 1: Atmospheric Dynamics & Real-Time Meteorology
| Layer ID | Name | Vertical Slice | Units | Public Sources |
| :--- | :--- | :--- | :--- | :--- |
| `geo_multispectral_satellite` | Geostationary Multispectral Satellite Imagery | Cloud-top pressure (`hPa`) | `Kelvin / Reflectance` | NOAA GOES-16/18 (ABI), EUMETSAT MTG-I1, JAXA Himawari-9 |
| `volumetric_doppler_radar` | 3D Volumetric Doppler Radar & Hydrometeor Classification | Altitude MSL / `hPa` | `dBZ / mm/hr / m/s` | NOAA MRMS, NEXRAD Level-II, EUMETNET OPERA, NASA/JAXA GPM DPR |
| `synoptic_fronts_isobars` | Synoptic Fronts & Isobaric Topology | Surface MSLP | `hPa` | NOAA WPC Coded Surface Fronts, ECMWF ERA5, NOAA GFS |
| `wind_vector_jet_streams` | 3D Wind Vector Fields, Jet Streams & CAPE | `1000..=10 hPa` | `m/s (Wind) / J/kg (CAPE)` | ECMWF IFS / ERA5, NOAA GFS / HRRR |
| `atmospheric_rivers_ivt` | Atmospheric Rivers & Integrated Vapor Transport (IVT) | `1000..300 hPa` column | `kg m-1 s-1` | CW3E Scripps AR Catalog, NASA MERRA-2, NOAA GFS |
| `global_lightning_telemetry` | Global Optical & RF Lightning Flash Density | Cloud column | `flashes km-2 min-1` | NOAA GOES-R GLM, EUMETSAT MTG Lightning Imager, NASA ISS LIS |

### Group 2: Trace Gases, Aerosols & Radiative Forcing
| Layer ID | Name | Vertical Slice | Units | Public Sources |
| :--- | :--- | :--- | :--- | :--- |
| `column_co2_flux` | Column-Averaged Carbon Dioxide (XCO2) & Surface Flux | Total atmospheric column | `ppm / gC m-2 day-1` | NASA OCO-2 / OCO-3, JAXA GOSAT-2, NOAA GML CarbonTracker |
| `methane_plumes_background` | Methane (CH4) Super-Emitter Plumes & Background | Total column `ppb` | `ppb / kg CH4 hr-1` | ESA Sentinel-5P TROPOMI, NASA EMIT, CarbonMapper (Tanager), MethaneSAT |
| `tropospheric_no2_so2` | Tropospheric NO2, SO2 & Formaldehyde (HCHO) | Tropospheric column | `molecules cm-2` | NASA TEMPO, ESA Sentinel-5P TROPOMI, NIER GEMS |
| `aerosol_optical_depth_speciated` | Speciated Aerosols (AOD, PM2.5, Black Carbon, Dust, Sulfate) | Surface PM2.5 & 550 nm AOD | `ug m-3 / Dimensionless` | Copernicus CAMS, NASA MODIS/VIIRS, NASA GEOS-FP, OpenAQ |
| `toa_radiative_energy_budget` | Top-of-Atmosphere (TOA) Radiative Energy Imbalance & Albedo | Top of atmosphere | `W m-2` | NASA CERES (Terra/Aqua/NOAA-20), DSCOVR EPIC/NISTAR |

### Group 3: Physical & Biogeochemical Oceanography
| Layer ID | Name | Vertical Slice | Units | Public Sources |
| :--- | :--- | :--- | :--- | :--- |
| `sst_marine_heatwaves` | Sea Surface Temperature (SST), Anomalies & Marine Heatwaves | Sea surface skin | `degC` | NOAA CoralTemp, NOAA OISST v2.1, ESA Sentinel-3 SLSTR, GHRSST MUR |
| `subsurface_ocean_heat_salinity_3d` | 3D Subsurface Ocean Heat Content (OHC) & Halocline Profiles | `0..=6000 m` | `Zettajoules / J m-2 / PSU` | International Argo Array, NOAA NCEI, Copernicus GLORYS12 |
| `thermohaline_currents_amoc` | Ocean Surface Currents, Mesoscale Eddies & AMOC Transport | `0..=6000 m` | `m/s / Sverdrups` | NASA OSCAR, Navy HYCOM, Copernicus CMEMS, RAPID-AMOC Array |
| `swot_sea_surface_height_altimetry` | Sea Surface Height (SSH), Steric/Barystatic Rise & SWOT Altimetry | Sea surface geoid offset | `mm / cm` | NASA/CNES SWOT, Sentinel-6 Michael Freilich, PSMSL Tide Gauges |
| `ocean_acidification_carbonate` | Ocean Acidification (pH), pCO2 & Aragonite/Calcite Saturation | `0..=6000 m` | `pH (Total) / Omega_Aragonite` | SOCAT, GLODAP, BGC-Argo, Copernicus Marine Biogeochemistry |
| `marine_chlorophyll_hypoxia` | Hyperspectral Phytoplankton, Chlorophyll-a & Hypoxic Dead Zones | Photic zone & benthic O2 | `mg m-3 / umol kg-1` | NASA PACE (OCI), MODIS-Aqua / Sentinel-3 OLCI, World Ocean Atlas |
| `teleconnection_oscillation_masks` | Teleconnection Basin Masks (ENSO, PDO, IOD, AMO, NAO) | Surface | `Standardized Index / degC` | NOAA CPC, JAMSTEC, NCAR Climate Data Guide |

### Group 4: Cryosphere & Glacial Mass Balance
| Layer ID | Name | Vertical Slice | Units | Public Sources |
| :--- | :--- | :--- | :--- | :--- |
| `sea_ice_concentration_drift` | Sea Ice Extent, Concentration, Polynyas & Drift Vectors | Sea surface | `Percent (%) / cm s-1` | NSIDC Passive Microwave, JAXA AMSR2, ESA Sentinel-1 SAR Drift |
| `sea_ice_thickness_freeboard_age` | Sea Ice Thickness, Freeboard & Multi-Year Ice Age | Ice draft & freeboard | `Meters (m) / Years` | NASA ICESat-2 (ATLAS), ESA CryoSat-2, UW PIOMAS, NSIDC Sea Ice Age |
| `gravimetric_ice_sheet_mass_balance` | Greenland & Antarctic Gravimetric Mass Balance & Grounding Lines | Ice sheet column | `cm Water Eq. / Gt yr-1` | NASA/GFZ GRACE-FO, IMBIE, MEaSUREs Antarctic Boundaries |
| `alpine_glacier_velocity_rgi` | Global Mountain Glacier Outlines, Flow Velocity & Elevation Change | Terrain surface | `m yr-1 / m w.e. yr-1` | NASA ITS_LIVE, Randolph Glacier Inventory (RGI 7.0), GLIMS, WGMS |
| `permafrost_thermokarst_subsidence` | Permafrost Probability, Active Layer Thickness & InSAR Subsidence | Subsurface `0..=20 m` | `Meters / degC` | ESA CCI Permafrost, GTN-P Boreholes, Sentinel-1 InSAR, NASA SMAP |

### Group 5: Terrestrial Biosphere, Hydrology & Carbon Sinks
| Layer ID | Name | Vertical Slice | Units | Public Sources |
| :--- | :--- | :--- | :--- | :--- |
| `canopy_height_aboveground_biomass` | 3D Forest Canopy Height & Aboveground Biomass Carbon Density | Canopy height (`m`) | `Mg C ha-1 / Meters` | NASA GEDI (ISS), ESA BIOMASS, ESA CCI Biomass, Sentinel-2/Landsat |
| `sif_photosynthesis_evaporative_stress` | Solar-Induced Fluorescence (SIF), GPP & Canopy Evaporative Stress | Vegetation canopy | `mW m-2 sr-1 nm-1 / ESI` | NASA ECOSTRESS (ISS), OCO-2/3 SIF, Sentinel-5P TROPOMI SIF, MODIS |
| `soil_moisture_groundwater_grace` | Root-Zone Soil Moisture & Deep Aquifer Groundwater Anomalies | Root-zone `1 m` & aquifer | `m3 m-3 / cm Water Eq.` | NASA SMAP, NASA/DLR GRACE-FO, NASA GLDAS, USGS/GGMN |
| `swot_surface_water_river_discharge` | Global River Discharge, Reservoir Volume & Flood Inundation | Water surface elevation | `m3 s-1 / km3` | NASA/CNES SWOT Hydrology, JRC Global Surface Water, Copernicus GloFAS, Sentinel-1 |
| `active_wildfires_frp_burn_severity` | Active Fire Radiative Power (FRP), Burn Scars (dNBR) & Carbon Emissions | Surface & plume height | `Megawatts (FRP) / dNBR` | NASA FIRMS (VIIRS 375 m / MODIS), GOES ABI Fire, GFED5, MTBS |
| `dynamic_land_cover_deforestation` | 10m Near-Real-Time Land Cover, Deforestation & Peatland/Mangrove Loss | Surface | `Categorical Class / Confidence` | Dynamic World (Google/WRI), GLAD/UMD Forest Loss, Global Mangrove Watch |

### Group 6: Climate Risk, Extreme Attribution & CMIP Projections
| Layer ID | Name | Vertical Slice | Units | Public Sources |
| :--- | :--- | :--- | :--- | :--- |
| `cmip6_downscaled_ensemble` | CMIP6 / CMIP7 Downscaled Multi-Model Projections (SSP1 to SSP5) | Surface `2 m` | `degC / mm day-1 / Days` | NASA NEX-GDDP-CMIP6, WCRP CORDEX, Copernicus CDS |
| `wet_bulb_globe_temp_heat_stress` | Wet-Bulb Globe Temperature (WBGT) & Human Survivability Thresholds | Surface `2 m` | `degC (WBGT / Tw)` | ECMWF ERA5-HEAT, NOAA HeatRisk / NDFD, CMIP6 Derived WBGT |
| `coastal_inundation_storm_surge` | Dynamic Coastal Sea-Level Inundation & Return-Period Storm Surge | Elevation (NAVD88 / EGM2008) | `Meters above MHHW` | NOAA SLR Viewer DEMs, Deltares GTSM, FABDEM / ICESat-2 Coastal |
| `compound_hazard_fire_weather_index` | Compound Hazard Index & Canadian Forest Fire Weather Index (FWI) | Surface | `FWI Index (0-100+)` | Copernicus GWIS, NOAA NCEP, WCRP Compound Extremes Atlas |
| `critical_infrastructure_socioeconomic_exposure` | Critical Infrastructure, Agro-Ecological Yield Shifts & Social Vulnerability | Surface infrastructure | `Risk Score (0-100) / MW / %` | OpenStreetMap / OpenInfraMap, NASA SEDAC GPWv4, FEMA NRI / CDC SVI, FAO GAEZ v4 |

---

## 5. CLI Usage & Subcommands

Global options available on all commands:
- `--db <PATH>` (default: `terratwin_climate.db`)
- `--schema <PATH>` (optional external JSON schema override)
- `--scenario <CODE>` (`SSP1-1.9`, `SSP1-2.6`, `SSP2-4.5`, `SSP3-7.0`, `SSP5-8.5`; default: `SSP2-4.5`)
- `--threads <N>` (Rayon thread count; `0` = all CPU cores)
- `--max-history-ticks <N>` (SQLite ring-buffer retention per scenario; `0` = unlimited; default: `96`)

```bash
# Launch GUI (default) or explicit `gui` subcommand
terratwin --db terratwin_climate.db gui

# Headless daemon mode (background ingest + simulation loop)
terratwin --headless --db terratwin_climate.db --scenario SSP2-4.5
terratwin daemon --ingest-interval-secs 300 --sim-interval-secs 30 --max-cycles 10

# Render offscreen 1600x1000 PNG screenshot (views: 1=3D Globe, 2=2D Cryosphere, 3=SSP5-8.5 Ocean & Transect)
terratwin --screenshot out_view3.png --screenshot-view 3

# Single synchronous simulation step (use --offline for air-gapped climatology seed)
terratwin step --year 2050.0 --scenario SSP3-7.0 --hpa 850 --depth-m 100 --offline

# Inspect SQLite database catalog and latest readouts
terratwin status --db terratwin_climate.db

# Export regional Zarr v3 store (zstd, gzip, or none) or JSON subset
terratwin export --layer sst_marine_heatwaves --output pacific_sst.zarr --codec zstd --compression-level 3 --min-lat 20 --max-lat 55 --min-lon -160 --max-lon -120 --chunk-rows 36 --chunk-cols 72
terratwin export --layer cmip6_downscaled_ensemble --output europe.json --json-subset --min-lat 35 --max-lat 70 --min-lon -12 --max-lon 35

# Prune expired ticks/observations and reclaim WAL disk space
terratwin prune --keep-ticks 96 --keep-raw-per-layer 4096 --vacuum
```
