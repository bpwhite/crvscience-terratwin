# TerraTwin HPC Climate Modeling & Telemetry Hub (`v2.5.0`)

A native Rust **headless background daemon + desktop global map GUI** (`terratwin`) paired with a standalone **HTML5 interactive browser workstation** (`web/terratwin_live_preview.html`), implementing the complete `climate_modeling_hub_schema.json` specification across **6 layer groups and 34 predictive layers** from `1950` to `2100`.

## Build & Run Quickstart

Ensure a standard Rust toolchain (`rustc` / `cargo` 1.80+) is installed. Both SQLite (`rusqlite` with `bundled`) and TLS (`reqwest` with `rustls-tls`) compile cleanly out of the box with no external OpenSSL or C library setup required. To explore the interactive web workstation immediately without compiling, open `web/terratwin_live_preview.html` directly in any modern browser.

```bash
# 1. Build optimized native binary (includes both Headless Daemon + Desktop GUI)
cargo build --release

# 2. Launch the Interactive Global Map GUI + Background Ingestion & Rayon Workers
cargo run --release -- --db terratwin_climate.db --max-history-ticks 96

# 3. Run purely Headless in the background (downloads telemetry + steps all 34 layers)
cargo run --release -- daemon --db terratwin_climate.db --ingest-interval-secs 300 --sim-interval-secs 30 --scenario SSP2-4.5

# 4. Build a minimal server-only binary with zero GUI/windowing dependencies
cargo build --release --no-default-features

# 5. Execute a single synchronous simulation pass (online or air-gapped --offline) and print JSON readouts
cargo run --release -- step --db terratwin_climate.db --year 2050.0 --scenario SSP3-7.0 --hpa 850 --depth-m 100

# 6. Export a chunked Zarr v3 binary directory store compressed with Zstandard (or Gzip)
cargo run --release -- export --db terratwin_climate.db --layer sst_marine_heatwaves --output pacific_sst.zarr --codec zstd --compression-level 3 --min-lat 20 --max-lat 55 --min-lon -160 --max-lon -120

# 7. Prune historical SQLite ticks & raw observations and reclaim disk space via WAL checkpoint + VACUUM
cargo run --release -- prune --db terratwin_climate.db --keep-ticks 96 --keep-raw-per-layer 4096 --vacuum

# 8. Query SQLite catalog & compression statistics, or run the integration test suite
cargo run --release -- status --db terratwin_climate.db
cargo test
```

## Crate & Web Workstation Architecture

| Path | Responsibility |
| :--- | :--- |
| `src/schema.rs` | Compile-time and runtime parser for `assets/climate_modeling_hub_schema.json` (6 layer groups, 34 layers, 4 projections, 10 pressure levels `1000..=10 hPa`, 10 ocean depths `0..=6000 m`, and 5 IPCC SSP pathways). |
| `src/db.rs` | Thread-safe SQLite WAL persistence with lossless `TTZ1` IEEE-754 byte-shuffled `lz4_flex` raster blob compression, automatic `RetentionPolicy` 4D tick ring-buffer pruning, and [Zarr v3 Core Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) chunked directory export (`zstd`, `gzip`, `none`). |
| `src/ingest.rs` | Non-blocking `tokio` + `reqwest` background downloader querying [Open-Meteo Weather, Marine, Air Quality & GloFAS Flood APIs](https://open-meteo.com/en/features), [NOAA GML Mauna Loa CO2 Trends](https://gml.noaa.gov/ccgg/trends/monthly.html), [NASA FIRMS Active Fires](https://firms.modaps.eosdis.nasa.gov/), [CoastWatch ERDDAP OISST v2.1](https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.html), and [NOAA CPC ENSO Diagnostics](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml). |
| `src/interp.rs` | Multi-threaded `rayon` preconditioned conjugate-gradient (`PCG`) [3D-Var Mass-Wind Balance](https://journals.ametsoc.org/view/journals/atot/25/11/2008jtecho594_1.pdf) assimilation solver with `quality_flag` error weights and flow-aligned Matérn-3/2 `SOAR` covariance, 16-case marching-squares `4 hPa` MSLP isobars, `L`/`H` extrema detection, baroclinic front detection, and `EPSG:4978` / `4326` / `3995` / `3031` projections. |
| `src/physics.rs` | Coupled geophysical solver running [Phillips 2-Layer Spherical QGPV](https://journals.ametsoc.org/view/journals/amsm/59/1/amsmonographs-d-18-0017.1.xml) with [Two-Layer QG Modal Helmholtz Inversion](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/quantitatively-mapping-the-eady-model-onto-a-twolayer-quasigeostrophic-model/B5FDCE9FA32565962769B8234057758B) and Charney Linear Balance MSLP, [Revised Cane-Zebiak 1.5-Layer Reduced-Gravity Ocean](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.899323/full) + [Stommel-Munk Wind-Driven Gyres](https://arxiv.org/html/1210.3630), `CQMSL` semi-Lagrangian advection, Skew-T soundings, and Ocean CTD profiles across all 34 layers. |
| `src/worker.rs` | Lock-free snapshot coordinator coupling the async Tokio HTTP ingestion loop with the Rayon multi-threaded physics worker for both Headless and GUI modes. |
| `src/gui.rs` | Feature-gated (`gui`) native `eframe` / `egui` desktop workstation with interactive 3D Globe / 2D / Polar Stereographic viewport, 4D SQLite tick timeline playback scrubber, marching-squares `4 hPa` isobar & cold/warm front overlays, Skew-T & CTD profiles, CMIP6 ensemble chart, and A-to-B transect tool. |
| `src/colormaps.rs` | Perceptually uniform scientific palettes (`Turbo`, `Viridis`, `Plasma`, `CoolWarm`, `CryoIce`, `Chlorophyll`, `ThermalIR`) and shaded relief basemaps. |
| `src/coastlines.rs` | Embedded continental coastline polylines, teleconnection oscillation boxes (`ENSO Niño 3.4`, `PDO`, `NAO`, `IOD`, `AMO`), and 20 major continental cities and polar stations spanning all 7 continents. |
| `web/terratwin_live_preview.html` | Single-file HTML5 Canvas browser workstation rendering [Natural Earth 1:110m World Atlas](https://www.jsdelivr.com/package/npm/world-atlas) coastlines, all 34 layers, 2-layer QGPV + Stommel-Munk ocean dynamics, 4D ring-buffer playback, Skew-T / CTD profiles, and Zarr v3 subset export. |

## Public Telemetry Sources & Plain-Text Physical Formulations

| Domain Group | Assimilated Public Sources | Simulation & Diagnostic Formulations |
| :--- | :--- | :--- |
| **1. Atmospheric Dynamics & Meteorology** (6 layers) | [Open-Meteo Forecast & ERA5](https://open-meteo.com/en/features), [Copernicus C3S Surface Air Temperature](https://climate.copernicus.eu/surface-air-temperature-october-2026), [NOAA GOES on AWS](https://registry.opendata.aws/noaa-goes/) | 2-layer spherical QGPV barotropic/baroclinic modal RB-SOR inversion, Charney linear balance MSLP, Bolton (1980) saturation vapor pressure, Marshall-Palmer reflectivity (`Z = 200 * R^1.6`), and Integrated Vapor Transport (`IVT`). |
| **2. Trace Gases, Aerosols & Radiation** (5 layers) | [NOAA GML Mauna Loa Observatory](https://gml.noaa.gov/obop/mlo/), [NOAA GML Carbon Cycle Trends](https://gml.noaa.gov/ccgg/trends/monthly.html), [Copernicus Climate Change Service](https://climate.copernicus.eu/about-data-and-analysis) | Myhre et al. (1998) logarithmic CO2 radiative forcing (`dF = 5.35 * ln(C / C0)`) and Budyko-Sellers latitudinal energy balance (`ASR - OLR`) with prognostic sea-ice albedo feedback. |
| **3. Physical & Biogeochemical Oceanography** (7 layers) | [NOAA NCEI OISST v2.1](https://www.ncei.noaa.gov/products/optimum-interpolation-sst), [CoastWatch ERDDAP OISST v2.1](https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.html), [International Argo Program](https://argo.ucsd.edu/), [Surface Ocean CO2 Atlas (SOCAT)](https://socat.info/) | 1.5-layer reduced-gravity thermocline anomaly + Stommel-Munk western-boundary gyre transport, UNESCO EOS-80 seawater density `rho(S, T, z)`, `0–6000 m` Ocean Heat Content, and marine carbonate `pH` equilibrium. |
| **4. Cryosphere & Glacial Mass Balance** (5 layers) | [NSIDC Sea Ice Index v4](https://nsidc.org/data/g02135/versions/4), [GEBCO Gridded Bathymetry Data](https://www.gebco.net/data-products/gridded-bathymetry-data) | Stefan analytical freezing/thawing degree-day sea-ice thickness and permafrost active-layer thaw depth (`z = alpha * sqrt(TDD)`), plus Glen's flow law alpine glacier velocity. |
| **5. Terrestrial Biosphere, Hydrology & Carbon** (6 layers) | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/), [Open-Meteo GloFAS River Discharge](https://open-meteo.com/en/features) | PAR light-use efficiency & CO2 fertilization Solar-Induced Fluorescence (`SIF`), root-zone bucket hydrology, GloFAS river discharge routing, and VIIRS active wildfire radiative power (`FRP`). |
| **6. Climate Risk & CMIP Projections** (5 layers) | [Open-Meteo Climate API](https://open-meteo.com/en/docs/climate-api), [NASA NEX-GDDP-CMIP6](https://planetarycomputer.microsoft.com/dataset/nasa-nex-gddp-cmip6) | [Stull (2011) Wet-Bulb Approximation](https://www.researchgate.net/publication/258495647_Wet-Bulb_Temperature_from_Relative_Humidity_and_Air_Temperature), outdoor `WBGT`, Van Wagner (1987) Canadian Forest Fire Weather Index (`FWI`), and 1-in-100-year storm surge inundation. |

## SQLite Storage & Compression Architecture (`terratwin_climate.db`)

* `layer_catalog`: Stores all 34 predictive layer definitions, units, spatial/temporal resolutions, render shaders, and source arrays synced from `assets/climate_modeling_hub_schema.json`.
* `raw_telemetry_samples`: Indexed `(layer_id, timestamp_utc)` and `(layer_id, latitude, longitude)` observations with quality weights assimilated by the 3D-Var `PCG` solver.
* `simulated_layer_grids`: Primary-keyed `(layer_id, scenario, timestamp_utc, vertical_level)` lossless `TTZ1` byte-shuffled `lz4_flex` raster blobs (`grid_f32_le`, `u_f32_le`, `v_f32_le`) with automatic transparent fallback for legacy uncompressed `f32` rows.
* `global_readouts_history` & `ingestion_sync_log`: Time-series audit logs of global climate anomalies, 4D playback tick timestamps, HTTP sync latencies, and `RetentionPolicy` ring-buffer pruning checkpoints.
