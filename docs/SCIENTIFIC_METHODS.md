# TerraTwin Scientific Methods & Numerical Formulations (`v2.5.0`)

This document details the physical equations, empirical parameterizations, numerical solvers, map projections, and validation benchmarks implemented across `src/physics.rs`, `src/interp.rs`, `src/colormaps.rs`, and `src/gui.rs` in `crvscience-terratwin`.

---

## 1. Moist Thermodynamics & Heat Stress

### 1.1 Bolton (1980) Saturation Vapor Pressure & Equivalent Potential Temperature
Saturation vapor pressure over liquid water (`bolton_saturation_vapor_pressure_hpa`) uses Bolton (1980, Eq. 10):

$$e_s(T_c) = 6.112 \exp\left(\frac{17.67\, T_c}{T_c + 243.5}\right) \quad [\text{hPa}]$$

To eliminate singular overflows or `NaN`/`Inf` outputs under extreme inputs, `bolton_saturation_vapor_pressure_hpa` returns `0.0` for `t_c <= -243.5` or `NaN`, evaluates the exact asymptotic limit for `+Inf`, and clamps the Magnus exponent to `[-100, 25]`.

Saturation specific humidity (`saturation_specific_humidity`) bounds $e_s \le 0.85\, p$ and $p \ge 1.0\text{ hPa}$:

$$q_s(T_c, p) = \frac{0.622\, e_s}{p - 0.378\, e_s} \in [10^{-6}, 0.045]\text{ kg kg}^{-1}$$

Equivalent potential temperature (`bolton_equivalent_potential_temp_k`, Bolton 1980 Eqs. 21 & 43) computes mixing ratio $r = q / \max(1 - q, 0.9)$, vapor pressure $e = \max(p r / (0.622 + r), 10^{-4})\text{ hPa}$, lifting condensation level temperature $T_L$, dry potential temperature at the LCL $\theta_{DL}$, and $\theta_e$:

$$T_L = \frac{2840}{\max(3.5 \ln T_K - \ln e - 4.805,\, 1.0)} + 55, \qquad \theta_{DL} = T_K \left(\frac{1000}{\max(p - e, 10)}\right)^{0.2854} \left(\frac{T_K}{T_L}\right)^{0.28 r}$$

$$\theta_e = \theta_{DL} \exp\left[\left(\frac{3036}{T_L} - 1.78\right) r (1 + 0.448 r)\right]$$

All inputs (`t_c`, `q_kg_kg`, `p_hpa`), LCL denominators, dry-pressure denominators, and exponential arguments are explicitly safeguarded against zero/negative humidity, sub-singularity temperatures, and non-finite floating-point values.

### 1.2 Stull (2011) Wet-Bulb Temperature & Outdoor WBGT
Wet-bulb temperature `stull_wet_bulb_celsius(t_c, rh_pct)` applies Stull's (2011) arctangent fit over $\text{RH} \in [5\%, 99\%]$ (with `NaN`/`Inf` input guards), and `compute_outdoor_wbgt` combines radiative globe heating $T_g = T_a + 0.014\, S_\downarrow / \max(|V|, 0.5)^{0.45}$ via $\text{WBGT} = 0.7 T_w + 0.2 T_g + 0.1 T_a$.

---

## 2. IPCC AR6 / Etminan (2016) Radiative Forcing & TCR/ECS Split

### 2.1 CO₂ Effective Radiative Forcing
Consistent with IPCC AR6 WG1 Chapter 7 (Forster et al., 2021) and Etminan et al. (2016) tropospherically adjusted effective radiative forcing ($F_{2\times\text{CO}_2} = 3.93\text{ W m}^{-2}$), `co2_radiative_forcing_wm2` evaluates forcing relative to the 1750 pre-industrial baseline $C_0 = 278.0\text{ ppm}$:

$$\Delta F_{\text{CO}_2}(C) = \frac{3.93}{\ln 2} \ln\left(\frac{\max(C, 180.0)}{278.0}\right) \approx 5.6698 \ln\left(\frac{C}{278.0}\right) \quad [\text{W m}^{-2}]$$

At $2 \times \text{CO}_2$ ($556.0\text{ ppm}$), $\Delta F_{\text{CO}_2} = 3.930\text{ W m}^{-2}$.

### 2.2 Equilibrium vs. Transient Global Warming Split
TerraTwin separates equilibrium and transient warming helpers in `src/physics.rs`:

- **Equilibrium warming (`equilibrium_global_warming_c`)** using IPCC AR6 best-estimate $\text{ECS} = 3.0\,^\circ\text{C}$ per $2\times\text{CO}_2$ forcing ($3.93\text{ W m}^{-2}$):

  $$\Delta T_{\text{eq}}(C) = \frac{\Delta F_{\text{CO}_2}(C)}{3.93} \times 3.0\,^\circ\text{C}$$

- **Transient warming (`transient_global_warming_c`)** representing 21st-century realized surface warming ($\text{TCR}_{\text{eff}} = 0.72 \times 3.0 = 2.16\,^\circ\text{C}$ per $2\times\text{CO}_2$):

  $$\Delta T_{\text{tr}}(C) = 0.72\,\Delta T_{\text{eq}}(C)$$

Coupled prognostic simulation ticks (`step_all_layers`), CMIP6 trajectories (`regional_cmip6_delta_t_c`), and land cryosphere/permafrost profiles (`compute_land_cryosphere_profile_impl`) consume `transient_global_warming_c` modulated by asymmetric polar amplification:

$$A(\phi) = \begin{cases} 0.85 + 1.75 \sin^2\phi, & \phi > 0\text{ (Arctic)} \\ 0.85 + 0.70 \sin^2\phi, & \phi \le 0\text{ (Antarctic)} \end{cases}$$

---

## 3. UNESCO EOS-80 Seawater Equation of State

`eos80_seawater_density_at_pressure_dbar(T, S, P_dbar)` and `eos80_seawater_density(T, S, z_m)` ($P_{\text{bar}} = 0.1\, P_{\text{dbar}}$ with $1\text{ m} \approx 1\text{ dbar}$) implement the complete UNESCO (1981) / Millero & Poisson (1981) / Fofonoff & Millard (1983) International Equation of State (EOS-80):

1. **Pure-Water (SMOW) 1-atm Density $\rho_w(T)$**:

   $$\rho_w(T) = 999.842594 + 6.793952\times 10^{-2} T - 9.095290\times 10^{-3} T^2 + 1.001685\times 10^{-4} T^3 - 1.120083\times 10^{-6} T^4 + 6.536332\times 10^{-9} T^5$$

2. **1-atm Seawater Density $\rho(S, T, 0)$**:

   $$\rho(S, T, 0) = \rho_w(T) + A(T) S + B(T) S^{3/2} + C S^2$$

   with $A(T)$ through $T^4$, $B(T) = -5.72466\times 10^{-3} + 1.0227\times 10^{-4} T - 1.6546\times 10^{-6} T^2$, and $C = 4.8314\times 10^{-4}$.

3. **Nonlinear Secant Bulk Modulus $K(S, T, P)$** ($P$ in bars):

   $$K(S, T, P) = K(S, T, 0) + A_w(S, T)\, P + B_w(S, T)\, P^2, \qquad \rho(S, T, P) = \frac{\rho(S, T, 0)}{1 - P / K(S, T, P)}$$

**Verified UNESCO EOS-80 Check Values**:
- $S = 35\text{ PSU},\ T = 25\,^\circ\text{C},\ P = 0\text{ dbar} \implies 1023.34306\text{ kg m}^{-3}$
- $S = 0\text{ PSU},\ T = 5\,^\circ\text{C},\ P = 0\text{ dbar} \implies 999.96675\text{ kg m}^{-3}$
- $S = 35\text{ PSU},\ T = 25\,^\circ\text{C},\ P = 10{,}000\text{ dbar}\ (1000\text{ bar}) \implies 1062.53817\text{ kg m}^{-3}$

---

## 4. Seawater Carbonate Equilibrium with Water Auto-Ionization ($K_w$)

`solve_seawater_carbonate_chemistry` computes ocean $\text{pH}_T$ (Total hydrogen ion scale) and aragonite saturation state $\Omega_{\text{arag}}$ using:
- **$\text{CO}_2$ solubility $K_0(T, S)$**: Weiss (1974)
- **Carbonic acid dissociation $K_1(T, S),\ K_2(T, S)$**: Lueker et al. (2000, Total scale)
- **Boric acid $K_B(T, S)$ & $B_T$**: Dickson (1990)
- **Water auto-ionization $K_w(T, S)$**: Millero (1995) / Dickson et al. (2007, Total scale):

  $$\ln K_w = 148.96502 - \frac{13847.26}{T_K} - 23.6521 \ln T_K + \left(-5.977 + \frac{118.67}{T_K} + 1.0495 \ln T_K\right)\sqrt{S} - 0.01615\, S$$

- **Total Alkalinity**: Lee et al. (2006) salinity-normalized $\text{TA} = 0.002305\,(S / 35)\text{ mol kg}^{-1}$
- **Aragonite solubility $K_{\text{sp}}(T, S, z)$**: Mucci (1983) + Millero (1995) hydrostatic pressure factor

The equilibrium proton concentration $[H^+] = 10^{-\text{pH}}$ solves the full proton/alkalinity balance:

$$f(\text{pH}) = [\text{HCO}_3^-] + 2[\text{CO}_3^{2-}] + [\text{B(OH)}_4^-] + [\text{OH}^-] - [\text{H}^+] - \text{TA} = 0, \qquad [\text{OH}^-] = \frac{K_w}{[H^+]}$$

using a bracketed Newton-Raphson solver with bisection fallback across $\text{pH} \in [5.5, 9.5]$ to $|f(\text{pH})| < 10^{-12}\text{ mol kg}^{-1}$, ensuring smooth monotonic convergence across $\text{pH } 7.3\text{–}8.4$ and beyond.

---

## 5. Van Wagner (1987) Canadian Fire Weather Index ($L_e$ / $L_f$ Daylength Tables)

`step_van_wagner_fwi_at_location` and `canadian_fire_weather_index_at` compute the full 6-component Canadian FWI system (`FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`) using Van Wagner (1987) / Lawson & Armitage (2008) latitude-band monthly daylength factors (`van_wagner_daylength_factors(lat_deg, month)`):

- **DMC Effective Daylength $L_e$** (`100 \cdot K_DMC` where $K_{\text{DMC}} = 1.894 (\max(T, -1.1) + 1.1)(100 - \text{RH}) L_e \times 10^{-6}$):
  - $\phi > 30^\circ\text{N}$: `[6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]`
  - $10^\circ\text{N} < \phi \le 30^\circ\text{N}$: `[7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8]`
  - $-10^\circ \le \phi \le 10^\circ$: `9.0` constant
  - $-30^\circ\text{S} \le \phi < -10^\circ\text{S}$ and $\phi < -30^\circ\text{S}$: 6-month hemisphere-shifted equivalents
- **DC Daylength Adjustment $L_f$** ($V_{\text{DC}} = \max(0.36(\max(T, -2.8) + 2.8) + L_f, 0)$):
  - $\phi > 20^\circ\text{N}$: `[-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]`
  - $-20^\circ \le \phi \le 20^\circ$: `1.4` constant
  - $\phi < -20^\circ\text{S}$: `[6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]`

---

## 6. Coupled Dynamical Cores, Assimilation & Projections

- **2-Layer Spherical QGPV & Linear Balance (`step_two_layer_qgpv`)**: Phillips (1956) two-level baroclinic potential vorticity (`Layer 1 ~ 250 hPa`, `Layer 2 ~ 800 hPa`) inverted via vertical Barotropic/Baroclinic normal-mode Helmholtz decomposition with parallel Red-Black SOR, coupled to Charney (1955) Linear Balance MSLP and Rayleigh-damped Ekman surface winds.
- **1.5-Layer Reduced-Gravity Ocean & Stommel-Munk Gyres (`step_ocean_reduced_gravity_and_gyre`)**: Forward-Backward Coriolis-split shallow-water equatorial wave dynamics ($c_1 = \sqrt{g' H_0} \approx 2.19\text{ m s}^{-1}$) combined with Stommel (1948) / Munk (1950) planetary-$\beta$ western-boundary gyres.
- **Transport & Diffusion**: Staniforth & Côté (1991) midpoint semi-Lagrangian trajectories with Bermejo & Staniforth (1992) quasi-monotone bicubic Catmull-Rom interpolation (`sample_grid_bicubic_monotone`), Priestley / Bermejo-Conde `cos(lat)` mass fixer (`conserve_spherical_mass`), and CFL-substepped finite-volume spherical diffusion (`spherical_diffusive_heat_step`).
- **3D-Var Data Assimilation (`assimilate_3dvar_cg`)**: Dual-space Jacobi-PCG PSAS solver with observation `quality_flag` error variance scaling and flow-aligned anisotropic Matérn-3/2 (`SOAR`) background covariance.
- **Projections & 3D Globe Illumination (`src/interp.rs`, `src/gui.rs`)**: Forward/inverse transforms for `EPSG:4978` (Orthographic 3D Globe with `bearing_deg` roll, Lambertian solar shading, Eddington/Minnaert limb darkening, and Fresnel Rayleigh atmospheric limb scatter), `EPSG:4326` (2:1 Plate Carrée), `EPSG:3995` (Arctic Polar Stereographic), and `EPSG:3031` (Antarctic Polar Stereographic), plus `zoom_camera_toward_ndc` cursor-anchored zooming.

---

## 7. Headless Validation Benchmarks

Empirical trajectories and physical diagnostics are generated reproducibly via `cargo test --test headless_validation` and written to `validation/ssp_projections_1950_2100.csv` and `validation/terratwin_validation_summary.json` (covering **T1** 1950–2100 SSP trajectories, **T2** 2100 cryosphere/ocean/SLR/aridity metrics, and **T3** Arctic-to-Tropics polar amplification ratios).

| Protocol / Scenario | $\text{CO}_2$ (ppm) | $\Delta F_{\text{CO}_2}$ ($\text{W m}^{-2}$) | Global 2m $\Delta T$ ($^\circ\text{C}$) | Thermosteric SLR (cm) | 0–2000 m OHC ($\text{GJ m}^{-2}$) | Arctic / Tropics Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SSP1-1.9 (2100)** | `393.0` | `1.963` | `1.554` | `14.935` | `66.856` | — |
| **SSP2-4.5 (2100)** | `603.0` | `4.390` | `3.169` | `53.833` | `68.765` | `2.498` |
| **SSP3-7.0 (2100)** | `867.0` | `6.449` | `4.544` | `86.850` | `70.435` | `2.537` |
| **SSP5-8.5 (2100)** | `1135.0` | `7.976` | `5.562` | `111.340` | `71.671` | `2.553` |

See [`validation/README.md`](../validation/README.md) and [`README.md`](../README.md) for full details.
