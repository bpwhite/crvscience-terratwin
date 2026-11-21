# Preprints for TerraTwin (`crvscience-terratwin`)

**Author:** Bryan P. White (ORCID: [0000-0003-4191-3511](https://orcid.org/0000-0003-4191-3511))  
**Hub:** https://crvscience.com/apps

This directory contains the self-contained LaTeX source (`.tex`) and single-column PDF (`.pdf`) editions for both:

1. **Companion Application Preprint (`APP-PREPRINT-10`)**
   - **Title:** TerraTwin: Fast Climate Projections for Global Warming, Sea-Level Rise, and Arctic Change
   - **Files:** `terratwin-coupled-climate-ssp.tex`, `terratwin-coupled-climate-ssp.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222392
   - **Figshare Mirror:** https://figshare.com/articles/preprint/TerraTwin_Fast_Climate_Projections_for_Global_Warming_Sea-Level_Rise_and_Arctic_Change/35013769
   - **Web Preprint:** https://crvscience.com/apps/preprints/terratwin-coupled-climate-ssp

2. **CRV Science Public Apps Network Overview (`APP-PREPRINT-01`)**
   - **Title:** The CRV Science App Network: Twenty Open-Source Rust Programs for Science, Data, and Simulation
   - **Files:** `crv-science-public-apps-network.tex`, `crv-science-public-apps-network.pdf`
   - **Canonical Zenodo DOI:** https://doi.org/10.5281/zenodo.24222398
   - **Figshare Mirror:** https://figshare.com/articles/preprint/The_CRV_Science_App_Network_Seventeen_Open-Source_Rust_Programs_for_Science_Data_and_Simulation/35013736
   - **Web Preprint:** https://crvscience.com/apps/preprints/crv-science-public-apps-network

## Compiling Standalone LaTeX

Both `.tex` files inline their bibliography so each compiles standalone with standard `pdflatex`:

```bash
pdflatex terratwin-coupled-climate-ssp.tex
pdflatex crv-science-public-apps-network.tex
```
