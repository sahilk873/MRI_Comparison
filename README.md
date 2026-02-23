# MRI_Comparison

Reproducible pipeline for the manuscript **"Quantifying Protocol Robustness of Apparent Diffusion Coefficient MRI Across 64 mT and 3 T Using Deep Embeddings and Classical Radiomics"** (JAIKE).

- **Dataset:** Paired 64 mT and 3 T brain MRI [Zenodo 15471394](https://zenodo.org/records/15471394).
- **Notebook:** `MRI_Comparison.ipynb` — preprocessing, registration, ResNet-50 embeddings, PyRadiomics extraction, paired similarity, and Field-Dominance Index (FDI).

## Added for manuscript revision (harmonization and analysis)

- **`harmonization.py`** — ComBat feature-domain batch correction for radiomics. Batch = protocol (LF, 3T_low, 3T_high); subject ID as covariate to preserve within-subject variation. Also provides `similarity_and_fdi()` to compute paired cosine, L2, and FDI from any feature matrix.
- **`frequency_domain.py`** — Spectral slope and high-frequency energy fraction (35% radius threshold) on mid-axial slice for frequency-domain summaries (manuscript Section 3.11).
- **`run_combat_radiomics.py`** — Script to run ComBat on the radiomics table produced by the notebook and print before/after stability (Table 5 in the manuscript).

### ComBat and Table 5

After running the notebook and saving `features_classical_radiomics.csv`:

```bash
python run_combat_radiomics.py --features-csv path/to/features_classical_radiomics.csv
```

Optional: save summary to CSV:

```bash
python run_combat_radiomics.py --features-csv path/to/features_classical_radiomics.csv --out-csv combat_before_after_summary.csv
```

If your CSV uses different condition names, pass `--condition-order 64mT 3T_lowres 3T_highres` (or your labels).

### Frequency-domain summaries

Use `frequency_domain.spectral_slope_and_hf_fraction(slice_2d, mask, hf_radius_percentile=35.0)` on the mid-axial slice (after robust scaling) to get slope and high-frequency fraction as in the manuscript.

## Dependencies

See the notebook (pip install cell): `nibabel`, `pandas`, `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`, `tqdm`, `joblib`, `SimpleITK`, `pyradiomics`, `torch`, `torchvision`. The harmonization and frequency-domain modules use only `numpy` and `pandas` (and `scipy` for cosine in `similarity_and_fdi`).
