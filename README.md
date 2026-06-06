# NIFTY Options IV Surface Reconstruction

Reconstructs missing Implied Volatility values across the NIFTY options surface using a multi-stage imputation pipeline.

## Pipeline

1. **RBF Interpolation** – Interpolates missing IVs across strikes within each timestamp
2. **SVI + PCHIP** – Fits a parametric volatility smile; fills remaining gaps with monotone cubic interpolation
3. **Forward-Fill** – Causal time-axis fill with a max gap of 12 periods
4. **ML Ensemble** – LightGBM + XGBoost + MLP ensemble for remaining missing values, with SABR-derived features
5. **Gaussian Smoothing** – CV-tuned temporal smoothing blended into predicted cells
6. **Convexity Enforcement** – No-arbitrage correction across strikes (4 passes)

## Requirements

```bash
pip install lightgbm xgboost scipy scikit-learn pandas numpy
```

## Usage

Place `dataset.csv` under `/kaggle/input/finclub-open-project-26/` and run all cells. The notebook outputs:

- `filled_dataset.csv` – Complete dataset with all missing IVs filled
- `submission.csv` – Competition-ready file with `id` and `value` columns
