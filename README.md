# Heat Equation Kernel – Diffusion Geometry for ETFs

Applies the heat equation to the ETF correlation graph. The heat kernel signature (HKS) measures thermal stability / centrality at each node. Diffusion time is adapted to macro conditions (VIX, DXY, yields). High HKS → ETF is thermally central and stable under diffusion.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- k‑NN graph from correlation distance (1 - |correlation|)
- Composite macro factor via PCA on all macro variables
- Diffusion time = base_time * (1 - macro_factor * time_range)
- Heat kernel signature via graph Laplacian eigen-decomposition
- Score = HKS (higher = more central)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-heat-equation-kernel-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High HKS → ETF is thermally stable and central in the diffusion geometry – likely to be influential.
- Low HKS → ETF is peripheral.

## Requirements

See `requirements.txt`.
