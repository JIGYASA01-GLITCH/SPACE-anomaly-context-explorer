# SPACE: Scientific Pattern Anomaly & Context Explorer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/notebooks/SPACE_v2_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An AI-assisted research tool, piloted on NASA's **TESS** light curves. It (1) finds unusual stretches in a star's brightness record with an unsupervised deep model and (2) places each finding in the published literature, with prior failures made visible. It is an **assistive tool, not an automatic discoverer**: every output must be verified by a human researcher.

> **Live dashboard:**[@VIEW] _https://theater-pilot-guitars-allocation.trycloudflare.com/_

## What it does

The app has two entry points that share one literature module (the *Research RAG*):

| Path | Page | What happens |
|---|---|---|
| **A, data to literature** | *Detect Anomalies* | Download a real TESS target, score every window with a 1D-CNN autoencoder, list and plot the unusual ones, then *Investigate*: data checks, literature, hypothesis, critic, downloadable report |
| **B, topic to landscape** | *Explore a Topic* | Type any topic, get ranked arXiv papers, flagged null results and a coverage label (plus an Established / Underexplored / Known failures / Start here summary if an LLM key is set) |

![Architecture](docs/images/architecture.png)

## How the anomaly detector works

1. **Data:** SPOC 2-minute light curves from MAST via `lightkurve`.
2. **Preprocessing:** 10-minute median bins, robust Savitzky-Golay detrending that protects transits and flares, per-star noise scaling (scores are in noise-sigma units), 400-bin windows (about 2.8 days).
3. **Model:** 1D convolutional autoencoder (133,649 parameters), trained **without labels** on windows from 15 TESS stars. Whole stars are held out for validation and the worst 3% of training windows are trimmed.
4. **Score:** mean of the largest 5% of squared reconstruction residuals (a plain whole-curve error hides short events). Pure noise scores about 6.
5. **Threshold:** validation median + 6 robust standard deviations.
6. **Evaluation:** injection-recovery of synthetic transit, flare and bump signals.

![Data flow](docs/images/data_flow.png)

## The Research RAG

arXiv retrieval (AND query with OR fallback) -> semantic ranking with `all-MiniLM-L6-v2` + FAISS -> failure / null-result phrase detection (optional zero-shot BART-MNLI) -> coverage label -> optional grounded LLM synthesis (Gemini or Claude). Without an API key the app falls back to extractive results and rule-based hypothesis and critic text.

## Example results

**pi Mensae (Sector 1):** five periodic dips about 6.3 days apart, consistent with the known planet pi Men c, are reconstructed by the model and treated as normal behaviour.

![pi Mensae](docs/images/pimen_light_curve.png)
![Window reconstruction](docs/images/pimen_window.png)

**Topic Explorer:** for "exoplanet transit detection TESS" the top-ranked paper is a null-result study.

![Topic Explorer](docs/images/topic_explorer_exoplanet.png)

## Quick start

### Option 1: Google Colab (recommended)
Click the **Open In Colab** badge, choose a GPU runtime and run the cells from top to bottom. The last cell prints a public link to the Streamlit dashboard (keep that cell running).

### Option 2: run locally
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py            # optional: trains on real TESS data (needs internet), writes checkpoint_v2.pt
streamlit run app.py
```
If no `checkpoint_v2.pt` is present, the app trains a quick synthetic fallback model and says so.

### Optional LLM features
Set a key before starting the app (never commit keys):
```bash
export GEMINI_API_KEY="..."        # or ANTHROPIC_API_KEY
```
On Streamlit Community Cloud add it under *Settings -> Secrets* as `GEMINI_API_KEY = "..."`.

## Repository structure
```
app.py                  Streamlit interface (landing, anomaly, investigation, topic pages)
core/data.py            MAST download, binning, detrending, noise scaling, windowing, synthetic curves
core/model.py           Autoencoder, top-5% residual score, checkpoint I/O
core/train.py           Training set, trimmed training, threshold, injection-recovery test
core/analysis.py        Window description, BLS / gap checks, rule-based hypothesis and critic, report
core/rag.py             arXiv retrieval, ranking, failure detection, coverage, explore()
core/llm.py             Gemini / Claude calls (synthesis, hypothesis, critic)
train_model.py          Train on real TESS data and save the checkpoint
notebooks/              Colab notebook that writes the files and launches the app
docs/images/            Figures used in this README
```

## Known limitations
- On real data the calibrated threshold was far too high (1727.5 vs. normal scores of about 6), so absolute flagging is not reliable yet. A recalibration plan is in the project report.
- pi Mensae was a training star, so its result is a sanity check, not an independent test.
- Injection-recovery numbers in the report come from synthetic data.
- Only arXiv abstracts are used; LLM output can be wrong and must be verified.

## Authors
- **Jigyasa Chaturvedi**, [@JIGYASA01-GLITCH](https://github.com/JIGYASA01-GLITCH)
- **Manshi Rahangdale** , [@manshirahangdale](https://github.com/manshirahangdale)

Supervisor: Ms. Archana Acharya, Assistant Professor, Centre for Artificial Intelligence, Madhav Institute of Technology & Science (MITS), Gwalior. Cornerstone Project, B.Tech AI & ML, Jul-Dec 2026.

## Acknowledgements
This work includes data collected by the TESS mission, obtained from the Mikulski Archive for Space Telescopes (MAST). Thank you to arXiv for use of its open access interoperability. Built with Lightkurve, Astropy, PyTorch, sentence-transformers, FAISS and Streamlit.
