# FloodAgent

A chat assistant that answers questions about **flood susceptibility in the Lower Levisa Fork basin (eastern Kentucky)**,
compares a machine-learning model with **FEMA flood zones**, and checks both against **real flood evidence**.
Free to run: Gemini API free tier, or a local Ollama model, or an offline keyword demo.

## What it does
The LLM never computes flood statistics. It chooses and calls 8 tools (`src/tools.py`); every number comes from the project's data.

| Tool | Purpose |
|---|---|
| `list_data` | What datasets, predictors and map layers exist |
| `load_basin` | Basin summary and sub-basin point counts |
| `get_features` | Flood vs non-flood predictor summaries |
| `train_model` | AdaBoost / random forest / gradient boosting with random, **spatial (leave-one-sub-basin-out)** or 2022-holdout validation |
| `compare_with_fema` | Four-group agreement (both / model only / FEMA only / neither) and flood-point catch rate |
| `validate_event` | July 2022 holdout, NOAA storm reports (2022, 2025), USGS high-water marks |
| `run_scenario` | What-if on predictors, with an automatic hydrology sanity check |
| `make_map` | Interactive folium map (probability, agreement, FEMA, points) |

## Run it
```
run_app.bat                      # Windows; or: streamlit run app.py
python eval/run_eval.py --backend demo      # 15-request evaluation (gemini / ollama / demo)
```
- **Gemini (free):** get a key at Google AI Studio, then `set GEMINI_API_KEY=...` (or paste in the sidebar). Never commit the key.
- **Ollama (offline):** install Ollama, `ollama pull qwen2.5:7b`, choose "Ollama" in the sidebar.

## Data and method
- 542 points: 271 observed floods (NOAA Storm Events, 2007-2024) and 271 non-flood points; 11 predictors (elevation, slope, aspect, flow accumulation, TWI, stream distance, hydrologic soil group, land cover, impervious surface, max precipitation, flow direction).
- FEMA NFHL downloaded 13-14 Nov 2026 (120 Kentucky counties, 128,770 polygons; 7,411 in the basin). "FEMA flood" = zones A/AE; "0.2%" zone reported as a variant.
- Basin = USGS HUC8 05070203 (2,828 km2); spatial folds = its five HUC10 sub-basins.

## Key results (so far)
| Validation | AUC |
|---|---|
| Random 5-fold (optimistic) | ~0.80-0.82 |
| **Leave-one-sub-basin-out** | **~0.77-0.81** |
| July 2022 points held out (n=21) | ~0.83-0.92 |

FEMA's A/AE zones overlap **~6%** of observed flood points; the model flags **~77%** (in-sample) at **~10%** false alarms on non-flood points.

## Limitations (stated up front)
- The model gives relative **susceptibility**, not flood extent, depth or a forecast. It was trained on 50/50 flood / non-flood points, so its flagged area (~1/3 of the basin) is inflated.
- Floyd/Mud Creek hold most flood points; Paint Creek is the weakest sub-basin (AUC ~0.66-0.74).
- Only 21 flood points from 2022 and 8 NOAA 2025 reports lie in the basin: validation is indicative, not conclusive.
- USGS high-water marks exist for Knott/Letcher/Perry (July 2022), **none inside this basin**. Sentinel-1 flood mapping found almost no water in the narrow forested valleys, so SAR is not used as ground truth.
- NOAA storm reports are rough locations. FEMA maps show regulatory risk, not everything that floods.
- `run_scenario` is a sensitivity test only. The precipitation layer is a fixed spatial input, and the model can use it as a location proxy: the tool flags results that contradict hydrology.
- The app retrains models on points whose predictors were sampled from the raster stack, so figures differ slightly from the original thesis tables.

## Layout
`src/tools.py` tools - `src/llm.py` swappable backends - `src/agent.py` tool-use loop and system prompt - `app.py` Streamlit UI -
`eval/` 15 test requests and runner - `app_data/` exported points, map overlays and summary - `results/` validation tables and rasters.
