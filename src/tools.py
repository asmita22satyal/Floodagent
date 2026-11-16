"""FloodAgent tools: the 8 functions the LLM agent can call.

Every tool returns a plain, JSON-serialisable dict so any LLM backend can read it.
All numbers come from the project's own data in ``app_data/`` -- the LLM never
computes flood statistics itself.
"""
from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

DATA = Path(os.environ.get("FLOODAGENT_DATA", Path(__file__).resolve().parent.parent / "app_data"))
FEATURES = ["Elevation_m", "Slope_deg", "Aspect_deg", "Flow_Accumulation", "TWI", "Stream_Distance",
            "Hydrologic_Soil_Group", "LULC", "Impervious_Surface_pct", "Max_Precipitation_mm", "Flow_Direction"]
MODELS = {
    "adaboost": lambda: AdaBoostClassifier(n_estimators=300, random_state=1),
    "random_forest": lambda: RandomForestClassifier(300, random_state=1, n_jobs=-1),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=1),
}
FLOOD_EVENT_TYPES = {"Flash Flood", "Flood"}
# +1: more of this -> more flood-prone; -1: more of this -> less flood-prone (basic hydrology)
EXPECTED_DIRECTION = {"Max_Precipitation_mm": 1, "Impervious_Surface_pct": 1, "TWI": 1, "Flow_Accumulation": 1,
                      "Elevation_m": -1, "Slope_deg": -1, "Stream_Distance": -1}


# ----------------------------------------------------------------------------- data helpers
@lru_cache(maxsize=1)
def _summary() -> dict:
    return json.loads((DATA / "summary.json").read_text())


@lru_cache(maxsize=1)
def _points() -> pd.DataFrame:
    return pd.read_csv(DATA / "points.csv")


@lru_cache(maxsize=1)
def _median() -> pd.Series:
    return _points()[FEATURES].median()


def _xy():
    df = _points()
    return df[FEATURES].fillna(_median()), df.label.values.astype(int), df


@lru_cache(maxsize=1)
def _sample():
    z = np.load(DATA / "basin_sample.npz", allow_pickle=True)
    return z["X"].astype("float64"), z["fema"]


def _youden(y, p) -> float:
    fpr, tpr, th = roc_curve(y, p)
    return float(th[np.argmax(tpr - fpr)])


def _subbasins() -> list[str]:
    return sorted(_points().subbasin.unique())


def _match_subbasin(name: str | None):
    if not name:
        return None, None
    hits = [s for s in _subbasins() if name.lower() in s.lower()]
    if len(hits) == 1:
        return hits[0], None
    return None, {"error": f"Unknown or ambiguous sub-basin '{name}'.", "available_subbasins": _subbasins()}


@lru_cache(maxsize=1)
def _final_model():
    X, y, _ = _xy()
    return MODELS["adaboost"]().fit(X, y)


def _rate(mask) -> float | None:
    return round(float(np.mean(mask)) * 100, 1) if len(mask) else None


# ----------------------------------------------------------------------------- 1. list_data
def list_data() -> dict:
    s, p = _summary(), _points()
    return {"datasets": [
        {"name": "flood_points", "what": "542 sample points (271 observed floods 2007-2024, 271 non-flood) with 11 predictors",
         "rows": int(len(p))},
        {"name": "susceptibility_map", "what": "AdaBoost flood-susceptibility probability (0-1) for the whole basin, ~33 m cells"},
        {"name": "fema_nfhl", "what": f"FEMA National Flood Hazard Layer, downloaded {s['fema_download_date']}: "
                                      f"{s['fema_counties']} Kentucky counties, {s['fema_polygons_state']:,} polygons statewide, "
                                      f"{s['fema_polygons_basin']:,} in the basin"},
        {"name": "noaa_storm_reports", "what": "NOAA flood/flash-flood reports 2022 and 2025 for 10 eastern Kentucky counties"},
        {"name": "usgs_high_water_marks", "what": "38 USGS STN marks (2022 Jul: Knott/Letcher/Perry, outside basin; 2025 Apr: central KY)"},
    ], "predictors": FEATURES, "layers_for_make_map": ["probability", "agreement", "fema", "points"],
        "basins_available": ["Lower Levisa"]}


# ----------------------------------------------------------------------------- 2. load_basin
def load_basin(name: str = "Lower Levisa") -> dict:
    if "levisa" not in name.lower():
        return {"error": f"Only the Lower Levisa Fork basin is loaded, not '{name}'.", "basins_available": ["Lower Levisa"]}
    s, p = _summary(), _points()
    sub = (p.groupby("subbasin").label.agg(flood="sum", total="count").reset_index())
    sub["non_flood"] = sub.total - sub.flood
    return {"basin": s["basin_name"], "area_km2": round(s["basin_km2"]), "sub_basins": sub[["subbasin", "flood", "non_flood"]]
            .rename(columns={"flood": "flood_points", "non_flood": "non_flood_points"}).to_dict("records"),
            "flood_points": int(p.label.sum()), "non_flood_points": int((1 - p.label).sum()),
            "model_threshold": s["threshold"], "note": "Mud Creek-Levisa Fork holds ~45% of all flood points."}


# ----------------------------------------------------------------------------- 3. get_features
def get_features(feature: str | None = None, subbasin: str | None = None) -> dict:
    sb, err = _match_subbasin(subbasin)
    if err:
        return err
    if feature and feature not in FEATURES:
        return {"error": f"Unknown feature '{feature}'.", "available": FEATURES}
    df = _points() if sb is None else _points()[_points().subbasin == sb]
    cols = [feature] if feature else FEATURES
    out = {}
    for c in cols:
        f, n = df[df.label == 1][c], df[df.label == 0][c]
        out[c] = {"flood_mean": round(float(f.mean()), 2), "flood_median": round(float(f.median()), 2),
                  "nonflood_mean": round(float(n.mean()), 2), "nonflood_median": round(float(n.median()), 2)}
    return {"scope": sb or "whole basin", "n_flood": int(df.label.sum()), "n_nonflood": int((1 - df.label).sum()),
            "feature_summary": out, "note": "Categorical codes (Hydrologic_Soil_Group, LULC, Flow_Direction) are codes, not measurements."}


# ----------------------------------------------------------------------------- 4. train_model
@lru_cache(maxsize=32)
def _validate(model: str, validation: str) -> dict:
    X, y, df = _xy()
    mk = MODELS[model]
    if validation == "random":
        P = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(X, y):
            P[te] = mk().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        t = _youden(y, P)
        return {"AUC": round(roc_auc_score(y, P), 3), "F1": round(f1_score(y, P >= t), 3), "n": len(y),
                "caution": "Random folds leak information between nearby points, so this is optimistic."}
    if validation == "spatial":
        P, per = np.zeros(len(y)), {}
        for s in df.subbasin.unique():
            te = (df.subbasin == s).values
            P[te] = mk().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
            per[s] = round(roc_auc_score(y[te], P[te]), 3)
        t = _youden(y, P)
        return {"AUC": round(roc_auc_score(y, P), 3), "F1": round(f1_score(y, P >= t), 3), "n": len(y),
                "auc_by_held_out_subbasin": per,
                "caution": "Each sub-basin was predicted by a model that never saw it (leave-one-sub-basin-out)."}
    # holdout_2022
    rng = np.random.RandomState(1)
    fl = np.where((df.year == 2022) & (y == 1))[0]
    nf = rng.choice(np.where(y == 0)[0], len(fl), replace=False)
    te = np.zeros(len(y), bool)
    te[fl] = te[nf] = True
    m = mk().fit(X[~te], y[~te])
    p = m.predict_proba(X[te])[:, 1]
    thr = _youden(y[~te], m.predict_proba(X[~te])[:, 1])
    return {"AUC": round(roc_auc_score(y[te], p), 3), "F1": round(f1_score(y[te], p >= thr), 3),
            "flood_recall": round(float((p[y[te] == 1] >= thr).mean()), 3), "n_test": int(te.sum()),
            "n_flood_test": int(len(fl)), "caution": "Only 21 flood points were held out, so treat this as indicative."}


def train_model(model: str = "adaboost", validation: str = "spatial") -> dict:
    if model not in MODELS:
        return {"error": f"Unknown model '{model}'.", "available": list(MODELS)}
    if validation not in ("random", "spatial", "holdout_2022"):
        return {"error": f"Unknown validation '{validation}'.", "available": ["random", "spatial", "holdout_2022"]}
    r = _validate(model, validation)
    return {"model": model, "validation": validation, **r, "features_used": FEATURES,
            "note": "gradient_boosting stands in for XGBoost in this app." if model == "gradient_boosting" else ""}


# ----------------------------------------------------------------------------- 5. compare_with_fema
def compare_with_fema(include_02pct_zone: bool = False, subbasin: str | None = None) -> dict:
    sb, err = _match_subbasin(subbasin)
    if err:
        return err
    s, T = _summary(), _summary()["threshold"]
    df = _points()
    if sb:
        df = df[df.subbasin == sb]
    df = df[df.model_prob.notna() & (df.fema_cls >= 0)]
    fema = (df.fema_cls >= 1) if include_02pct_zone else (df.fema_cls == 2)
    model = df.model_prob >= T
    fl, nf = df.label == 1, df.label == 0
    out = {"fema_definition": "A/AE zones plus 0.2% annual-chance zone" if include_02pct_zone else "high-risk zones A/AE (the 1% annual-chance floodplain)",
           "point_check": {
               "scope": sb or "whole basin",
               "observed_flood_points": int(fl.sum()),
               "model_flags_pct_of_flood_points": _rate(model[fl]), "fema_flags_pct_of_flood_points": _rate(fema[fl]),
               "model_false_alarm_pct_on_nonflood_points": _rate(model[nf]), "fema_false_alarm_pct_on_nonflood_points": _rate(fema[nf])},
           "caveat_point_check": "The model was trained on these points, so its rate is optimistic; see train_model(spatial) for honest accuracy."}
    if sb is None:
        key = "A_AE_plus_0.2pct" if include_02pct_zone else "A_AE"
        a = s["agreement"][key]
        out["area_agreement_km2"] = {"both_flood": a["both_flood"], "both_non_flood": a["both_nonflood"],
                                     "model_only": a["model_only"], "fema_only": a["fema_only"]}
        out["basin_area_km2"] = round(s["basin_km2"])
        out["model_flood_area_km2"], out["fema_flood_area_km2"] = a["model_flood_km2"], a["fema_flood_km2"]
        out["caveat_area"] = ("The model was trained on 50% flood / 50% non-flood points, so its flagged area (about a third of the basin) "
                              "is inflated. Read it as relative susceptibility, not as a flood extent.")
    else:
        out["note"] = "Area totals are only available for the whole basin; point check shown for this sub-basin."
    return out


# ----------------------------------------------------------------------------- 6. validate_event
def validate_event(event: str = "holdout_2022") -> dict:
    T = _summary()["threshold"]
    if event == "holdout_2022":
        r = _validate("adaboost", "holdout_2022")
        X, y, df = _xy()
        f22 = df[(df.year == 2022) & (df.label == 1) & (df.fema_cls >= 0)]
        return {"event": "July 2022 flood points held out of training (leakage-free)", "model_result": r,
                "fema_flags_pct_of_2022_flood_points": _rate(f22.fema_cls == 2), "n_2022_flood_points": int(len(f22)),
                "note": "Only part of the July 2022 flood hit this basin (worst was the North Fork Kentucky River)."}
    if event in ("noaa_2022", "noaa_2025"):
        yr = int(event[-4:])
        sr = pd.read_csv(DATA / "storm_reports.csv")
        sr = sr[(sr.year == yr) & sr.event_type.isin(FLOOD_EVENT_TYPES)]
        ins = sr[sr.in_basin & sr.model_prob.notna()]
        r = {"event": f"NOAA flood and flash-flood reports, {yr}", "reports_in_10_counties": int(len(sr)), "reports_inside_basin": int(len(ins)),
             "model_flags_pct": _rate(ins.model_prob >= T), "fema_high_risk_flags_pct": _rate(ins.fema_cls == 2)}
        r["caveat"] = ("Storm reports are rough locations (county/town level), so a miss does not always mean the map is wrong. "
                       + ("The model saw 2022 flood points in training, so this is not leakage-free; use holdout_2022 instead."
                          if yr == 2022 else "2025 is a true out-of-time test: the model never saw 2025 data."))
        if len(ins) < 20:
            r["warning"] = "Very few reports fall inside the basin, so the percentages are not reliable."
        return r
    if event == "hwm_2022":
        return {"event": "USGS high-water marks", "available": 38,
                "inside_basin": 0, "detail": "30 marks from July 2022 are in Knott, Letcher and Perry counties (North Fork Kentucky River basin); "
                                             "8 from April 2025 are in central Kentucky. None are in Lower Levisa, so they cannot validate this basin."}
    return {"error": f"Unknown event '{event}'.", "available": ["holdout_2022", "noaa_2022", "noaa_2025", "hwm_2022"]}


# ----------------------------------------------------------------------------- 7. run_scenario
def run_scenario(changes: list[dict]) -> dict:
    """changes: [{"feature": "Max_Precipitation_mm", "op": "multiply", "value": 1.2}, ...]"""
    if not changes:
        return {"error": "Give at least one change, e.g. {'feature':'Max_Precipitation_mm','op':'multiply','value':1.2}."}
    X, _ = _sample()
    T, area = _summary()["threshold"], _summary()["basin_km2"]
    Xs, warnings = X.copy(), []
    pts = _points()
    for c in changes:
        f, op, v = c.get("feature"), c.get("op"), c.get("value")
        if f not in FEATURES:
            return {"error": f"Unknown feature '{f}'.", "available": FEATURES}
        if op not in ("multiply", "add", "set") or v is None:
            return {"error": "op must be multiply, add or set, with a numeric value."}
        i = FEATURES.index(f)
        col = Xs[:, i]
        Xs[:, i] = col * v if op == "multiply" else col + v if op == "add" else v
        if f == "Impervious_Surface_pct":
            Xs[:, i] = np.clip(Xs[:, i], 0, 100)
        lo, hi = float(pts[f].min()), float(pts[f].max())
        if Xs[:, i].max() > hi * 1.001 or Xs[:, i].min() < lo - 1e-6:
            warnings.append(f"{f} goes outside the range seen in training ({lo:.1f}-{hi:.1f}); the model cannot extrapolate reliably.")
        if f in ("Hydrologic_Soil_Group", "LULC", "Flow_Direction"):
            warnings.append(f"{f} is a categorical code, so scaling it is not meaningful.")
    m = _final_model()
    med = _median().values
    base_in = pd.DataFrame(np.where(np.isfinite(X), X, med), columns=FEATURES)
    new_in = pd.DataFrame(np.where(np.isfinite(Xs), Xs, med), columns=FEATURES)
    pb, pn = m.predict_proba(base_in)[:, 1], m.predict_proba(new_in)[:, 1]
    fb, fn = float((pb >= T).mean()), float((pn >= T).mean())
    # Hydrology sanity check: a single change should move susceptibility the way physics expects.
    sanity = None
    if len(changes) == 1 and changes[0]["feature"] in EXPECTED_DIRECTION:
        f = changes[0]["feature"]
        delta = float(Xs[:, FEATURES.index(f)].mean() - X[:, FEATURES.index(f)].mean())
        expected = EXPECTED_DIRECTION[f] * np.sign(delta)
        actual = np.sign(round(float(pn.mean() - pb.mean()), 3))
        if expected != 0 and actual != 0 and expected != actual:
            sanity = (f"CONTRADICTS HYDROLOGY: changing {f} moved predicted susceptibility the opposite way to what physics expects. "
                      "The model learned this predictor as a spatial proxy (where floods were recorded), not as a cause, so this "
                      "scenario result should NOT be interpreted as a real-world effect.")
    return {"changes": changes, "baseline_flagged_pct": round(fb * 100, 1), "scenario_flagged_pct": round(fn * 100, 1),
            "baseline_flagged_km2": round(fb * area), "scenario_flagged_km2": round(fn * area),
            "change_km2": round((fn - fb) * area), "mean_probability_baseline": round(float(pb.mean()), 3),
            "mean_probability_scenario": round(float(pn.mean()), 3), "warnings": warnings,
            "hydrology_sanity_check": sanity or "passed (direction of change is physically plausible)",
            "note": "This shows how the susceptibility model responds to changed inputs. It is a sensitivity test, not a rainfall-runoff or hydraulic simulation. "
                    "Max_Precipitation_mm is a fixed spatial layer, not an event rainfall input."}


# ----------------------------------------------------------------------------- 8. make_map
def make_map(layer: str = "agreement", show_points: bool = False) -> dict:
    import folium
    layers = {
        "probability": ("probability.png", "Flood susceptibility (AdaBoost probability): yellow = low, dark red = high."),
        "agreement": ("agreement.png", "Model vs FEMA: blue = both flood, orange = model only, red = FEMA only, clear = both non-flood."),
        "fema": ("fema.png", "FEMA high-risk flood zones (A/AE) in dark blue."),
        "points": (None, "Observed flood points (blue) and non-flood points (grey)."),
    }
    if layer not in layers:
        return {"error": f"Unknown layer '{layer}'.", "available": list(layers)}
    s = _summary()
    b = s["bounds"]["probability"]
    m = folium.Map(location=[(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2], zoom_start=9, tiles="OpenStreetMap")
    png, desc = layers[layer]
    if png:
        folium.raster_layers.ImageOverlay(image=str(DATA / png), bounds=s["bounds"][png.split(".")[0]], opacity=0.85, name=layer).add_to(m)
    folium.GeoJson(str(DATA / "subbasins.geojson"), name="Sub-basins",
                   style_function=lambda f: {"color": "#334155", "weight": 1.5, "fill": False},
                   tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["Sub-basin"])).add_to(m)
    if layer == "points" or show_points:
        for r in _points().itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=3, weight=0.5, fill=True, fill_opacity=0.8,
                                color="#1d4ed8" if r.label == 1 else "#6b7280",
                                tooltip=f"{'Flood' if r.label == 1 else 'Non-flood'} | {r.county} | {r.subbasin}").add_to(m)
    folium.LayerControl().add_to(m)
    d = Path(tempfile.gettempdir()) / "floodagent_maps"
    d.mkdir(exist_ok=True)
    out = d / f"{layer}{'_points' if show_points and layer != 'points' else ''}.html"
    m.save(str(out))
    return {"map_file": str(out), "layer": layer, "legend": desc}


# ----------------------------------------------------------------------------- registry
def _spec(name, description, props=None, required=None):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []}}


TOOL_SPECS = [
    _spec("list_data", "List the datasets, predictors, map layers and basins available. Use for 'what data do you have' questions."),
    _spec("load_basin", "Load the study basin and return area, sub-basins and flood-point counts. Only Lower Levisa is available.",
          {"name": {"type": "string", "description": "Basin name, e.g. 'Lower Levisa'"}}),
    _spec("get_features", "Summarise the 11 predictor variables for flood vs non-flood points, optionally for one feature or one sub-basin.",
          {"feature": {"type": "string", "description": f"One of {FEATURES}"},
           "subbasin": {"type": "string", "description": "Sub-basin name or part of it, e.g. 'Paint Creek'"}}),
    _spec("train_model", "Train and validate a flood-susceptibility model; returns AUC and F1. 'spatial' holds out whole sub-basins (honest); 'random' is optimistic; 'holdout_2022' tests on 2022 floods never seen.",
          {"model": {"type": "string", "enum": list(MODELS)}, "validation": {"type": "string", "enum": ["random", "spatial", "holdout_2022"]}}),
    _spec("compare_with_fema", "Compare the model with FEMA flood zones: area agreement (both flood / model only / FEMA only / both non-flood) and how many observed flood points each map catches.",
          {"include_02pct_zone": {"type": "boolean", "description": "Also count FEMA's 0.2% annual-chance zone as flood"},
           "subbasin": {"type": "string", "description": "Restrict the point check to one sub-basin"}}),
    _spec("validate_event", "Check the model and FEMA against real flood evidence: holdout_2022 (July 2022 points held out), noaa_2022 / noaa_2025 (storm reports), hwm_2022 (USGS high-water marks).",
          {"event": {"type": "string", "enum": ["holdout_2022", "noaa_2022", "noaa_2025", "hwm_2022"]}}),
    _spec("run_scenario", "What-if test: change predictor values across the basin (e.g. +20% rainfall, +10 points impervious surface) and report how the flagged area changes.",
          {"changes": {"type": "array", "items": {"type": "object", "properties": {
              "feature": {"type": "string", "description": f"One of {FEATURES}"},
              "op": {"type": "string", "enum": ["multiply", "add", "set"]}, "value": {"type": "number"}},
              "required": ["feature", "op", "value"]}}}, ["changes"]),
    _spec("make_map", "Build an interactive map. Layers: probability (susceptibility), agreement (model vs FEMA), fema (FEMA high-risk zones), points (observed flood points).",
          {"layer": {"type": "string", "enum": ["probability", "agreement", "fema", "points"]},
           "show_points": {"type": "boolean", "description": "Overlay the flood points on the layer"}}),
]

_REGISTRY = {"list_data": list_data, "load_basin": load_basin, "get_features": get_features, "train_model": train_model,
             "compare_with_fema": compare_with_fema, "validate_event": validate_event, "run_scenario": run_scenario, "make_map": make_map}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_tool(name: str, args: dict | None = None) -> dict:
    fn = _REGISTRY.get(name)
    if fn is None:
        return {"error": f"No tool called '{name}'.", "available_tools": list(_REGISTRY)}
    try:
        return _clean(fn(**(args or {})))
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:  # keep the agent alive; let it explain the failure
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}
