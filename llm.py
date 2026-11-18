"""Swappable LLM backends. Every backend exposes the same method:

    step(system, messages, tool_specs) -> (text, tool_calls, raw)

``messages`` uses one neutral format (user / assistant / tool dicts), and each backend
translates it to its own API. Free options: Gemini API free tier, Ollama (local).
"""
from __future__ import annotations

import json
import os
import re

import requests


# ---------------------------------------------------------------------------- Gemini (free tier)
class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("No Gemini API key. Set the GEMINI_API_KEY environment variable (free key from Google AI Studio).")
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    def step(self, system, messages, tool_specs):
        from google.genai import types
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                if m.get("raw") is not None:           # reuse the model's own content (keeps any thought signatures)
                    contents.append(m["raw"])
                    continue
                parts = [types.Part(text=m["content"])] if m.get("content") else []
                parts += [types.Part(function_call=types.FunctionCall(name=c["name"], args=c["args"])) for c in m.get("tool_calls", [])]
                contents.append(types.Content(role="model", parts=parts))
            else:  # tool result
                part = types.Part(function_response=types.FunctionResponse(name=m["name"], response={"result": m["obj"]}))
                last = contents[-1] if contents else None
                if last is not None and last.role == "user" and last.parts and last.parts[0].function_response:
                    last.parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
        decls = [types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"])
                 for t in tool_specs]
        cfg = types.GenerateContentConfig(system_instruction=system, tools=[types.Tool(function_declarations=decls)], temperature=0.2)
        import time
        for attempt in range(5):   # free tier: 503 "high demand" and 429 are usually temporary
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
                break
            except Exception as ex:
                if attempt == 4 or not any(c in str(ex) for c in ("503", "429", "UNAVAILABLE")):
                    raise
                time.sleep(3 * 2 ** attempt)
        content = resp.candidates[0].content
        text, calls = "", []
        for p in content.parts or []:
            if getattr(p, "function_call", None):
                calls.append({"name": p.function_call.name, "args": dict(p.function_call.args or {})})
            elif getattr(p, "text", None):
                text += p.text
        return text.strip(), calls, content


# ---------------------------------------------------------------------------- Ollama (local, offline)
class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def step(self, system, messages, tool_specs):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                msgs.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
            elif m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}} for c in m["tool_calls"]]
                msgs.append(d)
            else:
                msgs.append({"role": "user", "content": m["content"]})
        tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
                 for t in tool_specs]
        r = requests.post(f"{self.host}/api/chat", json={"model": self.model, "messages": msgs, "tools": tools, "stream": False,
                                                         "options": {"temperature": 0.2}}, timeout=600)
        r.raise_for_status()
        msg = r.json()["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            a = c["function"].get("arguments") or {}
            calls.append({"name": c["function"]["name"], "args": json.loads(a) if isinstance(a, str) else a})
        return (msg.get("content") or "").strip(), calls, None


# ---------------------------------------------------------------------------- Offline demo (no LLM)
class KeywordBackend:
    """No language model: keyword routing + templated answers. Lets the app and tests run with no key and no GPU.
    It is a fallback / test harness, NOT the intelligent agent -- the evaluation reports it separately."""
    name = "keyword-demo"

    ROUTES = [
        (r"\b(what data|which data|datasets?|available)\b", [("list_data", {})]),
        (r"\b(what.?if|scenario|increase\w*|decrease\w*|more rain|rises?|\+\s*\d+\s*%)", [("run_scenario", None)]),
        (r"\b(accura\w*|auc|f1|cross.valid\w*|spatial|holdout|train\w*|how good|perform\w*|reliab\w*|never seen|unseen)", [("train_model", None)]),
        (r"\b(2022|2025|storm report|high.water|validate|event\w*)\b", [("validate_event", None)]),
        (r"\b(fema|nfhl|regulatory|disagree\w*|differ\w*|compar\w*|capture\w*)\b", [("compare_with_fema", {}), ("make_map", {"layer": "agreement"})]),
        (r"\b(map|show|where)\b", [("make_map", {"layer": "probability"})]),
        (r"\b(feature\w*|predictor\w*|slope|elevation|twi|soil|variable\w*|why|driver\w*|important)\b", [("get_features", {})]),
        (r"\b(basin|sub.?basin\w*|study area|levisa)\b", [("load_basin", {})]),
    ]
    TITLES = {"compare_with_fema": "Model vs FEMA", "train_model": "Model accuracy", "validate_event": "Flood event check",
              "run_scenario": "What-if scenario", "get_features": "Predictors", "load_basin": "Study basin",
              "list_data": "Available data", "make_map": "Map"}

    _WORDS = {"auc": "AUC", "fema": "FEMA", "f1": "F1", "km2": "km²", "pct": "%", "twi": "TWI", "lulc": "LULC", "n": "n",
              "noaa": "NOAA", "usgs": "USGS", "hwm": "HWM", "nfhl": "NFHL"}

    @classmethod
    def _label(cls, k):
        words = [cls._WORDS.get(w.lower(), w) for w in str(k).replace("_", " ").split()]
        text = " ".join(words)
        return text[:1].upper() + text[1:] if text else text

    @classmethod
    def _fmt(cls, v, d=0):
        pad = "  " * d
        if isinstance(v, dict):
            out = []
            for k, x in v.items():
                if k == "map_file" or x in ("", None):
                    continue
                if isinstance(x, (dict, list)) and x:
                    out.append(f"{pad}- **{cls._label(k)}**:")
                    out.append(cls._fmt(x, d + 1))
                else:
                    out.append(f"{pad}- **{cls._label(k)}**: {x}")
            return "\n".join(out)
        if isinstance(v, list):
            if all(not isinstance(i, (dict, list)) for i in v):
                return pad + "- " + ", ".join(map(str, v[:30]))
            out = []
            for i in v[:12]:
                if isinstance(i, dict):
                    out.append(pad + "- " + "; ".join(f"{cls._label(k)}: {x}" for k, x in i.items() if not isinstance(x, (dict, list))))
                else:
                    out.append(pad + f"- {i}")
            return "\n".join(out)
        return pad + str(v)

    def step(self, system, messages, tool_specs):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user = messages[last_user]["content"].lower()
        if messages[-1]["role"] == "tool":
            parts = []
            for m in messages[last_user + 1:]:
                if m["role"] != "tool":
                    continue
                try:
                    data = json.loads(m["content"])
                except Exception:
                    data = m["content"]
                title = self.TITLES.get(m["name"], m["name"])
                if m["name"] == "make_map" and isinstance(data, dict):
                    body = f"- {data.get('legend', 'Map drawn; see below.')}"
                else:
                    body = self._fmt(data)
                parts.append(f"### {title}\n{body[:2500]}")
            return ("*Offline demo mode: no language model is connected, so this shows the tool results directly. "
                    "Connect Gemini or Ollama in the sidebar for written answers.*\n\n" + "\n\n".join(parts)), [], None
        calls = []
        for pat, tools in self.ROUTES:
            if re.search(pat, user):
                for name, args in tools:
                    args = dict(args) if args is not None else self._args(name, user)
                    calls.append({"name": name, "args": args})
                break
        if not calls:
            return ("Offline demo mode could not match your question to a tool. Connect Gemini or Ollama for open-ended questions.", [], None)
        return "", calls, None

    @staticmethod
    def _args(name, user):
        if name == "validate_event":
            return {"event": "noaa_2025" if "2025" in user else "hwm_2022" if "high" in user else "noaa_2022" if "storm" in user else "holdout_2022"}
        if name == "train_model":
            return {"validation": "random" if "random" in user else "holdout_2022" if "2022" in user else "spatial"}
        if name == "run_scenario":
            pct = re.search(r"(\d+)\s*%", user)
            f = 1 + (int(pct.group(1)) / 100 if pct else 0.2) * (-1 if "decrease" in user else 1)
            feat = "Impervious_Surface_pct" if "imperv" in user or "urban" in user else "Max_Precipitation_mm"
            return {"changes": [{"feature": feat, "op": "multiply", "value": f}]}
        return {}


def make_backend(kind: str, **kw):
    kind = kind.lower()
    if kind.startswith("gemini"):
        return GeminiBackend(**kw)
    if kind.startswith("ollama"):
        return OllamaBackend(**kw)
    return KeywordBackend()
