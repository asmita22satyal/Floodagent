"""FloodAgent: a small tool-use loop around a swappable LLM backend."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from tools import TOOL_SPECS, run_tool

SYSTEM_PROMPT = """You are FloodAgent, an assistant for flood-susceptibility analysis of the Lower Levisa Fork basin in eastern Kentucky.
You answer by calling tools. All numbers must come from tool results; never invent or estimate statistics yourself.

What the project is:
- A machine-learning model (AdaBoost; also random forest and gradient boosting) trained on 271 observed flood points (2007-2024) and 271 non-flood points, using 11 terrain, soil, land-cover and rainfall predictors.
- It is compared with FEMA's National Flood Hazard Layer (regulatory flood zones: A/AE = high risk) and checked against real flood evidence.

Rules:
1. Call a tool whenever the user asks about data, accuracy, maps, FEMA, events or scenarios. Answer plain concepts (what is TWI, what is AUC) yourself, briefly.
2. Always state the key caveat from the tool result (e.g. 'trained on these points', 'only 8 reports in the basin', 'categorical codes'). Prefer the honest spatial-validation numbers over optimistic random-fold numbers.
3. The model gives relative SUSCEPTIBILITY, not a flood extent or depth, and not a forecast. FEMA maps are regulatory risk zones, not a record of what floods. Do not call either 'wrong'; explain what each measures.
4. If a run_scenario result says it CONTRADICTS HYDROLOGY, say so plainly and do not present the change as a real-world effect.
5. Decline, politely and briefly, anything outside scope: other basins, real-time or tomorrow's flooding, flood depth at a specific address, insurance or legal/permit advice, evacuation decisions. Suggest official sources (NWS Jackson KY, FEMA Map Service Center, local floodplain manager).
6. If a request is ambiguous, ask one short clarifying question instead of guessing.
7. After make_map, tell the user which layer you drew and how to read its colours (use the legend field).
Keep answers short and plain: lead with the finding, then one or two caveats."""


@dataclass
class AgentResult:
    answer: str
    trace: list = field(default_factory=list)      # [{"tool", "args", "result"}]
    map_file: str | None = None
    messages: list = field(default_factory=list)   # full history, pass back for multi-turn chat


class FloodAgent:
    def __init__(self, backend, max_steps: int = 6):
        self.backend, self.max_steps = backend, max_steps

    def ask(self, user_text: str, history: list | None = None) -> AgentResult:
        msgs = list(history or []) + [{"role": "user", "content": user_text}]
        trace, map_file, text = [], None, ""
        for _ in range(self.max_steps):
            text, calls, raw = self.backend.step(SYSTEM_PROMPT, msgs, TOOL_SPECS)
            msgs.append({"role": "assistant", "content": text, "tool_calls": calls, "raw": raw})
            if not calls:
                return AgentResult(text or "(no answer)", trace, map_file, msgs)
            for c in calls:
                res = run_tool(c["name"], c["args"])
                trace.append({"tool": c["name"], "args": c["args"], "result": res})
                map_file = res.get("map_file", map_file) if isinstance(res, dict) else map_file
                msgs.append({"role": "tool", "name": c["name"], "content": json.dumps(res)[:12000], "obj": res})
        return AgentResult(text or "I reached my step limit before finishing. Please ask a narrower question.", trace, map_file, msgs)
