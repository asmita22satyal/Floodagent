"""Run the 15 evaluation requests through a backend and save answers for grading.

    python eval/run_eval.py --backend gemini
    python eval/run_eval.py --backend ollama --model qwen2.5:7b
    python eval/run_eval.py --backend demo

Output: eval/results_<backend>.csv. Automatic column: tools_ok (did the agent call the expected tools?).
Fill the manual columns (correct 0/1, honest_caveat 0/1, notes) while reading the answers.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from agent import FloodAgent  # noqa: E402
from llm import make_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="demo")
ap.add_argument("--model", default=None)
ap.add_argument("--ids", default=None, help="comma-separated request ids to run, e.g. 9,10,12")
ap.add_argument("--tag", default="", help="suffix for the output file, e.g. _part2")
a = ap.parse_args()
only = {int(x) for x in a.ids.split(",")} if a.ids else None
kw = {"model": a.model} if a.model else {}
agent = FloodAgent(make_backend(a.backend, **kw))

rows = []
for r in pd.read_csv(ROOT / "eval" / "requests.csv").itertuples():
    if only and int(r.id) not in only:
        continue
    t0 = time.time()
    try:
        res = agent.ask(r.request)       # each request is a fresh conversation
        called = [t["tool"] for t in res.trace]
        answer, err = res.answer, ""
    except Exception as ex:
        called, answer, err = [], "", f"{type(ex).__name__}: {ex}"
    expected = [] if r.expected_tools == "none" else r.expected_tools.split("|")
    tools_ok = (not called) if not expected else all(e in called for e in expected)
    rows.append({"id": r.id, "category": r.category, "request": r.request, "expected_tools": r.expected_tools,
                 "tools_called": "|".join(called), "tools_ok": int(tools_ok), "seconds": round(time.time() - t0, 1),
                 "answer": answer, "error": err, "expected_behavior": r.expected_behavior,
                 "correct_0_1": "", "honest_caveat_0_1": "", "notes": ""})
    print(f"{r.id:>2} {r.category:<16} tools_ok={int(tools_ok)} called={called}")

out = ROOT / "eval" / f"results_{a.backend}{a.tag}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved {out}\nTools-as-expected: {sum(x['tools_ok'] for x in rows)}/{len(rows)}")
