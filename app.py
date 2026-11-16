"""FloodAgent -- Streamlit chat + map.  Run:  streamlit run app.py"""
import os
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent / "src"))
from agent import FloodAgent  # noqa: E402
from llm import make_backend  # noqa: E402

st.set_page_config(page_title="FloodAgent", page_icon="🌊", layout="wide")
st.title("🌊 FloodAgent")
st.caption("Ask questions about flood susceptibility in the Lower Levisa Fork basin (eastern Kentucky), "
           "compare it with FEMA flood zones, and check it against real events. Susceptibility, not a forecast.")

EXAMPLES = ["Show me where the model and FEMA disagree.",
            "How accurate is the model when it predicts a sub-basin it has never seen?",
            "How many real flood points does FEMA's high-risk zone catch compared with the model?",
            "How did the model do on the July 2022 flood?",
            "What if impervious surface increased by 15 points?",
            "What data do you have?"]

with st.sidebar:
    st.header("Language model")
    kind = st.radio("Backend", ["Gemini (free API)", "Ollama (local, offline)", "Offline demo (no LLM)"], index=2,
                    help="Gemini needs a free key from Google AI Studio. Ollama runs a model on this computer.")
    key = ""
    if kind.startswith("Gemini"):
        key = st.text_input("Gemini API key", type="password", value=os.environ.get("GEMINI_API_KEY", ""),
                            help="Kept in this session only. Never commit it to GitHub.")
        model = st.text_input("Model", value=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"))
        if key and st.button("List models my key can use"):
            try:
                from google import genai
                client = genai.Client(api_key=key)   # keep a reference, or it is closed mid-listing
                names = [m.name.replace("models/", "") for m in client.models.list()
                         if "generateContent" in (m.supported_actions or [])]
                st.code("\n".join(n for n in names if "gemini" in n))
            except Exception as ex:
                st.error(f"{type(ex).__name__}: {ex}")
    elif kind.startswith("Ollama"):
        model = st.text_input("Model", value=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    if st.button("Clear chat"):
        st.session_state.pop("history", None)
        st.session_state.pop("log", None)
        st.rerun()
    st.markdown("**Try:**")
    for e in EXAMPLES:
        if st.button(e, use_container_width=True):
            st.session_state["pending"] = e

st.session_state.setdefault("history", [])
st.session_state.setdefault("log", [])   # [{"role","content","trace","map_file"}]


def get_agent():
    if kind.startswith("Gemini"):
        return FloodAgent(make_backend("gemini", api_key=key or None, model=model))
    if kind.startswith("Ollama"):
        return FloodAgent(make_backend("ollama", model=model))
    return FloodAgent(make_backend("demo"))


def render(entry):
    st.markdown(entry["content"])
    if entry.get("trace"):
        with st.expander(f"Tools used ({len(entry['trace'])})"):
            for t in entry["trace"]:
                st.code(f"{t['tool']}({t['args']})", language="python")
                st.json(t["result"], expanded=False)
    if entry.get("map_file") and Path(entry["map_file"]).exists():
        components.html(Path(entry["map_file"]).read_text(encoding="utf-8"), height=560)


for entry in st.session_state["log"]:
    with st.chat_message(entry["role"]):
        render(entry)

prompt = st.chat_input("Ask about flood susceptibility, FEMA, events or what-if scenarios...")
prompt = st.session_state.pop("pending", None) or prompt
if prompt:
    st.session_state["log"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking and running tools..."):
            try:
                res = get_agent().ask(prompt, st.session_state["history"])
                st.session_state["history"] = res.messages
                entry = {"role": "assistant", "content": res.answer, "trace": res.trace, "map_file": res.map_file}
            except Exception as ex:  # surface API / connection problems clearly
                entry = {"role": "assistant", "content": f"**Could not get an answer.** {type(ex).__name__}: {ex}"}
        render(entry)
    st.session_state["log"].append(entry)
