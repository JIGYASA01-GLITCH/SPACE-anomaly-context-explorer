"""app.py -- SPACE: Scientific Pattern Anomaly & Context Explorer (Streamlit).
Path A: TESS anomaly detection -> investigation (literature + mini-experiments + hypothesis + critic -> report)
Path B: Topic Explorer (same Research RAG, no data)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import torch

from core import data as D, model as M, analysis as A, rag as R, llm as L, train as T

st.set_page_config(page_title="SPACE | Scientific Pattern Anomaly & Context Explorer", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.environ.get("DISCOVERY_ENGINE_CHECKPOINT", "checkpoint_v2.pt")
W = D.WINDOW

for k, v in {"page": "landing", "analysis": None, "sel": None, "investigation": None, "topic_result": None}.items():
    st.session_state.setdefault(k, v)


def go(page):
    st.session_state.page = page


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner="Loading anomaly model...")
def get_model():
    got = M.load_checkpoint(CKPT, DEVICE)
    if got:
        model, thr, meta = got
        return model, thr, meta, f"trained checkpoint ({meta.get('source', '?')})"
    per = T.build_synthetic_dataset(40)
    model, thr, meta = T.train_pipeline(per, epochs=40, out_path=CKPT, source="synthetic fallback (no checkpoint found)")
    return model, thr, meta, "NO trained checkpoint found: quick SYNTHETIC fallback model in use (run the training cell for real TESS)"


def sidebar():
    with st.sidebar:
        st.header("Settings")
        key = st.text_input("LLM API key: Gemini or Claude (optional)", type="password",
                            value=os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "",
                            help="Gemini keys (start with AIza) and Claude keys both work. Enables LLM synthesis, hypothesis and critic. Without it, rule-based fallbacks are used.")
        mdl = st.text_input("Model name (optional)", value="",
                            help=f"Leave empty for the default: {L.DEFAULT_GEMINI_MODEL} for Gemini keys, {L.DEFAULT_MODEL} for Claude keys.")
        emb = st.checkbox("Semantic re-ranking (HF sentence-transformers)", value=True)
        zs = st.checkbox("Zero-shot null-result classifier (HF BART-MNLI, ~1.6 GB first load)", value=False)
    return {"key": key or None, "model": mdl or None, "emb": emb, "zs": zs}


S = sidebar()


# ---------------------------------------------------------------- landing
def render_landing():
    st.title("🔭 SPACE")
    st.caption("Scientific Pattern Anomaly & Context Explorer: TESS light-curve anomaly detection + literature grounding + hypothesis + critic")
    st.write("What do you want to do?")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔭 Detect Anomalies")
        st.write("Analyze a real TESS star, flag unusual windows, then investigate them: literature, data checks, hypothesis, critic review.")
        st.button("Start anomaly detection", on_click=go, args=("anomaly",), key="go_a")
    with c2:
        st.subheader("📚 Explore a Research Topic")
        st.write("Skip the data. Type any topic and get a literature landscape: established, underexplored, known failures, where to start.")
        st.button("Start topic exploration", on_click=go, args=("topic",), key="go_t")


# ---------------------------------------------------------------- path A
def run_analysis(source, query, n_sectors, model, thr):
    if source == "synthetic":
        kind, seed = query
        curves = [D.synthetic_lightcurve(kind, seed=seed)]
    else:
        curves = D.fetch_tess_target(query, max_sectors=n_sectors)
    parts = []
    for c in curves:
        p = D.process_curve(c["time"], c["flux"])
        if len(p["windows"]):
            parts.append(p)
    if not parts:
        return None
    P = D.merge_processed(parts)
    errors, recon = M.score_curves(model, P["windows"], DEVICE)
    name = curves[0]["query"] if source == "synthetic" else f"{query}"
    return {"name": name, "sectors": [c["sector"] for c in curves], "P": P, "errors": errors, "recon": recon, "thr": thr,
            "flagged": np.where(errors > thr)[0]}


def render_anomaly():
    st.button("← Back", on_click=go, args=("landing",), key="b1")
    st.header("🔭 Anomaly Detection")
    model, thr, meta, note = get_model()
    st.caption(f"Model: {note}")
    with st.expander("Model details (training + injection-recovery test)"):
        st.write({k: v for k, v in meta.items() if k != "injection_recovery"})
        ir = meta.get("injection_recovery")
        if ir:
            st.write("Recall at threshold when synthetic signals of a given amplitude (in noise sigma) are injected into held-out windows:")
            st.dataframe(pd.DataFrame({k: ir[k] for k in ("transit", "flare", "bump")}).T.rename(columns=lambda a: f"{a}σ"))

    src = st.radio("Target source", ["Starter list", "Enter a name / TIC ID", "Synthetic demo (offline)"], horizontal=True)
    query, source = None, "real"
    if src == "Starter list":
        label = st.selectbox("Target", list(D.STARTER_TARGETS)); query = D.STARTER_TARGETS[label]
    elif src == "Enter a name / TIC ID":
        query = st.text_input("Star name or TIC ID", placeholder="e.g. TIC 261136679 or WASP-18").strip()
    else:
        source = "synthetic"
        kind = st.selectbox("Injected behaviour", ["transit", "flare", "weird", "quiet", "pulsator"])
        query = (kind, int(st.number_input("Random seed", 0, 9999, 7)))
    n_sec = st.slider("TESS sectors to analyze", 1, 3, 1) if source == "real" else 1

    if st.button("Fetch & Analyze", type="primary") and query:
        with st.spinner("Downloading from MAST and scoring windows..." if source == "real" else "Generating and scoring..."):
            try:
                res = run_analysis(source, query, n_sec, model, thr)
            except Exception as e:
                res = None; st.error(f"Download/analysis failed: {e}")
        if res is None:
            st.warning("No usable SPOC 2-min light curve found for this target. Try another name/TIC or the synthetic demo.")
        else:
            st.session_state.analysis, st.session_state.sel, st.session_state.investigation = res, None, None

    res = st.session_state.analysis
    if not res:
        return
    P, err = res["P"], res["errors"]
    t, z, starts = P["time"], P["z"], P["starts"]
    fl = res["flagged"]
    st.subheader(f"Result: {res['name']} (sectors {res['sectors']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Windows analysed", len(err)); c2.metric("Flagged above threshold", len(fl))
    c3.metric("Top score / threshold", f"{err.max():.1f} / {res['thr']:.1f}"); c4.metric("Noise per 10-min bin", f"{P['sigma']*1e6:.0f} ppm")

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, z, lw=0.6, color="#64748B")
    for i in fl:
        ax.axvspan(t[starts[i]], t[starts[i] + W - 1], color="#F472B6", alpha=0.18)
    ax.set_xlabel("time (BTJD)"); ax.set_ylabel("detrended flux (noise σ)"); ax.set_title("Full light curve; pink = flagged windows")
    show_fig(fig)

    order = fl[np.argsort(-err[fl])] if len(fl) else np.array([int(np.argmax(err))])
    if not len(fl):
        st.info("No window is above the anomaly threshold: this star looks consistent with normal behaviour at this sensitivity. Showing its most unusual window.")
    rows = []
    for r, i in enumerate(order[:15], 1):
        d = A.describe_window(P["windows"][i])
        rows.append({"rank": r, "window": int(i), "start (BTJD)": round(float(t[starts[i]]), 3), "score": round(float(err[i]), 1),
                     "score/threshold": round(float(err[i] / res["thr"]), 2), "pattern": d["kind"], "peak σ": round(d["peak_sigma"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    opts = [int(i) for i in order[:15]]
    sel = st.selectbox("Window to inspect", opts, format_func=lambda i: f"window {i} (score {err[i]:.1f}, t≈{t[starts[i]]:.2f})")
    st.session_state.sel = sel
    s = starts[sel]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t[s:s + W], P["windows"][sel], color="#F472B6", lw=1.1, label="actual (detrended)")
    ax.plot(t[s:s + W], res["recon"][sel], color="#64748B", lw=1.2, ls="--", label="model's expected 'normal'")
    ax.set_xlabel("time (BTJD)"); ax.set_ylabel("noise σ"); ax.legend()
    show_fig(fig)
    st.button("🔍 Investigate this window (literature, checks, hypothesis, critic)", on_click=go, args=("investigate",), key="inv")


def render_investigate():
    st.button("← Back to result", on_click=go, args=("anomaly",), key="b2")
    res, sel = st.session_state.analysis, st.session_state.sel
    if not res or sel is None:
        st.warning("Run detection first."); return
    P = res["P"]; s = int(P["starts"][sel]); zw = P["windows"][sel]
    desc = A.describe_window(zw)
    st.header(f"🧪 Investigation: {res['name']}, window {sel}")
    st.write(f"Pattern: **{desc['kind']}**, peak {desc['peak_sigma']:.1f}σ, ~{desc['duration_hours']:.1f} h, asymmetry {desc['asymmetry']:.1f}")
    query = st.text_input("Literature query (edit it to describe what you see)", A.build_query(desc))

    if st.button("Run investigation", type="primary"):
        notes, ctx = [], {"target": res["name"], "desc": desc, "score": float(res["errors"][sel]), "threshold": float(res["thr"]),
                          "t_start": float(P["time"][s]), "query": query}
        with st.status("Running investigation...", expanded=True) as status:
            st.write("1/4  Mini-experiments on the data (BLS periodicity, event counts, gap check)")
            ctx["experiment"] = A.mini_experiments(P["time"], P["z"], s, W, desc)
            st.write("2/4  Research RAG: arXiv retrieval + semantic ranking + failure detection")
            try:
                rag = R.explore(query, S["key"], use_embeddings=S["emb"], use_zeroshot=S["zs"], model=S["model"], synthesize=False)
            except Exception as e:
                rag = {"papers": [], "flagged_failures": [], "coverage": R.coverage([]), "ranking_method": "-", "zeroshot_error": None}
                notes.append(f"Literature retrieval failed: {e}")
            ctx["papers"], ctx["rag"] = rag["papers"], rag
            if rag.get("zeroshot_error"):
                notes.append(f"Zero-shot classifier unavailable, used keywords only: {rag['zeroshot_error']}")
            st.write("3/4  Hypothesis")
            hyp = None
            if S["key"] and rag["papers"]:
                try:
                    hyp = L.generate_hypothesis(ctx, rag["papers"], S["key"], S["model"]); hyp["source"] = "LLM"
                except Exception as e:
                    notes.append(f"LLM hypothesis failed, used template: {str(e)[:200]}")
            if hyp is None:
                hyp = A.rule_based_hypothesis(desc, ctx["experiment"]); hyp["source"] = "template"
            ctx["hypothesis"] = hyp
            st.write("4/4  Critic (adversarial review)")
            crit = None
            if S["key"] and rag["papers"]:
                try:
                    crit = L.critic_review(ctx, hyp, rag["papers"], S["key"], S["model"])
                except Exception as e:
                    notes.append(f"LLM critic failed, used rule-based critic: {str(e)[:200]}")
            ctx["critic"] = crit or A.rule_based_critic(desc, ctx["experiment"], hyp, rag["papers"], rag["coverage"])
            ctx["notes"] = notes
            ctx["report"] = A.build_report_md(ctx)
            status.update(label="Investigation complete", state="complete")
        st.session_state.investigation = ctx

    ctx = st.session_state.investigation
    if not ctx or ctx["target"] != res["name"]:
        return
    for n in ctx["notes"]:
        st.warning(n)
    st.subheader("1. Data checks")
    for n in ctx["experiment"]["notes"] or ["No additional notes."]:
        st.write("• " + n)
    st.subheader("2. Literature")
    render_rag_result(ctx["rag"], show_landscape=False)
    h, c = ctx["hypothesis"], ctx["critic"]
    st.subheader(f"3. Hypothesis ({h['source']})")
    st.write(f"**Classification:** {h.get('classification')}"); st.write(h.get("reasoning", ""))
    st.write(f"**Hypothesis:** {h.get('hypothesis')}"); st.write(f"**Falsifiable prediction:** {h.get('falsifiable_prediction')}")
    st.write(f"**Proposed check:** {h.get('proposed_check')}")
    st.subheader(f"4. Critic ({c.get('source', '')})")
    color = {"Supported": st.success, "Weakly supported": st.warning, "Not supported": st.error}.get(c["status"], st.info)
    color(f"{c['status']} (confidence: {c['confidence']})")
    st.write(f"**Strongest counter-argument:** {c['strongest_counterargument']}")
    for x in c["caveats"]:
        st.write("• " + x)
    st.subheader("Finding report")
    st.markdown(ctx["report"])
    st.download_button("Download report (.md)", ctx["report"], file_name="finding_report.md")


# ---------------------------------------------------------------- shared RAG renderer
def render_rag_result(r, show_landscape=True):
    cov = r["coverage"]
    st.caption(f"{len(r['papers'])} papers | ranking: {r['ranking_method']} | coverage heuristic: **{cov['label']}** "
               f"({cov['n_relevant']} papers with similarity ≥ {cov['sim_cut']}, best {cov['top_sim']:.2f})")
    if r.get("llm_error"):
        st.warning(f"LLM synthesis failed: {r['llm_error']}")
    if show_landscape and r.get("landscape"):
        st.markdown(r["landscape"])
    elif show_landscape:
        st.info("No API key (or LLM call failed): showing extractive results instead of a synthesized summary.")
    st.markdown("**Most relevant papers:**")
    for p in r["papers"][:8]:
        flag = f" ⚠️ _{p['failure_label']}_" if p.get("failure_label") else ""
        st.markdown(f"- [{p['title']}]({p['link']}) ({p['published']}) sim={p.get('sim', 0):.2f}{flag}")
    fl = r["flagged_failures"]
    st.markdown(f"**Papers flagging possible failures / null results / unresolved debate ({len(fl)}):**")
    for p in fl:
        st.markdown(f"- [{p['title']}]({p['link']})  \n  flags: `{', '.join(p['flags'])}`")
    if not fl:
        st.markdown("_None detected in the retrieved abstracts._")


# ---------------------------------------------------------------- path B
def render_topic():
    st.button("← Back", on_click=go, args=("landing",), key="b3")
    st.header("📚 Topic Explorer")
    st.write("Type any research topic to see what's known, what's underexplored, and what has already failed before you start.")
    topic = st.text_input("Topic", placeholder="e.g. stellar flares and exoplanet habitability")
    if st.button("Explore", type="primary") and topic:
        with st.spinner("Searching arXiv and ranking..."):
            try:
                st.session_state.topic_result = R.explore(topic, S["key"], use_embeddings=S["emb"], use_zeroshot=S["zs"], model=S["model"])
            except Exception as e:
                st.session_state.topic_result = None; st.error(str(e))
    if st.session_state.topic_result:
        st.subheader(f"Landscape: {st.session_state.topic_result['topic']}")
        render_rag_result(st.session_state.topic_result)


{"landing": render_landing, "anomaly": render_anomaly, "investigate": render_investigate, "topic": render_topic}[st.session_state.page]()
