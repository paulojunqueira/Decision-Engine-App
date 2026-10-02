import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import time
import io
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laya Evaluator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Laya Model Evaluator")
st.caption("A 4-step wizard to evaluate the Laya decision engine on your CSV data.")

# Device status banner
try:
    import torch as _t
    if _t.cuda.is_available():
        _gpu = _t.cuda.get_device_name(0)
        _mem = _t.cuda.get_device_properties(0).total_memory / 1024**3
        st.success(f"✅ GPU ready — {_gpu}  ({_mem:.1f} GB VRAM)", icon="🖥️")
    else:
        st.warning("⚠️ No CUDA GPU detected — running on CPU (slower)", icon="🐢")
except Exception:
    st.info("Could not detect GPU.")

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "df": None,
        "results_df": None,
        "metrics": None,
        "run_time": None,
        "questions": [],      # start empty — user builds their own
        "_next_qid": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build laya request list
# ─────────────────────────────────────────────────────────────────────────────
def build_requests(df, text_cols, questions):
    requests = []
    for _, row in df.iterrows():
        # Use "Column: value\n\n..." format when multiple cols — gives Laya clear structure
        if len(text_cols) == 1:
            state = str(row[text_cols[0]]) if pd.notna(row[text_cols[0]]) else ""
        else:
            parts = [f"{c}: {row[c]}" for c in text_cols if pd.notna(row[c]) and str(row[c]).strip()]
            state = "\n\n".join(parts)
        qs = {}
        for q in questions:
            qid = q["name"]
            qtype = q["qtype"]
            entry = {"type": qtype, "instructions": q.get("instructions", "")}
            if qtype == "choice":
                entry["criteria"] = {
                    item["label"]: item["description"] for item in q["criteria"]
                }
            elif qtype == "score":
                raw = q.get("score_levels", "Low,Medium,High")
                entry["criteria"] = [s.strip() for s in raw.split(",") if s.strip()]
            # noul: no criteria
            qs[qid] = entry
        requests.append({"state": state, "questions": qs})
    return requests


# ─────────────────────────────────────────────────────────────────────────────
# Helper: flatten results
# ─────────────────────────────────────────────────────────────────────────────
def flatten_results(results, questions, sample_df):
    records = []
    for i, res in enumerate(results):
        row = {}
        answers = res.get("answers", {})
        for q in questions:
            qid = q["name"]
            qtype = q["qtype"]
            ans = answers.get(qid, {})
            if qtype == "choice":
                row[f"{qid}_pred"] = ans.get("choice", None)
            elif qtype == "score":
                raw_levels = [s.strip() for s in q.get("score_levels", "Low,Medium,High").split(",") if s.strip()]
                score_val = ans.get("score", None)
                if score_val is not None:
                    idx = int(np.clip(round(float(score_val)), 0, len(raw_levels) - 1))
                    row[f"{qid}_pred"] = raw_levels[idx]
                    row[f"{qid}_score_raw"] = score_val
                else:
                    row[f"{qid}_pred"] = None
                    row[f"{qid}_score_raw"] = None
            elif qtype == "noul":
                row[f"{qid}_prob"] = ans.get("noul", None)
            conf = ans.get("confidence", None)
            row[f"{qid}_confidence"] = conf
        records.append(row)
    pred_df = pd.DataFrame(records)
    result_df = sample_df.reset_index(drop=True).join(pred_df)
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — 4-step wizard
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration Wizard")

    # ── Step 1: Upload CSV & Select Columns ──────────────────────────────────
    with st.expander("📂 Step 1 — Upload & Columns", expanded=st.session_state.df is None):
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(st.session_state.df):,} rows · {len(st.session_state.df.columns)} columns.")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

        if st.session_state.df is not None:
            cols = list(st.session_state.df.columns)
            st.caption("Columns: " + ", ".join(f"`{c}`" for c in cols))
            text_cols = st.multiselect(
                "Text columns to use as model input (concatenated)",
                options=cols,
                default=cols[:1],
            )
        else:
            text_cols = []

    df = st.session_state.df

    # ── Step 2: Sample & Model ────────────────────────────────────────────────
    with st.expander("⚙️ Step 2 — Sample & Model", expanded=False):
        total_rows = len(df) if df is not None else 500
        n_rows = st.slider(
            "Number of rows to evaluate",
            min_value=50,
            max_value=max(50, total_rows),
            value=min(500, max(50, total_rows)),
            step=50,
        )
        batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=32, step=8)
        checkpoint = st.selectbox(
            "Checkpoint",
            options=["auto (Router)", "english", "multilingual", "typed-decisions"],
        )
        sort_by_length = st.checkbox("Sort by length (faster batching)", value=True)
        random_seed = st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42, step=1,
            help="Controls which rows are sampled. Change to get a different subset.",
        )
        _cuda_available = False
        try:
            import torch as _torch
            _cuda_available = _torch.cuda.is_available()
        except Exception:
            pass
        _device_options = ["cuda", "cpu"] if _cuda_available else ["cpu"]
        device = st.selectbox(
            "Device",
            options=_device_options,
            help="CUDA is available ✅" if _cuda_available else "No CUDA GPU detected — running on CPU",
        )
        if _cuda_available:
            st.caption("✅ CUDA GPU detected")
        else:
            st.caption("⚠️ No CUDA GPU — CPU only")

    # ── Step 3: Define Questions ──────────────────────────────────────────────
    import json as _json

    with st.expander("❓ Step 3 — Define Questions", expanded=False):

        if st.button("➕ Add Question", use_container_width=True):
            st.session_state.questions.append({
                "id":           st.session_state._next_qid,
                "name":         f"question_{st.session_state._next_qid}",
                "qtype":        "choice",
                "instructions": "",
                "criteria":     [],
                "score_levels": "Low, Medium, High, Critical",
            })
            st.session_state._next_qid += 1
            st.rerun()

        if not st.session_state.questions:
            st.caption("No questions yet. Click the button above to add one.")

        to_delete    = []
        to_add_row   = []
        to_del_row   = []
        to_load_json = []

        for qi, q in enumerate(st.session_state.questions):
            qid        = q["id"]
            q_label    = q["name"] or f"question_{qi}"
            type_badge = {"choice": "🔵 choice", "score": "🟠 score", "noul": "🟣 noul"}.get(q["qtype"], q["qtype"])

            with st.expander(f"{q_label}  ·  {type_badge}", expanded=True):

                h1, h2, h3 = st.columns([3, 2, 1])
                with h1:
                    q["name"] = st.text_input(
                        "ID", value=q["name"], key=f"qname_{qid}",
                        placeholder="e.g. category",
                        label_visibility="collapsed",
                    )
                with h2:
                    q["qtype"] = st.selectbox(
                        "Type", options=["choice", "score", "noul"],
                        index=["choice", "score", "noul"].index(q["qtype"]),
                        key=f"qtype_{qid}", label_visibility="collapsed",
                    )
                with h3:
                    if st.button("🗑", key=f"del_{qid}", use_container_width=True, help="Remove question"):
                        to_delete.append(qi)

                q["instructions"] = st.text_input(
                    "Instructions",
                    value=q.get("instructions", ""),
                    key=f"qinstr_{qid}",
                    placeholder="e.g. What is the IT support category of this ticket?",
                )

                # ── choice ────────────────────────────────────────────────
                if q["qtype"] == "choice":

                    # Compact JSON import row
                    ji1, ji2, ji3 = st.columns([2, 3, 1])
                    with ji1:
                        json_file = st.file_uploader(
                            "Upload .json", type=["json"],
                            key=f"cjson_file_{qid}",
                            label_visibility="collapsed",
                            help='{"Label": "description"} or [{"label":…,"description":…}]',
                        )
                    with ji2:
                        json_text = st.text_input(
                            "Paste JSON",
                            key=f"cjson_text_{qid}",
                            placeholder='{"Label": "desc", ...}',
                            label_visibility="collapsed",
                        )
                    with ji3:
                        if st.button("Load", key=f"cjson_load_{qid}", use_container_width=True):
                            raw = None
                            try:
                                if json_file is not None:
                                    raw = _json.loads(json_file.read().decode("utf-8"))
                                elif json_text.strip():
                                    raw = _json.loads(json_text.strip())
                                else:
                                    st.warning("Upload or paste JSON first.")
                                if raw is not None:
                                    if isinstance(raw, dict):
                                        parsed = [{"label": k, "description": v} for k, v in raw.items()]
                                    elif isinstance(raw, list):
                                        parsed = [{"label": it.get("label",""), "description": it.get("description","")}
                                                  for it in raw if isinstance(it, dict)]
                                    else:
                                        st.error("Must be a JSON object or array.")
                                        parsed = None
                                    if parsed is not None:
                                        to_load_json.append((qi, parsed))
                            except Exception as _e:
                                st.error(f"Invalid JSON: {_e}")

                    # Criteria rows
                    if q["criteria"]:
                        st.caption(f"Criteria ({len(q['criteria'])})")
                    new_criteria = []
                    for ci, item in enumerate(q["criteria"]):
                        r1, r2, r3 = st.columns([2, 4, 0.4])
                        with r1:
                            lbl = st.text_input("Label", value=item["label"],
                                                key=f"clabel_{qid}_{ci}",
                                                label_visibility="collapsed",
                                                placeholder="Label")
                        with r2:
                            dsc = st.text_input("Desc", value=item["description"],
                                                key=f"cdesc_{qid}_{ci}",
                                                label_visibility="collapsed",
                                                placeholder="What this label means")
                        with r3:
                            if st.button("✖", key=f"cdel_{qid}_{ci}"):
                                to_del_row.append((qi, ci))
                            else:
                                new_criteria.append({"label": lbl, "description": dsc})
                    q["criteria"] = new_criteria

                    if not q["criteria"]:
                        st.caption("No criteria yet — upload/paste JSON or add rows.")
                    if st.button("＋ Add row", key=f"cadd_{qid}"):
                        to_add_row.append(qi)

                # ── score ─────────────────────────────────────────────────
                elif q["qtype"] == "score":
                    q["score_levels"] = st.text_input(
                        "Ordinal levels (comma-separated, low → high)",
                        value=q.get("score_levels", "Low, Medium, High, Critical"),
                        key=f"qscore_{qid}",
                        placeholder="Low, Medium, High, Critical",
                    )

                # ── noul ──────────────────────────────────────────────────
                else:
                    st.caption("Returns P(true) — a float 0.0→1.0. No criteria needed.")

        # ── Apply deferred mutations ──────────────────────────────────────
        changed = False
        for idx in sorted(to_delete, reverse=True):
            st.session_state.questions.pop(idx); changed = True
        for qi, parsed in to_load_json:
            if qi < len(st.session_state.questions):
                st.session_state.questions[qi]["criteria"] = parsed; changed = True
        for qi, ci in sorted(to_del_row, reverse=True):
            if qi < len(st.session_state.questions):
                crit = st.session_state.questions[qi]["criteria"]
                if ci < len(crit): crit.pop(ci)
            changed = True
        for qi in to_add_row:
            if qi < len(st.session_state.questions):
                st.session_state.questions[qi]["criteria"].append({"label": "", "description": ""}); changed = True
        if changed:
            st.rerun()

    # ── Step 4: Ground Truth ──────────────────────────────────────────────────
    with st.expander("🎯 Step 4 — Ground Truth (optional)", expanded=False):
        gt_col      = None
        gt_question = None   # kept for backward compat (single mapping)
        gt_mappings = {}     # {question_name: gt_column}

        if df is None:
            st.info("Upload a CSV first.")
        elif not st.session_state.questions:
            st.info("Define questions in Step 3 first.")
        else:
            cols     = ["— none —"] + list(df.columns)
            eligible = [q for q in st.session_state.questions if q["qtype"] in ("choice", "score")]

            if not eligible:
                st.caption("Only choice and score questions can be evaluated against a ground truth.")
            else:
                st.caption("Map each question to a ground-truth column (or leave as — none —).")
                for q in eligible:
                    qname  = q["name"] or f"question_{q['id']}"
                    qtype  = q["qtype"]
                    badge  = "🔵" if qtype == "choice" else "🟠"
                    sel = st.selectbox(
                        f"{badge} **{qname}**",
                        options=cols,
                        key=f"gt_map_{q['id']}",
                    )
                    if sel != "— none —":
                        gt_mappings[qname] = sel
                        # show a preview of unique values
                        uniq = sorted(str(v) for v in df[sel].dropna().unique())
                        st.caption(f"  {len(uniq)} values: {', '.join(uniq[:8])}" + (" …" if len(uniq) > 8 else ""))

                # legacy single-GT kept for run/results wiring
                if gt_mappings:
                    first_q = next(iter(gt_mappings))
                    gt_col      = gt_mappings[first_q]
                    gt_question = first_q

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    run_disabled = df is None or not text_cols or not st.session_state.questions
    run_clicked = st.button(
        "🚀 Run Evaluation",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Run inference & show results
# ─────────────────────────────────────────────────────────────────────────────
if run_clicked:
    st.session_state.results_df = None
    st.session_state.metrics = None

    # Drop rows missing any selected text column before sampling (matches notebook behaviour)
    clean_df = df.dropna(subset=text_cols).reset_index(drop=True)
    sample_df = clean_df.sample(n=min(n_rows, len(clean_df)), random_state=random_seed).reset_index(drop=True)

    requests = build_requests(sample_df, text_cols, st.session_state.questions)

    status_box = st.empty()
    progress_bar = st.progress(0, text="Loading model…")

    try:
        from laya import Router

        @st.cache_resource(show_spinner=False)
        def get_router(device_: str, checkpoint_: str):
            kwargs = {"device": device_}
            if checkpoint_ != "auto (Router)":
                kwargs["model"] = checkpoint_
            return Router(**kwargs)

        status_box.info("⏳ Loading Laya router…")
        router = get_router(device, checkpoint)

        status_box.info(f"⏳ Running inference on {len(requests):,} samples…")
        t0 = time.time()

        # Batch inference with progress updates
        results = []
        bs = int(batch_size)
        total_batches = (len(requests) + bs - 1) // bs

        for batch_idx in range(total_batches):
            batch = requests[batch_idx * bs : (batch_idx + 1) * bs]
            batch_results = router.predict_batch(
                batch, batch_size=bs, sort_by_length=sort_by_length
            )
            results.extend(batch_results)
            pct = (batch_idx + 1) / total_batches
            progress_bar.progress(pct, text=f"Batch {batch_idx+1}/{total_batches}")

        elapsed = time.time() - t0
        progress_bar.progress(1.0, text="Done!")
        st.session_state.run_time = elapsed

        result_df = flatten_results(results, st.session_state.questions, sample_df)
        st.session_state.results_df = result_df
        st.session_state.gt_col = gt_col
        st.session_state.gt_question = gt_question if gt_col else None
        st.session_state.gt_mappings = gt_mappings
        st.session_state.questions_snapshot = [q.copy() for q in st.session_state.questions]

        status_box.success(
            f"✅ Inference complete — {elapsed:.1f}s total, "
            f"{elapsed / len(requests) * 1000:.1f} ms/ticket"
        )

    except Exception as e:
        progress_bar.empty()
        status_box.error(f"Error during inference: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.titlesize":     11,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
}

if st.session_state.results_df is not None:
    result_df       = st.session_state.results_df
    gt_col_snap     = st.session_state.get("gt_col", None)
    questions_snap  = st.session_state.get("questions_snapshot", st.session_state.questions)
    elapsed         = st.session_state.run_time

    # ── 1. Results Table ──────────────────────────────────────────────────────
    st.subheader("📋 Results Table")
    st.dataframe(result_df, use_container_width=True, height=280)

    csv_buf = io.StringIO()
    result_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="laya_results.csv",
        mime="text/csv",
    )

    st.divider()

    # ── 2. Summary Run ────────────────────────────────────────────────────────
    st.subheader("⏱️ Summary Run")
    n_q = len(questions_snap)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows evaluated", f"{len(result_df):,}")
    m2.metric("Total time", f"{elapsed:.2f}s")
    m3.metric("ms / row", f"{elapsed / len(result_df) * 1000:.1f}")
    m4.metric("Questions", str(n_q))

    st.divider()

    # ── 3. Analytics ──────────────────────────────────────────────────────────
    st.subheader("📈 Analytics")

    for q in questions_snap:
        qid      = q["name"]
        qtype    = q["qtype"]
        pred_col = f"{qid}_pred"
        conf_col = f"{qid}_confidence"
        prob_col = f"{qid}_prob"

        with st.container():
            st.markdown(f"**`{qid}`** — *{qtype}*")

            with plt.rc_context(PLOT_STYLE):

                if qtype in ("choice", "score") and pred_col in result_df.columns:
                    col_a, col_b = st.columns(2)

                    # Prediction distribution
                    with col_a:
                        vc = result_df[pred_col].value_counts().sort_index()
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(vc)))
                        bars = ax.barh(vc.index.astype(str), vc.values, color=colors)
                        ax.set_title("Prediction Distribution")
                        ax.set_xlabel("Count")
                        for bar, v in zip(bars, vc.values):
                            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                                    str(v), va="center", fontsize=7)
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    # Confidence histogram
                    with col_b:
                        if conf_col in result_df.columns:
                            fig, ax = plt.subplots(figsize=(4.5, 3))
                            ax.hist(result_df[conf_col].dropna(), bins=25,
                                    color="#4A90D9", alpha=0.85, edgecolor="white")
                            ax.axvline(result_df[conf_col].mean(), color="#E05C5C",
                                       linestyle="--", linewidth=1.2,
                                       label=f"mean={result_df[conf_col].mean():.2f}")
                            ax.set_title("Confidence Distribution")
                            ax.set_xlabel("Confidence")
                            ax.set_ylabel("Count")
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

                elif qtype == "noul" and prob_col in result_df.columns:
                    col_a, col_b = st.columns(2)

                    with col_a:
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        ax.hist(result_df[prob_col].dropna(), bins=25,
                                color="#7B61FF", alpha=0.8, edgecolor="white")
                        ax.axvline(0.5, color="#E05C5C", linestyle="--",
                                   linewidth=1.2, label="threshold=0.5")
                        flagged = (result_df[prob_col] >= 0.5).mean()
                        ax.set_title(f"P(yes) Distribution  |  {flagged*100:.1f}% flagged")
                        ax.set_xlabel("P(yes)")
                        ax.set_ylabel("Count")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with col_b:
                        if conf_col in result_df.columns:
                            fig, ax = plt.subplots(figsize=(4.5, 3))
                            ax.hist(result_df[conf_col].dropna(), bins=25,
                                    color="#4A90D9", alpha=0.85, edgecolor="white")
                            ax.axvline(result_df[conf_col].mean(), color="#E05C5C",
                                       linestyle="--", linewidth=1.2,
                                       label=f"mean={result_df[conf_col].mean():.2f}")
                            ax.set_title("Confidence Distribution")
                            ax.set_xlabel("Confidence")
                            ax.set_ylabel("Count")
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

        st.write("")  # spacing between questions

    st.divider()

    # ── 4. Ground Truth ───────────────────────────────────────────────────────
    gt_mappings_snap = st.session_state.get("gt_mappings", {})
    if gt_mappings_snap:
        st.subheader("🎯 Ground Truth Evaluation")

        for q in questions_snap:
            qid      = q["name"]
            qtype    = q["qtype"]
            pred_col = f"{qid}_pred"
            conf_col = f"{qid}_confidence"

            # Only evaluate questions that have a GT mapping
            gt_col_for_q = gt_mappings_snap.get(qid)
            if not gt_col_for_q or gt_col_for_q not in result_df.columns:
                continue
            if qtype not in ("choice", "score") or pred_col not in result_df.columns:
                continue

            st.markdown(f"**`{qid}`** — *{qtype}*  ·  GT: `{gt_col_for_q}`")

            y_true = result_df[gt_col_for_q].astype(str)
            y_pred = result_df[pred_col].astype(str)
            valid  = y_true.notna() & y_pred.notna()
            y_true_v, y_pred_v = y_true[valid], y_pred[valid]

            acc         = accuracy_score(y_true_v, y_pred_v)
            f1_macro    = f1_score(y_true_v, y_pred_v, average="macro",    zero_division=0)
            f1_weighted = f1_score(y_true_v, y_pred_v, average="weighted", zero_division=0)
            mean_conf   = result_df[conf_col].mean() if conf_col in result_df.columns else None

            gm1, gm2, gm3, gm4 = st.columns(4)
            gm1.metric("Accuracy",    f"{acc:.3f}")
            gm2.metric("Macro F1",    f"{f1_macro:.3f}")
            gm3.metric("Weighted F1", f"{f1_weighted:.3f}")
            if mean_conf is not None:
                gm4.metric("Mean confidence", f"{mean_conf:.3f}")

            labels = sorted(set(y_true_v) | set(y_pred_v))
            cm     = confusion_matrix(y_true_v, y_pred_v, labels=labels)

            with plt.rc_context(PLOT_STYLE):

                # Row 1: confusion matrix + per-class accuracy
                col_a, col_b = st.columns(2)

                with col_a:
                    fig, ax = plt.subplots(figsize=(4.5, 3.8))
                    sns.heatmap(
                        cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels,
                        linewidths=0.4, ax=ax, cbar=False,
                        annot_kws={"size": 8},
                    )
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.tick_params(axis="y", rotation=0, labelsize=7)
                    plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
                             rotation_mode="anchor", fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with col_b:
                    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
                    colors = ["#4A90D9" if v >= acc else "#E05C5C" for v in per_class_acc]
                    fig, ax = plt.subplots(figsize=(4.5, 3.8))
                    bars = ax.bar(labels, per_class_acc, color=colors, edgecolor="white")
                    ax.axhline(acc, color="gray", linestyle="--", linewidth=1,
                               label=f"overall={acc:.2f}")
                    ax.set_ylim(0, 1.12)
                    ax.set_title("Per-Class Accuracy")
                    ax.set_xlabel("Class")
                    ax.set_ylabel("Accuracy")
                    plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
                             rotation_mode="anchor", fontsize=7)
                    for bar, v in zip(bars, per_class_acc):
                        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                                f"{v:.2f}", ha="center", fontsize=7)
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # Row 2: calibration + confidence by correctness
                if conf_col in result_df.columns:
                    col_c, col_d = st.columns(2)
                    tmp = result_df[[gt_col_for_q, pred_col, conf_col]].copy()
                    tmp["correct"] = tmp[gt_col_for_q].astype(str) == tmp[pred_col].astype(str)

                    with col_c:
                        bins = np.arange(0, 1.05, 0.1)
                        tmp["bucket"] = pd.cut(tmp[conf_col], bins=bins)
                        cal = tmp.groupby("bucket", observed=True).agg(
                            acc_mean=("correct", "mean"),
                            conf_mean=(conf_col, "mean"),
                            count=(conf_col, "count"),
                        ).dropna()
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        ax.bar(cal["conf_mean"], cal["acc_mean"], width=0.07,
                               color="#4A90D9", alpha=0.8, label="Accuracy")
                        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1,
                                label="Perfect calibration")
                        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
                        ax.set_title("Calibration Curve")
                        ax.set_xlabel("Mean confidence")
                        ax.set_ylabel("Accuracy")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                    with col_d:
                        correct_conf   = tmp[tmp["correct"]][conf_col].dropna()
                        incorrect_conf = tmp[~tmp["correct"]][conf_col].dropna()
                        fig, ax = plt.subplots(figsize=(4.5, 3))
                        ax.hist(correct_conf,   bins=20, alpha=0.65, color="#4A90D9",
                                label=f"Correct (n={len(correct_conf)})")
                        ax.hist(incorrect_conf, bins=20, alpha=0.65, color="#E05C5C",
                                label=f"Incorrect (n={len(incorrect_conf)})")
                        ax.axvline(correct_conf.mean(),   color="#4A90D9", linestyle="--", linewidth=1)
                        ax.axvline(incorrect_conf.mean(), color="#E05C5C", linestyle="--", linewidth=1)
                        ax.set_title("Confidence: Correct vs Incorrect")
                        ax.set_xlabel("Confidence")
                        ax.set_ylabel("Count")
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            st.write("")

elif df is None:
    st.info("👈 Start by uploading a CSV in Step 1 of the sidebar.")
else:
    st.info("👈 Configure your evaluation in the sidebar and click **🚀 Run Evaluation**.")
