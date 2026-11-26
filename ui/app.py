"""Streamlit dashboard for the bean leaf disease pipeline.

Run locally:  streamlit run ui/app.py      (expects the API at API_URL, default http://localhost:8000)
"""
import os
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]
LABELS = {"angular_leaf_spot": "Angular Leaf Spot", "bean_rust": "Bean Rust", "healthy": "Healthy"}
COLORS = {"Angular Leaf Spot": "#2a78d6", "Bean Rust": "#eb6834", "Healthy": "#1baf7a"}
COLOR_SCALE = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))

st.set_page_config(page_title="Bean Disease Classifier", page_icon="🌱", layout="wide")


def api(method, path, timeout=60, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kw)
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {}, r.status_code
    except requests.RequestException as e:
        return {"detail": str(e)}, 0


def fmt_uptime(sec):
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


@st.cache_data(ttl=600)
def load_features():
    data, code = api("GET", "/features")
    return pd.DataFrame(data) if code == 200 else pd.DataFrame()


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌱 Bean Disease AI")
page = st.sidebar.radio("Navigate", ["Dashboard", "Predict", "Data insights", "Upload & retrain"])
st.sidebar.caption(f"API: `{API_URL}`")
st.sidebar.markdown(f"[Swagger API docs]({API_URL}/docs)")


@st.fragment(run_every=5)
def health_panel():
    h, code = api("GET", "/health", timeout=10)
    cols = st.columns(4)
    if code != 200:
        cols[0].metric("API status", "🔴 Offline")
        st.error(f"Cannot reach the API at {API_URL}. On Render's free tier the service may take ~1 min to wake up.")
        return
    cols[0].metric("API status", "🟢 Online" if h["model_loaded"] else "🟠 No model")
    cols[1].metric("Model up-time", fmt_uptime(h["uptime_seconds"]))
    cols[2].metric("Model version", h["model_version"] or "–")
    cols[3].metric("Retraining", "⏳ running" if h["retraining"] else "idle")
    st.caption(f"Up since {h['started_at']} (UTC) · refreshes every 5 s")


# ---------------------------------------------------------------- pages
if page == "Dashboard":
    st.title("Model monitoring dashboard")
    health_panel()

    metrics, code = api("GET", "/model/metrics")
    stats, _ = api("GET", "/stats")
    if code == 200:
        st.subheader("Production model — test-set performance")
        tm = metrics["test_metrics"]
        c = st.columns(6)
        for col, (k, name) in zip(c, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                                      ("f1", "F1-score"), ("roc_auc", "ROC-AUC"), ("loss", "Loss")]):
            col.metric(name, f"{tm.get(k, 0):.3f}")
        st.caption(f"Version {metrics['version']} · saved {metrics['saved_at']} · trigger: {metrics.get('trigger', '–')}"
                   f" · evaluated on {tm.get('n_samples', '?')} held-out test images")

    if stats:
        st.subheader("Usage")
        p = stats["predictions"]
        c = st.columns(4)
        c[0].metric("Predictions served", p["total"])
        c[1].metric("Avg. inference latency", f"{p['avg_latency_ms']} ms" if p["avg_latency_ms"] else "–")
        c[2].metric("Training images", sum(stats["dataset"]["train"].values()))
        c[3].metric("Uploads waiting for retraining", stats["uploads"]["pending_total"])

    st.subheader("Evaluate the model in production")
    st.write("Runs the model that is currently serving requests on the held-out test set, live on the server.")
    if st.button("▶ Run live evaluation"):
        with st.spinner("Evaluating on the test set..."):
            ev, code = api("POST", "/model/evaluate", timeout=300)
        if code == 200:
            c = st.columns(5)
            for col, k in zip(c, ["accuracy", "precision", "recall", "f1", "loss"]):
                col.metric(k.capitalize(), f"{ev[k]:.3f}")
            cm = pd.DataFrame(ev["confusion_matrix"], index=[LABELS[x] for x in CLASSES],
                              columns=[LABELS[x] for x in CLASSES])
            left, right = st.columns(2)
            with left:
                st.markdown("**Confusion matrix** (rows = true, columns = predicted)")
                st.dataframe(cm.style.background_gradient(cmap="Blues"), width="stretch")
            with right:
                st.markdown("**Per-class report**")
                rep = pd.DataFrame(ev["per_class"]).T.loc[CLASSES].rename(index=LABELS)
                st.dataframe(rep[["precision", "recall", "f1-score", "support"]].round(3), width="stretch")
            st.caption(f"Model {ev['model_version']} · {ev['n_samples']} images · {ev['duration_s']} s")
        else:
            st.error(ev.get("detail", "Evaluation failed"))

elif page == "Predict":
    st.title("Diagnose a bean leaf")
    st.write("Upload one photo of a bean leaf. The model predicts whether it is healthy, has angular leaf spot "
             "or bean rust.")
    f = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if f:
        left, right = st.columns([1, 1.3])
        left.image(f, width="stretch")
        with right:
            with st.spinner("Predicting..."):
                res, code = api("POST", "/predict", files={"file": (f.name, f.getvalue(), f.type)})
            if code == 200:
                icon = "✅" if res["predicted_class"] == "healthy" else "⚠️"
                st.markdown(f"## {icon} {res['label']}")
                st.markdown(f"Confidence **{res['confidence']:.1%}** · {res['latency_ms']} ms · model {res['model_version']}")
                probs = pd.DataFrame({"Class": [LABELS[k] for k in res["probabilities"]],
                                      "Probability": list(res["probabilities"].values())})
                chart = alt.Chart(probs).mark_bar(cornerRadiusEnd=4, height=22).encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    y=alt.Y("Class:N", sort=list(COLORS), title=None, axis=alt.Axis(labelOverlap=False)),
                    color=alt.Color("Class:N", scale=COLOR_SCALE, legend=None),
                    tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
                ).properties(height=170)
                st.altair_chart(chart, width="stretch")
                st.info(res["advice"])
            else:
                st.error(res.get("detail", "Prediction failed"))

elif page == "Data insights":
    st.title("What the data tells us")
    df = load_features()
    if df.empty:
        st.warning("Feature table not available from the API.")
        st.stop()
    df["Class"] = df["class"].map(LABELS)
    train = df[df["split"] == "train"]

    st.subheader("Class balance")
    counts = train.groupby("Class").size().reset_index(name="Images")
    st.altair_chart(alt.Chart(counts).mark_bar(cornerRadiusEnd=4, height=24).encode(
        x="Images:Q", y=alt.Y("Class:N", sort=list(COLORS), title=None),
        color=alt.Color("Class:N", scale=COLOR_SCALE, legend=None), tooltip=["Class", "Images"],
    ).properties(height=120), width="stretch")
    st.caption("The three classes are balanced (~390 training images each), so accuracy is a fair metric.")

    def density(col, title, xtitle):
        base = alt.Chart(df).transform_density(col, groupby=["Class"], as_=[col, "density"], extent=[0, 1])
        lines = base.mark_area(opacity=0.15).encode(
            x=alt.X(f"{col}:Q", title=xtitle, axis=alt.Axis(format="%")), y=alt.Y("density:Q", title="density"),
            color=alt.Color("Class:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)))
        edge = base.mark_line(strokeWidth=2).encode(x=f"{col}:Q", y="density:Q",
                                                    color=alt.Color("Class:N", scale=COLOR_SCALE))
        med = df.groupby("Class")[col].median().reset_index()
        rules = alt.Chart(med).mark_rule(strokeDash=[4, 3], strokeWidth=1.5).encode(
            x=f"{col}:Q", color=alt.Color("Class:N", scale=COLOR_SCALE),
            tooltip=["Class", alt.Tooltip(f"{col}:Q", format=".1%", title="median")])
        st.altair_chart((lines + edge + rules).properties(title=title, height=260), width="stretch")
        tbl = df.groupby("Class")[col].agg(["median", "mean"]).mul(100).round(1).rename(
            columns={"median": "median %", "mean": "mean %"})
        with st.expander("Show values"):
            st.dataframe(tbl)

    st.subheader("1 · Green leaf area")
    density("green_ratio", "Share of green-dominant pixels (dashed = median)", "green leaf area")
    st.markdown("**Interpretation:** healthy leaves are ~70 % green, diseased leaves only ~43 %. Disease destroys "
                "chlorophyll-rich tissue, so green area separates *healthy vs. diseased* well, but the two disease "
                "curves overlap: green area alone cannot say *which* disease it is.")

    st.subheader("2 · Lesion-coloured area")
    density("lesion_ratio", "Share of brown / yellow / rust-coloured pixels", "lesion-coloured area")
    st.markdown("**Interpretation:** bean rust has the most rust-brown pixels (median ≈ 17 %), angular leaf spot "
                "≈ 12 % and healthy ≈ 7 % (soil in the background). Rust pustules are reddish-brown and spread "
                "over the leaf; angular leaf spot lesions are greyer and bounded by veins.")

    st.subheader("3 · Texture (edge density)")
    density("edge_density", "Share of strong-gradient pixels", "edge density")
    st.markdown("**Interpretation:** texture is almost identical across classes (≈ 14–15 %) because field photos "
                "include soil, veins and other leaves. Global statistics cannot separate the two diseases, so the "
                "model must learn the *shape* of the lesions, which is why a convolutional network (a custom CNN trained from scratch) "
                "is used.")

    st.subheader("Putting it together")
    sample = df.sample(min(900, len(df)), random_state=1)
    st.altair_chart(alt.Chart(sample).mark_circle(size=40, opacity=0.6, stroke="white", strokeWidth=0.5).encode(
        x=alt.X("green_ratio:Q", title="green leaf area", axis=alt.Axis(format="%")),
        y=alt.Y("lesion_ratio:Q", title="lesion-coloured area", axis=alt.Axis(format="%")),
        color=alt.Color("Class:N", scale=COLOR_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Class", "file", alt.Tooltip("green_ratio:Q", format=".1%"), alt.Tooltip("lesion_ratio:Q", format=".1%")],
    ).properties(height=380).interactive(), width="stretch")
    st.markdown("**The story:** healthy leaves form their own cluster (lots of green, few lesions). The two diseases "
                "overlap in colour space, so the CNN has to rely on the lesions' spatial patterns. Augmentation "
                "(flip, rotate, zoom, contrast) makes it robust to the varied lighting and camera angles seen here.")

elif page == "Upload & retrain":
    st.title("Upload new data & retrain")
    stats, code = api("GET", "/stats")
    if code != 200:
        st.error("API unreachable")
        st.stop()
    up = stats["uploads"]

    st.subheader("1 · Upload labelled images (bulk)")
    st.write("Select the class, then drop many images at once, **or** upload a `.zip` containing folders named "
             "`angular_leaf_spot/`, `bean_rust/`, `healthy/`. Files are validated and saved to the database.")
    label = st.selectbox("Class of these images", CLASSES, format_func=LABELS.get)
    files = st.file_uploader("Images or zip", type=["jpg", "jpeg", "png", "webp", "bmp", "zip"],
                             accept_multiple_files=True)
    if st.button("⬆ Upload", disabled=not files, type="primary"):
        with st.spinner(f"Uploading {len(files)} file(s)..."):
            payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
            res, code = api("POST", "/upload", files=payload, data={"label": label}, timeout=600)
        if code == 200:
            st.success(f"Saved {res['saved']} image(s). {res['pending_for_retraining']} image(s) now waiting "
                       "for retraining.")
            if res["rejected"]:
                st.warning(f"{len(res['rejected'])} file(s) rejected")
                st.json(res["rejected"], expanded=False)
            if res["auto_retrain_started"]:
                st.info(f"Threshold of {res['retrain_threshold']} reached → automatic retraining started "
                        f"(run {res['auto_retrain_started']}).")
            time.sleep(1)
            st.rerun()
        else:
            st.error(res.get("detail", "Upload failed"))

    st.subheader("2 · Data waiting for retraining")
    pend = pd.DataFrame({"Class": [LABELS[c] for c in CLASSES], "Pending": [up["pending"][c] for c in CLASSES],
                         "Already used": [up["used"][c] for c in CLASSES]})
    c1, c2 = st.columns([1, 1.4])
    c1.dataframe(pend, hide_index=True, width="stretch")
    thr = stats["retrain_threshold"]
    c2.progress(min(up["pending_total"] / thr, 1.0),
                text=f"Automatic retraining trigger: {up['pending_total']} / {thr} new images"
                     + ("" if stats["auto_retrain"] else " (auto-trigger disabled)"))

    st.subheader("3 · Trigger retraining")
    st.write("Fine-tunes the current production model on the uploaded images (plus a replay sample of the "
             "original training data). The new model replaces the old one only if its test accuracy does not drop.")
    if st.button("🔁 Retrain model now", disabled=up["pending_total"] == 0):
        res, code = api("POST", "/retrain")
        if code == 200:
            st.session_state["run_id"] = res["run_id"]
        else:
            st.error(res.get("detail", "Could not start retraining"))

    run_id = st.session_state.get("run_id")
    if run_id:
        box = st.empty()
        failures = 0
        with st.spinner(f"Retraining (run {run_id}) — this can take a few minutes..."):
            while True:
                run, code = api("GET", f"/retrain/status/{run_id}")
                if code != 200:                       # API unreachable or run lost: retry for ~1 min
                    failures += 1
                    if failures >= 15:
                        break
                    box.warning(f"Run {run_id}: waiting for the API to respond ({failures}/15) ...")
                    time.sleep(4)
                    continue
                failures = 0
                if run.get("status") != "running":
                    break
                box.info(f"Run {run_id} running since {run.get('started_at')} ...")
                time.sleep(4)
        box.empty()
        st.session_state.pop("run_id")
        if code != 200:
            st.error(f"Run {run_id} was lost: the API stopped responding or restarted during retraining. "
                     "Training needs about 850 MB of RAM, so hosts with 512 MB (such as Render's free tier) "
                     "kill the server. Run retraining locally or on an instance with at least 1 GB of RAM.")
        else:
            (st.success if run.get("status") == "completed" else st.warning)(
                f"Run {run_id}: {run.get('status')} — {run.get('message')}")
        if run.get("metrics"):
            m = run["metrics"]
            cols = st.columns(5)
            for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "loss"]):
                col.metric(k.capitalize(), f"{m['after'][k]:.3f}", f"{m['after'][k] - m['before'][k]:+.3f}",
                           delta_color="inverse" if k == "loss" else "normal")
            hist = m.get("history", {})
            if hist:
                h = pd.DataFrame({"epoch": range(1, len(hist["loss"]) + 1), "train": hist["loss"],
                                  "validation": hist.get("val_loss")}).melt("epoch", var_name="set", value_name="loss")
                st.altair_chart(alt.Chart(h).mark_line(point=True, strokeWidth=2).encode(
                    x="epoch:O", y="loss:Q",
                    color=alt.Color("set:N", scale=alt.Scale(range=["#2a78d6", "#eb6834"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["epoch", "set", alt.Tooltip("loss:Q", format=".4f")]).properties(
                    title="Retraining loss", height=220), width="stretch")

    st.subheader("Retraining history")
    hist, _ = api("GET", "/retrain/history")
    if hist:
        rows = [{"Run": r["id"], "Trigger": r["trigger"], "Status": r["status"], "Started": r["started_at"],
                 "New images": r["n_new_images"],
                 "Acc. before": (r["metrics"] or {}).get("before", {}).get("accuracy"),
                 "Acc. after": (r["metrics"] or {}).get("after", {}).get("accuracy"),
                 "Model version": r["model_version"], "Message": r["message"]} for r in hist]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No retraining runs yet.")
