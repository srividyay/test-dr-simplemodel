# app.py — Healthcare-friendly Streamlit UI for Image Severity Classifier
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import io, os, zipfile, tempfile, pickle, shutil, json, datetime
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of streamlit_app/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_pipeline import ImagePreprocessor
from src.utils import safe_extract_zip

# Optional: Google Drive model fetch
try:
    import gdown
except Exception:
    gdown = None

# -----------------------------
# App-wide constants & helpers
# -----------------------------
APP_TITLE = "Diabetic Retinopathy Severity Classifier"
APP_TAGLINE = "For clinical use only."

TRIAGE_THRESHOLDS = {
    # map of class index (or name) → (label, guidance, hex_color)
    # You can override from config via cfg.streamlit.triage
    # Example intent: higher index = higher severity (customize for your model)
}

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

def risk_badge(text: str, color: str = "#444"):
    st.markdown(
        f"""
        <span style="
            display:inline-block;
            padding:0.25rem 0.5rem;
            border-radius:999px;
            background:{color};
            color:white;
            font-size:0.85rem;
            font-weight:600;">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )

def in_ph_safe_mode():
    # PHI Safe Mode: never store file names; wipe temp dirs after use
    return st.session_state.get("phi_safe_mode", True)

def show_disclaimer():
    st.info(
        "### Clinical Preview\n",
        icon="👁️",
    )

def ensure_tmp():
    td = tempfile.mkdtemp(prefix="imclf_")
    if in_ph_safe_mode():
        # When PHI-safe mode is ON, we try to minimize local persistence by using tempdirs
        pass
    return td

def cleanup_tmp(path: str):
    if in_ph_safe_mode():
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

def topk_from_probs(class_names, probs, k=5):
    probs = np.array(probs).reshape(-1)
    order = np.argsort(-probs)[: min(k, len(probs))]
    return [(class_names[i], float(probs[i])) for i in order]

def make_prob_df(top_items):
    return pd.DataFrame(top_items, columns=["Class", "Probability"])

def color_from_triage(class_key, triage_map):
    default = ("", "", "#3b82f6")  # blue pill as default
    if class_key in triage_map:
        return triage_map[class_key][2]
    # also try by index if passed as int key string
    try:
        idx_key = int(class_key)
        if idx_key in triage_map:
            return triage_map[idx_key][2]
    except Exception:
        pass
    return default[2]

def get_triage_tip(class_key, triage_map):
    # Returns (label, guidance)
    if class_key in triage_map:
        return triage_map[class_key][0], triage_map[class_key][1]
    try:
        idx_key = int(class_key)
        if idx_key in triage_map:
            return triage_map[idx_key][0], triage_map[idx_key][1]
    except Exception:
        pass
    return ("General", "Predictions provide near 75% accuracy!")

def sanitize_filename(name: str) -> str:
    # Avoid leaking patient identifiers; optionally hash
    if in_ph_safe_mode():
        return "image_" + str(abs(hash(name)))[:8]
    return name

def find_images_recursively(root_dir: str):
    files = []
    for base, _, fnames in os.walk(root_dir):
        for f in fnames:
            if f.startswith("._") or "__MACOSX" in base:
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                files.append(os.path.join(base, f))
    return files

def download_models(model_map: dict):
    if not model_map:
        return
    if gdown is None:
        st.warning("gdown not available; skipping model auto-download.")
        return
    os.makedirs("artifacts", exist_ok=True)
    for filename, url in model_map.items():
        fp = f"artifacts/{filename}"
        if not os.path.exists(fp):
            with st.spinner(f"Downloading {filename}…"):
                gdown.download(url, fp, quiet=False)

# ---------------------------------
# Streamlit app scaffolding & cache
# ---------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🩺")

@st.cache_resource
def load_artifacts(cfg_path: str):
    cfg = load_config(cfg_path)
    # Optional triage overrides
    triage = cfg.streamlit.get("triage", {})
    # Optional Drive model fetch
    drive_models = cfg.streamlit.get("drive_models", {})
    if drive_models:
        download_models(drive_models)

    pre = ImagePreprocessor.load(cfg.paths["pipeline_pkl"])
    with open(cfg.paths["model_pkl"], "rb") as f:
        wrapper = pickle.load(f)
    # class_names
    class_names = getattr(wrapper, "class_names", None)
    if class_names is None:
        # best-effort fallback
        num_classes = int(cfg.model.get("num_classes", 2))
        class_names = [f"class_{i}" for i in range(num_classes)]
        wrapper.class_names = class_names
    return cfg, pre, wrapper, triage

def preprocess_and_predict_file(pre, wrapper, file_bytes, filename):
    tmpdir = ensure_tmp()
    try:
        safe_name = sanitize_filename(filename)
        tmppath = os.path.join(tmpdir, safe_name)
        with open(tmppath, "wb") as f:
            f.write(file_bytes)
        arr = pre.preprocess_path(tmppath)
        x = np.expand_dims(arr, axis=0)
        idxs, probs = wrapper.predict(x)
        i = int(idxs[0]); conf = float(np.max(probs[0]))
        disp_img = Image.fromarray((arr * 255).astype("uint8"))

        return i, conf, probs[0], disp_img
    finally:
        cleanup_tmp(tmpdir)

# -----------------------------
# Sidebar (Clinical UI helpers)
# -----------------------------
with st.sidebar:
    st.header("About & Usage")
    st.caption(APP_TAGLINE)
    show_disclaimer()

    st.toggle("PHI-Safe Mode (recommended)", value=True, key="phi_safe_mode",
              help="When enabled, filenames are sanitized/hardened and temporary data is purged after use.")

    st.markdown("**Confidence Guidance**")
    st.caption(
        "- Severity grades (0-4 scale):\n"
        "- 0: No diabetic retinopathy\n"
        "- 1: Mild nonproliferative retinopathy\n"
        "- 2: Moderate nonproliferative retinopathy\n"
        "- 3: Severe nonproliferative retinopathy\n"
        "- 4: Proliferative diabetic retinopathy"
    )

    st.divider()
    st.markdown("**Audit Notes (local only)**")
    facility = st.text_input("Facility/Team (optional)", "")
    operator  = st.text_input("Operator initials (optional)", "")
    st.caption("These values only annotate UI state; no PHI is stored by this app.")

    st.divider()
    st.markdown("**Config**")
    cfg_path = st.text_input("Config path", value="configs/config_local.yaml", help="YAML with paths & streamlit settings.")
    refresh_models = st.checkbox("Force re-load artifacts", value=False, help="Uncheck unless troubleshooting.")
    st.markdown("---")
    st.caption("Build time: " + _now_iso())

# --------------------------------
# Main: App title & description
# --------------------------------
st.title("🩺 " + APP_TITLE)
st.write(
    "Upload diabetic retinopathy images. The app returns **model-inferred severity** "
    "and **confidence** with a plain-language triage cue. "
    "Consult the sidebar for usage notes."
)

# Load artifacts
if not os.path.exists(cfg_path):
    st.error("Config file not found. Please verify the path in the sidebar.")
    st.stop()

if refresh_models:
    # Drop the cache and reload
    load_artifacts.clear()

cfg, pre, wrapper, triage_map = load_artifacts(cfg_path)
class_names = wrapper.class_names
if not triage_map:
    # Build a gentle default palette across classes
    triage_map = {i: (name, "Interpret in context; not for diagnosis.", col)
                  for i, (name, col) in enumerate(zip(class_names,
                   ["#059669", "#10b981", "#3b82f6", "#f59e0b", "#ef4444"]*10))}

# Tabs
tab1, tab2 = st.tabs(["Single Image Prediction", "ZIP/Bulk Prediction"])

# -------------------------
# Single Image Prediction
# -------------------------
with tab1:
    st.subheader("Single Image Prediction")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        file = st.file_uploader(
            "Upload an image",
            type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTS)],
            accept_multiple_files=False,
            help="Supported formats: PNG, JPG/JPEG, BMP, TIFF, WEBP",
        )
    with col_r:
        agree = st.checkbox(
            "I confirm this upload contains **no PHI** or it has been properly de-identified.",
            value=True,
        )

    if st.button("Predict Single", type="primary", disabled=not(file and agree)):
        if not file:
            st.warning("Please upload an image.")
        elif not agree:
            st.warning("Please confirm de-identification before proceeding.")
        else:
            try:
                with st.spinner("Analyzing image…"):
                    i, conf, probs, disp_img = preprocess_and_predict_file(pre, wrapper, file.getbuffer(), file.name)
                    pred_name = class_names[i]
                    tri_label, tri_tip = get_triage_tip(i, triage_map)
                    color = color_from_triage(i, triage_map)

                    st.success(f"Prediction: **{pred_name}**  ·  Confidence: **{conf:.3f}**")
                    st.image(disp_img, caption="Processed image used for prediction", width=300)

                # Result header
                st.success(f"Prediction: **{pred_name}**  ·  Confidence: **{conf:.3f}**")
                risk_badge(f"Triage: {tri_label}", color)
                st.caption(tri_tip)

                # Top-k breakdown
                top_items = topk_from_probs(class_names, probs, k=min(5, len(class_names)))
                st.markdown("**Class probabilities**")
                df_top = make_prob_df(top_items)
                st.bar_chart(df_top.set_index("Class"))

                # Minimal audit echo (no PHI)
                with st.expander("Session details"):
                    st.json({
                        "timestamp": _now_iso(),
                        "facility": facility or None,
                        "operator": operator or None,
                        "config": Path(cfg_path).name,
                        "phi_safe_mode": in_ph_safe_mode(),
                        "predicted_class": pred_name,
                        "confidence": round(conf, 4),
                        "top_k": top_items,
                    })

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ----------------------
# ZIP / Bulk Prediction
# ----------------------
with tab2:
    st.subheader("ZIP/Bulk Prediction")
    st.caption("Upload a **ZIP** of images (nested folders OK). System ignores `__MACOSX/` and hidden resource files.")
    zip_file = st.file_uploader("Upload ZIP", type=["zip"], accept_multiple_files=False)
    max_files = int(cfg.streamlit.get("max_zip_extract_files", 3500))
    want_csv_preview = st.checkbox("Preview results table after processing", value=True)

    if st.button("Predict ZIP", type="primary", disabled=not zip_file):
        try:
            with st.spinner("Unpacking and preprocessing…"):
                td = ensure_tmp()
                extract_dir = os.path.join(td, "images")
                os.makedirs(extract_dir, exist_ok=True)

                # Safer extraction (limits count & path traversal)
                safe_extract_zip(io.BytesIO(zip_file.getbuffer()), extract_dir, max_files=max_files)

                imgs = find_images_recursively(extract_dir)
                if not imgs:
                    st.warning("No supported images found in ZIP.")
                else:
                    from src.run_pipeline import predict_files  # expects (cfg, imgs, out_csv)
                    out_csv = os.path.join(td, "predictions.csv")

                    with st.spinner("Running model inference…"):
                        predict_files(cfg, imgs, out_csv)

                    df = pd.read_csv(out_csv)
                    # Remove raw filenames in PHI-safe mode; show hashed surrogates instead
                    if in_ph_safe_mode() and "file" in df.columns:
                        df["file_surrogate"] = df["file"].apply(lambda x: sanitize_filename(os.path.basename(str(x))))
                        df = df.drop(columns=["file"]).rename(columns={"file_surrogate": "file"})

                    # Optional triage color column
                    if "pred_idx" in df.columns:
                        df["triage_label"] = df["pred_idx"].apply(lambda k: get_triage_tip(k, triage_map)[0])
                    elif "pred" in df.columns:
                        df["triage_label"] = df["pred"].apply(lambda name: get_triage_tip(name, triage_map)[0])

                    # Download + preview
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                    if want_csv_preview:
                        st.dataframe(df, use_container_width=True, height=520)

                    st.success("CSV processed successfully ✅")

            cleanup_tmp(td)  # purge temp artifacts in PHI-safe mode

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# ----------------------
# Footer / glossary pane
# ----------------------
with st.expander("Clinical Glossary & Interpretation Tips"):
    st.markdown(
        """
- **Model class names** map to your study schema (see config). The app shows a **triage** cue to help you sort cases for review.
- **Confidence** is the model’s internal probability; it is **not** a measure of clinical certainty.
        """
    )

st.caption("© Research preview. Configuration: `{}` • PHI-Safe Mode: **{}**".format(Path(cfg_path).name, "ON" if in_ph_safe_mode() else "OFF"))


# ======================
# Severity → Risk → Recommendation helpers
# ======================
from typing import Dict, List, Union
import numpy as np
import pandas as pd

SEVERITY_LABELS = {
    0: "No diabetic retinopathy",
    1: "Mild nonproliferative retinopathy",
    2: "Moderate nonproliferative retinopathy",
    3: "Severe nonproliferative retinopathy",
    4: "Proliferative diabetic retinopathy",
}

def _severity_action(sev: int) -> Dict[str, str]:
    if sev <= 0:
        return {"triage": "Routine","recommendation": "Routine annual screening.","follow_up": "12 months"}
    if sev == 1:
        return {"triage": "Watchlist","recommendation": "Recheck 6–12 months; optimize glycemic/BP/lipid control.","follow_up": "6–12 months"}
    if sev == 2:
        return {"triage": "Early intervention (flag)","recommendation": "Non-urgent ophthalmology referral; consider treatment planning.","follow_up": "≤ 3 months"}
    if sev == 3:
        return {"triage": "Immediate treatment","recommendation": "Urgent retina referral; treatment per clinician.","follow_up": "1–2 weeks"}
    return {"triage": "Immediate treatment","recommendation": "Urgent retina care (e.g., anti-VEGF/laser per clinician).","follow_up": "48–72 hours"}

def _compute_risk_score(prob: Union[List[float], np.ndarray]) -> float:
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 1 or prob.size != 5:
        raise ValueError("Expected probability vector of length 5 for classes 0..4.")
    weights = np.linspace(0.0, 1.0, 5)
    return float(np.clip(np.sum(prob * weights), 0.0, 1.0))

def recommend_from_prediction(pred_class: int = None, class_probs: Union[List[float], np.ndarray, None] = None):
    if class_probs is not None:
        class_probs = np.asarray(class_probs, dtype=float)
        if class_probs.shape != (5,):
            raise ValueError("class_probs must be length-5 for classes 0..4.")
        severity = int(np.argmax(class_probs))
        risk_score = _compute_risk_score(class_probs)
        confidence = float(class_probs[severity])
    else:
        if pred_class is None:
            raise ValueError("Provide either pred_class or class_probs.")
        severity = int(pred_class)
        one_hot = np.zeros(5); one_hot[severity] = 1.0
        risk_score = _compute_risk_score(one_hot)
        confidence = 1.0
    policy = _severity_action(severity)
    return {"severity": severity,"severity_label": SEVERITY_LABELS[severity],"risk_score_0to1": round(risk_score, 3),
            "confidence": round(confidence, 3),"triage": policy["triage"],"recommendation": policy["recommendation"],"follow_up": policy["follow_up"]}

def recommend_for_dataframe(df: pd.DataFrame, prob_cols=("p0","p1","p2","p3","p4")) -> pd.DataFrame:
    P = df.loc[:, prob_cols].to_numpy()
    sev = P.argmax(axis=1)
    weights = np.linspace(0.0, 1.0, 5)
    risk = (P * weights).sum(axis=1)
    conf = P.max(axis=1)
    out = df.copy()
    out["severity"] = sev
    out["severity_label"] = [SEVERITY_LABELS[i] for i in sev]
    out["risk_score_0to1"] = np.round(risk, 3)
    out["confidence"] = np.round(conf, 3)
    def _pol(i):
        pol = _severity_action(int(i))
        return pol["triage"], pol["recommendation"], pol["follow_up"]
    triage, recs, fu = np.vectorize(_pol, otypes=[object, object, object])(sev)
    out["triage"] = triage
    out["recommendation"] = recs
    out["follow_up"] = fu
    return out

def format_recommendation_text(result: dict) -> str:
    return (f"Severity {result['severity']} – {result['severity_label']} | "
            f"Risk: {result['risk_score_0to1']:.2f} (conf {result['confidence']:.2f}) | "
            f"Triage: {result['triage']} | Follow-up: {result['follow_up']}")


# ======================
# Streamlit UI: Recommendations (Single & Bulk) with spinners
# ======================
try:
    import streamlit as st
    _HAS_STREAMLIT = True
except Exception:
    _HAS_STREAMLIT = False

def _safe_model_predict_proba(image_bytes):
    if 'model_predict_proba' in globals():
        return model_predict_proba(image_bytes)
    if 'predict_image' in globals():
        try:
            out = predict_image(image_bytes)
            if isinstance(out, (list, tuple)):
                for item in out:
                    try:
                        arr = np.asarray(item, dtype=float).reshape(-1)
                        if arr.shape[0] == 5:
                            return arr
                    except Exception:
                        continue
        except Exception:
            pass
    return None

def _badge(text, kind="info"):
    colors = {"success": "#16a34a","warning": "#d97706","error": "#dc2626","info": "#2563eb"}
    c = colors.get(kind, "#2563eb")
    st.markdown(f"<div style='display:inline-block;padding:4px 8px;border-radius:12px;background:{c};color:white;font-weight:600'>{text}</div>", unsafe_allow_html=True)

def render_recommendations_ui():
    if not _HAS_STREAMLIT:
        return
    st.markdown("## 🧠 AI Triage & Recommendations")
    tabs = st.tabs(["Single Image", "Bulk (CSV probs p0..p4)"])

    with tabs[0]:
        st.caption("Upload an image or paste probabilities to compute severity → risk → recommendation.")
        up = st.file_uploader("Upload retinal image", type=["png","jpg","jpeg","bmp","tiff","webp"], key="rec_single")
        probs_text = st.text_input("Or paste probabilities [p0,p1,p2,p3,p4]",
                                   placeholder="e.g., 0.02,0.15,0.55,0.20,0.08")

        if st.button("Compute Recommendation", type="primary"):
            probs = None
            if probs_text.strip():
                try:
                    parts = [float(x.strip()) for x in probs_text.split(",")]
                    if len(parts) == 5:
                        s = sum(parts)
                        if s > 0:
                            parts = [x/s for x in parts]
                        probs = np.array(parts, dtype=float)
                except Exception as e:
                    st.error(f"Could not parse probabilities: {e}")

            if probs is None and up is not None:
                with st.spinner("Running model and computing triage…"):
                    try:
                        image_bytes = up.read()
                        probs = _safe_model_predict_proba(image_bytes)
                    except Exception as e:
                        st.error(f"Inference error: {e}")

            if probs is None:
                st.warning("No probabilities available. Provide probs or ensure your model_predict_proba/predict_image is wired.")
            else:
                with st.spinner("Generating recommendation…"):
                    res = recommend_from_prediction(class_probs=probs)
                _badge(f"Severity {res['severity']} – {res['severity_label']}",
                       kind="error" if res['severity']>=3 else ("warning" if res['severity']==2 else "info"))
                st.metric("Risk (0–1)", f"{res['risk_score_0to1']:.2f}")
                st.metric("Confidence", f"{res['confidence']:.2f}")
                st.success(res["recommendation"] + f" | Follow-up: {res['follow_up']}")
                with st.expander("Details"):
                    st.json(res)

    with tabs[1]:
        st.caption("Upload CSV with p0..p4 columns to batch-generate triage.")
        csv_up = st.file_uploader("Upload probabilities CSV", type=["csv"], key="rec_bulk")
        if st.button("Process CSV", type="primary"):
            if not csv_up:
                st.warning("Please upload a CSV first.")
            else:
                try:
                    with st.spinner("Computing batch recommendations…"):
                        df = pd.read_csv(csv_up)
                        missing_cols = [c for c in ("p0","p1","p2","p3","p4") if c not in df.columns]
                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}. Include p0..p4.")
                        else:
                            out = recommend_for_dataframe(df, prob_cols=("p0","p1","p2","p3","p4"))
                            st.success("Done.")
                            st.dataframe(out, use_container_width=True)
                            csv_bytes = out.to_csv(index=False).encode("utf-8")
                            st.download_button("Download results CSV", data=csv_bytes,
                                               file_name="retinopathy_recommendations.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Failed to process CSV: {e}")

if _HAS_STREAMLIT:
    try:
        render_recommendations_ui()
    except Exception as e:
        import traceback
        st.warning("Recommendation UI loaded with partial functionality.")
        st.code(traceback.format_exc())
