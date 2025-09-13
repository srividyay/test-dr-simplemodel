"""
app.py —
---------------------------------------------------------------------------
- No Google service account; uses public Drive File ID or URL.
- Downloads once to MODEL_LOCAL_PATH (default: artifacts/model.bin).
- Optionally mirrors/copies to multiple expected paths (MODEL_TARGETS env, comma-separated)
  so existing code that looks in a different path still finds the file.
- Exposes MODEL_LOCAL_PATH via env for downstream code.

Env/Secrets:
  - GDRIVE_FILE_ID    : Google Drive file ID or full URL (public access)
  - MODEL_LOCAL_PATH  : Primary save path (default: artifacts/model.bin)
  - MODEL_TARGETS     : Comma-separated extra paths to copy the model to (e.g. "artifacts/model.pkl,models/best.h5")
  - MODEL_SHA256      : Optional expected SHA-256 (lowercase hex) for integrity
"""
import os, re, hashlib, requests, shutil
from pathlib import Path
import streamlit as st

_DEF_CHUNK = 1 << 20  # 1 MiB
BASE_DIR = Path(__file__).resolve().parent

def _abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p)

# --- repo-root import bootstrap (so `src` resolves) ---
import sys, os, time
from pathlib import Path
import streamlit as st
from urllib.request import urlopen, Request, urlretrieve
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- end bootstrap ---

# Optional: read “secrets” if you deploy on Streamlit Cloud
# Put in .streamlit/secrets.toml:
# [artifacts]
# model_gdrive_id = "1AbCDEF..."
# pipeline_gdrive_id = "1HiJKL..."
SECRETS = st.secrets.get("artifacts", {})

# Where to cache artifacts locally (relative to repo root by default)
CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ROOT / "artifacts"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _gdrive_download(file_id: str, dst: Path) -> None:
    """
    Download a PUBLIC Google Drive file id to dst without gdown/service account.
    For large files with virus scan warning, we follow the confirm token cookie.
    """
    # Simple path via export=download
    base = "https://drive.google.com/uc?export=download&id="
    url = f"{base}{file_id}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req) as r:
            data = r.read()
    except HTTPError as e:
        raise RuntimeError(f"GDrive download failed: HTTP {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"GDrive download failed: {e.reason}") from e

    # For big files, Google adds an interstitial page. Heuristic: if content looks HTML, retry with confirm.
    if data[:100].lstrip().startswith(b"<!DOCTYPE html") or b"confirm=" in data:
        # Try to extract confirm token
        text = data.decode("utf-8", errors="ignore")
        import re
        m = re.search(r'href="(/uc\?export=download[^"]*?confirm=([^"&]+)[^"]*)"', text)
        if not m:
            # Fallback: write what we got; sometimes it is the file
            dst.write_bytes(data)
            return
        confirm_url = "https://drive.google.com" + m.group(1).replace("&amp;", "&")
        req2 = Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req2) as r2:
            data2 = r2.read()
        dst.write_bytes(data2)
        return
    # Small file path
    dst.write_bytes(data)


def _download_if_missing(dst: Path, *, file_id: str | None = None, url: str | None = None) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if file_id:
        _gdrive_download(file_id, dst)
    elif url:
        # Generic URL
        urlretrieve(url, dst)
    else:
        raise RuntimeError(f"No download source provided for {dst}")


@st.cache_resource(show_spinner=True)
def ensure_and_load_artifacts(cfg_path: str):
    """
    Ensures model/pipeline exist locally by downloading if needed, then loads them.
    Cached so it runs once per session.
    """
    from src.config import load_config
    from src.data_pipeline import ImagePreprocessor
    import pickle

    cfg = load_config(cfg_path)

    # Resolve local paths
    pipeline_pkl = Path(cfg.paths["pipeline_pkl"])
    model_pkl = Path(cfg.paths["model_pkl"])

    # Optionally override with a cache dir
    if not pipeline_pkl.is_absolute():
        pipeline_pkl = CACHE_DIR / Path(cfg.paths["pipeline_pkl"]).name
    if not model_pkl.is_absolute():
        model_pkl = CACHE_DIR / Path(cfg.paths["model_pkl"]).name

    # Sources (choose whichever you use)
    # 1) Prefer Streamlit secrets
    pipe_id = SECRETS.get("pipeline_gdrive_id", None)
    model_id = SECRETS.get("model_gdrive_id", None)
    # 2) Or encode in your YAML as:
    # paths:
    #   pipeline_gdrive_id: "..."
    #   model_gdrive_id: "..."
    pipe_id = pipe_id or cfg.paths.get("pipeline_gdrive_id")
    model_id = model_id or cfg.paths.get("model_gdrive_id")
    pipe_url = cfg.paths.get("pipeline_url")  # optional direct URL
    model_url = cfg.paths.get("model_url")    # optional direct URL

    with st.spinner("Ensuring model & pipeline artifacts..."):
        if not pipeline_pkl.exists():
            if pipe_id or pipe_url:
                _download_if_missing(pipeline_pkl, file_id=pipe_id, url=pipe_url)
            else:
                # If you train locally first, you can upload your pkl to the repo/artifacts
                raise FileNotFoundError(
                    f"Preprocessor missing: {pipeline_pkl}. Provide pipeline_gdrive_id or pipeline_url in config/secrets."
                )
        if not model_pkl.exists():
            if model_id or model_url:
                _download_if_missing(model_pkl, file_id=model_id, url=model_url)
            else:
                raise FileNotFoundError(
                    f"Model missing: {model_pkl}. Provide model_gdrive_id or model_url in config/secrets."
                )

    # Now load them
    pre = ImagePreprocessor.load(str(pipeline_pkl))
    with open(model_pkl, "rb") as f:
        wrapper = pickle.load(f)
    return cfg, pre, wrapper

# ---------------------------
# Hashing & safe file writes
# ---------------------------
def _sha256_file(path: str, chunk_size: int = _DEF_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _stream_to_file(resp: requests.Response, dst_path: Path, chunk_size: int = _DEF_CHUNK):
    resp.raise_for_status()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(dst_path.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
    tmp.replace(dst_path)

# -----------------------------------------
# Google Drive (PUBLIC) download, cached
# -----------------------------------------
def _gdrive_confirm_token(resp: requests.Response):
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    if "confirm=" in resp.url:
        return resp.url.split("confirm=")[1].split("&")[0]
    return None

def _extract_file_id(maybe_id_or_url: str) -> str:
    s = maybe_id_or_url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", s):
        return s
    m = re.search(r"/file/d/([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", s)
    if m:
        return m.group(1)
    raise ValueError("Could not extract Google Drive file ID. Provide a file ID or a valid Drive URL.")

def download_gdrive_public_cached(file_id_or_url: str, dst_path: Path, expected_sha256: str | None = None, timeout: int = 120) -> Path:
    # Cache hit?
    if dst_path.exists() and dst_path.stat().st_size > 0:
        if expected_sha256:
            try:
                if _sha256_file(str(dst_path)) == expected_sha256.lower():
                    return dst_path
            except Exception:
                pass
        else:
            return dst_path

    file_id = _extract_file_id(file_id_or_url)
    sess = requests.Session()
    base = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}

    r = sess.get(base, params=params, stream=True, timeout=timeout)
    token = _gdrive_confirm_token(r)
    if token:
        params["confirm"] = token
        r = sess.get(base, params=params, stream=True, timeout=timeout)

    # If Drive returns an HTML error page (quota/permissions), bail out with clear message.
    ctype = r.headers.get("Content-Type", "").lower()
    if "text/html" in ctype:
        head = r.text[:1024].lower()
        if any(x in head for x in ("quota", "access denied", "sign in", "permission")):
            raise RuntimeError(
                "Google Drive public link error: quota or permission issue. "
                "Make a copy of the file to get a new File ID or adjust sharing to 'Anyone with the link'."
            )

    _stream_to_file(r, dst_path)

    if expected_sha256 and _sha256_file(str(dst_path)) != expected_sha256.lower():
        raise RuntimeError("SHA-256 mismatch after download.")
    return dst_path

def ensure_model_cached_and_mirrored() -> Path:
    """
    Download the model once (if GDRIVE_FILE_ID provided), save to MODEL_LOCAL_PATH,
    and also copy to each path listed in MODEL_TARGETS (comma-separated).
    Returns the primary local path.
    """
    file_id_or_url = os.getenv("GDRIVE_FILE_ID", "") or st.secrets.get("GDRIVE_FILE_ID", "")
    primary_path = _abs(os.getenv("MODEL_LOCAL_PATH", "../artifacts"))
    expected_sha = (os.getenv("MODEL_SHA256", "") or st.secrets.get("MODEL_SHA256", "") or "").lower() or None

    # Parse additional targets
    if not file_id_or_url:
        # No download configured; still ensure folders exist
        primary_path.parent.mkdir(parents=True, exist_ok=True)

    with st.spinner("Fetching model (first run only)…"):
        local_path = download_gdrive_public_cached(file_id_or_url, primary_path, expected_sha)

    # Expose for downstream loaders
    os.environ.setdefault("MODEL_LOCAL_PATH", str(primary_path))
    return primary_path 

# ============================================================
# === YOUR EXISTING APP CODE GOES BELOW — UNCHANGED =========
# ============================================================
# Paste your original Streamlit app code below this line.
# This patch only ensures the model file is present where your code expects it.
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import io, os, zipfile, tempfile, pickle

ROOT = Path(__file__).resolve().parents[1]  # repo root (parent of streamlit_app/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_pipeline import ImagePreprocessor
from src.utils import safe_extract_zip

#pip install gdown
import gdown 
import os

st.set_page_config(page_title='Image Severity Classifier', layout='wide')

@st.cache_resource
def load_artifacts(cfg_path: str):
    cfg = load_config(cfg_path)
    pre = ImagePreprocessor.load(cfg.paths['pipeline_pkl'])
    with open(cfg.paths['model_pkl'], 'rb') as f:
        wrapper = pickle.load(f)
    return cfg, pre, wrapper

def preprocess_and_predict_file(pre, wrapper, file_bytes, filename):
    tmpdir = tempfile.mkdtemp()
    tmppath = os.path.join(tmpdir, filename)
    with open(tmppath, 'wb') as f:
        f.write(file_bytes)
    arr = pre.preprocess_path(tmppath)
    x = np.expand_dims(arr, axis=0)
    idxs, probs = wrapper.predict(x)
    i = int(idxs[0]); conf = float(np.max(probs[0]))
    return i, conf

def app():
    # Call this before loading models
    #ensure_model_cached_and_mirrored()
    st.title('Image Severity Classifier')
    #cfg_path = st.text_input('Config path', value='configs/config_local.yaml')
    
  if not os.path.exists(cfg_path):
        st.error('Config file not found.')
        st.stop()
    cfg, pre, wrapper = ensure_and_load_artifacts(cfg_path)
    class_names = wrapper.class_names

    tab1, tab2 = st.tabs(['Single Image Prediction', 'ZIP/Bulk Prediction'])

    with tab1:
        st.subheader('Single Image Prediction')
        file = st.file_uploader('Upload an image', type=['png','jpg','jpeg','bmp','tiff','tif','webp'])
        if st.button('Predict Single', type='primary') and file:
            try:
                i, conf = preprocess_and_predict_file(pre, wrapper, file.getbuffer(), file.name)
                st.success(f'Prediction: {class_names[i]} (confidence {conf:.3f})')
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    with tab2:
        st.subheader('ZIP/Bulk Prediction')
        zip_file = st.file_uploader('Upload a ZIP of images', type=['zip'])
        if st.button('Predict ZIP', type='primary') and zip_file:
            try:
                with st.spinner("Processing CSV..."):
                    td = tempfile.mkdtemp()
                    extract_dir = os.path.join(td, 'images')
                    os.makedirs(extract_dir, exist_ok=True)
                    safe_extract_zip(io.BytesIO(zip_file.getbuffer()), extract_dir, max_files=cfg.streamlit.get('max_zip_extract_files', 3500))
                    imgs = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if os.path.splitext(f)[1].lower() in ['.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp']]
                    if not imgs:
                        st.warning('No images found in ZIP.')
                    else:
                        from src.run_pipeline import predict_files
                        out_csv = os.path.join(td, 'predictions.csv')
                        predict_files(cfg, imgs, out_csv)
                        df = pd.read_csv(out_csv)
                        st.download_button('Download CSV', data=df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')
                        st.dataframe(df, use_container_width=True, height=520)
                st.success("CSV processed successfully ✅")
            except Exception as e:
                st.error(f'Batch prediction failed: {e}')

if __name__ == '__main__':
    app()
