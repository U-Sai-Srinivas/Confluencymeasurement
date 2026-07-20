"""
Cell Confluency Analyzer
========================

Upload a folder of brightfield / phase-contrast microscopy images and get a
per-image % confluency read-out, downloadable as Excel, plus exportable
segmentation masks for raw-data filing.

Two segmentation engines:

  1. Cellpose (AI, recommended) - a pretrained deep-learning cell segmentation
     model (cyto3). On translucent, spread adherent cells (e.g. A549) it is far
     more accurate than classical thresholding and needs almost no tuning.
     Requires `cellpose` + `torch` (see requirements.txt). Pinned to cellpose<4
     so it uses the small, fast `cyto3` model rather than the very large,
     CPU-slow Cellpose-SAM model.

  2. Classical (fast, no AI) - illumination-flattened edge+texture segmentation
     with a single "sensitivity" control. Runs anywhere, in a fraction of a
     second, with no heavy dependencies. Good for a quick look or when Cellpose
     is not installed.

Both engines share the same preprocessing (illumination flattening + optional
contrast enhancement), the same scale-bar / burnt-in-annotation removal, and the
same outputs, so you can switch between them freely.
"""

import io
import os
import gc
import glob
import shutil
import zipfile
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cell Confluency Analyzer", layout="wide")

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# =============================================================================
# Cellpose availability (optional dependency) - detected once, cached.
# =============================================================================

@st.cache_resource(show_spinner=False)
def cellpose_status():
    """Return (available: bool, message: str). Imported lazily so the app still
    runs (classical engine only) on environments without cellpose/torch."""
    try:
        import cellpose  # noqa: F401
        from cellpose import models  # noqa: F401
        return True, getattr(cellpose, "version", "unknown")
    except Exception as e:  # pragma: no cover - depends on deployment
        return False, str(e)


@st.cache_resource(show_spinner="Loading the Cellpose model (first run downloads ~25 MB)...")
def load_cellpose_model():
    """Load and cache the pretrained model. Handles both the cellpose v3 API
    (models.Cellpose with a size model, supports auto-diameter) and the v4 API
    (models.CellposeModel). Returns (api_kind, model)."""
    from cellpose import models
    # Prefer v3-style Cellpose (cyto3) - small, fast on CPU, has a size model.
    try:
        return "v3", models.Cellpose(gpu=False, model_type="cyto3")
    except Exception:
        # v4 (Cellpose-SAM): single generalist model, no model_type/channels.
        return "v4", models.CellposeModel(gpu=False)


# =============================================================================
# Image loading (robust to 8/16-bit, grayscale/BGR/BGRA, tif/png/jpg)
# =============================================================================

def _to_gray_bgr(img):
    """Given a decoded image of any depth/channel layout, return
    (gray_uint8, bgr_uint8) suitable for display and processing."""
    if img is None:
        return None, None
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        bgr = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        bgr = None

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if bgr is None:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif bgr.dtype != np.uint8:
        bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray, bgr


# NOTE: these are deliberately NOT cached. @st.cache_data would keep every
# decoded full-resolution image (~30 MB each) in RAM for the whole session,
# which is what made large batches (>20 images) run out of memory and crash.
def load_from_bytes(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    return _to_gray_bgr(img)


def load_from_path(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return _to_gray_bgr(img)


# =============================================================================
# Shared preprocessing
# =============================================================================

def flatten_illumination(gray, sigma_frac=0.06):
    """Remove broad illumination gradients / vignetting (and the central Airy
    artifact common in these scans) by subtracting a very large-sigma Gaussian
    background estimate. Returns a uint8 image with a flat background."""
    sigma = max(10.0, max(gray.shape) * sigma_frac)
    bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma)
    flat = gray.astype(np.float32) - bg
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def enhance(gray, clip=2.0):
    """CLAHE local contrast enhancement - makes faint, translucent spread cells
    visible to the segmenter without blowing out illumination gradients."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def preprocess(gray, do_flatten=True, do_enhance=True):
    out = gray
    if do_flatten:
        out = flatten_illumination(out)
    if do_enhance:
        out = enhance(out)
    return out


def saturated_mask(gray, thresh=250, dilate=9):
    """Detect near-white, saturated pixels - scale bars and burnt-in
    annotations are pure white and would otherwise be counted as cells."""
    m = (gray >= thresh).astype(np.uint8) * 255
    if np.count_nonzero(m) == 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
    return cv2.dilate(m, k)


def _remove_small(mask, min_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_px
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def downsample(gray, max_dim):
    h, w = gray.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s >= 1.0:
        return gray, 1.0
    return cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA), s


# =============================================================================
# Engine 1: Classical (edge + texture), fast, no AI
# =============================================================================

def segment_classical(gray, sensitivity=0.6, min_object_px=400,
                      close_frac=0.012, drop_saturated=True):
    """Illumination-flattened gradient-magnitude segmentation. Cells (halos,
    membrane edges, internal texture) produce gradient; smooth background does
    not. `sensitivity` in [0,1] monotonically increases detected area."""
    h, w = gray.shape
    flat = flatten_illumination(gray)
    gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 2.0)
    gnorm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    t_otsu, _ = cv2.threshold(gnorm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = float(np.clip(t_otsu * (1.4 - 0.8 * sensitivity), 1, 254))
    mask = (gnorm > thresh).astype(np.uint8) * 255

    ck = max(3, int(round(min(h, w) * close_frac)) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # fill enclosed holes (spread-cell interiors are smooth -> low gradient)
    ff = mask.copy()
    m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    mask = mask | cv2.bitwise_not(ff)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, ck // 2) | 1,) * 2))

    if drop_saturated:
        mask[saturated_mask(gray) > 0] = 0
    mask = _remove_small(mask, min_object_px)

    # labels for cell counting (approximate for classical)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    return mask, labels.astype(np.int32)


# =============================================================================
# Engine 2: Cellpose (AI), recommended
# =============================================================================

def segment_cellpose(gray, diameter=25, auto_diameter=False, flow_threshold=0.6,
                     cellprob_threshold=-1.0, do_flatten=True, do_enhance=True,
                     min_object_px=15, drop_saturated=True):
    """Run pretrained Cellpose on the (preprocessed) image and return a binary
    mask + integer instance-label map. Contrast enhancement is applied first
    because raw brightfield A549 cells are too low-contrast for the model to
    detect reliably."""
    kind, model = load_cellpose_model()
    proc = preprocess(gray, do_flatten=do_flatten, do_enhance=do_enhance)

    diam = None if auto_diameter else diameter
    if kind == "v3":
        out = model.eval(proc, diameter=diam, channels=[0, 0],
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold)
    else:
        try:
            out = model.eval(proc, diameter=diam, flow_threshold=flow_threshold,
                             cellprob_threshold=cellprob_threshold)
        except TypeError:
            out = model.eval(proc)
    labels = np.asarray(out[0]).astype(np.int32)

    if drop_saturated:
        labels[saturated_mask(gray) > 0] = 0
    mask = (labels > 0).astype(np.uint8) * 255
    if min_object_px > 0:
        cleaned = _remove_small(mask, min_object_px)
        labels[cleaned == 0] = 0
        mask = cleaned
    return mask, labels


# =============================================================================
# Unified per-image processing
# =============================================================================

def make_overlay(bgr, mask, labels=None, alpha=0.4):
    """Green fill on detected cells + red outlines. If instance labels are
    given, outline each cell so touching cells stay visually distinct."""
    colored = np.zeros_like(bgr)
    colored[mask > 0] = (0, 255, 0)
    blended = cv2.addWeighted(bgr, 1 - alpha, colored, alpha, 0)
    if labels is not None and labels.max() > 0:
        edges = np.zeros(mask.shape, np.uint8)
        # per-instance boundaries via morphological gradient on each label id
        grad = cv2.morphologyEx((labels > 0).astype(np.uint8) * 255,
                                cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        edges = grad
        # also draw internal boundaries between adjacent instances
        lab_grad = cv2.morphologyEx(labels.astype(np.float32), cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        edges[lab_grad > 0] = 255
        blended[edges > 0] = (0, 0, 255)
    else:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cnts, -1, (0, 0, 255), 2)
    return blended


def process(gray, bgr, params, want_overlay=True, want_enhanced=True,
            want_labels=True):
    """Run the selected engine at the working resolution, then scale results
    back to full resolution for display / export. Returns a result dict.

    Confluency (an area ratio) is scale-invariant, so downsampling for speed
    does not bias it. The want_* flags let batch runs skip building large arrays
    (overlays, enhanced previews, full-res label maps) they don't need, which
    keeps memory low on big folders."""
    work, s = downsample(gray, params["work_dim"])

    if params["engine"] == "Cellpose (AI)":
        mask_w, labels_w = segment_cellpose(
            work,
            diameter=params["diameter"], auto_diameter=params["auto_diameter"],
            flow_threshold=params["flow_threshold"],
            cellprob_threshold=params["cellprob_threshold"],
            do_flatten=params["do_flatten"], do_enhance=params["do_enhance"],
            min_object_px=params["min_object_px"], drop_saturated=params["drop_saturated"])
        enhanced_view = preprocess(work, params["do_flatten"], params["do_enhance"]) if want_enhanced else None
    else:
        mask_w, labels_w = segment_classical(
            work, sensitivity=params["sensitivity"],
            min_object_px=params["min_object_px"],
            drop_saturated=params["drop_saturated"])
        enhanced_view = flatten_illumination(work) if want_enhanced else None

    confluency = 100.0 * np.count_nonzero(mask_w) / mask_w.size
    cell_count = int(labels_w.max())

    # area stats in ORIGINAL-image pixels (undo the downsample factor)
    fg_work = int(np.count_nonzero(mask_w))
    mean_area_orig = (fg_work / max(cell_count, 1)) / (s * s) if cell_count else 0.0

    # scale results back to full resolution for overlay + export
    h, w = gray.shape
    mask_full = cv2.resize(mask_w, (w, h), interpolation=cv2.INTER_NEAREST)
    labels_full = None
    if want_labels or want_overlay:
        labels_full = cv2.resize(labels_w, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = make_overlay(bgr, mask_full, labels_full) if want_overlay else None

    return {
        "confluency": confluency,
        "cell_count": cell_count,
        "mean_cell_area_px": mean_area_orig,
        "mask": mask_full,
        "labels": labels_full,
        "enhanced": enhanced_view,
        "overlay": overlay,
    }


# =============================================================================
# Excel + ZIP builders
# =============================================================================

def build_excel(rows, params, engine_label):
    """Return xlsx bytes with a Results sheet and a Summary sheet."""
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"] if "status" in df else df
    conf = pd.to_numeric(ok.get("confluency_percent"), errors="coerce").dropna()

    summary = {
        "Analysis date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Segmentation engine": engine_label,
        "Images processed": int(len(df)),
        "Images OK": int(len(ok)),
        "Mean confluency (%)": round(float(conf.mean()), 2) if len(conf) else None,
        "Median confluency (%)": round(float(conf.median()), 2) if len(conf) else None,
        "Std confluency (%)": round(float(conf.std()), 2) if len(conf) > 1 else None,
        "Min confluency (%)": round(float(conf.min()), 2) if len(conf) else None,
        "Max confluency (%)": round(float(conf.max()), 2) if len(conf) else None,
    }
    for k, v in params.items():
        summary[f"param: {k}"] = v
    summary_df = pd.DataFrame({"Field": list(summary.keys()),
                               "Value": list(summary.values())})

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Results")
        summary_df.to_excel(xw, index=False, sheet_name="Summary")
        for sheet, frame in (("Results", df), ("Summary", summary_df)):
            ws = xw.sheets[sheet]
            for col in ws[1]:
                col.font = col.font.copy(bold=True)
            for i, name in enumerate(frame.columns, start=1):
                width = max(12, min(40, int(frame[name].astype(str).str.len().max() or 0) + 2,
                                    len(str(name)) + 2))
                ws.column_dimensions[chr(64 + i) if i <= 26 else "AA"].width = max(width, len(str(name)) + 2)
    buf.seek(0)
    return buf.getvalue()


def add_pngs_to_zip(zf, fname, res, export_binary=True, export_overlay=True,
                    export_labels=False):
    base = os.path.splitext(fname)[0]
    if export_binary and res.get("mask") is not None:
        ok, png = cv2.imencode(".png", res["mask"])
        if ok:
            zf.writestr(f"masks/{base}_mask.png", png.tobytes())
    if export_overlay and res.get("overlay") is not None:
        # overlays are QC visuals, so JPEG (much smaller than PNG) is fine and
        # keeps the archive from blowing up on large batches.
        ok, jpg = cv2.imencode(".jpg", res["overlay"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            zf.writestr(f"overlays/{base}_overlay.jpg", jpg.tobytes())
    if export_labels and res.get("labels") is not None:
        # 16-bit instance-label PNG: each cell has a unique id (raw data)
        lab16 = np.clip(res["labels"], 0, 65535).astype(np.uint16)
        ok, png = cv2.imencode(".png", lab16)
        if ok:
            zf.writestr(f"label_masks/{base}_labels.png", png.tobytes())


# =============================================================================
# UI
# =============================================================================

cp_available, cp_info = cellpose_status()

st.title("🔬 Cell Confluency Analyzer")
st.caption("Upload brightfield / phase-contrast images, get a % confluency read-out per image, "
           "download the results as Excel, and export segmentation masks for your records.")

with st.sidebar:
    st.header("1 · Images")
    source_mode = st.radio(
        "Where are the images?",
        ["Upload images", "Local folder path"],
        help="Upload lets you drag in a whole folder's worth of images at once. "
             "Local folder path only works when you run this app on your own computer.",
    )
    uploaded_files, folder, output_folder = None, "", ""
    if source_mode == "Upload images":
        uploaded_files = st.file_uploader(
            "Drop images here (select many at once)",
            type=[e.strip(".") for e in VALID_EXT], accept_multiple_files=True)
    else:
        folder = st.text_input("Image folder path", placeholder=r"C:\path\to\images")
        output_folder = st.text_input("Output folder", placeholder="defaults to <folder>/confluency_output")

    st.divider()
    st.header("2 · Segmentation engine")
    engine_options = []
    if cp_available:
        engine_options.append("Cellpose (AI)")
    engine_options.append("Classical (fast)")
    engine = st.radio(
        "Method", engine_options,
        format_func=lambda m: "🤖 Cellpose AI — recommended, most accurate"
        if m == "Cellpose (AI)" else "⚡ Classical — fast, no AI needed",
        help="Cellpose is a pretrained deep-learning model; it segments spread, "
             "translucent cells much better and needs little tuning. Classical is "
             "instant and dependency-free but tends to undercount faint cells.")
    if not cp_available:
        st.info("Cellpose isn't installed here, so only the Classical engine is available. "
                "To enable the AI engine, install the packages in requirements.txt.")

    st.divider()
    st.header("3 · Settings")

    # Defaults are passed explicitly to each widget (value=). Widget state then
    # persists across reruns via its key. (Pre-seeding session_state before a
    # lazily-created widget is unreliable, so we don't rely on it.)
    ss = st.session_state

    if engine == "Cellpose (AI)":
        st.caption("Good defaults are set — usually you only touch **cell diameter**.")
        auto_diameter = st.checkbox("Auto-estimate cell diameter", value=False, key="auto_diameter",
                                    help="Let Cellpose estimate typical cell size. If it detects "
                                         "too few cells, turn this off and set the diameter manually.")
        diameter = st.slider("Approx. cell diameter (px, at working resolution)", 5, 60,
                             value=25, key="diameter", disabled=auto_diameter,
                             help="Typical width of one cell in the analysis image. For these 4X "
                                  "A549 scans, ~20–30 works well.")
        do_enhance = st.checkbox("Enhance faint cells (recommended)", value=True, key="do_enhance",
                                 help="Contrast-boost before segmentation. Essential for translucent "
                                      "spread cells in brightfield — without it many cells are missed.")
        with st.expander("Advanced detection tuning"):
            cellprob_threshold = st.slider("Detection sensitivity (cell-probability threshold)",
                                           -6.0, 6.0, value=-1.0, key="cellprob_threshold", step=0.5,
                                           help="Lower = detect more (including fainter) cells; "
                                                "higher = only confident detections.")
            flow_threshold = st.slider("Flow error threshold", 0.1, 3.0, value=0.6,
                                       key="flow_threshold", step=0.1,
                                       help="Higher allows less cell-like shapes through.")
        sensitivity = ss.get("sensitivity", 0.6)  # classical-only; unused here
        min_object_px = st.slider("Ignore objects smaller than (px)", 0, 300,
                                  value=15, step=5, key="min_object_px_cp",
                                  help="Removes debris / speckle. Measured at the working "
                                       "resolution, where a whole cell is only a few hundred px — "
                                       "keep this small (~15) so real cells aren't deleted.")
    else:
        sensitivity = st.slider("Sensitivity", 0.0, 1.0, value=0.6, key="sensitivity", step=0.05,
                                help="Higher detects more area (and more noise). Watch the live "
                                     "preview and pick the value where the green covers the cells "
                                     "without spilling into empty background.")
        min_object_px = st.slider("Ignore objects smaller than (px)", 0, 4000,
                                  value=400, step=50, key="min_object_px_cl",
                                  help="Removes speckle noise smaller than this.")
        # Cellpose-only params (unused by the classical engine, kept for the params dict)
        auto_diameter = ss.get("auto_diameter", False)
        diameter = ss.get("diameter", 25)
        do_enhance = ss.get("do_enhance", True)
        cellprob_threshold = ss.get("cellprob_threshold", -1.0)
        flow_threshold = ss.get("flow_threshold", 0.6)

    with st.expander("Common options"):
        do_flatten = st.checkbox("Correct uneven illumination", value=True, key="do_flatten",
                                 help="Removes vignetting / bright-center gradients before analysis.")
        drop_saturated = st.checkbox("Exclude scale bar & burnt-in text", value=True, key="drop_saturated",
                                     help="Ignores pure-white regions (scale bars, annotations) so "
                                          "they aren't counted as cells.")
        work_dim = st.select_slider("Working resolution (px, long edge)",
                                    options=[768, 1024, 1280, 1536, 2048], value=1536, key="work_dim",
                                    help="Images are analyzed at this size for speed. Higher = more "
                                         "accurate but slower (Cellpose on CPU).")

    st.divider()
    st.header("4 · Mask export")
    export_binary = st.checkbox("Binary masks (.png)", value=True)
    export_overlay = st.checkbox("Overlays (.png)", value=True)
    export_labels = st.checkbox("Instance label masks (16-bit .png)",
                                value=(engine == "Cellpose (AI)"),
                                help="Each cell gets a unique ID — useful raw data for downstream "
                                     "per-cell analysis. Cellpose only.")

    run_button = st.button("▶ Run analysis", type="primary", use_container_width=True)

params = {
    "engine": engine, "work_dim": work_dim, "diameter": diameter,
    "auto_diameter": auto_diameter, "flow_threshold": flow_threshold,
    "cellprob_threshold": cellprob_threshold, "do_flatten": do_flatten,
    "do_enhance": do_enhance, "sensitivity": sensitivity,
    "min_object_px": min_object_px, "drop_saturated": drop_saturated,
}
engine_label = ("Cellpose cyto3 (AI)" if engine == "Cellpose (AI)"
                else "Classical edge+texture")


# ---- live-preview helpers: list names cheaply, decode only the chosen image ----
def preview_names():
    if source_mode == "Upload images" and uploaded_files:
        return [uf.name for uf in uploaded_files]
    if source_mode == "Local folder path" and folder and os.path.isdir(folder):
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "*"))
                      if p.lower().endswith(VALID_EXT))
    return []


def load_preview(name):
    if source_mode == "Upload images" and uploaded_files:
        uf = next((u for u in uploaded_files if u.name == name), None)
        return load_from_bytes(uf.getvalue()) if uf else (None, None)
    if source_mode == "Local folder path" and folder:
        return load_from_path(os.path.join(folder, name))
    return None, None


names = preview_names()

# =============================================================================
# Live preview
# =============================================================================
st.subheader("🔍 Live preview")
if not names:
    st.info("Add images in the sidebar to preview segmentation before running the full batch.")
else:
    choice = st.selectbox("Preview image", names, key="preview_choice")
    gray, bgr = load_preview(choice)

    if gray is None:
        st.warning(f"Couldn't read {choice} for preview.")
        res = None
    elif engine == "Cellpose (AI)":
        st.caption("Cellpose preview runs the AI model on one image (a few seconds). "
                   "Click to refresh after changing settings.")
        do_prev = st.button("🔄 Update Cellpose preview")
        sig = str(params) + choice
        if do_prev or st.session_state.get("_prev_sig") == sig:
            with st.spinner("Segmenting preview with Cellpose..."):
                res = process(gray, bgr, params, want_labels=False)
            st.session_state["_prev_sig"] = sig
            st.session_state["_prev_res"] = res
        res = st.session_state.get("_prev_res") if st.session_state.get("_prev_sig") == sig else None
    else:
        res = process(gray, bgr, params, want_labels=False)

    if res is not None:
        c = st.columns(4)
        c[0].image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        c[1].image(res["enhanced"], caption="Preprocessed", use_container_width=True)
        c[2].image(res["mask"], caption="Mask", use_container_width=True)
        c[3].image(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB),
                   caption=f"{res['confluency']:.1f}% confluent · {res['cell_count']} cells",
                   use_container_width=True)
    elif engine == "Cellpose (AI)":
        st.info("Click **Update Cellpose preview** to see segmentation for the current settings.")

st.divider()

# =============================================================================
# Batch run
# =============================================================================
MAX_THUMBS = 9          # how many overlay thumbnails to show after a run
THUMB_MAX_DIM = 500     # px, long edge of each thumbnail


def input_specs():
    """List of (name, kind, ref) WITHOUT decoding images or reading bytes, so a
    big folder costs almost nothing until each image is processed in turn."""
    if source_mode == "Upload images":
        return [(uf.name, "upload", uf) for uf in (uploaded_files or [])]
    if source_mode == "Local folder path" and folder and os.path.isdir(folder):
        return [(os.path.basename(p), "path", p)
                for p in sorted(glob.glob(os.path.join(folder, "*")))
                if p.lower().endswith(VALID_EXT)]
    return []


def make_thumb(overlay_bgr, conf):
    h, w = overlay_bgr.shape[:2]
    sc = min(1.0, THUMB_MAX_DIM / max(h, w))
    small = cv2.resize(overlay_bgr, (max(1, int(w * sc)), max(1, int(h * sc))),
                       interpolation=cv2.INTER_AREA) if sc < 1.0 else overlay_bgr
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpg.tobytes() if ok else None


def _drop_old_results():
    """Delete the temp zip from a previous run so they don't accumulate."""
    old = st.session_state.get("results", {})
    p = old.get("zip_path")
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass


if run_button:
    specs = input_specs()
    if not specs:
        if source_mode == "Local folder path" and folder and not os.path.isdir(folder):
            st.error("That folder path doesn't exist.")
        else:
            st.warning("Add images (upload some, or point to a folder with "
                       "png/jpg/jpeg/tif/tiff/bmp) before running.")
        st.stop()

    _drop_old_results()
    total = len(specs)
    st.info(f"Processing {total} image(s) with {engine_label}...")
    progress = st.progress(0.0)
    status = st.empty()
    rows, thumbs = [], []

    # Stream results straight to a zip on disk (not RAM): the archive can grow to
    # hundreds of MB on big batches without ever being held in memory at once.
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="confluency_")
    os.close(zip_fd)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (fname, kind, ref) in enumerate(specs):
            status.caption(f"{i + 1}/{total} · {fname}")
            gray = bgr = res = None
            try:
                gray, bgr = (load_from_bytes(ref.getvalue()) if kind == "upload"
                             else load_from_path(ref))
                if gray is None:
                    rows.append({"filename": fname, "confluency_percent": None,
                                 "cell_count": None, "mean_cell_area_px": None,
                                 "status": "unreadable"})
                else:
                    want_overlay = export_overlay or len(thumbs) < MAX_THUMBS
                    res = process(gray, bgr, params, want_overlay=want_overlay,
                                  want_enhanced=False, want_labels=export_labels)
                    rows.append({
                        "filename": fname,
                        "confluency_percent": round(res["confluency"], 2),
                        "cell_count": res["cell_count"],
                        "mean_cell_area_px": round(res["mean_cell_area_px"], 1),
                        "status": "ok",
                    })
                    add_pngs_to_zip(zf, fname, res, export_binary, export_overlay, export_labels)
                    if len(thumbs) < MAX_THUMBS and res.get("overlay") is not None:
                        th = make_thumb(res["overlay"], res["confluency"])
                        if th:
                            thumbs.append((fname, th, res["confluency"]))
            except Exception as e:
                rows.append({"filename": fname, "confluency_percent": None,
                             "cell_count": None, "mean_cell_area_px": None,
                             "status": f"error: {e}"})
            finally:
                del gray, bgr, res
                if (i + 1) % 5 == 0:
                    gc.collect()
            progress.progress((i + 1) / total)

        excel_bytes = build_excel(rows, params, engine_label)
        zf.writestr("confluency_results.xlsx", excel_bytes)

    status.empty()
    gc.collect()

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # In folder mode, copy outputs next to the images so nothing has to be
    # downloaded through the browser at all.
    saved_dir = None
    if source_mode == "Local folder path" and n_ok > 0:
        try:
            out_dir = output_folder.strip() or os.path.join(folder, "confluency_output")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"confluency_results_{stamp}.xlsx"), "wb") as f:
                f.write(excel_bytes)
            shutil.copyfile(zip_path, os.path.join(out_dir, f"confluency_masks_{stamp}.zip"))
            saved_dir = out_dir
        except OSError as e:
            st.warning(f"Couldn't write to the output folder: {e}")

    st.session_state["results"] = {
        "rows": rows, "stamp": stamp, "excel_bytes": excel_bytes,
        "zip_path": zip_path, "zip_size": os.path.getsize(zip_path),
        "thumbs": thumbs, "engine_label": engine_label, "n_ok": n_ok,
        "saved_dir": saved_dir,
    }


# =============================================================================
# Results (rendered from session_state so downloads survive Streamlit reruns)
# =============================================================================
R = st.session_state.get("results")
if R:
    rows = R["rows"]
    df = pd.DataFrame(rows)
    conf = pd.to_numeric(df["confluency_percent"], errors="coerce").dropna()

    if R["n_ok"] == 0:
        st.error("No images could be processed. See details below.")
        st.dataframe(df, use_container_width=True)
    else:
        st.success(f"Done — {R['n_ok']}/{len(rows)} images processed with "
                   f"{R['engine_label']}. Mean confluency {conf.mean():.1f}% "
                   f"(range {conf.min():.1f}–{conf.max():.1f}%).")

        m = st.columns(4)
        m[0].metric("Images", R["n_ok"])
        m[1].metric("Mean confluency", f"{conf.mean():.1f}%")
        m[2].metric("Median", f"{conf.median():.1f}%")
        m[3].metric("Std dev", f"{conf.std():.1f}%" if len(conf) > 1 else "—")

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        stamp = R["stamp"]
        dl = st.columns(2)
        dl[0].download_button(
            "⬇ Download Excel results", R["excel_bytes"],
            file_name=f"confluency_results_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

        zip_path = R.get("zip_path")
        if zip_path and os.path.exists(zip_path):
            size_mb = R["zip_size"] / 1e6
            with open(zip_path, "rb") as f:
                dl[1].download_button(
                    f"⬇ Download masks + overlays (.zip · {size_mb:.0f} MB)", f.read(),
                    file_name=f"confluency_masks_{stamp}.zip",
                    mime="application/zip", use_container_width=True)
        else:
            dl[1].info("Mask ZIP no longer available — re-run to regenerate.")

        if R.get("saved_dir"):
            st.caption(f"Also saved to: {R['saved_dir']}")

        if R["thumbs"]:
            st.subheader("Overlays (green = detected cells, red = outlines)")
            shown = len(R["thumbs"])
            if R["n_ok"] > shown:
                st.caption(f"Showing the first {shown} of {R['n_ok']} — all overlays "
                           "are in the ZIP.")
            per_row = 3
            for i in range(0, len(R["thumbs"]), per_row):
                cols = st.columns(per_row)
                for col, (fn, jpg, cf) in zip(cols, R["thumbs"][i:i + per_row]):
                    arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    col.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
                              caption=f"{fn} — {cf:.1f}%", use_container_width=True)
else:
    st.info("Set up your images and settings in the sidebar, then click **Run analysis**.")
