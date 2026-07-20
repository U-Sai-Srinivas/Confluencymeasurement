import io
import os
import gc
import glob
import zipfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cell Confluency Estimator", layout="wide")

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ----------------------------- Core image processing (pure, no I/O) -----------------------------

def correct_background_morph(gray, kernel_frac=0.05, cells_darker=True):
    """Remove uneven illumination using a large-kernel morphological
    opening/closing (estimated background), then subtract it so cells always
    end up as BRIGHT blobs on a near-zero background, regardless of whether
    cells are naturally darker or brighter than the background.
    kernel_frac scales the kernel size to image size so it works across
    resolutions. Works well when the shading is fairly local/patchy."""
    h, w = gray.shape
    k = max(15, int(min(h, w) * kernel_frac))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if cells_darker:
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        corrected = cv2.subtract(background, gray)
    else:
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        corrected = cv2.subtract(gray, background)

    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def correct_background_flatfield(gray, sigma_frac=0.20, cells_darker=True):
    """Removes broad, whole-image illumination gradients (vignetting, uneven
    light source) via large-sigma Gaussian flat-field division: estimate a
    smooth background with a very wide blur, then divide it out. This
    handles gradients spanning most/all of the image far better than
    morphological opening/closing, which is limited by its kernel size.
    Better choice for sparse cultures where the shading pattern is broader
    than any individual cell cluster."""
    h, w = gray.shape
    sigma = max(10.0, min(h, w) * sigma_frac)
    bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma)
    flat = (gray.astype(np.float32) / (bg + 1e-3)) * bg.mean()
    flat = np.clip(flat, 0, 255).astype(np.uint8)
    if not cells_darker:
        flat = 255 - flat
    corrected = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8)


def correct_background(gray, kernel_frac=0.05, cells_darker=True, method="morphological"):
    if method == "flatfield":
        return correct_background_flatfield(gray, sigma_frac=kernel_frac, cells_darker=cells_darker)
    return correct_background_morph(gray, kernel_frac=kernel_frac, cells_darker=cells_darker)


def denoise(gray, strength=0):
    """Edge-preserving denoise to suppress fine sensor grain/JPEG texture
    before contrast enhancement, without blurring away real cell edges the
    way a plain Gaussian/median blur of the same strength would. strength=0
    skips this step entirely."""
    if strength <= 0:
        return gray
    d = 5 + 2 * strength
    return cv2.bilateralFilter(gray, d=d, sigmaColor=10 * strength, sigmaSpace=d)


def enhance_contrast(gray, clip_limit=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)




def segment_cells_threshold(gray, min_object_px=50, close_kernel=5, use_adaptive=False,
                             adaptive_block=51, adaptive_c=2):
    """Original approach: threshold pixel INTENSITY after background
    correction. Works well when cells form solid, evenly-lit patches.
    Struggles on sparse, grainy, or unevenly-illuminated phase-contrast
    images because the threshold is still sensitive to residual shading and
    sensor noise. Returns binary mask (0/255) where 255 = cell/foreground."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if use_adaptive:
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, adaptive_block, adaptive_c
        )
    else:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return _remove_small_objects(mask, min_object_px)


def segment_cells_local_contrast(gray, window=15, min_object_px=20):
    """PHANTAST-style local contrast thresholding (Jaccard et al. 2014,
    Biotechnol. Bioeng.): phase-contrast cells are textured (edges, halos,
    internal structure) while empty background is smooth, REGARDLESS of
    the image's absolute brightness at that point. So instead of
    thresholding pixel intensity (which a broad illumination gradient or
    vignette corrupts), this thresholds local intensity variance — flat
    regions (background, including any residual shading gradient) score
    low, textured regions (cells) score high. This is naturally robust to
    whole-image illumination gradients that defeat intensity-based
    thresholding, and is dramatically faster since it needs no large-kernel
    background estimation step at all. Best choice for sparse, low-contrast
    phase-contrast/brightfield cultures."""
    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window, window))
    sqmean = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(window, window))
    local_std = np.sqrt(np.clip(sqmean - mean * mean, 0, None))
    std_u8 = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(std_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _remove_small_objects(mask, min_object_px), std_u8


def _remove_small_objects(mask, min_object_px):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_object_px
    keep[0] = False  # background label is never kept
    return (keep[labels] * 255).astype(np.uint8)


def compute_confluency(mask):
    foreground = np.count_nonzero(mask)
    total = mask.size
    return 100.0 * foreground / total


def make_overlay(original_bgr, mask, color=(0, 255, 0), alpha=0.4):
    overlay = original_bgr.copy()
    colored = np.zeros_like(original_bgr)
    colored[mask > 0] = color
    blended = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 1)
    return blended


def process_image_array(raw, params):
    """Runs the full pipeline on an already-decoded BGR image array."""
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    if params.get("segmentation_approach", "threshold") == "local_contrast":
        # Local contrast is already illumination-invariant, so background
        # correction and CLAHE aren't needed for segmentation itself — but
        # we still compute a corrected+enhanced view for display consistency.
        mask, enhanced = segment_cells_local_contrast(
            gray, window=params.get("contrast_window", 15), min_object_px=params["min_object_px"]
        )
    else:
        corrected = correct_background(gray, kernel_frac=params["bg_kernel_frac"], cells_darker=params["invert"],
                                        method=params.get("bg_method", "morphological"))
        denoised = denoise(corrected, strength=params.get("denoise_strength", 0))
        enhanced = enhance_contrast(denoised, clip_limit=params["clahe_clip"])
        mask = segment_cells_threshold(
            enhanced,
            min_object_px=params["min_object_px"],
            close_kernel=params["morph_kernel"],
            use_adaptive=params["use_adaptive"],
            adaptive_block=params["adaptive_block"],
            adaptive_c=params["adaptive_c"],
        )

    confluency = compute_confluency(mask)
    overlay = make_overlay(raw, mask)

    return {"raw": raw, "gray": gray, "enhanced": enhanced, "mask": mask, "overlay": overlay, "confluency": confluency}


def process_image_from_path(path, params):
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw is None:
        return None
    return process_image_array(raw, params)


def process_image_from_bytes(file_bytes, params):
    file_array = np.frombuffer(file_bytes, np.uint8)
    raw = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if raw is None:
        return None
    return process_image_array(raw, params)


def score_segmentation(gray_enhanced, mask):
    """Unsupervised quality proxy — there's no ground truth to compare against,
    so this rewards a segmentation that (a) isn't degenerate (~0% or ~100%
    foreground), (b) isn't fragmented into lots of tiny speckle objects, and
    (c) has mask boundaries that sit on genuinely high-contrast edges in the
    image rather than falling on flat, arbitrary regions. Higher is better."""
    total = mask.size
    fg = np.count_nonzero(mask)
    conf = 100.0 * fg / total

    degeneracy_penalty = 0.0
    if conf < 1 or conf > 99:
        degeneracy_penalty = 60
    elif conf < 3 or conf > 97:
        degeneracy_penalty = 20

    n_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    n_components = max(0, n_labels - 1)
    frag_penalty = min((n_components / max(fg, 1)) * 1000.0 * 3, 40)

    edges = cv2.Canny(mask, 50, 150)
    gx = cv2.Sobel(gray_enhanced, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_enhanced, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(gx, gy)
    boundary_contrast = float(grad_mag[edges > 0].mean()) if np.count_nonzero(edges) > 0 else 0.0

    score = boundary_contrast - degeneracy_penalty - frag_penalty
    return score, {"confluency": round(conf, 1), "components": n_components, "boundary_contrast": round(boundary_contrast, 1)}


@st.cache_data(show_spinner=False)
def _decode_bytes(file_bytes):
    return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def _decode_path(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def get_sample_images(source_mode, uploaded_files, folder, max_n=3):
    """Returns up to max_n (name, raw_bgr_array) pairs for preview/auto-tune, without touching disk output."""
    samples = []
    if source_mode == "Upload images" and uploaded_files:
        for uf in uploaded_files[:max_n]:
            raw = _decode_bytes(uf.getvalue())
            if raw is not None:
                samples.append((uf.name, raw))
    elif source_mode == "Local folder path" and folder and os.path.isdir(folder):
        paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(VALID_EXT))
        for p in paths[:max_n]:
            raw = _decode_path(p)
            if raw is not None:
                samples.append((os.path.basename(p), raw))
    return samples


AUTO_TUNE_MAX_DIM = 480  # px — search runs on a downsampled copy so it stays fast regardless of source resolution


def _downsample_for_tuning(raw, max_dim=AUTO_TUNE_MAX_DIM):
    h, w = raw.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return raw
    return cv2.resize(raw, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def auto_suggest_params(sample_images, base_params, progress_cb=None):
    """Small grid search scoped to whichever segmentation approach is
    currently selected (base_params['segmentation_approach']), scored with
    score_segmentation() and averaged across sample images.

    Deliberately does NOT search across both segmentation approaches: the
    unsupervised score can't reliably tell "many small regions = real sparse
    cells" apart from "many small regions = noise fragments", so letting it
    pick between fundamentally different approaches can silently overrule a
    correct manual choice (e.g. local contrast for a sparse phase-contrast
    culture) in favor of one that merely LOOKS more confident by this proxy.
    Choosing the approach itself is left to the person, informed by the
    Live Preview; auto-tune only refines the knobs within it.

    Keeps 'invert' and 'use_adaptive' as the user set them, since those
    reflect known facts about the imaging modality, not something to guess
    blindly. Runs entirely on downsampled copies (<= AUTO_TUNE_MAX_DIM per
    side) so it stays fast even on full-resolution microscopy scans.
    Size-dependent parameters are searched as fractions of image size, then
    translated back to absolute pixel sizes for the caller's actual
    (full-resolution) images.
    """
    ref_h, ref_w = sample_images[0][1].shape[:2]
    ref_area = ref_h * ref_w
    approach = base_params.get("segmentation_approach", "threshold")

    if approach == "local_contrast":
        # Local contrast is fast even at full resolution (no large-kernel
        # morphology involved), so search directly on the real images —
        # this avoids a subtle bug where a small-window floor clamped
        # during downsampled search doesn't correspond to the same
        # effective window once translated back to full resolution.
        search_images = sample_images
        # Kept deliberately small — a heuristic score with no ground truth
        # will happily drift toward large windows/thresholds that merge
        # everything into a few big blobs (higher average boundary
        # contrast, fewer "fragments"), which defeats the point of this
        # method for sparse cultures. Capping the search keeps results in
        # single-cell-scale territory; use the Live Preview to go further.
        contrast_window_fracs = [0.003, 0.006, 0.010, 0.016]
        min_object_fracs = [0.000005, 0.00001, 0.00003, 0.00006]
        combos = [(cwf, mof) for cwf in contrast_window_fracs for mof in min_object_fracs]
    else:
        search_images = [(name, _downsample_for_tuning(raw)) for name, raw in sample_images]
        bg_methods = ["morphological", "flatfield"]
        bg_kernel_fracs = [0.08, 0.15, 0.25, 0.35]
        denoise_strengths = [0, 2]
        clahe_clips = [1.5, 2.0, 3.0]
        morph_kernel_fracs = [0.010, 0.020, 0.035]
        combos = [(bgm, bgf, dn, clip, mkf)
                  for bgm in bg_methods for bgf in bg_kernel_fracs for dn in denoise_strengths
                  for clip in clahe_clips for mkf in morph_kernel_fracs]

    best_score, best_combo_frac, best_info = None, None, None

    for i, combo in enumerate(combos):
        scores, infos = [], []
        for _, img_for_search in search_images:
            h, w = img_for_search.shape[:2]
            if approach == "local_contrast":
                cwf, mof = combo
                win = max(5, int(round(cwf * min(h, w))) | 1)
                min_px = max(3, int(round(mof * h * w)))
                trial_params = dict(base_params, segmentation_approach="local_contrast",
                                     contrast_window=win, min_object_px=min_px)
            else:
                bgm, bgf, dn, clip, mkf = combo
                mk_px = max(3, int(round(mkf * min(h, w))) | 1)
                trial_params = dict(base_params, segmentation_approach="threshold",
                                     bg_method=bgm, bg_kernel_frac=bgf, denoise_strength=dn,
                                     clahe_clip=clip, morph_kernel=mk_px)
            res = process_image_array(img_for_search, trial_params)
            s, info = score_segmentation(res["gray"], res["mask"])
            scores.append(s)
            infos.append(info)
        avg_score = sum(scores) / len(scores)
        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_combo_frac = combo
            best_info = infos[0]
        if progress_cb:
            progress_cb((i + 1) / len(combos), i + 1, len(combos))

    if approach == "local_contrast":
        cwf, mof = best_combo_frac
        best_combo = {
            "segmentation_approach": "local_contrast",
            "contrast_window": max(5, int(round(cwf * min(ref_h, ref_w))) | 1),
            "min_object_px": max(3, int(round(mof * ref_area))),
        }
    else:
        bgm, bgf, dn, clip, mkf = best_combo_frac
        best_combo = {
            "segmentation_approach": "threshold",
            "bg_method": bgm,
            "bg_kernel_frac": bgf,
            "denoise_strength": dn,
            "clahe_clip": clip,
            "morph_kernel": min(max(3, int(round(mkf * min(ref_h, ref_w))) | 1), 25),
        }
    return best_combo, best_score, best_info


# ----------------------------- Streamlit UI -----------------------------

st.title("🔬 Cell Confluency Estimator")
st.caption(
    "Analyze brightfield/phase-contrast microscopy images. "
    "Each image is background-corrected, contrast-enhanced, segmented, and scored for % confluency."
)

with st.sidebar:
    st.header("Image Source")
    source_mode = st.radio(
        "How are you providing images?",
        ["Upload images", "Local folder path"],
        help="Use 'Upload images' on Streamlit Cloud — the server can't see your computer's files. "
             "'Local folder path' only works when you run this app on your own machine.",
    )

    uploaded_files = None
    folder = ""
    output_folder = ""

    if source_mode == "Upload images":
        uploaded_files = st.file_uploader(
            "Upload microscopy images",
            type=[e.strip(".") for e in VALID_EXT],
            accept_multiple_files=True,
        )
    else:
        folder = st.text_input("Image folder path", value="", placeholder=r"C:\path\to\images")
        output_folder = st.text_input("Output folder (masks/overlays/Excel)", value="",
                                       placeholder="defaults to <folder>/confluency_output")

    st.divider()
    st.subheader("Segmentation")

    DEFAULTS = {
        "segmentation_approach": "local_contrast", "contrast_window": 15,
        "use_adaptive": False, "adaptive_block": 51, "adaptive_c": 2, "invert": True,
        "bg_method": "morphological", "bg_kernel_frac": 0.08, "denoise_strength": 0,
        "clahe_clip": 2.0, "morph_kernel": 5, "min_object_px": 30,
    }
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

    # Apply any auto-suggested values BEFORE the widgets below are instantiated —
    # Streamlit forbids writing to session_state for a key after its widget exists.
    if "_apply_suggestion" in st.session_state:
        for k, v in st.session_state.pop("_apply_suggestion").items():
            st.session_state[k] = v

    segmentation_approach = st.radio(
        "Segmentation approach", ["local_contrast", "threshold"], key="segmentation_approach",
        format_func=lambda m: "Local contrast (recommended — phase-contrast/brightfield)" if m == "local_contrast" else "Intensity threshold (confluent, evenly-lit patches)",
        help="Local contrast: separates cells from background by local TEXTURE rather than brightness — naturally ignores illumination gradients and vignetting. Best default for phase-contrast microscopy. Intensity threshold: the original approach, better suited to solid, evenly-lit confluent patches (e.g. some fluorescence images).",
    )

    if segmentation_approach == "local_contrast":
        contrast_window = st.slider("Contrast window size (px)", 5, 51, step=2, key="contrast_window",
                                     help="Size of the neighborhood used to measure local texture. Should be roughly the width of a single cell — too small picks up pixel noise as texture, too large blurs separate cells together.")
        min_object_px = st.slider("Minimum object size (pixels)", 0, 500, step=5, key="min_object_px",
                                   help="Removes speckle noise smaller than this many pixels.")
        # Keep these defined (used only by the threshold branch / auto-tune) so downstream code is uniform
        use_adaptive, adaptive_block, adaptive_c = False, 51, 2
        invert = st.session_state["invert"]
        bg_method, bg_kernel_frac, denoise_strength, clahe_clip, morph_kernel = "morphological", 0.08, 0, 2.0, 5
    else:
        use_adaptive = st.checkbox("Use adaptive thresholding instead of Otsu", key="use_adaptive",
                                    help="Otsu (automatic global threshold) works well for evenly lit images. Switch to adaptive if lighting varies a lot across the image.")
        adaptive_block = st.slider("Adaptive block size (odd)", 11, 151, step=2, disabled=not use_adaptive, key="adaptive_block")
        adaptive_c = st.slider("Adaptive C offset", -10, 10, disabled=not use_adaptive, key="adaptive_c")

        invert = st.checkbox("Cells are darker than background", key="invert",
                              help="Typical for brightfield: cells often appear darker than the empty well/background. Uncheck if cells appear brighter (e.g. some fluorescence images).")

        bg_method = st.radio(
            "Background correction method", ["morphological", "flatfield"], key="bg_method",
            format_func=lambda m: "Morphological (confluent/patchy cultures)" if m == "morphological" else "Flat-field (uneven illumination, sparse cultures)",
            help="Morphological: good when cells form solid patches. Flat-field: use this when there's a broad brightness gradient across the whole image (vignetting, off-center light) — very common with sparse, low-density cultures.",
        )
        bg_kernel_frac = st.slider("Background correction strength", 0.01, 0.50, step=0.01, key="bg_kernel_frac",
                                    help="For morphological: fraction of image size used for the background-estimation kernel — should exceed your biggest confluent cell patch. For flat-field: how wide the smoothing is — increase this if the background still looks unevenly lit after correction (try 0.20–0.35 for whole-image gradients).")
        denoise_strength = st.slider("Denoise (before contrast enhancement)", 0, 5, step=1, key="denoise_strength",
                                      help="Edge-preserving smoothing to suppress camera/JPEG grain before contrast enhancement amplifies it into false detections. Start at 0; increase if the mask looks speckled with tiny noise-sized objects rather than real cell shapes.")
        clahe_clip = st.slider("Contrast enhancement (CLAHE clip limit)", 0.5, 5.0, step=0.5, key="clahe_clip")

        morph_kernel = st.slider("Morphology kernel size (odd)", 3, 25, step=2, key="morph_kernel")
        min_object_px = st.slider("Minimum object size (pixels)", 0, 2000, step=10, key="min_object_px",
                                   help="Removes speckle noise smaller than this many pixels.")

    st.divider()
    auto_tune_clicked = st.button("🪄 Auto-suggest settings", use_container_width=True,
                                   help="Tries a range of background-correction, contrast, and morphology settings on a few of your images and picks the combination that looks cleanest by an automated quality check. Always confirm with the live preview below — it can't know what's biologically correct, only what looks well-segmented.")

    run_button = st.button("Run analysis", type="primary", use_container_width=True)

params = {
    "segmentation_approach": segmentation_approach,
    "contrast_window": st.session_state.get("contrast_window", 15),
    "bg_method": bg_method,
    "bg_kernel_frac": bg_kernel_frac,
    "denoise_strength": denoise_strength,
    "clahe_clip": clahe_clip,
    "min_object_px": min_object_px,
    "morph_kernel": morph_kernel,
    "invert": invert,
    "use_adaptive": use_adaptive,
    "adaptive_block": adaptive_block,
    "adaptive_c": adaptive_c,
}

sample_images = get_sample_images(source_mode, uploaded_files, folder, max_n=3)

if auto_tune_clicked:
    if not sample_images:
        st.warning("Add at least one image first (upload one, or point to a folder) so there's something to tune on.")
    else:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _report(frac, done, total):
            progress_bar.progress(frac)
            status_text.caption(f"Testing combination {done}/{total} (on a downsampled copy for speed)...")

        best_combo, best_score, best_info = auto_suggest_params(sample_images, params, progress_cb=_report)
        progress_bar.empty()
        status_text.empty()
        st.session_state["_apply_suggestion"] = best_combo
        st.session_state["_auto_tune_info"] = best_info
        st.rerun()

if st.session_state.get("_auto_tune_info"):
    info = st.session_state.pop("_auto_tune_info")
    st.success(
        f"Applied suggested settings — resulting sample confluency ≈ {info['confluency']}%, "
        f"{info['components']} detected object(s). Check the live preview below and adjust further if needed."
    )

# ----------------------------- Live preview (updates as you move sliders, no need to hit Run) -----------------------------
st.subheader("🔍 Live Preview")
if not sample_images:
    st.info("Upload an image (or set a valid local folder) to see a live preview of segmentation as you adjust settings.")
else:
    preview_names = [name for name, _ in sample_images]
    preview_choice = st.selectbox("Preview image", preview_names, key="preview_choice")
    preview_raw = dict(sample_images)[preview_choice]

    try:
        preview_res = process_image_array(preview_raw, params)
        p_cols = st.columns(4)
        with p_cols[0]:
            st.image(cv2.cvtColor(preview_raw, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with p_cols[1]:
            st.image(preview_res["enhanced"], caption="Corrected + enhanced", use_container_width=True)
        with p_cols[2]:
            st.image(preview_res["mask"], caption="Segmentation mask", use_container_width=True)
        with p_cols[3]:
            st.image(cv2.cvtColor(preview_res["overlay"], cv2.COLOR_BGR2RGB),
                      caption=f"Overlay — {preview_res['confluency']:.1f}% confluent", use_container_width=True)
    except Exception as e:
        st.warning(f"Couldn't render a preview with the current settings: {e}")

st.divider()


def make_thumbnail(bgr_image, max_dim=400):
    """Small copy for on-screen preview grids. Keeping full-resolution
    overlay arrays for every image in a large batch is what actually blows
    past Streamlit Cloud's free-tier memory limit — 40 images at ~4K
    resolution is roughly 1GB just for the preview overlays alone. The full
    resolution version still goes into the downloadable zip; only the
    on-screen preview is downsized."""
    h, w = bgr_image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return bgr_image.copy()
    return cv2.resize(bgr_image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def render_results(results, preview_slots, excel_bytes, excel_name, zip_bytes=None, zip_name=None,
                    masks_dir=None, overlays_dir=None, excel_path=None, total_processed=None):
    df = pd.DataFrame(results)
    st.success(f"Done. Processed {len(results)} image(s).")
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    dl_cols = st.columns(2) if zip_bytes is not None else st.columns(1)
    with dl_cols[0]:
        st.download_button("⬇️ Download Excel results", excel_bytes, file_name=excel_name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if zip_bytes is not None:
        with dl_cols[1]:
            st.download_button("⬇️ Download masks + overlays (.zip)", zip_bytes, file_name=zip_name, mime="application/zip")

    if masks_dir and overlays_dir and excel_path:
        st.caption(f"Masks saved to: {masks_dir}")
        st.caption(f"Overlays saved to: {overlays_dir}")
        st.caption(f"Excel saved to: {excel_path}")

    MAX_PREVIEWS = 24
    if preview_slots:
        shown = preview_slots[:MAX_PREVIEWS]
        title = "Preview (overlay = red contour + green fill on detected cell area)"
        if len(preview_slots) > MAX_PREVIEWS:
            title += f" — showing first {MAX_PREVIEWS} of {len(preview_slots)}; all images are in the downloads above"
        st.subheader(title)
        cols_per_row = 3
        for i in range(0, len(shown), cols_per_row):
            row = shown[i:i + cols_per_row]
            cols = st.columns(len(row))
            for col, (fname, overlay, conf) in zip(cols, row):
                with col:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"{fname} — {conf:.1f}%", use_container_width=True)


if run_button:
    # ---------------- Upload mode: everything stays in memory, zipped for download ----------------
    if source_mode == "Upload images":
        if not uploaded_files:
            st.error("Please upload at least one image.")
            st.stop()

        st.info(f"Processing {len(uploaded_files)} image(s)...")
        progress = st.progress(0)
        results, preview_slots = [], []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, uf in enumerate(uploaded_files):
                fname = uf.name
                try:
                    res = process_image_from_bytes(uf.getvalue(), params)
                    if res is None:
                        results.append({"filename": fname, "confluency_percent": None, "status": "unreadable image"})
                        progress.progress((idx + 1) / len(uploaded_files))
                        continue

                    ok_mask, mask_png = cv2.imencode(".png", res["mask"])
                    ok_overlay, overlay_png = cv2.imencode(".png", res["overlay"])
                    if ok_mask:
                        zf.writestr(f"masks/mask_{fname}.png", mask_png.tobytes())
                    if ok_overlay:
                        zf.writestr(f"overlays/overlay_{fname}.png", overlay_png.tobytes())

                    results.append({
                        "filename": fname,
                        "confluency_percent": round(res["confluency"], 2),
                        "status": "ok",
                    })
                    preview_slots.append((fname, make_thumbnail(res["overlay"]), res["confluency"]))
                    del res  # don't hold the full-res arrays past this iteration

                except Exception as e:
                    results.append({"filename": fname, "confluency_percent": None, "status": f"error: {e}"})

                progress.progress((idx + 1) / len(uploaded_files))
                if (idx + 1) % 10 == 0:
                    gc.collect()  # nudge cleanup along for large batches instead of waiting for it to accumulate

        if not any(r["status"] == "ok" for r in results):
            st.warning("No images could be processed.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.stop()

        df = pd.DataFrame(results)
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        render_results(
            results, preview_slots,
            excel_bytes=excel_buffer, excel_name=f"confluency_results_{stamp}.xlsx",
            zip_bytes=zip_buffer.getvalue(), zip_name=f"confluency_masks_overlays_{stamp}.zip",
        )

    # ---------------- Local folder mode: original disk-based behavior ----------------
    else:
        if not folder or not os.path.isdir(folder):
            st.error("Please provide a valid, existing image folder path.")
            st.stop()

        out_dir = output_folder.strip() or os.path.join(folder, "confluency_output")
        masks_dir = os.path.join(out_dir, "masks")
        overlays_dir = os.path.join(out_dir, "overlays")
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

        image_paths = sorted(
            p for p in glob.glob(os.path.join(folder, "*"))
            if p.lower().endswith(VALID_EXT)
        )

        if not image_paths:
            st.warning("No images found in that folder (looked for png/jpg/jpeg/tif/tiff/bmp).")
            st.stop()

        st.info(f"Found {len(image_paths)} images. Processing...")
        progress = st.progress(0)
        results, preview_slots = [], []

        for idx, path in enumerate(image_paths):
            fname = os.path.basename(path)
            try:
                res = process_image_from_path(path, params)
                if res is None:
                    results.append({"filename": fname, "confluency_percent": None, "status": "unreadable image"})
                    progress.progress((idx + 1) / len(image_paths))
                    continue

                mask_path = os.path.join(masks_dir, f"mask_{fname}")
                overlay_path = os.path.join(overlays_dir, f"overlay_{fname}")
                cv2.imwrite(mask_path, res["mask"])
                cv2.imwrite(overlay_path, res["overlay"])

                results.append({
                    "filename": fname,
                    "confluency_percent": round(res["confluency"], 2),
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "status": "ok",
                })
                preview_slots.append((fname, make_thumbnail(res["overlay"]), res["confluency"]))
                del res  # don't hold the full-res arrays past this iteration

            except Exception as e:
                results.append({"filename": fname, "confluency_percent": None, "status": f"error: {e}"})

            progress.progress((idx + 1) / len(image_paths))
            if (idx + 1) % 10 == 0:
                gc.collect()

        df = pd.DataFrame(results)
        excel_path = os.path.join(out_dir, f"confluency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(excel_path, index=False)
        with open(excel_path, "rb") as f:
            excel_bytes = f.read()

        render_results(
            results, preview_slots,
            excel_bytes=excel_bytes, excel_name=os.path.basename(excel_path),
            masks_dir=masks_dir, overlays_dir=overlays_dir, excel_path=excel_path,
        )
else:
    if source_mode == "Upload images":
        st.info("Upload one or more images in the sidebar and click **Run analysis** to begin.")
    else:
        st.info("Set an image folder in the sidebar and click **Run analysis** to begin.")
