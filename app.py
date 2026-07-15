import io
import os
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

def correct_background(gray, kernel_frac=0.05, cells_darker=True):
    """Remove uneven illumination using a large-kernel morphological
    opening/closing (estimated background), then subtract it so cells always
    end up as BRIGHT blobs on a near-zero background, regardless of whether
    cells are naturally darker or brighter than the background.
    kernel_frac scales the kernel size to image size so it works across
    resolutions."""
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
    return corrected.astype(np.uint8), background


def enhance_contrast(gray, clip_limit=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def segment_cells(gray, min_object_px=50, close_kernel=5, use_adaptive=False,
                   adaptive_block=51, adaptive_c=2):
    """Threshold + clean up morphology. Input `gray` is assumed to already be
    background-corrected such that cells are BRIGHT on a near-zero
    background (see correct_background). Returns binary mask (0/255) where
    255 = cell/foreground."""
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

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_object_px:
            clean_mask[labels == i] = 255

    return clean_mask


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

    corrected, _ = correct_background(gray, kernel_frac=params["bg_kernel_frac"], cells_darker=params["invert"])
    enhanced = enhance_contrast(corrected, clip_limit=params["clahe_clip"])

    mask = segment_cells(
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
    use_adaptive = st.checkbox("Use adaptive thresholding instead of Otsu", value=False,
                                help="Otsu (automatic global threshold) works well for evenly lit images. Switch to adaptive if lighting varies a lot across the image.")
    adaptive_block = st.slider("Adaptive block size (odd)", 11, 151, 51, step=2, disabled=not use_adaptive)
    adaptive_c = st.slider("Adaptive C offset", -10, 10, 2, disabled=not use_adaptive)

    invert = st.checkbox("Cells are darker than background", value=True,
                          help="Typical for brightfield: cells often appear darker than the empty well/background. Uncheck if cells appear brighter (e.g. some fluorescence images).")

    bg_kernel_frac = st.slider("Background correction strength", 0.01, 0.30, 0.08, step=0.01,
                                help="Fraction of image size used for the background-estimation kernel. Should be larger than your biggest confluent cell patch — increase if large cell clusters are missed, decrease if fine detail is lost.")
    clahe_clip = st.slider("Contrast enhancement (CLAHE clip limit)", 0.5, 5.0, 2.0, step=0.5)

    morph_kernel = st.slider("Morphology kernel size (odd)", 3, 25, 5, step=2)
    min_object_px = st.slider("Minimum object size (pixels)", 0, 2000, 50, step=10,
                               help="Removes speckle noise smaller than this many pixels.")

    run_button = st.button("Run analysis", type="primary")

params = {
    "bg_kernel_frac": bg_kernel_frac,
    "clahe_clip": clahe_clip,
    "min_object_px": min_object_px,
    "morph_kernel": morph_kernel,
    "invert": invert,
    "use_adaptive": use_adaptive,
    "adaptive_block": adaptive_block,
    "adaptive_c": adaptive_c,
}


def render_results(results, preview_slots, excel_bytes, excel_name, zip_bytes=None, zip_name=None,
                    masks_dir=None, overlays_dir=None, excel_path=None):
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

    if preview_slots:
        st.subheader("Preview (overlay = red contour + green fill on detected cell area)")
        cols_per_row = 3
        for i in range(0, len(preview_slots), cols_per_row):
            row = preview_slots[i:i + cols_per_row]
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
                    preview_slots.append((fname, res["overlay"], res["confluency"]))

                except Exception as e:
                    results.append({"filename": fname, "confluency_percent": None, "status": f"error: {e}"})

                progress.progress((idx + 1) / len(uploaded_files))

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
                preview_slots.append((fname, res["overlay"], res["confluency"]))

            except Exception as e:
                results.append({"filename": fname, "confluency_percent": None, "status": f"error: {e}"})

            progress.progress((idx + 1) / len(image_paths))

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
