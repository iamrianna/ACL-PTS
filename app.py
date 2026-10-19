"""
PTS Workstation - web-based PACS-style tool for FULLY AUTOMATED Posterior Tibial
Slope (PTS) measurement on sagittal knee MRI (e.g. 1.5T proton-density, slice 16/27).

Pipeline (no manual landmarks):
  load (.pck/.pkl/.npy/.jpg/.png) -> tibia segmentation -> shaft axis + plateau line
  -> PTS -> bootstrap/ensemble 95% CI -> QC flag -> risk tier (Low/High)
  -> biomechanics (anterior shear, ACL load increase) -> ACL rupture module
  -> validation against reference values (optional) -> watermarked exports.

Made by Rianna-Maria Tanase
"""
import io
import os
import pickle
import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.stats import theilslopes, pearsonr
from skimage import filters, measure, morphology, transform

WATERMARK = "Made by Rianna-Maria Tanase"
SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)  # threshold scales used for the ensemble


# ----------------------------------------------------------------------------
# 1. LOADING (.pck is loaded with a RESTRICTED unpickler: arrays only)
# ----------------------------------------------------------------------------
class _SafeUnpickler(pickle.Unpickler):
    _np_mods = {"numpy", "numpy.core.multiarray", "numpy._core.multiarray",
                "numpy.core.numeric", "numpy._core.numeric"}
    _np_names = {"_reconstruct", "ndarray", "dtype", "scalar", "_frombuffer"}
    _builtins = {"dict", "list", "tuple", "set", "frozenset", "bytearray",
                 "complex", "slice", "range"}

    def find_class(self, module, name):
        if module in self._np_mods and name in self._np_names:
            return super().find_class(module, name)
        if module == "builtins" and name in self._builtins:
            return super().find_class(module, name)
        if module == "collections" and name == "OrderedDict":
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Blocked unsafe object: {module}.{name}")


def _extract_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for k in ("volume", "image", "pixel_array", "data", "array", "img"):
            if k in obj:
                return _extract_array(obj[k])
        for v in obj.values():
            try:
                return _extract_array(v)
            except ValueError:
                continue
    if isinstance(obj, (list, tuple)) and obj:
        arrs = [np.asarray(_extract_array(o)) for o in obj]
        return np.stack(arrs)
    raise ValueError("No image array found in file.")


def to_volume(arr, slice_axis="auto"):
    """Return float32 array (N, H, W)."""
    a = np.asarray(arr)
    if a.ndim == 4:                       # (N,H,W,C) -> gray
        a = a.mean(-1)
    if a.ndim == 3 and a.shape[-1] in (3, 4) and a.shape[0] > 4:  # RGB(A) image
        a = a[..., :3].mean(-1)[None]
    elif a.ndim == 3:
        ax = int(np.argmin(a.shape)) if slice_axis == "auto" else int(slice_axis)
        a = np.moveaxis(a, ax, 0)
    elif a.ndim == 2:
        a = a[None]
    else:
        raise ValueError(f"Unsupported array shape {a.shape}")
    return a.astype(np.float32)


def load_file(uploaded, slice_axis="auto"):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith((".pck", ".pkl", ".pickle")):
        arr = _extract_array(_SafeUnpickler(io.BytesIO(data)).load())
    elif name.endswith(".npy"):
        arr = np.load(io.BytesIO(data), allow_pickle=False)
    else:
        arr = np.array(Image.open(io.BytesIO(data)).convert("L"))
    return to_volume(arr, slice_axis)


def prep(a, max_side=512):
    lo, hi = np.percentile(a, (1, 99.5))
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)
    s = max_side / max(a.shape)
    if s < 1:
        a = transform.resize(a, (int(a.shape[0] * s), int(a.shape[1] * s)),
                             anti_aliasing=True).astype(np.float32)
    return a


# ----------------------------------------------------------------------------
# 2. AUTOMATIC SEGMENTATION (classical baseline + optional ONNX hook)
# ----------------------------------------------------------------------------
def _segment_onnx(img, path):
    import onnxruntime as ort  # optional dependency
    sess = ort.InferenceSession(path)
    x = transform.resize(img, (256, 256))[None, None].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0].squeeze()
    out = 1 / (1 + np.exp(-out)) if out.min() < 0 or out.max() > 1 else out
    m = transform.resize(out, img.shape) > 0.5
    return m


def segment_tibia(img, scale=1.0):
    """Returns boolean tibia mask. Plug in a trained model with env TIBIA_SEG_ONNX."""
    onnx_path = os.environ.get("TIBIA_SEG_ONNX", "")
    if onnx_path and os.path.exists(onnx_path):
        return _segment_onnx(img, onnx_path)
    g = filters.gaussian(img, 1.5)
    m = g > filters.threshold_otsu(g) * scale
    m = ndi.binary_opening(m, morphology.disk(3))
    m = ndi.binary_closing(m, morphology.disk(3))
    m = ndi.binary_fill_holes(m)
    lab = measure.label(m)
    H, W = m.shape
    best, best_score = 0, 0.0
    for r in measure.regionprops(lab):
        if r.area < 0.01 * H * W:
            continue
        cy = r.centroid[0] / H
        hfrac = (r.bbox[2] - r.bbox[0]) / H
        score = r.area * hfrac * (1.0 if cy > 0.4 else 0.3)
        if score > best_score:
            best, best_score = r.label, score
    return lab == best if best else np.zeros_like(m)


# ----------------------------------------------------------------------------
# 3. PTS GEOMETRY
# ----------------------------------------------------------------------------
def measure_pts(mask, anterior="Left", rng=None, jitter=False):
    """PTS = angle between the tibial plateau line and the perpendicular to the
    proximal tibial shaft axis. Positive = plateau descends posteriorly."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 300:
        return None
    y0, y1 = ys.min(), ys.max()
    Ht = y1 - y0
    if Ht < 40:
        return None
    u = (lambda lo, hi: rng.uniform(lo, hi)) if jitter else (lambda lo, hi: (lo + hi) / 2)

    # --- shaft axis: midpoints of cortex-to-cortex extent, robust line fit
    lo, hi = u(0.30, 0.42), u(0.88, 0.98)
    rr, mids = [], []
    for r in range(int(y0 + lo * Ht), int(y0 + hi * Ht)):
        c = np.flatnonzero(mask[r])
        if c.size > 5:
            rr.append(r)
            mids.append((c.min() + c.max()) / 2)
    if len(rr) < 15:
        return None
    rr, mids = np.array(rr), np.array(mids)
    if jitter:
        idx = rng.integers(0, len(rr), len(rr))
        rr, mids = rr[idx], mids[idx]
    a, b = np.polyfit(rr, mids, 1)          # x = a*y + b   (image coords)

    # --- plateau: superior tibial contour over the central AP window
    prox = mask[y0:int(y0 + 0.15 * Ht) + 1]
    cols = np.flatnonzero(prox.any(0))
    xmin, xmax = cols.min(), cols.max()
    W = xmax - xmin
    wlo, whi = u(0.15, 0.30), u(0.70, 0.85)
    xw = np.arange(int(xmin + wlo * W), int(xmin + whi * W) + 1)
    xw = np.array([x for x in xw if mask[:, x].any()])
    if len(xw) < 15:
        return None
    ytop = np.array([np.argmax(mask[:, x]) for x in xw], dtype=float)
    ytop = ndi.median_filter(ytop, size=5, mode="nearest")
    if jitter:
        idx = rng.integers(0, len(xw), len(xw))
        xf, yf = xw[idx], ytop[idx]
    else:
        xf, yf = xw, ytop
    m, c, _, _ = theilslopes(yf, xf)        # y = m*x + c   (image coords)
    rmse = float(np.sqrt(np.mean((ytop - (m * xw + c)) ** 2)))

    # --- angle (y-up frame)
    s = 1.0 if anterior == "Left" else -1.0     # sign of posterior direction in x
    t = np.array([s, -m * s])                   # plateau direction, posterior
    n = math.hypot(1, a)
    uvec = np.array([-a, 1.0]) / n              # shaft, proximal
    w = s * np.array([1.0, a]) / n              # perpendicular, posterior
    pts = math.degrees(math.atan2(-float(t @ uvec), float(t @ w)))
    return dict(pts=pts, a=a, b=b, m=m, c=c, x0=int(xw[0]), x1=int(xw[-1]),
                y0=int(y0), y1=int(y1), n_pts=len(xw), plateau_rmse=rmse,
                area=int(mask.sum()))


def analyze_slice(img, anterior="Left", n_boot=150, seed=0):
    img = prep(img)
    masks = {s: segment_tibia(img, s) for s in SCALES}
    nom = measure_pts(masks[1.0], anterior)
    if nom is None:
        return dict(ok=False, img=img, mask=masks[1.0], reason="Tibia/plateau not found")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        r = measure_pts(masks[SCALES[rng.integers(len(SCALES))]], anterior, rng, True)
        if r:
            samples.append(r["pts"])
    samples = np.array(samples)
    if len(samples) < 10:
        return dict(ok=False, img=img, mask=masks[1.0], reason="Unstable measurement")
    lo, hi = np.percentile(samples, (2.5, 97.5))
    sd = float(samples.std(ddof=1))
    reasons = []
    if sd > 2.0:
        reasons.append("high ensemble SD")
    if hi - lo > 6.0:
        reasons.append("wide CI")
    if not -5 <= nom["pts"] <= 25:
        reasons.append("implausible PTS")
    if nom["plateau_rmse"] > 4.0:
        reasons.append("irregular plateau contour")
    return dict(ok=True, img=img, mask=masks[1.0], geom=nom, pts=nom["pts"],
                ci=(float(lo), float(hi)), sd=sd, samples=samples,
                qc="PASS" if not reasons else "REVIEW", qc_reasons=reasons)


# ----------------------------------------------------------------------------
# 4. RISK ENGINE + BIOMECHANICS (simplified static model, parameters editable)
# ----------------------------------------------------------------------------
def biomech(pts, mass_kg, load_bw, ref_pts, acl_share):
    """Axial joint load F acts on a plateau inclined by PTS -> anterior shear
    component F*sin(PTS). ACL load increase = share * (shear - shear at ref slope)."""
    pts = np.asarray(pts, dtype=float)
    F = mass_kg * 9.81 * load_bw
    shear = F * np.sin(np.radians(pts))
    ref = F * np.sin(np.radians(ref_pts))
    d_acl = acl_share * (shear - ref)
    pct = 100 * d_acl / (acl_share * ref) if ref > 0 else np.full_like(d_acl, np.nan)
    return shear, d_acl, pct


def risk_tier(pts, ci, thr):
    tier = "High" if pts >= thr else "Low"
    borderline = ci[0] < thr <= ci[1]
    return tier, borderline


def acl_status(img, model_path):
    """Rupture status needs a TRAINED classifier. Contract for the optional ONNX model:
    input (1,1,256,256) float32 in [0,1]; output = probability of ACL tear."""
    if not model_path or not os.path.exists(model_path):
        return None
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path)
    x = transform.resize(img, (256, 256))[None, None].astype(np.float32)
    p = float(np.ravel(sess.run(None, {sess.get_inputs()[0].name: x})[0])[0])
    return 1 / (1 + math.exp(-p)) if not 0 <= p <= 1 else p


# ----------------------------------------------------------------------------
# 5. AGGREGATION OVER A VOLUME
# ----------------------------------------------------------------------------
def pool(results):
    good = [r for r in results if r["ok"]]
    if not good:
        return None
    use = [r for r in good if r["qc"] == "PASS"] or good
    pts = float(np.median([r["pts"] for r in use]))
    samples = np.concatenate([r["samples"] for r in use])
    lo, hi = np.percentile(samples, (2.5, 97.5))
    return dict(pts=pts, ci=(float(lo), float(hi)), sd=float(samples.std(ddof=1)),
                samples=samples, n_used=len(use),
                qc="PASS" if all(r["qc"] == "PASS" for r in use) else "REVIEW")


# ----------------------------------------------------------------------------
# 6. WATERMARK + OVERLAY + REPORTS
# ----------------------------------------------------------------------------
def _font(size):
    for p in ("DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def add_watermark(im):
    im = im.convert("RGBA")
    W, H = im.size
    fs = max(12, W // 22)
    f = _font(fs)
    tile = Image.new("RGBA", (int(fs * 15), int(fs * 2)), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((5, 5), WATERMARK, font=f, fill=(255, 255, 255, 70))
    tile = tile.rotate(30, expand=True)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    for y in range(-tile.height, H + tile.height, int(tile.height * 1.6)):
        for x in range(-tile.width, W + tile.width, int(tile.width * 1.3)):
            layer.paste(tile, (x, y), tile)
    im = Image.alpha_composite(im, layer)
    d = ImageDraw.Draw(im)
    f2 = _font(max(11, W // 38))
    d.text((8, H - max(11, W // 38) - 10), WATERMARK, font=f2, fill=(255, 255, 0, 255))
    return im.convert("RGB")


def render_overlay(res, size=640):
    img = res["img"]
    H, W = img.shape
    sc = size / H
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((int(W * sc), size)).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    if res["ok"]:
        edge = res["mask"] ^ ndi.binary_erosion(res["mask"])
        ey, ex = np.nonzero(edge)
        for x, y in zip(ex[::2], ey[::2]):
            d.point((x * sc, y * sc), fill=(0, 255, 120, 160))
        g = res["geom"]
        ya, yb = g["y0"], g["y1"]
        d.line([(g["a"] * ya + g["b"]) * sc, ya * sc, (g["a"] * yb + g["b"]) * sc, yb * sc],
               fill=(255, 220, 0, 255), width=3)
        d.line([g["x0"] * sc, (g["m"] * g["x0"] + g["c"]) * sc,
                g["x1"] * sc, (g["m"] * g["x1"] + g["c"]) * sc], fill=(255, 40, 40, 255), width=3)
        xm = (g["x0"] + g["x1"]) / 2
        ym = g["m"] * xm + g["c"]
        n = math.hypot(1, g["a"])
        dx, dy = 1 / n, -g["a"] / n          # perpendicular to shaft (image coords)
        L = (g["x1"] - g["x0"]) / 2
        d.line([(xm - dx * L) * sc, (ym - dy * L) * sc, (xm + dx * L) * sc, (ym + dy * L) * sc],
               fill=(0, 220, 255, 220), width=2)
        d.text((10, 10), f"PTS {res['pts']:.1f} deg  95% CI [{res['ci'][0]:.1f}, {res['ci'][1]:.1f}]",
               font=_font(18), fill=(255, 255, 255, 255))
    else:
        d.text((10, 10), "AUTO-MEASUREMENT FAILED: " + res.get("reason", ""),
               font=_font(18), fill=(255, 80, 80, 255))
    return add_watermark(im)


def make_pdf(rows, overlay):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Posterior Tibial Slope - Automated Report", fontsize=15, weight="bold")
        ax = fig.add_axes([0.1, 0.45, 0.8, 0.45])
        ax.imshow(overlay)
        ax.axis("off")
        txt = "\n".join(f"{k}: {v}" for k, v in rows)
        fig.text(0.1, 0.42, txt, va="top", fontsize=9, family="monospace")
        fig.text(0.5, 0.5, WATERMARK, fontsize=34, color="gray", alpha=0.18,
                 rotation=30, ha="center", va="center")
        fig.text(0.5, 0.02, WATERMARK + " | Research use only - not a medical device",
                 ha="center", fontsize=8, color="gray")
        pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# 7. VALIDATION METRICS (AI vs reference)
# ----------------------------------------------------------------------------
def icc21(x, y):
    d = np.c_[x, y]
    n, k = d.shape
    gm = d.mean()
    ssr = k * ((d.mean(1) - gm) ** 2).sum()
    ssc = n * ((d.mean(0) - gm) ** 2).sum()
    sse = ((d - gm) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def validation_metrics(df, thr):
    e = df["ai_pts"] - df["reference_pts"]
    out = {
        "n": len(df), "MAE (deg)": e.abs().mean(), "RMSE (deg)": math.sqrt((e ** 2).mean()),
        "Bias (deg)": e.mean(), "LoA lower": e.mean() - 1.96 * e.std(ddof=1),
        "LoA upper": e.mean() + 1.96 * e.std(ddof=1),
        "Within 2 deg (%)": 100 * (e.abs() <= 2).mean(), "Within 3 deg (%)": 100 * (e.abs() <= 3).mean(),
        "Reference inside AI 95% CI (%)": 100 * ((df.reference_pts >= df.ci_low) & (df.reference_pts <= df.ci_high)).mean(),
    }
    if len(df) > 2:
        out["Pearson r"] = pearsonr(df.ai_pts, df.reference_pts)[0]
        out["ICC(2,1)"] = icc21(df.ai_pts.values, df.reference_pts.values)
    ra, ai = df.reference_pts >= thr, df.ai_pts >= thr
    out["Risk-tier agreement (%)"] = 100 * (ra == ai).mean()
    if ra.any():
        out["Sensitivity High (%)"] = 100 * (ai & ra).sum() / ra.sum()
    if (~ra).any():
        out["Specificity High (%)"] = 100 * (~ai & ~ra).sum() / (~ra).sum()
    return out


def bland_altman(df):
    m = (df.ai_pts + df.reference_pts) / 2
    e = df.ai_pts - df.reference_pts
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.scatter(m, e)
    ax.axhline(e.mean(), color="k")
    for s in (-1.96, 1.96):
        ax.axhline(e.mean() + s * e.std(ddof=1), color="r", ls="--")
    ax.set_xlabel("Mean of AI & reference PTS (deg)")
    ax.set_ylabel("AI - reference (deg)")
    ax.set_title("Bland-Altman")
    fig.text(0.99, 0.01, WATERMARK, ha="right", fontsize=7, color="gray")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 8. STREAMLIT UI
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_slice(img, anterior, n_boot):
    return analyze_slice(img, anterior, n_boot)


def case_summary(vol, anterior, n_boot, idxs):
    res = [cached_slice(vol[i], anterior, n_boot) for i in idxs]
    return res, pool(res)


def main():
    st.set_page_config(page_title="PTS Workstation", layout="wide", page_icon="🦴")
    st.markdown(
        f"<div style='position:fixed;bottom:8px;right:14px;opacity:.55;z-index:999;"
        f"font-size:13px;pointer-events:none'>{WATERMARK}</div>", unsafe_allow_html=True)
    st.title("PTS Workstation - automated posterior tibial slope on sagittal knee MRI")
    st.caption(WATERMARK + " | Research prototype - not a medical device")

    with st.sidebar:
        st.header("Settings")
        anterior = st.radio("Anterior side of image (metadata)", ["Left", "Right"], horizontal=True)
        n_boot = st.slider("Ensemble/bootstrap runs (CI)", 50, 400, 150, 50)
        thr = st.number_input("High-risk PTS threshold (deg)", 5.0, 20.0, 12.0, 0.5)
        st.subheader("Biomechanical model")
        mass = st.number_input("Body mass (kg)", 30.0, 200.0, 70.0)
        load_bw = st.number_input("Axial joint load (x body weight)", 0.5, 6.0, 1.0, 0.25)
        ref_pts = st.number_input("Reference (normal) PTS (deg)", 0.0, 15.0, 9.0, 0.5)
        acl_share = st.slider("Fraction of shear resisted by ACL", 0.1, 1.0, 1.0, 0.05)
        st.subheader("Optional trained models")
        acl_path = st.text_input("ACL tear ONNX classifier path", "models/acl_classifier.onnx")
        st.text_input("Tibia segmentation ONNX (set env TIBIA_SEG_ONNX)", os.environ.get("TIBIA_SEG_ONNX", ""),
                      disabled=True)
        slice_axis = st.selectbox("Slice axis of 3D arrays", ["auto", 0, 1, 2])

    tab1, tab2 = st.tabs(["Workstation", "Validation (AI vs reference)"])

    # ------------------------------------------------------------ WORKSTATION
    with tab1:
        up = st.file_uploader("Upload MRI (.pck, .pkl, .npy, .jpg, .png)",
                              type=["pck", "pkl", "pickle", "npy", "jpg", "jpeg", "png"])
        if not up:
            st.info("Upload a sagittal knee MRI slice or volume. Measurement is fully automatic.")
        else:
            try:
                vol = load_file(up, slice_axis)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()
            N = vol.shape[0]
            mode = st.radio("Analysis", ["Current slice", "Whole volume (central 50%)"], horizontal=True) \
                if N > 1 else "Current slice"
            i = st.slider(f"Slice (of {N})", 1, N, (N + 1) // 2) - 1 if N > 1 else 0

            with st.spinner("Running automatic measurement..."):
                cur = cached_slice(vol[i], anterior, n_boot)
                if mode.startswith("Whole"):
                    lo_i, hi_i = int(N * 0.25), max(int(N * 0.75), int(N * 0.25) + 1)
                    allres, pooled = case_summary(vol, anterior, n_boot, range(lo_i, hi_i))
                else:
                    allres, pooled = [cur], (pool([cur]) if cur["ok"] else None)

            c1, c2 = st.columns([3, 2])
            overlay = render_overlay(cur)
            c1.image(overlay, caption=f"Slice {i + 1}/{N} - yellow: shaft axis | red: plateau | cyan: perpendicular",
                     use_container_width=True)

            with c2:
                if not pooled:
                    st.error("Automatic measurement failed on the analysed slice(s): "
                             + cur.get("reason", ""))
                else:
                    pts, ci = pooled["pts"], pooled["ci"]
                    tier, borderline = risk_tier(pts, ci, thr)
                    shear, dacl, pct = biomech(pooled["samples"], mass, load_bw, ref_pts, acl_share)
                    s_pt, d_pt, p_pt = [x[0] for x in biomech([pts], mass, load_bw, ref_pts, acl_share)]
                    sc = lambda x: np.percentile(x, (2.5, 97.5))
                    st.metric("PTS (deg)", f"{pts:.1f}", f"95% CI {ci[0]:.1f} to {ci[1]:.1f}", delta_color="off")
                    st.write(f"**Auto-QC:** {pooled['qc']}  |  ensemble SD {pooled['sd']:.2f} deg"
                             f"  |  slices used: {pooled['n_used']}")
                    if cur["ok"] and cur["qc_reasons"]:
                        st.warning("Current slice flagged: " + ", ".join(cur["qc_reasons"]))
                    (st.error if tier == "High" else st.success)(f"Risk tier: **{tier}** (threshold {thr:.1f} deg)")
                    if borderline:
                        st.warning("CI straddles the threshold - tier is borderline.")
                    st.write(f"**Anterior tibial shear:** {s_pt:.0f} N  (95% CI {sc(shear)[0]:.0f} to {sc(shear)[1]:.0f})")
                    st.write(f"**ACL load increase vs {ref_pts:.0f} deg reference:** {d_pt:+.0f} N "
                             f"({p_pt:+.0f}%)  (95% CI {sc(dacl)[0]:+.0f} to {sc(dacl)[1]:+.0f} N)")

                    p = acl_status(cur["img"], acl_path) if cur["ok"] else None
                    if p is None:
                        st.info("**ACL rupture status: NOT ASSESSED.** PTS cannot show a rupture; it only "
                                f"indicates mechanical risk (**{tier}**). Load a trained ONNX classifier to "
                                "obtain Ruptured / Intact.")
                        acl_txt = f"Not assessed (risk tier {tier})"
                    else:
                        verdict = "RUPTURED (suspected)" if p >= 0.5 else "INTACT (suspected)"
                        (st.error if p >= 0.5 else st.success)(f"ACL: **{verdict}** - model probability {p:.2f}")
                        acl_txt = f"{verdict}, p={p:.2f}"

                    ref_txt = "-"
                    ref = st.number_input("Reference PTS for validation only (optional, deg)", 0.0, 30.0, 0.0, 0.1,
                                          help="Used ONLY to compare against the AI; never fed to the algorithm.")
                    if ref > 0:
                        err = pts - ref
                        inside = ci[0] <= ref <= ci[1]
                        st.write(f"**AI - reference = {err:+.1f} deg** | reference inside AI CI: {'yes' if inside else 'no'}")
                        ref_txt = f"{ref:.1f} (error {err:+.1f})"

                    rows = [("File", up.name), ("Slice", f"{i + 1}/{N} ({mode})"),
                            ("PTS (deg)", f"{pts:.2f}"), ("95% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]"),
                            ("Ensemble SD", f"{pooled['sd']:.2f}"), ("Auto-QC", pooled["qc"]),
                            ("Risk tier", tier + (" (borderline)" if borderline else "")),
                            ("Anterior shear (N)", f"{s_pt:.0f} [{sc(shear)[0]:.0f}, {sc(shear)[1]:.0f}]"),
                            ("ACL load increase (N)", f"{d_pt:+.0f} [{sc(dacl)[0]:+.0f}, {sc(dacl)[1]:+.0f}]"),
                            ("ACL status", acl_txt), ("Reference PTS", ref_txt),
                            ("Model", f"mass {mass} kg, load {load_bw} BW, ref slope {ref_pts} deg")]
                    st.dataframe(pd.DataFrame(rows, columns=["Item", "Value"]), hide_index=True)
                    png = io.BytesIO(); overlay.save(png, format="PNG")
                    d1, d2, d3 = st.columns(3)
                    d1.download_button("PNG", png.getvalue(), "pts_overlay.png", "image/png")
                    d2.download_button("CSV", pd.DataFrame(rows, columns=["item", "value"]).to_csv(index=False),
                                       "pts_report.csv", "text/csv")
                    d3.download_button("PDF report", make_pdf(rows, np.array(overlay)),
                                       "pts_report.pdf", "application/pdf")

            if mode.startswith("Whole"):
                st.subheader("Per-slice results")
                st.dataframe(pd.DataFrame([
                    dict(slice=int(N * 0.25) + k + 1, ok=r["ok"],
                         pts=round(r["pts"], 2) if r["ok"] else None,
                         ci_low=round(r["ci"][0], 2) if r["ok"] else None,
                         ci_high=round(r["ci"][1], 2) if r["ok"] else None,
                         qc=r.get("qc", "FAIL")) for k, r in enumerate(allres)]), hide_index=True)

    # ------------------------------------------------------------- VALIDATION
    with tab2:
        st.write("Upload several cases plus a CSV with columns **filename, reference_pts** "
                 "(optional: **reference_acl** = 1 ruptured / 0 intact). The AI runs fully automatically; "
                 "the reference values are used only to score it.")
        files = st.file_uploader("Images / volumes", accept_multiple_files=True, key="batch",
                                 type=["pck", "pkl", "pickle", "npy", "jpg", "jpeg", "png"])
        ref_csv = st.file_uploader("Reference CSV", type=["csv"], key="refcsv")
        if files and ref_csv and st.button("Run validation"):
            ref_df = pd.read_csv(ref_csv)
            rows_ = []
            bar = st.progress(0.0)
            for k, f in enumerate(files):
                bar.progress((k + 1) / len(files))
                try:
                    v = load_file(f, slice_axis)
                    idxs = range(0, 1) if v.shape[0] == 1 else range(int(v.shape[0] * .25), max(int(v.shape[0] * .75), int(v.shape[0] * .25) + 1))
                    _, pl = case_summary(v, anterior, n_boot, idxs)
                    if pl is None:
                        continue
                    mid = cached_slice(v[v.shape[0] // 2], anterior, n_boot)
                    pr = acl_status(mid["img"], acl_path) if mid["ok"] else None
                    rows_.append(dict(filename=f.name, ai_pts=pl["pts"], ci_low=pl["ci"][0],
                                      ci_high=pl["ci"][1], qc=pl["qc"], ai_acl_prob=pr))
                except Exception as e:
                    st.warning(f"{f.name}: {e}")
            res = pd.DataFrame(rows_).merge(ref_df, on="filename", how="inner")
            if len(res) < 2:
                st.error("Need at least 2 matched cases (check that filenames match the CSV).")
            else:
                m = validation_metrics(res, thr)
                st.dataframe(pd.DataFrame(m.items(), columns=["metric", "value"]).round(3), hide_index=True)
                st.pyplot(bland_altman(res))
                if "reference_acl" in res and res.ai_acl_prob.notna().all():
                    pred = (res.ai_acl_prob >= .5).astype(int)
                    st.write(f"**ACL rupture accuracy:** {100 * (pred == res.reference_acl).mean():.1f}%")
                st.dataframe(res.round(2), hide_index=True)
                st.download_button("Download results CSV", res.to_csv(index=False), "validation_results.csv")


if __name__ == "__main__":
    main()
