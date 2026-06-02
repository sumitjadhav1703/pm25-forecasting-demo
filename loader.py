# loader.py
import os
import json
import numpy as np

# ── UPDATE THIS after creating your HF dataset repo ──────────────────────────
HF_REPO_ID = "sumit1703/pm25-forecasting-data"
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HUB_CACHE_DIR  = "/tmp/pm25_data"

REQUIRED_FILES = [
    "demo_preds.npy",
    "demo_inputs.npy",
    "lat_lon.npy",
    "demo_stats.json",
    "sample_indices.npy",
]

H, W = 140, 124


def load_demo_data() -> dict:
    """
    Returns a dict with keys:
        preds          np.ndarray  (22, 140, 124, 16)
        inputs         np.ndarray  (22, 10, 140, 124)
        lat            np.ndarray  (140, 124)  or None
        lon            np.ndarray  (140, 124)  or None
        stats          dict        {feature: {mean, std}}
        sample_indices np.ndarray  (22,)
    """
    # Check if all files exist locally
    all_local = all(
        os.path.exists(os.path.join(LOCAL_DATA_DIR, f))
        for f in REQUIRED_FILES
    )

    if all_local:
        print("[loader] Using local data/ folder.")
        data_dir = LOCAL_DATA_DIR
    else:
        print("[loader] Local data/ not found — downloading from HF Hub...")
        data_dir = _download_from_hub()

    return _load_files(data_dir)


def _download_from_hub() -> str:
    from huggingface_hub import hf_hub_download

    os.makedirs(HUB_CACHE_DIR, exist_ok=True)

    for fname in REQUIRED_FILES:
        dest = os.path.join(HUB_CACHE_DIR, fname)
        if os.path.exists(dest):
            print(f"  [cache hit] {fname}")
            continue
        print(f"  [downloading] {fname} ...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=fname,
            repo_type="dataset",
            local_dir=HUB_CACHE_DIR,
        )
        print(f"  [done] {fname}")

    return HUB_CACHE_DIR


def _load_files(data_dir: str) -> dict:
    preds          = np.load(os.path.join(data_dir, "demo_preds.npy"))
    inputs         = np.load(os.path.join(data_dir, "demo_inputs.npy"))
    lat_lon        = np.load(os.path.join(data_dir, "lat_lon.npy"))
    sample_indices = np.load(os.path.join(data_dir, "sample_indices.npy"))

    with open(os.path.join(data_dir, "demo_stats.json")) as f:
        stats = json.load(f)

    # Validate shapes
    assert preds.ndim  == 4, f"demo_preds.npy expected 4D, got {preds.shape}"
    assert inputs.ndim == 4, f"demo_inputs.npy expected 4D, got {inputs.shape}"

    # Parse lat/lon — handle both (2, H, W) and (H, W, 2)
    lat, lon = None, None
    if lat_lon.shape == (2, H, W):
        lat, lon = lat_lon[0], lat_lon[1]
    elif lat_lon.shape == (H, W, 2):
        lat, lon = lat_lon[..., 0], lat_lon[..., 1]
    else:
        print(f"[loader] Warning: unexpected lat_lon shape {lat_lon.shape} — axes will show pixel indices")

    print(f"[loader] Loaded — preds:{preds.shape}  inputs:{inputs.shape}")

    return {
        "preds":          preds,
        "inputs":         inputs,
        "lat":            lat,
        "lon":            lon,
        "stats":          stats,
        "sample_indices": sample_indices,
    }
