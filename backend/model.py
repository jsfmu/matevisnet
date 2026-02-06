# backend/model.py
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "san_leandro_products.csv"
EMBS_PATH = PROJECT_ROOT / "data" / "image_embs.npy"

# Globals initialised at startup
raw_df: pd.DataFrame | None = None
df: pd.DataFrame | None = None
image_embs: np.ndarray | None = None
nn: NearestNeighbors | None = None


def _str_or_none(value: Any) -> Optional[str]:
    """Convert values to trimmed strings, returning None for empties/NaNs."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if pd.notna(value):
        return str(value)
    return None


def _normalize_sku(value: Any) -> Optional[str]:
    """Return a canonical SKU string (strip whitespace and trailing .0)."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return str(int(value))

    text = str(value).strip()
    if not text:
        return None

    if "." in text:
        left, right = text.split(".", 1)
        if left and left.isdigit() and right.strip("0") == "":
            return left

    return text


def compute_material_bucket(row: pd.Series) -> str:
    """Rough-type bucket so wood planks don't mix with tile/vinyl/etc."""
    cat = str(row.get("category_slug", "")).lower()
    name = str(row.get("name", "")).lower()

    if "installation-materials" in cat:
        return "install"
    if any(k in name for k in ["stair nose", "stairnose", "stair-nose"]):
        return "trim"

    if "/wood" in cat:
        return "wood"
    if "laminate" in cat:
        return "laminate"
    if "vinyl" in cat or "rigid-core" in cat or "nucore" in cat:
        return "vinyl"
    if "stone" in cat:
        return "stone"
    if "porcelain" in cat or "ceramic" in cat or "tile" in cat:
        return "tile"
    if "decoratives" in cat or "mosaic" in cat or "wall-tile" in cat:
        return "decor"

    return "other"


def init_model() -> None:
    """Load CSV + embeddings and build the kNN index."""
    global raw_df, df, image_embs, nn

    raw_df = pd.read_csv(CSV_PATH)
    raw_df["sku_norm"] = raw_df["sku"].apply(_normalize_sku)

    image_col = "image_filename"
    if image_col not in raw_df.columns:
        raise ValueError(f"Column {image_col!r} not found in CSV")

    df_local = raw_df[
        raw_df[image_col].notna() & (raw_df[image_col].astype(str) != "")
    ].reset_index(drop=True)
    df_local["sku_norm"] = df_local["sku"].apply(_normalize_sku)

    image_embs = np.load(EMBS_PATH)
    if image_embs.shape[0] != len(df_local):
        raise ValueError(
            f"Embeddings/CSV mismatch: {image_embs.shape[0]} emb vs {len(df_local)} rows"
        )

    df_local["material_bucket"] = df_local.apply(compute_material_bucket, axis=1)

    df = df_local

    nn = NearestNeighbors(n_neighbors=50, metric="cosine")
    nn.fit(image_embs)


def get_index_by_sku(query_sku: str) -> int:
    """Return index in df for the given SKU (filtered set, with embeddings)."""
    assert df is not None and raw_df is not None
    sku_str = _normalize_sku(query_sku)
    if not sku_str:
        raise ValueError("SKU must be provided")

    matches = df.index[df["sku_norm"] == sku_str].tolist()
    if matches:
        return matches[0]

    exists_in_raw = raw_df["sku_norm"].eq(sku_str).any()
    if exists_in_raw:
        raise ValueError(
            f"SKU {sku_str!r} exists in CSV but not in the embedded set "
            "(likely missing image_filename)."
        )
    raise ValueError(f"SKU {sku_str!r} not found in CSV at all.")


def search_similar_by_index(
    query_idx: int,
    top_k: int = 10,
    exclude_same: bool = True,
) -> List[Dict[str, Any]]:
    """Return top_k similar products (same material bucket) for df.iloc[query_idx]."""
    assert df is not None and image_embs is not None and nn is not None

    query_bucket = df.loc[query_idx, "material_bucket"]

    distances, indices = nn.kneighbors(image_embs[query_idx].reshape(1, -1), n_neighbors=top_k + 10)
    distances = distances[0]
    indices = indices[0]

    results: List[Dict[str, Any]] = []

    for dist, idx in zip(distances, indices):
        idx = int(idx)
        if exclude_same and idx == query_idx:
            continue

        row = df.iloc[idx]
        if row["material_bucket"] != query_bucket:
            continue

        results.append(
            {
                "index": idx,
                "sku": row.get("sku_norm") or _normalize_sku(row.get("sku")),
                "name": str(row["name"]),
                "category_slug": row.get("category_slug"),
                "material_bucket": row["material_bucket"],
                "distance": float(dist),
                "image_filename": _str_or_none(row.get("image_filename")),
                "image_url": _str_or_none(row.get("image_url")),
            }
        )
        if len(results) >= top_k:
            break

    return results


def similar_by_sku(query_sku: str, top_k: int = 10) -> Dict[str, Any]:
    """High-level API used by your web/mobile UI."""
    assert df is not None

    query_idx = get_index_by_sku(query_sku)
    query_row = df.iloc[query_idx]
    neighbors = search_similar_by_index(query_idx, top_k=top_k)

    def to_dict(row: pd.Series) -> Dict[str, Any]:
        return {
            "sku": row.get("sku_norm") or _normalize_sku(row.get("sku")),
            "name": str(row["name"]),
            "category_slug": row.get("category_slug"),
            "material_bucket": row.get("material_bucket"),
            "image_filename": _str_or_none(row.get("image_filename")),
            "image_url": _str_or_none(row.get("image_url")),
        }

    return {
        "query": to_dict(query_row),
        "results": neighbors,
    }


def get_remote_image_url(identifier: str) -> Optional[str]:
    """Return the remote CDN URL for an image filename or SKU."""
    if not identifier:
        return None

    token = identifier.strip()
    if not token:
        return None

    if raw_df is None:
        return None

    # Try matching image_filename first
    matches = raw_df[raw_df["image_filename"].astype(str) == token]
    if matches.empty and "." in token:
        # Allow calling with just the basename (e.g. SKU)
        basename = token.rsplit(".", 1)[0]
        matches = raw_df[raw_df["image_filename"].astype(str).str.split(".").str[0] == basename]
        if matches.empty:
            matches = raw_df[raw_df["sku_norm"] == basename]
    elif matches.empty:
        matches = raw_df[raw_df["sku_norm"] == token]

    if matches.empty:
        return None

    url = matches.iloc[0].get("image_url")
    return _str_or_none(url)
