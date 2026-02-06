#!/usr/bin/env python3
"""
Scrape Floor & Decor product metadata & images for the
San Leandro store (storeID=238).

Outputs:
  - data/san_leandro_products.csv
  - images/<SKU>.jpg

Run from repo root:
  .\.venv\Scripts\Activate.ps1
  python scrape_238_products.py
"""

import csv
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

# ---------------------- CONFIG ----------------------

BASE_URL = "https://www.flooranddecor.com"
STORE_ID = 238  # San Leandro

# Seed category slugs to make sure we at least cover these
CATEGORY_SLUGS = [
    "/tile",
    "/stone",
    "/wood",
    "/decoratives",
]

DATA_DIR = "data"
IMAGES_DIR = "images"
METADATA_CSV = os.path.join(DATA_DIR, "san_leandro_products.csv")

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds between requests

# Safety valve so we don't accidentally spider the entire internet
MAX_PAGES_PER_CATEGORY = 10_000

# Product URL pattern:
# e.g. https://www.flooranddecor.com/...-101363893.html
PRODUCT_URL_RE = re.compile(
    r"https://www\.flooranddecor\.com/[A-Za-z0-9_\-/]+-(\d{6,})\.html"
)

# ---------------------- CSV FIELD MANAGEMENT ----------------------
# You said you want to keep *all* of these columns, without deleting anything:
# sku,name,category_slug,product_url,image_url,image_filename,
# row_id,cluster_id,pca_x,pca_y,surface_type,material_group,cluster_size

REQUIRED_FIELD_ORDER = [
    "sku",
    "name",
    "category_slug",
    "product_url",
    "image_url",
    "image_filename",
    "row_id",
    "cluster_id",
    "pca_x",
    "pca_y",
    "surface_type",
    "material_group",
    "cluster_size",
]


def compute_fieldnames(rows: List[Dict[str, str]]) -> List[str]:
    """
    Build the final CSV header:

      1. Start with REQUIRED_FIELD_ORDER (in that order).
      2. Append *any other* columns that already exist in the data.

    This way we:
      - Never lose your clustering columns.
      - Also keep any extra columns you might add later in notebooks.
    """
    fieldnames: List[str] = []
    seen: Set[str] = set()

    # 1) Seed with required order
    for col in REQUIRED_FIELD_ORDER:
        if col not in seen:
            seen.add(col)
            fieldnames.append(col)

    # 2) Add any additional keys from the rows
    for row in rows:
        for col in row.keys():
            if col not in seen:
                seen.add(col)
                fieldnames.append(col)

    return fieldnames


# ---------------------- EXCLUSION FILTERS ----------------------

# Category / URL substrings we want to completely avoid crawling
EXCLUDE_CATEGORY_SUBSTRINGS = [
    # Big buckets
    "installation-materials",
    "installation",
    "decorative-hardware",
    "cabinet-hardware",
    "vanities",
    "vanity",
    "countertops",
    "countertop",
    "slab",
    "doors",
    "door",  # e.g. /doors/interior-doors/...
    "moulding",
    "molding",
    "baseboard",
    "trim",
    "transitions",
    "transition-strip",
    # Installation-material-ish
    "thinset",
    "thin-set",
    "mortar",
    "mortars",
    "grout",
    "adhesive",
    "adhesives",
    "underlayment",
    "membrane",
]

# Product name tokens we consider "not tiles/wood/deco"
EXCLUDE_PRODUCT_NAME_TOKENS = [
    # Installation materials
    "thinset",
    "thin-set",
    "mortar",
    "grout",
    "adhesive",
    "underlayment",
    "membrane",
    "primer",
    "spacer",
    "trowel",
    "saw",
    "blade",
    "bucket",
    "tape",
    "screed",
    # Vanities / countertops
    "vanity",
    "vanities",
    "countertop",
    "counter top",
    "counter tops",
    "slab",
    # Doors / knobs / hardware
    "door",
    "doors",
    "knob",
    "knobs",
    "pull",
    "pulls",
    "handle",
    "handles",
    "hinge",
    "hinges",
    "lever",
    "levers",
    "cabinet hardware",
    "cabinet knob",
    "cabinet pull",
]


def url_is_excluded(url: str) -> bool:
    """
    Return True if this URL looks like an installation-material / vanity /
    countertop / door / hardware thing that we don't want.
    """
    path = urlparse(url).path.lower()
    return any(bad in path for bad in EXCLUDE_CATEGORY_SUBSTRINGS)


# ---------------------- LOGGING ----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------- HTTP HELPERS ----------------------


def make_session() -> requests.Session:
    """Create a requests session with a decent User-Agent."""
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def set_store_context(session: requests.Session) -> None:
    """
    Hit the store selector URL so cookies / context are set
    for the San Leandro store (ID=238).
    """
    url = f"{BASE_URL}/store?storeID={STORE_ID}"
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        logging.info("Store context response: %s %s", r.status_code, url)
    except Exception as e:
        logging.warning("Failed to set store context: %s", e)


# ---------------------- CATEGORY DISCOVERY ----------------------


def discover_category_slugs(session: requests.Session) -> List[str]:
    """
    Hit the site-wide sitemap and collect a *broad* set of category-like URLs.
    We:
      * start from CATEGORY_SLUGS
      * add any internal, non-product URL whose path contains flooring-ish
        keywords (tile, wood, vinyl, laminate, floor, wall, etc.), while
        skipping installation materials, vanities, countertops, doors, hardware.
    """
    sitemap_url = urljoin(BASE_URL, "/sitemap")
    slugs: Set[str] = set(CATEGORY_SLUGS)  # start from your existing list

    try:
        r = session.get(sitemap_url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            logging.warning("Sitemap %s returned status %s", sitemap_url, r.status_code)
            return sorted(slugs)
    except Exception as e:
        logging.warning("Failed to fetch sitemap %s: %s", sitemap_url, e)
        return sorted(slugs)

    soup = BeautifulSoup(r.text, "html.parser")

    category_keywords = [
        "tile",
        "stone",
        "wood",
        "vinyl",
        "laminate",
        "bathroom",
        "kitchen",
        "backsplash",
        "decor",
        "mosaic",
        # "countertop",  # intentionally excluded via EXCLUDE_CATEGORY_SUBSTRINGS
        "floor",
        "wall",
        "stair",
        # "installation",  # filtered below
        "fixtures",
        "tools",
        "grout",
        "thinset",
        "adhesive",
    ]

    skip_prefixes = (
        "/customer-care",
        "/about-us",
        "/company",
        "/pro-center",
        "/pro",
        "/account",
        "/wishlist",
        "/cart",
        "/order",
        "/signin",
        "/login",
        "/register",
        "/faq",
        "/help",
        "/design-services",
        "/blog",
        "/videos",
        "/galleries",
    )

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/"):
            continue

        # Strip query params / fragments
        href = href.split("?", 1)[0].split("#", 1)[0]
        if not href or href == "/":
            continue

        low = href.lower()

        # Skip obvious non-product sections
        if low.startswith(skip_prefixes):
            continue

        # NEW: skip categories we don't care about at all
        if any(bad in low for bad in EXCLUDE_CATEGORY_SUBSTRINGS):
            continue

        # Skip product detail URLs (.html)
        if low.endswith(".html"):
            continue

        # Only keep URLs that look flooring-related
        if any(kw in low for kw in category_keywords):
            slugs.add(href)

    logging.info("Discovered %d category-like slugs from sitemap", len(slugs))
    return sorted(slugs)


# ---------------------- URL DISCOVERY ----------------------


def fetch_product_urls_for_category(
    session: requests.Session, category_slug: str
) -> Set[str]:
    """
    Given a category slug like "/tile", crawl ALL listing / filter / search pages
    reachable under that slug and extract product detail URLs using PRODUCT_URL_RE.

    This is intentionally aggressive:
      * Any internal link whose URL mentions the category token ("/tile" or "tile")
        is followed (up to MAX_PAGES_PER_CATEGORY pages).
      * We don't limit ourselves only to 'start=', 'page=', etc. query params anymore.
      * But we *do* skip unwanted categories (installation materials, vanities,
        countertops, doors, knobs, etc.)
    """
    start_url = urljoin(BASE_URL, category_slug)

    to_visit: List[str] = [start_url]
    visited: Set[str] = set()
    found_urls: Set[str] = set()

    cat_token = category_slug.strip("/").lower()

    while to_visit and len(visited) < MAX_PAGES_PER_CATEGORY:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        logging.info("Scanning category page (%s pages seen) %s", len(visited), url)
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
        except Exception as e:
            logging.warning("Failed to fetch category %s: %s", url, e)
            continue

        content_type = r.headers.get("Content-Type", "")
        if r.status_code != 200 or "text/html" not in content_type:
            logging.warning(
                "Category %s returned status %s (Content-Type=%s)",
                url,
                r.status_code,
                content_type,
            )
            continue

        html = r.text

        # --- collect product URLs from this page ---
        for match in PRODUCT_URL_RE.finditer(html):
            full_url = match.group(0)
            if url_is_excluded(full_url):
                # Skip installation materials / vanities / countertops / doors / knobs
                continue
            found_urls.add(full_url)

        # --- discover more listing pages to crawl ---
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#"):
                continue
            if href.lower().startswith("javascript:"):
                continue

            full = urljoin(BASE_URL, href)
            if full in visited:
                continue

            parsed = urlparse(full)

            # Stay inside main domain
            if parsed.netloc and "flooranddecor.com" not in parsed.netloc:
                continue

            # If this anchor is itself a product detail URL, just record it
            if PRODUCT_URL_RE.search(full):
                if url_is_excluded(full):
                    continue
                found_urls.add(full)
                continue

            href_low = href.lower()

            # NEW: don't even walk into "bad" sections
            if any(bad in href_low for bad in EXCLUDE_CATEGORY_SUBSTRINGS):
                continue

            # Decide if this looks like a listing / filter / search page
            in_same_family = False

            # Same slug (/tile, /tile/..., /tile?... etc.)
            if category_slug.lower() in href_low:
                in_same_family = True
            # Or category token in query path (/search?cgid=tile-something, etc.)
            elif cat_token and cat_token in href_low:
                in_same_family = True

            if not in_same_family:
                continue

            # Looks relevant to this category; follow it
            to_visit.append(full)

        # be nice to the server
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info(
        "Category %s -> %d candidate product URLs from %d listing pages (capped at %d)",
        category_slug,
        len(found_urls),
        len(visited),
        MAX_PAGES_PER_CATEGORY,
    )
    return found_urls


def fetch_all_product_urls(
    session: requests.Session,
    category_slugs: List[str],
) -> Dict[str, Set[str]]:
    """
    For each category, collect product URLs.
    Returns {category_slug: {url1, url2, ...}}.
    """
    result: Dict[str, Set[str]] = {}
    for slug in category_slugs:
        urls = fetch_product_urls_for_category(session, slug)
        if urls:
            result[slug] = urls
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    total = sum(len(v) for v in result.values())
    logging.info(
        "Total unique product URLs found across %d categories: %d",
        len(result),
        total,
    )
    return result


# ---------------------- AVAILABILITY FILTER ----------------------


def is_available_in_store(soup: BeautifulSoup) -> bool:
    """
    Heuristically determine if the product is actually stocked at the current store
    (after set_store_context has been called).

    Logic:
      - Find the 'In-Store Pickup' block.
      - If that block says 'Item will be shipped and should arrive in X days',
        treat it as NOT stocked locally (ship-to-store only) and skip.
      - Otherwise, keep it.
    """

    # Find the "In-Store Pickup" label as a text node
    pickup_node = soup.find(
        string=lambda t: isinstance(t, NavigableString)
        and "In-Store Pickup" in t
    )

    # If we can't find it, don't over-filter; assume it's fine
    if not pickup_node:
        return True

    # Walk up ancestors to find the "card" that contains the pickup text
    current = pickup_node.parent
    for _ in range(5):
        if current is None:
            break

        text = current.get_text(separator=" ", strip=True).lower()

        if "in-store pickup" in text:
            # Phrases that mean "not actually in stock at this store"
            bad_phrases = [
                "item will be shipped and should arrive in",
                "not available at this store",
                "online only",
                "not sold in stores",
            ]
            if any(p in text for p in bad_phrases):
                return False

            # No bad phrase in the pickup block -> treat as in-store available
            return True

        current = current.parent

    # Fallback if we never find a good container
    return True


# ---------------------- PARSING HELPERS ----------------------


def extract_sku_from_text(text: str) -> Optional[str]:
    """Look for 'SKU: 101363893' style patterns in the raw HTML."""
    m = re.search(r"SKU[:\s]+(\d{6,})", text)
    if m:
        return m.group(1)
    return None


def extract_sku_from_url(url: str) -> Optional[str]:
    """Fallback: get SKU from the trailing '-digits.html' in the URL."""
    m = re.search(r"-(\d{6,})\.html", url)
    if m:
        return m.group(1)
    return None


def extract_product_image_url(soup: BeautifulSoup, sku: Optional[str]) -> Optional[str]:
    """
    Try several strategies to get the *actual* product hero image.

    Order:
      1) Any <img> whose src contains the SKU (most specific)
      2) JSON-LD Product schema "image" field
      3) <meta property="og:image"> (but skip generic nav banners)
      4) Heuristic scan of remaining <img> tags (skipping logos/icons/banners)
    """

    def is_bad_image_url(url: str) -> bool:
        """Filter out obvious non-product or generic banner assets."""
        low = url.lower()
        # Generic/global stuff we never want
        bad_tokens = [
            "logo",
            "icon",
            "sprite",
            "facebook",
            "twitter",
            "pinterest",
            "instagram",
            "youtube",
            "favicon",
        ]
        if any(t in low for t in bad_tokens):
            return True

        # The generic bathroom inspiration banner that was showing up everywhere
        if "bathroom_inspriationnavigation" in low or "inspirationnavigation" in low:
            return True

        return False

    # 1) If we know the SKU, look for img src that contains it
    if sku:
        for img in soup.find_all("img"):
            src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-lazy")
                or img.get("data-original")
            )
            if not src:
                continue
            src = src.strip()
            if not src or src.startswith("data:"):
                continue

            full = urljoin(BASE_URL, src)
            if sku in full and not is_bad_image_url(full):
                return full

    # 2) JSON-LD Product schema with an "image" field
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or script.get_text(strip=True)
            if not raw:
                continue
            data = json.loads(raw)
        except Exception:
            continue

        def find_product_image(obj):
            if isinstance(obj, dict):
                if obj.get("@type") == "Product" and obj.get("image"):
                    img = obj["image"]
                    if isinstance(img, list):
                        img = img[0]
                    return img
                for v in obj.values():
                    res = find_product_image(v)
                    if res:
                        return res
            elif isinstance(obj, list):
                for v in obj:
                    res = find_product_image(v)
                    if res:
                        return res
            return None

        img = find_product_image(data)
        if img:
            full = urljoin(BASE_URL, img)
            if not is_bad_image_url(full):
                return full

    # 3) og:image – skip if it's a generic banner
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        candidate = og["content"].strip()
        if candidate:
            full = urljoin(BASE_URL, candidate)
            if not is_bad_image_url(full):
                return full

    # 4) Fallback: scan <img> tags and pick a reasonable candidate
    candidates: List[str] = []
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy")
            or img.get("data-original")
        )
        if not src:
            continue
        src = src.strip()
        if not src or src.startswith("data:"):
            continue

        full = urljoin(BASE_URL, src)
        if is_bad_image_url(full):
            continue

        candidates.append(full)

    if candidates:
        return candidates[0]

    return None


# ---------------------- PRODUCT PAGE PARSING ----------------------


def parse_product_page(
    session: requests.Session, url: str, category_slug: str
) -> Optional[Dict[str, str]]:
    """Fetch and parse a single product page into a metadata dict."""
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logging.warning("Failed to fetch product %s: %s", url, e)
        return None

    if r.status_code != 200:
        logging.warning("Product %s returned status %s", url, r.status_code)
        return None

    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # ---- AVAILABILITY FILTER ----
    if not is_available_in_store(soup):
        logging.info("Skipping %s (not stocked at store %s)", url, STORE_ID)
        return None
    # -----------------------------

    # --- SKU ---
    sku = extract_sku_from_text(html)
    if not sku:
        sku = extract_sku_from_url(url)
    if not sku:
        logging.debug("Could not extract SKU from %s", url)
        return None

    # --- Name ---
    name: Optional[str] = None
    h1 = soup.find("h1")
    if h1:
        name = h1.get_text(strip=True)

    if not name:
        meta_title = soup.find("meta", property="og:title")
        if meta_title and meta_title.get("content"):
            name = meta_title["content"].strip()

    if not name:
        # Fallback: use URL slug
        name = (
            url.split("/")[-1]
            .split(".html")[0]
            .replace("-", " ")
            .title()
        )

    # --- Filter out unwanted product types by name / URL ---
    combined_for_filter = " ".join(
        part for part in [name, category_slug, url] if part
    ).lower()

    if any(tok in combined_for_filter for tok in EXCLUDE_PRODUCT_NAME_TOKENS):
        logging.info("Skipping %s (excluded product type: %s)", url, name)
        return None

    if url_is_excluded(url):
        logging.info("Skipping %s (excluded by URL path)", url)
        return None

    # --- Image URL ---
    image_url = extract_product_image_url(soup, sku)
    logging.debug("SKU %s -> image_url: %s", sku, image_url)

    row: Dict[str, str] = {
        "sku": sku,
        "name": name,
        "category_slug": category_slug,
        "product_url": url,
        "image_url": image_url or "",
        # image_filename set below
    }

    # Default filename for new rows if we have an image
    if row["image_url"]:
        row["image_filename"] = f"{sku}.jpg"
    else:
        row["image_filename"] = ""

    # Extra fields start blank for new products
    for extra in [
        "row_id",
        "cluster_id",
        "pca_x",
        "pca_y",
        "surface_type",
        "material_group",
        "cluster_size",
    ]:
        row.setdefault(extra, "")

    return row


def scrape_products(
    session: requests.Session, product_urls_by_category: Dict[str, Set[str]]
) -> List[Dict[str, str]]:
    """Scrape all product pages and return a list of metadata dicts."""
    rows: List[Dict[str, str]] = []
    seen_skus: Set[str] = set()

    for category, urls in product_urls_by_category.items():
        logging.info("Scraping %d products for category %s", len(urls), category)
        for url in urls:
            meta = parse_product_page(session, url, category)
            if not meta:
                continue

            sku = meta["sku"]
            if sku in seen_skus:
                continue
            seen_skus.add(sku)

            rows.append(meta)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info("Parsed %d unique products successfully", len(rows))
    return rows


# ---------------------- OUTPUT HELPERS ----------------------


def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_metadata(rows: List[Dict[str, str]], csv_path: str) -> None:
    """
    Write out *all* metadata.

    - Uses REQUIRED_FIELD_ORDER first, then any extra columns found in the rows.
    - For every row, we make sure every fieldname exists (missing -> "").
    - This preserves your clustering columns (and any other notebook columns)
      instead of wiping them out.
    """
    if not rows:
        logging.info("No rows to save, skipping CSV write.")
        return

    fieldnames = compute_fieldnames(rows)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            out_row = {}
            for col in fieldnames:
                out_row[col] = row.get(col, "")
            writer.writerow(out_row)

    logging.info("Metadata written to %s (%d rows, %d columns)", csv_path, len(rows), len(fieldnames))


def download_images(session: requests.Session, rows: List[Dict[str, str]]) -> None:
    if not rows:
        logging.info("No rows for image download.")
        return

    logging.info("Downloading images for %d products...", len(rows))
    for row in rows:
        sku = row["sku"]
        img_url = row.get("image_url")
        img_filename = row.get("image_filename")
        if not img_url or not img_filename:
            continue

        img_path = os.path.join(IMAGES_DIR, img_filename)

        # Skip if image already exists
        if os.path.exists(img_path):
            continue

        logging.info("Downloading image for SKU %s: %s", sku, img_url)
        try:
            resp = session.get(img_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                with open(img_path, "wb") as f:
                    f.write(resp.content)
            else:
                logging.warning(
                    "Image download failed for SKU %s (%s): status %s",
                    sku,
                    img_url,
                    resp.status_code,
                )
        except Exception as e:
            logging.warning("Exception downloading image for SKU %s: %s", sku, e)


def load_existing_metadata(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load existing CSV (if present) into a dict keyed by SKU.
    Keeps *all* columns found in the CSV.
    """
    existing: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(csv_path):
        return existing

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = row.get("sku")
            if not sku:
                continue
            existing[sku] = row
    logging.info("Loaded %d existing rows from %s", len(existing), csv_path)
    return existing


# ---------------------- MAIN ----------------------


def main() -> None:
    ensure_dirs()
    session = make_session()

    # Scope to San Leandro store (cookies / context for store #238)
    set_store_context(session)

    # 1) Load any existing metadata so we can merge
    existing_by_sku = load_existing_metadata(METADATA_CSV)

    # Only crawl tile, stone, wood, and decorative surface categories
    category_slugs = CATEGORY_SLUGS
    logging.info(
        "Using fixed surface category slugs (tile/stone/wood/decoratives): %s",
        category_slugs,
    )

    logging.info("Discovering product URLs from category pages...")
    product_urls_by_category = fetch_all_product_urls(session, category_slugs)

    logging.info("Starting product scrape...")
    scraped_rows = scrape_products(session, product_urls_by_category)

    # 2) Build dict for new scrape
    scraped_by_sku: Dict[str, Dict[str, str]] = {}
    for row in scraped_rows:
        sku = row.get("sku")
        if not sku:
            continue
        scraped_by_sku[sku] = row

    # 3) Figure out which SKUs are truly new
    new_skus = [sku for sku in scraped_by_sku.keys() if sku not in existing_by_sku]
    new_rows = [scraped_by_sku[sku] for sku in new_skus]
    logging.info("Found %d new SKUs (out of %d scraped)", len(new_rows), len(scraped_rows))

    # 4) Merge existing + scraped WITHOUT clobbering extra columns
    merged_by_sku: Dict[str, Dict[str, str]] = {}

    # Start with existing rows as-is
    for sku, row in existing_by_sku.items():
        merged_by_sku[sku] = row.copy()

    # Update/insert scraped rows
    base_fields_to_update = ["name", "category_slug", "product_url", "image_url", "image_filename"]

    for sku, scraped_row in scraped_by_sku.items():
        if sku in merged_by_sku:
            # Existing row: keep all existing keys, patch only base metadata if scraped has non-empty value
            existing_row = merged_by_sku[sku]
            for key in base_fields_to_update:
                val = scraped_row.get(key)
                if val:  # only overwrite if scraper found something non-empty
                    existing_row[key] = val
        else:
            # Brand new SKU: take scraped row, make sure extra fields exist
            new_row = scraped_row.copy()
            for extra in [
                "row_id",
                "cluster_id",
                "pca_x",
                "pca_y",
                "surface_type",
                "material_group",
                "cluster_size",
            ]:
                new_row.setdefault(extra, "")
            merged_by_sku[sku] = new_row

    merged_rows = list(merged_by_sku.values())

    # 5) Download images only for truly new SKUs
    download_images(session, new_rows)

    # 6) Save full merged metadata back to the same CSV
    logging.info("Saving metadata (merged existing + new)...")
    save_metadata(merged_rows, METADATA_CSV)

    logging.info(
        "Done. Metadata -> %s (%d products total), images -> %s/ (new: %d)",
        METADATA_CSV,
        len(merged_rows),
        IMAGES_DIR,
        len(new_rows),
    )


if __name__ == "__main__":
    main()
