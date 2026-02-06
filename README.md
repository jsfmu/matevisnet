# Material Visual Network

Image-based product matcher for flooring and tile — given a photo of a plank, tile, or decorative piece, return the closest matching SKUs from a store catalog (currently Floor & Decor, San Leandro – Store 238).

---

## Getting Started (TL;DR)

### 1. Clone the repository

```
git clone <your-repo-url>.git
cd Mat-Vis-Net
```

### 2. Set up the Python environment

```
python -m venv venv
```

Activate it:

- Windows: `venv\Scripts\activate`
- macOS / Linux: `source venv/bin/activate`

Install backend dependencies:

```
pip install fastapi "uvicorn[standard]" numpy pandas scikit-learn
```

### 3. Pull the latest Store 238 catalog

```
python scrape_238_products.py
```

The scraper narrows the crawl to surface materials, enforces in-store availability, updates `data/san_leandro_products.csv`, and downloads product photos into `images/`.

### 4. Build image embeddings

Use `notebooks/01_embeddings_and_knn.ipynb` to generate `data/image_embs.npy` aligned with the CSV rows (automation script coming soon). The FastAPI service expects both the CSV and the embedding file.

### 5. Start the API

```
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010
```

The service exposes `/similar-by-sku` and serves product images at `/images/<filename>`.

### 6. Launch the web client

```
cd matvis-web
npm install
npm run dev
```

By default the React app talks to `http://127.0.0.1:8010`. Set `VITE_API_BASE_URL` in `matvis-web/.env.local` if you run the API elsewhere.

---

## Overview

Material Visual Network is an experiment in visual search for building materials.

Given a customer photo (phone picture of a plank, tile, or decorative piece), the system aims to:

- Identify what kind of product it is (wood, porcelain tile, vinyl plank, decorative, etc.).
- Suggest visually similar products based on color, texture, pattern, and finish.
- Map those matches back to **real SKUs** in a specific store’s catalog.
- Act as a building block for an assistant that can answer questions like:
  - “What in-stock products look like this?”
  - “What’s the closest match to this plank in the San Leandro store?”
  - “Show me alternatives that have a similar look, but are porcelain instead of wood.”

The current focus is on:

- Building a clean **image + metadata dataset** from a single Floor & Decor location.
- Prototyping a **visual embedding pipeline** using a pretrained vision–language model.
- Using **nearest-neighbor search** over those embeddings to retrieve the closest products.
- Understanding the structure of the catalog in embedding space (e.g., how wood vs tile vs stone cluster).

---

## Data Pipeline

### 1. Store-specific scraper (`scrape_238_products.py`)

The scraper is designed to maintain a high-quality, surface-focused dataset:

- **Scope**: only surface categories:
  - `/tile`, `/stone`, `/wood`, `/decoratives`
- **Exclusions by URL / category**:
  - Installation materials (thinset, mortar, grout, adhesives, membranes, underlayments)
  - Tools and accessories (trowels, saws, spacers, etc.)
  - Vanities, countertops, slabs
  - Doors and cabinet hardware (knobs, pulls, handles)
- **Availability check**:
  - Uses the “In-Store Pickup” section to distinguish true in-store stock from ship-to-store / online-only items.
  - Skips products that are not actually available at Store 238.
- **Metadata extraction**:
  - SKU
  - Product name
  - `category_slug`
  - Product URL
  - Best-guess product hero image URL
- **Image download**:
  - Saves images as `images/<SKU>.jpg`.
  - Skips generic banners, logos, and navigation assets.
  - Only downloads images for **new** SKUs so you don’t redownload everything on every run.
- **CSV merge behavior**:
  - Loads any existing `data/san_leandro_products.csv`.
  - Preserves all existing columns (including analysis columns such as `row_id`, `cluster_id`, `pca_x`, `pca_y`, `surface_type`, `material_group`, `cluster_size`).
  - Updates base metadata (`name`, `category_slug`, `product_url`, `image_url`, `image_filename`) when better data is found.
  - Appends truly new SKUs, initializing analysis fields as empty strings for later filling.

### 2. Embeddings & kNN index (`notebooks/01_embeddings_and_knn.ipynb`)

This notebook builds the core vector space that powers the matcher:

- Loads `data/san_leandro_products.csv` and filters to rows with valid `image_filename`.
- Uses a pretrained **CLIP-style vision encoder** to embed each product image into a high-dimensional vector.
- Stacks embeddings into a matrix aligned with the rows of the CSV.
- Builds a **k-Nearest Neighbors** index (e.g., scikit-learn’s `NearestNeighbors`) using **cosine distance**.
- Demonstrates query flow:
  - Load a query image (either a product image or a user photo).
  - Compute its embedding.
  - Run kNN search to get top-K nearest products.
  - Post-filter results by material or other metadata if desired.
  - Optionally drop exact self-matches so recommendations show **alternatives**, not just the same SKU.

### 3. Clustering & visualization (`notebooks/02_clustering_and_viz.ipynb`)

This notebook focuses on understanding the structure of the catalog in embedding space:

- Runs **PCA** on the embedding matrix to reduce to 2-D for a global, linear view.
- Optionally runs **UMAP / t-SNE** for a nonlinear, neighborhood-preserving view.
- Colors points by:
  - `material_group` (e.g., wood, tile/porcelain, stone, decorative, other)
  - `cluster_id` from k-means or similar clustering.
- Observations from early plots (for Floor & Decor San Leandro):
  - Wood SKUs tend to form a distinct, compact cluster.
  - Tile/porcelain products are relatively concentrated.
  - Stone products are more spread out, reflecting higher visual diversity.

These visualizations support storytelling such as:

- “Off-the-shelf CLIP embeddings already separate wood from other materials.”
- “Tile/porcelain occupy a tight region in embedding space, while stone spans a broader manifold.”

---

## Backend API

- `backend/model.py` loads `data/san_leandro_products.csv` together with `data/image_embs.npy`, filters out rows without images, tags each item with a coarse `material_bucket`, and builds a cosine kNN index using scikit-learn.
- `backend/app.py` boots a FastAPI service that:
  - initialises the model on startup,
  - exposes `GET /similar-by-sku?sku=<SKU>&k=<int>` returning the query product plus neighbours (including distances and `image_url` values),
  - exposes `GET /health` for simple probes, and
  - mounts `/images` so the web client can render product photography straight from disk.
- Run it locally with `python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010`. Adjust the host/port as needed and set `VITE_API_BASE_URL` for the frontend if you diverge from that default.

---

## Web UI

- `matvis-web/` is a Vite + React app that lets you enter a SKU and browse similar products.
- `npm run dev` serves the UI on `http://localhost:5173` and forwards API calls to the base URL defined by `VITE_API_BASE_URL` (defaults to `http://127.0.0.1:8010`).
- Cards render the query result and neighbours with SKU, product name, category slug, material bucket, cosine distance, and the hero image streamed from FastAPI (`/images/<filename>`).
- For a production build, run `npm run build` and deploy the generated static assets behind a proxy that routes API calls to the FastAPI service.

---

## Algorithm

The retrieval pipeline is a **content-based image retrieval** system on top of a pretrained vision–language model and a kNN index.

1. **Product Image Embedding**

   - Each product image is passed through a pretrained CLIP-style vision encoder.
   - The encoder outputs a fixed-length embedding vector.
   - These embeddings capture visual patterns (color, texture, pattern, overall style) without relying solely on text descriptions.

2. **Catalog Index Construction**

   - All product embeddings are stacked into a single matrix.
   - A k-Nearest Neighbors index is built over this matrix using **cosine distance**:
     - `NearestNeighbors(metric="cosine")` or a similar implementation.
   - This index supports efficient retrieval of visually similar products.

3. **Similarity Search & Post-Filtering**

   For a given query image:

   - Compute the query embedding with the same encoder.
   - Use the kNN index to find the top-K nearest product embeddings.
   - Apply business rules and metadata filters, for example:
     - Match on `material_group` (e.g., “only return wood if the input is wood”).
     - Restrict to in-stock products for Store 238.
     - Exclude the **exact same SKU** when you want recommendations rather than echoing the input.
   - Return a ranked list of candidate SKUs along with distances and metadata.

Effectively, the system does a **vector similarity search** over the catalog: products that look similar live near each other in embedding space, and the kNN lookup surfaces those neighbors.

---

## Current Status

What’s implemented:

- Store-aware scraper for **Floor & Decor San Leandro (Store 238)**.
- Surface-focused dataset:
  - Filters out installation materials, vanities, countertops, doors, and hardware.
  - Keeps tile, stone, wood, and decorative surface products.
- Robust metadata + image download:
  - Merges into a single CSV with stable columns.
  - Preserves analysis fields and only appends new SKUs.
- Prototype notebooks for:
  - Embedding product images with a CLIP-style encoder.
  - Building a kNN index for similarity search.
  - Visualizing embeddings with PCA and clustering by material.
- FastAPI service (`backend/`) that loads pre-computed embeddings, hosts `/similar-by-sku`, serves `/images`, and provides a `/health` probe.
- React/Vite dev client (`matvis-web/`) that queries the API, renders similarity results, and can be built for static hosting.

Planned / in progress:

- More complete metadata schema:
  - Material, finish, size, price, coverage, in-stock flags, etc.
- Stable embedding pipeline (CLI / script) outside the notebooks.
- Image-based search endpoint (e.g., `POST /search/image`) with automatic embedding.
- FAISS or another ANN backend for scalable similarity search.
- Enhanced UI flows (image upload, filters, analytics dashboards).
- More advanced clustering & analysis:
  - Identify “visual families” inside each material group.
  - Use clusters to drive faceted browsing and recommendations.

---

## Tech Stack

In use:

- Python
- Data collection: `requests`, `beautifulsoup4`
- Data handling: `pandas`, `csv`, standard library
- Embeddings: CLIP-style vision encoder (via Hugging Face or similar)
- Similarity search: scikit-learn `NearestNeighbors`
- API: FastAPI + Uvicorn
- Frontend: React + Vite
- Exploration / analysis: Jupyter Notebook, PCA, clustering

Planned / in progress:

- Vector search: FAISS or similar for large catalogs
- Infra: deployment scripts and configs under `infra/`

---

## Repository Structure

    .
    ├── venv/                         # Local Python virtual environment (ignored by git)
    ├── backend/
    │   ├── app.py                    # FastAPI app (health check, similar-by-sku, static images)
    │   └── model.py                  # CSV/embedding loader, kNN search helpers
    ├── apps/
    │   └── client/                   # Legacy HTML upload tester (kept for reference)
    ├── data/
    │   └── san_leandro_products.csv  # Scraped product metadata for Store 238
    ├── images/
    │   └── <SKU>.jpg                 # Downloaded product images named by SKU
    ├── infra/                        # Infrastructure and deployment configuration (WIP)
    ├── matvis-mobile/                # Expo prototype (placeholder)
    ├── matvis-web/                   # React + Vite SKU similarity client
    │   ├── src/
    │   │   ├── assets/App.jsx        # UI rendering similar results
    │   │   └── main.jsx              # React entry point
    │   ├── package.json
    │   └── vite.config.js
    ├── notebooks/
    │   ├── 01_embeddings_and_knn.ipynb   # Build embeddings + kNN index
    │   └── 02_clustering_and_viz.ipynb   # PCA/UMAP + clustering visualizations
    ├── .gitignore                    # Git ignore rules
    ├── flow.drawio.png               # System / data flow diagram
    ├── scrape_238_products.py        # Scraper for Floor & Decor Store 238
    └── README.md                     # Project documentation (this file)

Core workflow:

1. Run `scrape_238_products.py` to refresh the dataset (metadata + images).  
2. Generate embeddings via `notebooks/01_embeddings_and_knn.ipynb` and save them as `data/image_embs.npy`.  
3. Start the FastAPI service (`python -m uvicorn backend.app:app --host 127.0.0.1 --port 8010`).  
4. Launch the React client from `matvis-web/` (`npm run dev`) to explore results interactively.  
5. Use `notebooks/02_clustering_and_viz.ipynb` for deeper analysis and storytelling.
