# backend/app.py
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from .model import get_remote_image_url, init_model, similar_by_sku

REMOTE_IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.flooranddecor.com/",
}

app = FastAPI(
    title="Mat-Vis-Net API",
    description="SKU similarity search for Floor & Decor San Leandro",
)

# CORS so your web/mobile app can hit it during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Product(BaseModel):
    sku: str
    name: str
    category_slug: str | None = None
    material_bucket: str | None = None
    image_filename: str | None = None
    distance: float | None = None
    image_url: str | None = None
    image_proxy_url: str | None = None


class SimilarResponse(BaseModel):
    query: Product
    results: List[Product]


@app.on_event("startup")
def on_startup() -> None:
    init_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/similar-by-sku", response_model=SimilarResponse)
def similar_by_sku_endpoint(sku: str, k: int = 5):
    try:
        payload = similar_by_sku(sku, top_k=k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def with_proxy_metadata(prod: Dict[str, Any]) -> Dict[str, Any]:
        image_filename = prod.get("image_filename")
        proxy = f"/images/{image_filename}" if image_filename else None
        return {**prod, "image_proxy_url": proxy}

    query = with_proxy_metadata(payload["query"])
    results = [with_proxy_metadata(r) for r in payload["results"]]

    return {"query": query, "results": results}


@app.get("/images/{image_id}")
async def image_proxy(image_id: str):
    remote_url = get_remote_image_url(image_id)
    if not remote_url:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                remote_url,
                headers=REMOTE_IMAGE_HEADERS,
                follow_redirects=True,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {exc}") from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Upstream image fetch failed ({resp.status_code})",
        )

    headers = {"cache-control": resp.headers.get("cache-control", "public, max-age=86400")}
    content_type = resp.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    return Response(content=resp.content, headers=headers)