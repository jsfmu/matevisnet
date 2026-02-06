import { useState } from "react";

const envApiBase = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE = envApiBase || "http://127.0.0.1:8000";

function App() {
  const [sku, setSku] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!sku.trim()) return;

    setLoading(true);
    setError("");
    setData(null);

    try {
      const params = new URLSearchParams({
        sku: sku.trim(),
        k: String(topK),
      });
      const res = await fetch(`${API_BASE}/similar-by-sku?${params.toString()}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "1.5rem" }}>
      <h1>Material Visual Network – SKU Matcher</h1>

      <form onSubmit={handleSearch} style={{ marginBottom: "1rem" }}>
        <label style={{ display: "block", marginBottom: "0.5rem" }}>
          Query SKU:
          <input
            value={sku}
            onChange={(e) => setSku(e.target.value)}
            placeholder="101156321"
            style={{ marginLeft: 8, padding: 4 }}
          />
        </label>

        <label style={{ display: "block", marginBottom: "0.5rem" }}>
          Top K:
          <input
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            style={{ marginLeft: 8, width: 60, padding: 4 }}
          />
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && (
        <div style={{ color: "red", marginBottom: "1rem" }}>
          Error: {error}
        </div>
      )}

      {data && (
        <>
          <section style={{ marginBottom: "1.5rem" }}>
            <h2>Query Product</h2>
            <ProductCard product={data.query} />
          </section>

          <section>
            <h2>Similar Products</h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                gap: "1rem",
              }}
            >
              {data.results.map((r) => (
                <ProductCard key={r.sku} product={r} />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

function ProductCard({ product }) {
  let imgUrl =
    product.image_proxy_url?.trim() ||
    product.image_url?.trim() ||
    null;
  if (imgUrl && !/^https?:\/\//i.test(imgUrl)) {
    const suffix = imgUrl.startsWith("/") ? imgUrl : `/${imgUrl}`;
    imgUrl = `${API_BASE}${suffix}`;
  }

  return (
    <div
      style={{
        border: "1px solid #ccc",
        borderRadius: 8,
        padding: 10,
      }}
    >
      {imgUrl && (
        <img
          src={imgUrl}
          alt={product.name}
          style={{
            width: "100%",
            height: 160,
            objectFit: "cover",
            borderRadius: 6,
            marginBottom: 8,
          }}
        />
      )}
      <div style={{ fontWeight: "bold" }}>{product.name}</div>
      <div>SKU: {product.sku}</div>
      {product.category_slug && (
        <div>Category: {product.category_slug}</div>
      )}
      {product.material_bucket && (
        <div>Bucket: {product.material_bucket}</div>
      )}
    </div>
  );
}

export default App;
