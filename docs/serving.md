# Serving

## Local

```bash
uv run uvicorn brain_mri_segmentation.serving.main:app --host 0.0.0.0 --port 8000
```

Set `BRAIN_MRI_SEGMENTATION_CHECKPOINT=artifacts/checkpoints/best.ckpt` before launching so the app loads your trained weights.

## Docker

```bash
docker compose up api
```

Image: `ghcr.io/kiselyovd/brain-mri-segmentation`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/segment` | Run segmentation on one image |
| `GET` | `/metrics` | Prometheus metrics |

### `GET /health`

```json
{"status": "ok"}
```

### `POST /segment`

Multipart upload of one image (RGB PNG, JPEG, or TIF; resized server-side to 256 × 256):

```bash
curl -X POST -F "file=@slice.png" http://localhost:8000/segment
```

Response — binary mask as a flat list of 0/1 pixel values (row-major, 256 × 256):

```json
{
  "mask": [[0, 0, 1, ...], ...],
  "shape": [256, 256]
}
```

### `GET /metrics`

Prometheus metrics including request count, latency histograms, and in-flight counts. Wire to Grafana via the standard scrape endpoint.

## Headers

Every response carries `X-Request-ID` for log correlation — propagate it from your upstream gateway to make traces end-to-end.
