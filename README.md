# OME Arrow benchmarks

Benchmarking [OME Arrow](https://github.com/WayScience/ome-arrow) through Parquet, Vortex, LanceDB, and more.

## Running benchmarks

1. Create and sync a uv environment (includes parquet, lancedb, vortex-data):

```bash
uv venv
uv sync
```

2. Launch Jupyter and open `notebooks/compare_parquet_vortex_lance.ipynb`:

```bash
uv run python <benchmark file>
```

The benchmarks defaults to ~100,000 rows x ~4,000 columns of `float64` data and ~50 columns of `string` data. Lower `N_ROWS`/`N_COLS` in the config cell if you hit memory pressure (especially before converting to pandas for the CSV benchmark).

An OME-Arrow variant lives at `notebooks/compare_parquet_vortex_lance_ome.py` which adds a single OME image column (random 100x100) alongside the existing columns.

An OME-Arrow-only + OME-Zarr benchmark lives at `notebooks/compare_ome_arrow_only.pyt`, focusing on a single OME image column and a directory-per-image OME-Zarr comparison.
