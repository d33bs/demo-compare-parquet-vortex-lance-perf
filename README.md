# demo-compare-parquet-vortex-lance-perf

Roughly comparing Parquet, Vortex, and Lance performance on a wide synthetic dataset.

## Running the notebook

1) Create and sync a uv environment (includes parquet, lancedb, vortex-data):

```bash
uv venv
uv sync
```

2) Launch Jupyter and open `notebooks/compare_parquet_vortex_lance.ipynb`:

```bash
uv run jupyter lab
```

The notebook defaults to ~100,000 rows x ~4,000 columns of `float64` data and ~50 columns of `string` data. Lower `N_ROWS`/`N_COLS` in the config cell if you hit memory pressure (especially before converting to pandas for the CSV benchmark).

An OME-Arrow variant lives at `notebooks/compare_parquet_vortex_lance_ome.ipynb` (or `.py` via jupytext) which adds a single OME image column (random 100x100) alongside the existing columns.

## Notes on Vortex

The notebook uses `vortex.io.write` to persist data and `vortex.open(...).to_arrow().read_all()` to read.

## Included formats

- Parquet (pyarrow, zstd)
- Lance (lancedb)
- Vortex
- CSV (pandas, gzip)
- DuckDB table
