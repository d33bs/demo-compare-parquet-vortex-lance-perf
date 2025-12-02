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

The notebook defaults to ~100,000 rows x ~3,000 columns of `float32` data (~1.2 GB in-memory). Lower `N_ROWS`/`N_COLS` in the config cell if you hit memory pressure.

## Notes on Vortex

The notebook uses `vortex.io.write` to persist data and `vortex.open(...).to_arrow().read_all()` to read.
