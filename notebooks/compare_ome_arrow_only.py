# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # OME-Arrow single-column benchmarks
#
# Compare on-disk formats for a table containing only `row_id` and one OME-Arrow image column, plus a separate benchmark for a directory-per-image OME-Zarr layout.
# Measures: write all, read all, random-read some, size on disk.
#
# Requires `pyarrow`, `lancedb`, `vortex-data`, `duckdb`, `ome-arrow` (installed via uv). OME-Zarr export requires optional deps (`bioio-ome-zarr`, `zarr`, `numcodecs`).
#
# **Setup**
# - Run `uv run poe lab` (or `uv venv && uv sync && uv run jupyter lab`).
# - Artifacts are written under `data/` (git-ignored).
# - If you want the OME-Zarr timings, install the extra deps: `uv pip install bioio-ome-zarr zarr numcodecs`.

# +
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import lancedb
import duckdb
import vortex
import vortex.io as vxio
from ome_arrow import OMEArrow

pd.set_option("display.precision", 4)


# +
DATA_DIR = Path("data")
shutil.rmtree(DATA_DIR, ignore_errors=True)
DATA_DIR.mkdir(exist_ok=True)

N_ROWS = 1_000
OME_SHAPE = (100, 100)
OME_DTYPE = np.uint8
REPEATS = 5
RANDOM_READ_REPEATS = REPEATS
RANDOM_ROW_COUNT = 10
SEED = 13

RUN_DUCKDB = True
DUCK_ROWS = None  # use full table
DUCK_REPEATS = REPEATS

PARQUET_PATH = DATA_DIR / "ome_only.parquet"
PARQUET_DUCK_PATH = DATA_DIR / "ome_only_duck.parquet"
LANCE_PATH = DATA_DIR / "ome_only_lance"
VORTEX_PATH = DATA_DIR / "ome_only.vortex"
DUCK_PATH = DATA_DIR / "ome_only.duckdb"
OME_ZARR_DIR = DATA_DIR / "ome_zarr_runs"
LANCE_TABLE = "bench"
DUCK_TABLE = "bench"

# Resolve versions
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


def _pkg_version(name: str, default: str = 'missing') -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return default

try:
    vortex_version = getattr(vortex, '__version__', None) or importlib_metadata.version('vortex-data')
except importlib_metadata.PackageNotFoundError:
    try:
        vortex_version = importlib_metadata.version('vortex')
    except importlib_metadata.PackageNotFoundError:
        vortex_version = 'unknown'

VERSIONS = {
    'pyarrow': pa.__version__,
    'lancedb': getattr(lancedb, '__version__', 'unknown'),
    'duckdb': duckdb.__version__,
    'vortex': vortex_version,
    'ome-arrow': getattr(__import__('ome_arrow'), '__version__', 'unknown'),
    'bioio_ome_zarr': _pkg_version('bioio-ome-zarr'),
    'zarr': _pkg_version('zarr'),
    'numcodecs': _pkg_version('numcodecs'),
}
FORMAT_VERSIONS = {
    'Parquet (pyarrow, zstd)': f"pyarrow {VERSIONS['pyarrow']}",
    'Parquet (duckdb, zstd)': f"duckdb {VERSIONS['duckdb']}",
    'Lance (lancedb)': f"lancedb {VERSIONS['lancedb']}",
    'Vortex': f"vortex {VERSIONS['vortex']}",
    'DuckDB (file table)': f"duckdb {VERSIONS['duckdb']}",
}


# +
rng = np.random.default_rng(SEED)
# Add an explicit row id column so we can filter for random access reads later.
row_ids = pa.array(np.arange(N_ROWS, dtype=np.int64))

# Build the OME-Arrow column with random images.
# OMEArrow defaults to dim_order="TCZYX", so we supply a 5D array with singleton T,C,Z.
ome_pylist = []
for _ in range(N_ROWS):
    img = rng.integers(0, 256, size=(1, 1, 1, *OME_SHAPE), dtype=OME_DTYPE)
    ome_scalar = OMEArrow(data=img).data.as_py()
    ome_scalar.pop('masks', None)  # drop Null field to avoid Arrow null append issue
    ome_pylist.append(ome_scalar)
ome_column = pa.array(ome_pylist)

column_names = ['ome_image']
columns = [row_ids, ome_column]
column_names = ['row_id'] + column_names
table = pa.Table.from_arrays(columns, names=column_names)
print('table rows:', table.num_rows, 'cols:', table.num_columns)

# -

duck_table = table
RANDOM_INDICES = sorted(rng.choice(table.num_rows, size=RANDOM_ROW_COUNT, replace=False).tolist())


# +
def drop_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    return 0


def run_benchmarks(table: pa.Table, configs, repeats: int = 3):
    results = []
    for cfg in configs:
        cfg_repeats = cfg.get('repeats', repeats)
        cfg_random_repeats = cfg.get('random_repeats', cfg_repeats)
        write_times = []
        read_times = []
        random_read_times = []
        print(f"[format start] {cfg['name']}", flush=True)
        for run_idx in range(cfg_repeats):
            if cfg.get('cleanup', True):
                drop_path(cfg['path'])
            t0 = time.perf_counter()
            cfg_table = cfg.get('table', table)
            cfg['write'](cfg_table, cfg['path'])
            elapsed = time.perf_counter() - t0
            write_times.append(elapsed)
            print(f"[write] {cfg['name']} run {run_idx + 1}/{cfg_repeats}: {elapsed:.2f}s", flush=True)

        size_bytes = path_size_bytes(cfg['path'])

        for run_idx in range(cfg_repeats):
            gc.collect()
            t0 = time.perf_counter()
            _ = cfg['read'](cfg['path'])
            elapsed = time.perf_counter() - t0
            read_times.append(elapsed)
            print(f"[read ] {cfg['name']} run {run_idx + 1}/{cfg_repeats}: {elapsed:.2f}s", flush=True)

        for run_idx in range(cfg_random_repeats):
            gc.collect()
            t0 = time.perf_counter()
            random_fn = cfg.get('random_read')
            if random_fn is None:
                tbl = cfg['read'](cfg['path'])
                _ = tbl.take(RANDOM_INDICES)
            else:
                _ = random_fn(cfg['path'], indices=RANDOM_INDICES)
            elapsed = time.perf_counter() - t0
            random_read_times.append(elapsed)
            print(f"[rnd  ] {cfg['name']} run {run_idx + 1}/{cfg_random_repeats}: {elapsed:.2f}s", flush=True)

        results.append({
            'format': cfg['name'],
            'write_seconds': write_times,
            'read_seconds': read_times,
            'random_read_seconds': random_read_times,
            'size_mb': size_bytes / (1024 * 1024),
        })
        print(f"[format end] {cfg['name']}", flush=True)
    return results



# +
drop_path(LANCE_PATH)
LANCE_DB = lancedb.connect(LANCE_PATH)

def reset_lance_table(db, table_name):
    try:
        if hasattr(db, 'table_names') and table_name in db.table_names():
            if hasattr(db, 'drop_table'):
                db.drop_table(table_name)
            else:
                drop_path(LANCE_PATH)
                return lancedb.connect(LANCE_PATH)
    except Exception:
        drop_path(LANCE_PATH)
        return lancedb.connect(LANCE_PATH)
    return db



# +
def lance_write(tbl, path=LANCE_PATH, table_name=LANCE_TABLE):
    global LANCE_DB
    LANCE_DB = reset_lance_table(LANCE_DB, table_name)
    LANCE_DB.create_table(table_name, tbl, mode="overwrite")


def lance_read(path=LANCE_PATH, table_name=LANCE_TABLE):
    return LANCE_DB.open_table(table_name).to_arrow()


def vortex_write(tbl, path=VORTEX_PATH):
    drop_path(path)
    vxio.write(tbl, str(path))


def vortex_read(path=VORTEX_PATH):
    return vortex.open(str(path)).to_arrow().read_all()


def duck_write(tbl, path=DUCK_PATH, table_name=DUCK_TABLE):
    drop_path(path)
    con = duckdb.connect(str(path))
    con.register('tmp_tbl', tbl)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM tmp_tbl")
    con.close()


def duck_read(path=DUCK_PATH, table_name=DUCK_TABLE):
    with duckdb.connect(str(path)) as con:
        return con.execute(f"SELECT * FROM {table_name}").fetch_arrow_table()


def parquet_duck_write(tbl, path=PARQUET_DUCK_PATH):
    drop_path(path)
    con = duckdb.connect()
    con.register('tmp_tbl', tbl)
    con.execute(f"COPY (SELECT * FROM tmp_tbl) TO '{path}' WITH (FORMAT 'PARQUET', COMPRESSION 'ZSTD')")
    con.close()


def parquet_duck_read(path=PARQUET_DUCK_PATH):
    with duckdb.connect() as con:
        return con.execute(f"SELECT * FROM read_parquet('{path}')").fetch_arrow_table()


def parquet_random_read(path=PARQUET_PATH, indices=None):
    try:
        dataset = ds.dataset(path, format="parquet")
        filt = ds.field('row_id').isin(pa.array(indices, type=pa.int64()))
        return dataset.to_table(filter=filt)
    except Exception:
        return pq.read_table(path).take(indices)


def parquet_duck_random_read(path=PARQUET_DUCK_PATH, indices=None):
    idx_list = ",".join(str(int(i)) for i in indices)
    with duckdb.connect() as con:
        return con.execute(f"SELECT * FROM read_parquet('{path}') WHERE row_id IN ({idx_list})").fetch_arrow_table()


def lance_random_read(path=LANCE_PATH, table_name=LANCE_TABLE, indices=None):
    idx_list = ",".join(str(int(i)) for i in indices)
    table = LANCE_DB.open_table(table_name)
    try:
        return table.query().where(f"row_id IN ({idx_list})").to_arrow()
    except Exception:
        return table.to_arrow().take(indices)


def vortex_random_read(path=VORTEX_PATH, indices=None):
    reader = vortex.open(str(path)).to_arrow()
    target = sorted(int(i) for i in indices)
    collected = None
    offset = 0
    for batch in reader:
        batch_len = batch.num_rows
        batch_matches = [idx - offset for idx in target if offset <= idx < offset + batch_len]
        if not batch_matches:
            offset += batch_len
            continue
        if collected is None:
            collected = {name: [] for name in batch.schema.names}
        for name, column in zip(batch.schema.names, batch.columns):
            for rel_idx in batch_matches:
                collected[name].append(column[rel_idx].as_py())
        offset += batch_len
        if len(next(iter(collected.values()))) >= len(target):
            break
    if not collected:
        return pa.Table.from_arrays([], names=[])
    arrays = [pa.array(collected[name]) for name in collected.keys()]
    return pa.Table.from_arrays(arrays, names=list(collected.keys()))


def duck_random_read(path=DUCK_PATH, table_name=DUCK_TABLE, indices=None):
    idx_list = ",".join(str(int(i)) for i in indices)
    with duckdb.connect(str(path)) as con:
        return con.execute(f"SELECT * FROM {table_name} WHERE row_id IN ({idx_list})").fetch_arrow_table()


# OME-Zarr helpers (dir-per-image)
def ome_zarr_deps_available() -> bool:
    try:
        import bioio_ome_zarr  # noqa: F401
        import zarr  # noqa: F401
        import numcodecs  # noqa: F401
        return True
    except Exception:
        return False


def ome_zarr_write_all(records, base_path=OME_ZARR_DIR):
    if not ome_zarr_deps_available():
        raise RuntimeError("OME-Zarr deps missing: install bioio-ome-zarr zarr numcodecs")
    drop_path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    from ome_arrow import to_ome_zarr
    for idx, rec in enumerate(records):
        out_dir = base_path / f"img_{idx:05d}.zarr"
        to_ome_zarr(rec, str(out_dir))


def ome_zarr_read_all(base_path=OME_ZARR_DIR):
    if not ome_zarr_deps_available():
        raise RuntimeError("OME-Zarr deps missing: install bioio-ome-zarr zarr numcodecs")
    from ome_arrow import from_ome_zarr
    out = []
    for zarr_dir in sorted(base_path.glob("*.zarr")):
        out.append(from_ome_zarr(str(zarr_dir)))
    return out


def ome_zarr_random_read(indices, base_path=OME_ZARR_DIR):
    if not ome_zarr_deps_available():
        raise RuntimeError("OME-Zarr deps missing: install bioio-ome-zarr zarr numcodecs")
    from ome_arrow import from_ome_zarr
    paths = sorted(base_path.glob("*.zarr"))
    out = []
    for idx in indices:
        if 0 <= idx < len(paths):
            out.append(from_ome_zarr(str(paths[idx])))
    return out


format_configs = [
    {
        'name': 'Parquet (pyarrow, zstd)',
        'path': PARQUET_PATH,
        'write': lambda tbl, path=PARQUET_PATH: pq.write_table(tbl, path, compression='zstd'),
        'read': lambda path=PARQUET_PATH: pq.read_table(path),
        'random_read': parquet_random_read,
        'random_repeats': RANDOM_READ_REPEATS,
    },
    {
        'name': 'Parquet (duckdb, zstd)',
        'path': PARQUET_DUCK_PATH,
        'write': parquet_duck_write,
        'read': parquet_duck_read,
        'random_read': parquet_duck_random_read,
        'random_repeats': RANDOM_READ_REPEATS,
    },
    {
        'name': 'Lance (lancedb)',
        'path': LANCE_PATH,
        'write': lance_write,
        'read': lance_read,
        'cleanup': False,
        'random_read': lance_random_read,
        'random_repeats': RANDOM_READ_REPEATS,
    },
    {
        'name': 'Vortex',
        'path': VORTEX_PATH,
        'write': vortex_write,
        'read': vortex_read,
        'random_read': vortex_random_read,
        'random_repeats': RANDOM_READ_REPEATS,
    },
    {
        'name': 'DuckDB (file table)',
        'path': DUCK_PATH,
        'write': duck_write,
        'read': duck_read,
        'table': duck_table,
        'repeats': DUCK_REPEATS,
        'random_read': duck_random_read,
        'random_repeats': RANDOM_READ_REPEATS,
    },
]

if ome_zarr_deps_available():
    format_configs.append({
        'name': 'OME-Zarr (dir-per-image)',
        'path': OME_ZARR_DIR,
        'write': lambda records, path=OME_ZARR_DIR: ome_zarr_write_all(records, path),
        'read': lambda path=OME_ZARR_DIR: ome_zarr_read_all(path),
        'random_read': lambda path=OME_ZARR_DIR, indices=None: ome_zarr_random_read(indices, path),
        'table': ome_pylist,  # run_benchmarks passes as cfg_table; here it's a list of records
        'random_repeats': RANDOM_READ_REPEATS,
        'version': (
            f"ome-arrow {VERSIONS.get('ome-arrow', '')}; "
            f"bioio-ome-zarr {VERSIONS.get('bioio_ome_zarr', '')}; "
            f"zarr {VERSIONS.get('zarr', '')}; "
            f"numcodecs {VERSIONS.get('numcodecs', '')}"
        ),
    })
else:
    print("Skipping OME-Zarr format (install bioio-ome-zarr zarr numcodecs to enable).")

print('Formats:', [cfg['name'] for cfg in format_configs])
# -

results = run_benchmarks(table, format_configs, repeats=REPEATS)
results_df = pd.DataFrame({
    'format': [r['format'] for r in results],
    'write_avg_s': [np.mean(r['write_seconds']) for r in results],
    'write_std_s': [np.std(r['write_seconds']) for r in results],
    'read_all_avg_s': [np.mean(r['read_seconds']) for r in results],
    'read_all_std_s': [np.std(r['read_seconds']) for r in results],
        'read_random_avg_s': [np.mean(r['random_read_seconds']) for r in results],
        'read_random_std_s': [np.std(r['random_read_seconds']) for r in results],
        'size_mb': [r['size_mb'] for r in results],
        'version': [r.get('version', FORMAT_VERSIONS.get(r['format'], '')) for r in results],
})
results_df


# +
timings = []
for r in results:
    for idx, t in enumerate(r['write_seconds']):
        timings.append({'format': r['format'], 'kind': 'write', 'run': idx, 'seconds': t})
    for idx, t in enumerate(r['read_seconds']):
        timings.append({'format': r['format'], 'kind': 'read', 'run': idx, 'seconds': t})
    for idx, t in enumerate(r['random_read_seconds']):
        timings.append({'format': r['format'], 'kind': 'random_read', 'run': idx, 'seconds': t})

pd.DataFrame(timings)
