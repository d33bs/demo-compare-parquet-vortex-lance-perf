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

# # Parquet vs. Vortex vs. Lance performance
#
# Generate a wide random dataset (~100k rows x 4k float columns + 50 string columns) and benchmark on-disk formats.
#
# Requires `pyarrow`, `lancedb`, `vortex-data`, `duckdb` (installed via uv).
#

# **Setup**
# - Run `uv run poe lab` (or `uv venv && uv sync && uv run jupyter lab`).
# - Artifacts are written under `data/` (git-ignored).
#

# +
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lancedb
import duckdb
import vortex
import vortex.io as vxio

pd.set_option("display.precision", 4)


# +
DATA_DIR = Path("data")
shutil.rmtree(DATA_DIR, ignore_errors=True)
DATA_DIR.mkdir(exist_ok=True)

N_ROWS = 100_000
N_COLS = 4_000  # float columns
STR_COLS = 50
DTYPE = np.float64
REPEATS = 5
SEED = 13

RUN_DUCKDB = True
DUCK_ROWS = None  # use full table
DUCK_REPEATS = REPEATS

PARQUET_PATH = DATA_DIR / "wide.parquet"
PARQUET_DUCK_PATH = DATA_DIR / "wide_duck.parquet"
LANCE_PATH = DATA_DIR / "lance_db"
VORTEX_PATH = DATA_DIR / "wide.vortex"
DUCK_PATH = DATA_DIR / "wide.duckdb"
LANCE_TABLE = "bench"
DUCK_TABLE = "bench"

# Resolve versions
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

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
}
FORMAT_VERSIONS = {
    'Parquet (pyarrow, zstd)': f"pyarrow {VERSIONS['pyarrow']}",
    'Parquet (duckdb, zstd)': f"duckdb {VERSIONS['duckdb']}",
    'Lance (lancedb)': f"lancedb {VERSIONS['lancedb']}",
    'Vortex': f"vortex {VERSIONS['vortex']}",
    'DuckDB (file table)': f"duckdb {VERSIONS['duckdb']}",
}

N_ROWS, N_COLS, STR_COLS, DTYPE


# +
rng = np.random.default_rng(SEED)
float_names = [f"col_{i:04d}" for i in range(N_COLS)]
float_columns = [pa.array(rng.standard_normal(N_ROWS, dtype=DTYPE)) for _ in range(N_COLS)]

str_names = [f"str_{i:04d}" for i in range(STR_COLS)]
str_columns = []
for _ in range(STR_COLS):
    ints = rng.integers(0, 1_000_000, size=N_ROWS, dtype=np.int32)
    strings = np.char.add('s', ints.astype(str))
    str_columns.append(pa.array(strings))

column_names = float_names + str_names
columns = float_columns + str_columns
table = pa.Table.from_arrays(columns, names=column_names)
print('table rows:', table.num_rows, 'cols:', table.num_columns)

# -

duck_table = table


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
        write_times = []
        read_times = []
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

        results.append({
            'format': cfg['name'],
            'write_seconds': write_times,
            'read_seconds': read_times,
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


format_configs = [
    {
        'name': 'Parquet (pyarrow, zstd)',
        'path': PARQUET_PATH,
        'write': lambda tbl, path=PARQUET_PATH: pq.write_table(tbl, path, compression='zstd'),
        'read': lambda path=PARQUET_PATH: pq.read_table(path),
    },
    {
        'name': 'Parquet (duckdb, zstd)',
        'path': PARQUET_DUCK_PATH,
        'write': parquet_duck_write,
        'read': parquet_duck_read,
    },
    {
        'name': 'Lance (lancedb)',
        'path': LANCE_PATH,
        'write': lance_write,
        'read': lance_read,
        'cleanup': False,
    },
    {
        'name': 'Vortex',
        'path': VORTEX_PATH,
        'write': vortex_write,
        'read': vortex_read,
    },
    {
        'name': 'DuckDB (file table)',
        'path': DUCK_PATH,
        'write': duck_write,
        'read': duck_read,
        'table': duck_table,
        'repeats': DUCK_REPEATS,
    },
]

print('Formats:', [cfg['name'] for cfg in format_configs])

# -

results = run_benchmarks(table, format_configs, repeats=REPEATS)
results_df = pd.DataFrame({
    'format': [r['format'] for r in results],
    'write_avg_s': [np.mean(r['write_seconds']) for r in results],
    'write_std_s': [np.std(r['write_seconds']) for r in results],
    'read_avg_s': [np.mean(r['read_seconds']) for r in results],
    'read_std_s': [np.std(r['read_seconds']) for r in results],
    'size_mb': [r['size_mb'] for r in results],
    'version': [FORMAT_VERSIONS.get(r['format'], '') for r in results],
})
results_df


# +
timings = []
for r in results:
    for idx, t in enumerate(r['write_seconds']):
        timings.append({'format': r['format'], 'kind': 'write', 'run': idx, 'seconds': t})
    for idx, t in enumerate(r['read_seconds']):
        timings.append({'format': r['format'], 'kind': 'read', 'run': idx, 'seconds': t})

pd.DataFrame(timings)

