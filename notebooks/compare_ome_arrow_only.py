"""OME-Arrow single-column benchmarks

Compare on-disk formats for a table containing only `row_id` and one OME-Arrow
image column, plus a separate benchmark for a directory-per-image OME-Zarr
layout. Measures: write all, read all, random-read some, size on disk.

Requires `pyarrow`, `lancedb`, `vortex-data`, `duckdb`, `ome-arrow` (installed
via uv). OME-Zarr export requires optional deps (`bioio-ome-zarr`, `zarr`,
`numcodecs`).

Setup:
- Artifacts are written under `data/` (git-ignored).
- If you want the OME-Zarr timings, install the extra deps:
  `uv pip install bioio-ome-zarr zarr numcodecs`.
"""
import gc
import shutil
import time
from pathlib import Path
import bioio_ome_zarr
import zarr
import numcodecs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import lancedb
import duckdb
import vortex
import vortex.io as vxio
from ome_arrow import OMEArrow

pd.set_option("display.precision", 4)
PLOT_TITLE = f"{(__doc__ or 'OME-Arrow single-column benchmarks').splitlines()[0]} (lower is better)"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)
SUMMARY_PARQUET = DATA_DIR / "compare_ome_arrow_only_summary.parquet"
RUNS_PARQUET = DATA_DIR / "compare_ome_arrow_only_runs.parquet"
RUN_BENCHMARKS = not (SUMMARY_PARQUET.exists() and RUNS_PARQUET.exists())

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
    'ome-zarr': _pkg_version('ome-zarr'),
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
if RUN_BENCHMARKS:
    rng = np.random.default_rng(SEED)
    # Add an explicit row id column so we can filter for random access reads later.
    row_ids = pa.array(np.arange(N_ROWS, dtype=np.int64))

    # Build the OME-Arrow column with random images.
    # OMEArrow defaults to dim_order="TCZYX", so we supply a 5D array with singleton T,C,Z.
    ome_pylist = []
    ome_arrays = []
    for _ in range(N_ROWS):
        img = rng.integers(0, 256, size=(1, 1, 1, *OME_SHAPE), dtype=OME_DTYPE)
        ome_arrays.append(img)
        ome_scalar = OMEArrow(data=img).data.as_py()
        ome_scalar.pop('masks', None)  # drop Null field to avoid Arrow null append issue
        ome_pylist.append(ome_scalar)
    ome_column = pa.array(ome_pylist)

    # Assemble the full table with row_id + OME-Arrow payloads.
    column_names = ['ome_image']
    columns = [row_ids, ome_column]
    column_names = ['row_id'] + column_names
    table = pa.Table.from_arrays(columns, names=column_names)
    print('table rows:', table.num_rows, 'cols:', table.num_columns)

    # DuckDB can ingest Arrow directly; keep the full table for parity.
    duck_table = table
    # Precompute row ids for random-read benchmarks.
    RANDOM_INDICES = sorted(rng.choice(table.num_rows, size=RANDOM_ROW_COUNT, replace=False).tolist())
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
            'version': cfg.get('version', FORMAT_VERSIONS.get(cfg['name'], '')),
        })
        print(f"[format end] {cfg['name']}", flush=True)
    return results



if RUN_BENCHMARKS:
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

# OME-Zarr (native) helpers — dir-per-image layout
def ome_zarr_native_available() -> bool:
    try:
        import ome_zarr  # noqa: F401
        import zarr  # noqa: F401
        import numcodecs  # noqa: F401
        from ome_zarr.io import parse_url  # noqa: F401
        from ome_zarr.writer import write_image  # noqa: F401
        from ome_zarr.reader import Reader  # noqa: F401
        return True
    except Exception:
        return False


def ome_zarr_write_all_native(arrays, base_path=OME_ZARR_DIR):
    if not ome_zarr_native_available():
        raise RuntimeError("OME-Zarr native deps missing: install ome-zarr zarr numcodecs")
    drop_path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    import zarr

    for idx, arr in enumerate(arrays):
        out_dir = base_path / f"img_{idx:05d}.zarr"
        store = parse_url(str(out_dir), mode="w").store
        root = zarr.group(store=store)
        write_image(arr, root, axes="tczyx")


def ome_zarr_read_all_native(base_path=OME_ZARR_DIR):
    if not ome_zarr_native_available():
        raise RuntimeError("OME-Zarr native deps missing: install ome-zarr zarr numcodecs")
    import zarr

    out = []
    for zarr_dir in sorted(base_path.glob("*.zarr")):
        grp = zarr.open_group(str(zarr_dir), mode="r")
        if "0" in grp:
            out.append(grp["0"][...])
    return out


def ome_zarr_random_read_native(indices, base_path=OME_ZARR_DIR):
    if not ome_zarr_native_available():
        raise RuntimeError("OME-Zarr native deps missing: install ome-zarr zarr numcodecs")
    import zarr

    paths = sorted(base_path.glob("*.zarr"))
    out = []
    for idx in indices:
        if 0 <= idx < len(paths):
            grp = zarr.open_group(str(paths[idx]), mode="r")
            if "0" in grp:
                out.append(grp["0"][...])
    return out


if RUN_BENCHMARKS:
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

    if ome_zarr_native_available():
        format_configs.append({
            'name': 'OME-Zarr (dir-per-image)',
            'path': OME_ZARR_DIR,
            'write': lambda arrays, path=OME_ZARR_DIR: ome_zarr_write_all_native(arrays, path),
            'read': lambda path=OME_ZARR_DIR: ome_zarr_read_all_native(path),
            'random_read': lambda path=OME_ZARR_DIR, indices=None: ome_zarr_random_read_native(indices, path),
            'table': ome_arrays,  # run_benchmarks passes as cfg_table; here it's a list of numpy arrays
            'random_repeats': RANDOM_READ_REPEATS,
            'version': (
                f"ome-zarr {VERSIONS.get('ome-zarr', '')}; "
                f"zarr {VERSIONS.get('zarr', '')}; "
                f"numcodecs {VERSIONS.get('numcodecs', '')}"
            ),
        })
    else:
        raise RuntimeError("OME-Zarr format requires ome-zarr, zarr, and numcodecs. Install them to proceed.")

    print('Formats:', [cfg['name'] for cfg in format_configs])

if RUN_BENCHMARKS:
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
    results_df.to_parquet(SUMMARY_PARQUET, index=False)

    timings = []
    for r in results:
        for idx, t in enumerate(r['write_seconds']):
            timings.append({'format': r['format'], 'kind': 'write', 'run': idx, 'seconds': t})
        for idx, t in enumerate(r['read_seconds']):
            timings.append({'format': r['format'], 'kind': 'read', 'run': idx, 'seconds': t})
        for idx, t in enumerate(r['random_read_seconds']):
            timings.append({'format': r['format'], 'kind': 'random_read', 'run': idx, 'seconds': t})

    timings_df = pd.DataFrame(timings)
    timings_df.to_parquet(RUNS_PARQUET, index=False)
else:
    results_df = pd.read_parquet(SUMMARY_PARQUET)
    timings_df = pd.read_parquet(RUNS_PARQUET)
    print('Loaded existing benchmark data from', SUMMARY_PARQUET, 'and', RUNS_PARQUET)

print(results_df)
print(timings_df)

summary = results_df.copy()
metrics = [
    ('write_avg_s', 'Write avg (s)'),
    ('read_all_avg_s', 'Read all avg (s)'),
    ('read_random_avg_s', 'Read random avg (s)'),
    ('size_mb', 'Size (MB)'),
]

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
fig.suptitle(PLOT_TITLE)
x = np.arange(len(summary))
COLOR_MAP = {
    "Parquet (pyarrow, zstd)": "#2F5C8A",
    "Parquet (duckdb, zstd)": "#3D7FBF",
    "Lance (lancedb)": "#C86A1B",
    "Vortex": "#2E7D4F",
    "DuckDB (file table)": "#B23B3B",
    "OME-Zarr (dir-per-image)": "#7A5A3C",
}
colors = [COLOR_MAP.get(name, "#BAB0AC") for name in summary["format"]]
def label_bars(ax, bars, fmt="%.3f"):
    values = [bar.get_height() for bar in bars]
    if not values:
        return
    y_max = max(values)
    if y_max <= 0:
        return
    ax.set_ylim(0, y_max * 1.12)
    offset = 0.01 * y_max
    small_threshold = 0.08 * y_max
    ylim_top = ax.get_ylim()[1]
    for bar, value in zip(bars, values):
        if value < small_threshold:
            y = value + offset
            va = "bottom"
            color = "black"
        else:
            y = value - offset
            va = "top"
            color = "white"
        y = min(y, ylim_top * 0.98)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt % value,
            ha="center",
            va=va,
            color=color,
            fontsize=9,
            clip_on=False,
        )
for ax, (col, title) in zip(axes.flat, metrics):
    bars = ax.bar(x, summary[col], color=colors)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(summary['format'], rotation=30, ha='right')
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    if col != 'size_mb':
        ax.set_ylabel('seconds')
    else:
        ax.set_ylabel('MB')
    label_bars(ax, bars)

fig.savefig(IMAGES_DIR / "compare_ome_arrow_only_summary.png", dpi=150)
plt.close(fig)
