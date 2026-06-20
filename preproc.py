# preprocess.py

import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter
import sys

# =====================================================
# CONFIGURATION
# =====================================================

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = "processed.csv"

TIME_COLUMN = None
PRESSURE_COLUMN = None

# Set to None to disable resampling
RESAMPLE_INTERVAL = "5min"

# "median" recommended
AGG_METHOD = "median"

# Outlier clipping
REMOVE_OUTLIERS = True
LOWER_QUANTILE = 0.01
UPPER_QUANTILE = 0.99

# Median filter
MEDIAN_KERNEL = 5

# Savitzky-Golay
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3

# =====================================================
# LOAD DATA
# =====================================================

print("Loading CSV...")

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

# =====================================================
# DETECT COLUMNS
# =====================================================

print("\nColumns found:")
print(df.columns.tolist())

for col in df.columns:

    cl = col.lower().strip()

    # time column
    if (
        "time" in cl
        or "date" in cl
        or "timestamp" in cl
    ):
        TIME_COLUMN = col

    # pressure column
    if (
        "pressure" in cl
        or "dhp" in cl
        or "bhp" in cl
    ):
        PRESSURE_COLUMN = col

print("Using time column:", TIME_COLUMN)
print("Using pressure column:", PRESSURE_COLUMN)

if TIME_COLUMN is None:
    raise ValueError(
        f"Could not detect time column.\n"
        f"Available columns: {df.columns.tolist()}"
    )

if PRESSURE_COLUMN is None:
    raise ValueError(
        f"Could not detect pressure column.\n"
        f"Available columns: {df.columns.tolist()}"
    )
# =====================================================
# PARSE TIMESTAMPS
# =====================================================

print("Parsing timestamps...")

print("Detecting timestamp format...")

sample = (
    df[TIME_COLUMN]
    .dropna()
    .astype(str)
    .iloc[0]
)

print("Sample timestamp:", sample)

time_format = None

formats = [
    "%Y/%m/%d %H:%M:%S.%fT",  # <--- Just add T right after %f
    "%Y/%m/%d %H:%M:%ST",     # <--- Just add T right after %S
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d-%b-%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f"
]
for fmt in formats:
    try:
        pd.to_datetime(sample, format=fmt)
        time_format = fmt
        break
    except:
        pass

if time_format is None:

    print(
        "Could not determine format automatically."
    )

    df[TIME_COLUMN] = pd.to_datetime(
        df[TIME_COLUMN],
        utc=True,
        errors="coerce"
    )

else:

    print("Detected format:", time_format)

    ts = df[TIME_COLUMN].astype(str)
    ts = ts.str.rstrip("T")

    try:
        df[TIME_COLUMN] = pd.to_datetime(
            ts,
            format="%Y/%m/%d %H:%M:%S.%f",
            utc=True,
            errors="raise"
        )
    except:
        df[TIME_COLUMN] = pd.to_datetime(
            ts,
            format="%Y/%m/%d %H:%M:%S",
            utc=True,
            errors="coerce"
        )

bad_times = df[TIME_COLUMN].isna().sum()

if bad_times > 0:
    print(f"Removing {bad_times:,} rows with invalid timestamps")

df = df.dropna(subset=[TIME_COLUMN])

# Keep only required columns
df = df[[TIME_COLUMN, PRESSURE_COLUMN]]

# Ensure pressure is numeric
df[PRESSURE_COLUMN] = pd.to_numeric(
    df[PRESSURE_COLUMN],
    errors="coerce"
)

df = df.dropna(subset=[PRESSURE_COLUMN])

# =====================================================
# SORT + INDEX
# =====================================================

df = df.sort_values(TIME_COLUMN)

df = df.set_index(TIME_COLUMN)

print("\nIndex type:")
print(type(df.index))
print(df.index.dtype)

print(f"\nRows before cleanup: {len(df):,}")

# =====================================================
# HANDLE DUPLICATES
# =====================================================

df = df.groupby(level=0).median()

print(f"Rows after duplicate merge: {len(df):,}")

# =====================================================
# RESAMPLE
# =====================================================

if RESAMPLE_INTERVAL is not None:

    print(f"\nResampling to {RESAMPLE_INTERVAL}")

    if AGG_METHOD == "median":
        df = df.resample(RESAMPLE_INTERVAL).median()

    elif AGG_METHOD == "mean":
        df = df.resample(RESAMPLE_INTERVAL).mean()

    else:
        raise ValueError(
            "AGG_METHOD must be 'median' or 'mean'"
        )

    df = df.dropna()

print(f"Rows after resampling: {len(df):,}")

# =====================================================
# OUTLIER CLIPPING
# =====================================================

if REMOVE_OUTLIERS:

    print("Clipping extreme outliers...")

    q_low = df[PRESSURE_COLUMN].quantile(
        LOWER_QUANTILE
    )

    q_high = df[PRESSURE_COLUMN].quantile(
        UPPER_QUANTILE
    )

    df[PRESSURE_COLUMN] = (
        df[PRESSURE_COLUMN]
        .clip(q_low, q_high)
    )

# =====================================================
# MEDIAN FILTER
# =====================================================

print("Applying median filter...")

kernel = MEDIAN_KERNEL

if kernel % 2 == 0:
    kernel += 1

df["pressure_med"] = medfilt(
    df[PRESSURE_COLUMN].values,
    kernel_size=kernel
)

# =====================================================
# SAVITZKY-GOLAY FILTER
# =====================================================

print("Applying Savitzky-Golay filter...")

window = min(SAVGOL_WINDOW, len(df))

if window % 2 == 0:
    window -= 1

if window < 5:
    window = 5

poly = min(SAVGOL_POLYORDER, window - 1)

df["pressure_smooth"] = savgol_filter(
    df["pressure_med"].values,
    window_length=window,
    polyorder=poly
)

# =====================================================
# DERIVATIVE
# =====================================================

print("Calculating dp/dt...")

time_seconds = (
    df.index.view("int64") / 1e9
)

df["dpdt"] = np.gradient(
    df["pressure_smooth"],
    time_seconds
)

# =====================================================
# EXPORT
# =====================================================

print(f"Saving -> {OUTPUT_FILE}")

df.reset_index().to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDone.")
print(f"Final rows: {len(df):,}")

print("\nOutput columns:")
print(df.reset_index().columns.tolist())