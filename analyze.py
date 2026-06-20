# analyze.py
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# SETTINGS
# =====================================================

INPUT_FILE = sys.argv[1]

OUTPUT_CSV = "result.csv"

PLOTS_FOLDER = "static/plots"

TIME_COL = None

PRESSURE_COL = "pressure_smooth"

DERIVATIVE_COL = "dpdt"

# -----------------------------------------------------
# Detection parameters
# -----------------------------------------------------

# Automatically determine derivative threshold
DERIVATIVE_PERCENTILE = 99.5

# Used when refining start/end
SMALL_SLOPE_FACTOR = 0.15

# Challenge requirements
MIN_DURATION_HOURS = 2
MIN_PRESSURE_GAIN = 600

BUFFER_HOURS = 1.5

MERGE_GAP_HOURS = 6

# =====================================================
# LOAD
# =====================================================

print("Loading...")

df = pd.read_csv(INPUT_FILE)

print("\nColumns:")
print(df.columns.tolist())

# Detect time column automatically
for col in df.columns:

    cl = col.lower().strip()

    if (
        "time" in cl
        or "date" in cl
        or "timestamp" in cl
    ):
        TIME_COL = col
        break

if TIME_COL is None:

    raise ValueError(
        f"Could not detect time column.\n"
        f"Available columns: {df.columns.tolist()}"
    )

print("Using time column:", TIME_COL)

# Parse timestamps
df[TIME_COL] = pd.to_datetime(
    df[TIME_COL],
    errors="coerce"
)

# Remove invalid timestamps
df = df.dropna(subset=[TIME_COL])

pressure = df[PRESSURE_COL].values

print("\nRows:")
print(len(df))

print("\nPressure range:")
print(df[PRESSURE_COL].min())
print(df[PRESSURE_COL].max())

print("\nMedian sample spacing (sec):")
print(
    df[TIME_COL]
    .diff()
    .dt.total_seconds()
    .median()
)
valid = pressure > 100

df = df.loc[valid].copy()

pressure = df[PRESSURE_COL].values
dpdt = df[DERIVATIVE_COL].values
from scipy.signal import medfilt

dpdt = medfilt(dpdt, kernel_size=5)

os.makedirs(PLOTS_FOLDER, exist_ok=True)

# =====================================================
# AUTOMATIC THRESHOLDS
# =====================================================

START_THRESHOLD = np.percentile(dpdt, DERIVATIVE_PERCENTILE)

END_THRESHOLD = np.percentile(dpdt, 100 - DERIVATIVE_PERCENTILE)

SMALL_SLOPE = START_THRESHOLD * SMALL_SLOPE_FACTOR

print(f"Start threshold : {START_THRESHOLD:.5g}")
print(f"End threshold   : {END_THRESHOLD:.5g}")

# =====================================================
# FIND RAW INTERVALS
# =====================================================
sample_sec = (
    df[TIME_COL]
    .diff()
    .dt.total_seconds()
    .median()
)

window = max(
    2,
    int(2 * 3600 / sample_sec)
)

gain = np.zeros(len(pressure))

gain[window:] = (
    pressure[window:]
    - pressure[:-window]
)

positive = gain > 300

raw = []

start = None

for i, flag in enumerate(positive):

    if flag and start is None:
        start = i

    elif not flag and start is not None:
        raw.append([start, i - 1])
        start = None

if start is not None:
    raw.append([start, len(df) - 1])

print("Raw candidates:", len(raw))

# =====================================================
# MERGE CLOSE INTERVALS
# =====================================================

merged = []

for event in raw:

    if not merged:
        merged.append(event)
        continue

    prev = merged[-1]

    gap = (
        df.iloc[event[0]][TIME_COL]
        - df.iloc[prev[1]][TIME_COL]
    ).total_seconds()/3600

    if gap <= MERGE_GAP_HOURS:
        prev[1] = event[1]
    else:
        merged.append(event)

# =====================================================
# REFINE START/END
# =====================================================

results = []

for start, end in merged:

    # -------------------------
    # Backtrack to actual start
    # -------------------------

    s = start

    while s > 1:

        if abs(dpdt[s]) < SMALL_SLOPE:
            break

        s -= 1

    # Search local minimum before rise
    LOOKBACK_HOURS = 3

    samples_per_hour = int(round(
        3600 /
        (
            df[TIME_COL]
            .diff()
            .dt.total_seconds()
            .median()
        )
    ))

    LOOKBACK = LOOKBACK_HOURS * samples_per_hour

    search_start = max(0, s - LOOKBACK)

    baseline = np.median(
        pressure[search_start:s+1]
    )

    while s < start:

        if pressure[s] > baseline + 5:
            break

        s += 1

    # -------------------------
    # Search forward for drop
    # -------------------------

    e = end

    while e < len(df)-2:

        if dpdt[e] < END_THRESHOLD:
            break

        e += 1

    # Backtrack to plateau
    plateau = np.max(
        pressure[start:e+1]
    )

    while e > start:

        if pressure[e] > plateau - 5:
            break

        e -= 1

    # -------------------------
    # Validate
    # -------------------------

    gain = pressure[e] - pressure[s]
    # Make sure pressure mostly increases

    segment = pressure[s:e+1]

    positive_steps = np.sum(
        np.diff(segment) > 0
    )

    if positive_steps < 0.7 * len(segment):
        continue

    duration = (
        df.iloc[e][TIME_COL]
        - df.iloc[s][TIME_COL]
    ).total_seconds()/3600

    if gain < MIN_PRESSURE_GAIN:
        continue

    if duration < MIN_DURATION_HOURS:
        continue

    exact_start = df.iloc[s][TIME_COL]
    exact_end = df.iloc[e][TIME_COL]

    results.append({

        "Exact Start": exact_start,

        "Interval Start":
            exact_start
            - pd.Timedelta(hours=BUFFER_HOURS),

        "Exact End": exact_end,

        "Interval End":
            exact_end
            + pd.Timedelta(hours=BUFFER_HOURS),

        "Duration (hr)": duration,

        "Pressure Gain": gain

    })
merged_results = []
results = pd.DataFrame(results)
if len(results) == 0:

    print("No PBUs detected.")

    results.to_csv(
        OUTPUT_CSV,
        index=False
    )

    raise SystemExit
for _, row in results.iterrows():

    if not merged_results:
        merged_results.append(row.to_dict())
        continue

    prev = merged_results[-1]

    if row["Exact Start"] <= prev["Exact End"]:

        prev["Exact End"] = max(
            prev["Exact End"],
            row["Exact End"]
        )

        prev["Interval End"] = max(
            prev["Interval End"],
            row["Interval End"]
        )

    else:
        merged_results.append(
            row.to_dict()
        )

results = pd.DataFrame(merged_results)

results.to_csv(
    OUTPUT_CSV,
    index=False
)

print(results)

print(f"\nDetected {len(results)} PBUs")

# =====================================================
# PLOTS
# =====================================================

for i, row in results.iterrows():

    interval = df[
        (df[TIME_COL] >= row["Interval Start"])
        &
        (df[TIME_COL] <= row["Interval End"])
    ]

    plt.figure(figsize=(12,6))

    plt.plot(
        interval[TIME_COL],
        interval[PRESSURE_COL],
        label="Pressure"
    )

    plt.axvline(
        row["Interval Start"],
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Interval Start"
    )

    plt.axvline(
        row["Interval End"],
        color="blue",
        linestyle="-.",
        linewidth=2,
        label="Interval End"
    )

    plt.axvline(
        row["Exact Start"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Exact Start"
    )

    plt.axvline(
        row["Exact End"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Exact End"
    )

    plt.title(f"PBU #{i+1}")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_FOLDER,
            f"pbu_{i+1:03d}.png"
        ),
        dpi=200
    )

    plt.close()

print(f"Saved plots to '{PLOTS_FOLDER}'")