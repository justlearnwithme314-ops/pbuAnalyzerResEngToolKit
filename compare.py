import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

# ===========================================================
# Output folder
# ===========================================================

OUTPUT_FOLDER = "static/comparison"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===========================================================
# Supported timestamp formats
# ===========================================================

FORMATS = [
    "%Y/%m/%d %H:%M:%S.%fT",
    "%Y/%m/%d %H:%M:%ST",
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
    "%Y-%m-%dT%H:%M:%S.%f",
]


def parse_timestamp(value):
    """
    Parse many timestamp formats.
    """

    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    value = value.replace("Z", "")
    value = value.replace("+00:00", "")

    try:
        return pd.to_datetime(value)
    except:
        pass

    for fmt in FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except:
            continue

    return pd.NaT


# ===========================================================
# Read input.csv
# ===========================================================
print("Reading input.csv...")
t0 = time.time()

if len(sys.argv) < 4:
    print(
        "Usage:\n"
        "python sec2.py processed.csv result.csv train_data.csv [output.csv]"
    )
    raise SystemExit

INPUT_FILE = sys.argv[1]
PROCESSED_FILE = sys.argv[2]
RESULT_FILE = sys.argv[3]
input_df = pd.read_csv(
    INPUT_FILE,
    usecols=["timestamp", "dhp"]
)

print(f"CSV loaded in {time.time()-t0:.1f}s")
print("Rows:", len(input_df))

input_df.columns = input_df.columns.str.lower()

print("Starting timestamp parsing...")
t0 = time.time()

input_df["timestamp"] = pd.to_datetime(
    input_df["timestamp"],
    errors="coerce",
    utc=True
).dt.tz_localize(None)

print(f"Timestamp parsing took {time.time()-t0:.1f}s")

print("Dropping bad timestamps...")
input_df = input_df.dropna(subset=["timestamp"])

print("Flooring timestamps...")
input_df["timestamp"] = input_df["timestamp"].dt.floor("min")

print("Converting DHP...")
input_df["dhp"] = pd.to_numeric(
    input_df["dhp"],
    errors="coerce"
)

print("Grouping...")
t0 = time.time()

input_df = (
    input_df
    .groupby("timestamp", as_index=False)
    .mean()
)

print(f"Groupby took {time.time()-t0:.1f}s")
print("Input rows:", len(input_df))

# ===========================================================
# Read processed.csv
# ===========================================================

print("Reading processed.csv...")

processed = pd.read_csv(PROCESSED_FILE)

processed.columns = processed.columns.str.lower()

processed["timestamp"] = pd.to_datetime(
    processed["timestamp"],
    errors="coerce",
    utc=True
).dt.tz_localize(None)
processed = processed.dropna(subset=["timestamp"])

processed["timestamp"] = processed["timestamp"].dt.floor("min")

processed["dhp"] = pd.to_numeric(processed["dhp"], errors="coerce")

processed = (
    processed
    .groupby("timestamp", as_index=False)
    .mean()
)

print("Processed rows:", len(processed))


# ===========================================================
# Merge datasets
# ===========================================================
print("INPUT:", input_df["timestamp"].dtype)
print("PROC :", processed["timestamp"].dtype)
merged = pd.merge_asof(
    input_df.sort_values("timestamp"),
    processed.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("2min"),
    suffixes=("_input", "_processed")
)

merged = merged.dropna()

print("Matched rows:", len(merged))


# ===========================================================
# Difference calculation
# ===========================================================

merged["difference_percent"] = (
    np.abs(
        merged["dhp_input"] -
        merged["dhp_processed"]
    )
    /
    merged["dhp_input"]
    * 100
)

bad_points = merged[
    merged["difference_percent"] > 5
]

print("Points exceeding 5%:", len(bad_points))


# ===========================================================
# Comparison plot
# ===========================================================

plt.figure(figsize=(16, 7))

plt.plot(
    merged["timestamp"],
    merged["dhp_input"],
    label="Input"
)

plt.plot(
    merged["timestamp"],
    merged["dhp_processed"],
    label="Processed"
)

if len(bad_points):

    plt.scatter(
        bad_points["timestamp"],
        bad_points["dhp_input"],
        s=20,
        label=">5% Difference"
    )

plt.xlabel("Timestamp")
plt.ylabel("Pressure")

plt.title("Input vs Processed Pressure")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "comparison.png"
    ),
    dpi=300
)

plt.close()

print("Saved comparison plot.")


# ===========================================================
# Read PBU results
# ===========================================================

results = pd.read_csv(RESULT_FILE)

results.columns = results.columns.str.lower()

results["exact start"] = pd.to_datetime(
    results["exact start"],
    errors="coerce",
    utc=True
).dt.tz_localize(None)

results["exact end"] = pd.to_datetime(
    results["exact end"],
    errors="coerce",
    utc=True
).dt.tz_localize(None)

# ===========================================================
# PBU overview
# ===========================================================

plt.figure(figsize=(18,7))

plt.plot(
    processed["timestamp"],
    processed["dhp"],
    linewidth=1
)

for _, row in results.iterrows():

    plt.axvline(
        row["exact start"],
        linestyle="--",
        alpha=0.7
    )

    plt.axvline(
        row["exact end"],
        linestyle=":"
    )

plt.title("Detected PBUs")

plt.xlabel("Timestamp")

plt.ylabel("Pressure")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "pbu_overview.png"
    ),
    dpi=300
)

plt.close()

print("Saved PBU overview.")
# ===========================================================
# Calculate derivative (dp/dt)
# ===========================================================

processed = processed.sort_values("timestamp").reset_index(drop=True)

time_seconds = (
    processed["timestamp"] - processed["timestamp"].iloc[0]
).dt.total_seconds()

processed["seconds"] = time_seconds

processed["dpdt"] = np.gradient(
    processed["dhp"],
    processed["seconds"]
)

print("Calculated dp/dt.")


# ===========================================================
# Generate zoomed PBU plots
# ===========================================================

MAX_PLOTS = min(20, len(results))

print(f"Generating {MAX_PLOTS} zoomed PBU plots...")

for i, row in results.head(MAX_PLOTS).iterrows():

    start = row["exact start"]
    end = row["exact end"]

    if pd.isna(start) or pd.isna(end):
        continue

    duration = end - start

    if duration.total_seconds() <= 0:
        duration = pd.Timedelta(minutes=10)

    padding = duration * 0.5

    window_start = start - padding
    window_end = end + padding

    window = processed[
        (processed["timestamp"] >= window_start)
        &
        (processed["timestamp"] <= window_end)
    ]

    if len(window) < 3:
        continue

    # =======================================================
    # Pressure plot
    # =======================================================

    plt.figure(figsize=(12, 5))

    plt.plot(
        window["timestamp"],
        window["dhp"],
        linewidth=1.5,
        label="Pressure"
    )

    plt.axvline(
        start,
        linestyle="--",
        label="PBU Start"
    )

    plt.axvline(
        end,
        linestyle=":",
        label="PBU End"
    )

    plt.title(f"PBU {i+1}")

    plt.xlabel("Timestamp")
    plt.ylabel("Pressure")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"pbu_{i+1:02d}.png"
        ),
        dpi=300
    )

    plt.close()

    # =======================================================
    # Derivative plot
    # =======================================================

    plt.figure(figsize=(12,5))

    plt.plot(
        window["timestamp"],
        window["dpdt"],
        linewidth=1.5,
        label="dp/dt"
    )

    plt.axvline(
        start,
        linestyle="--",
        label="PBU Start"
    )

    plt.axvline(
        end,
        linestyle=":",
        label="PBU End"
    )

    plt.title(f"PBU {i+1} Derivative")

    plt.xlabel("Timestamp")
    plt.ylabel("dp/dt")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            f"derivative_{i+1:02d}.png"
        ),
        dpi=300
    )

    plt.close()


# ===========================================================
# Summary
# ===========================================================

print()
print("=" * 60)
print("Comparison finished successfully.")
print(f"Matched samples      : {len(merged)}")
print(f">5% difference points: {len(bad_points)}")
print(f"PBUs detected        : {len(results)}")
print(f"Zoomed plots created : {MAX_PLOTS}")
print(f"Output folder        : {OUTPUT_FOLDER}")
print("=" * 60)