import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# ============================================
# INPUTS
# ============================================

INPUT_FILE = sys.argv[1]
NUM_GRAPHS = int(sys.argv[2])

NUM_GRAPHS = max(1, min(NUM_GRAPHS, 100))

OUTPUT_FOLDER = "static/view"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================
# LOAD CSV
# ============================================

print("Loading CSV...")

df = pd.read_csv(INPUT_FILE)

df.columns = df.columns.str.strip()

# ============================================
# DETECT COLUMNS
# ============================================

TIME_COLUMN = None
PRESSURE_COLUMN = None

for col in df.columns:

    cl = col.lower()

    if (
        "time" in cl
        or "date" in cl
        or "timestamp" in cl
    ):
        TIME_COLUMN = col

    if (
        "dhp" in cl
        or "bhp" in cl
        or "pressure" in cl
    ):
        PRESSURE_COLUMN = col

if TIME_COLUMN is None:
    raise ValueError("Could not find timestamp column")

if PRESSURE_COLUMN is None:
    raise ValueError("Could not find pressure column")

print("Time column:", TIME_COLUMN)
print("Pressure column:", PRESSURE_COLUMN)

# ============================================
# PARSE
# ============================================

# ============================================
# TIMESTAMP FORMATS
# ============================================

formats = [
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
    "%Y-%m-%dT%H:%M:%S.%f"
]

# ============================================
# PARSE TIMESTAMPS
# ============================================

print("Parsing timestamps...")

ts = (
    df[TIME_COLUMN]
    .astype(str)
    .str.strip()
    .str.replace("Z", "", regex=False)
    .str.replace("+00:00", "", regex=False)
)
print("\nSample timestamps:")

for x in ts.head(10):
    print(x)
df[TIME_COLUMN] = pd.to_datetime(
    ts,
    errors="coerce",
    utc=False
)

bad = df[TIME_COLUMN].isna().sum()

print(
    f"Invalid timestamps: {bad:,}"
)

df = df.dropna(
    subset=[TIME_COLUMN]
)
df[PRESSURE_COLUMN] = pd.to_numeric(
    df[PRESSURE_COLUMN],
    errors="coerce"
)

df = df.dropna()

# ============================================
# CLEAR OLD PLOTS
# ============================================

for f in os.listdir(OUTPUT_FOLDER):

    if f.endswith(".png"):

        os.remove(
            os.path.join(
                OUTPUT_FOLDER,
                f
            )
        )

# ============================================
# SPLIT DATA
# ============================================

chunk_size = len(df) // NUM_GRAPHS

print(
    f"Rows: {len(df)}"
)

print(
    f"Chunk size: {chunk_size}"
)

# ============================================
# CREATE PLOTS
# ============================================

for i in range(NUM_GRAPHS):

    start = i * chunk_size

    if i == NUM_GRAPHS - 1:

        end = len(df)

    else:

        end = (i + 1) * chunk_size

    chunk = df.iloc[start:end]

    if len(chunk) < 2:
        continue

    plt.figure(
        figsize=(12,5)
    )

    plt.plot(
        chunk[TIME_COLUMN],
        chunk[PRESSURE_COLUMN]
    )

    start_time = (
        chunk[TIME_COLUMN]
        .iloc[0]
    )

    end_time = (
        chunk[TIME_COLUMN]
        .iloc[-1]
    )

    plt.title(
        f"Segment {i+1}\n"
        f"{start_time} -> {end_time}"
    )

    plt.xlabel("Time")

    plt.ylabel("Pressure")

    plt.grid(True)

    plt.tight_layout()

    filename = (
        f"view_{i+1:02d}.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300
    )

    plt.close()

print(
    f"Created {NUM_GRAPHS} plots."
)