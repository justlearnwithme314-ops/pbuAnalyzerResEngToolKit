import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.impute import SimpleImputer
from scipy.stats import linregress

# =====================================================
# FILES
# =====================================================

if len(sys.argv) < 4:
    print(
        "Usage:\n"
        "python sec2.py processed.csv result.csv train_data.csv [output.csv]"
    )
    raise SystemExit

PROCESSED_FILE = sys.argv[1]
PBU_FILE = sys.argv[2]
TRAIN_FILE = sys.argv[3]

OUTPUT_FILE = (
    sys.argv[4]
    if len(sys.argv) > 4
    else "reservoir_predictions.csv"
)

# =====================================================
# KNOWN RESERVOIR PRESSURES
# =====================================================



MATCH_WINDOW_DAYS = 7

# =====================================================
# LOAD
# =====================================================

print("Loading files...")

df = pd.read_csv(PROCESSED_FILE)
pbus = pd.read_csv(PBU_FILE)
train_pressures = pd.read_csv(TRAIN_FILE)

train_pressures["date"] = pd.to_datetime(
    train_pressures["date"]
).dt.tz_localize(None)
print("Processed columns:")
print(df.columns.tolist())

TIME_COL = None

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

df[TIME_COL] = pd.to_datetime(
    df[TIME_COL],
    errors="coerce"
).dt.tz_localize(None)

pbus["Exact Start"] = pd.to_datetime(
    pbus["Exact Start"]
).dt.tz_localize(None)

pbus["Exact End"] = pd.to_datetime(
    pbus["Exact End"]
).dt.tz_localize(None)
PRESSURE_COL = None

for col in df.columns:

    cl = col.lower().strip()

    if (
        "pressure" in cl
        or "dhp" in cl
        or "bhp" in cl
    ):
        PRESSURE_COL = col
        break

if PRESSURE_COL is None:

    if "pressure_smooth" in df.columns:
        PRESSURE_COL = "pressure_smooth"

print("Using pressure column:", PRESSURE_COL)
# =====================================================
# HORNER FEATURE
# =====================================================

def horner_feature(
    pressure,
    dt_hours,
    tp_hours=1000
):

    try:

        pressure = np.asarray(pressure)
        dt_hours = np.asarray(dt_hours)

        mask = dt_hours > 0

        pressure = pressure[mask]
        dt_hours = dt_hours[mask]

        if len(pressure) < 10:
            return np.nan

        H = (tp_hours + dt_hours) / dt_hours

        tail = len(H) // 2

        x = np.log10(H[tail:])
        y = pressure[tail:]

        slope, intercept, *_ = linregress(
            x,
            y
        )

        # reject garbage values

        if intercept < 3000:
            return np.nan

        if intercept > 4500:
            return np.nan

        return intercept

    except Exception:

        return np.nan

# =====================================================
# FEATURE EXTRACTION
# =====================================================

records = []

for idx, pbu in pbus.iterrows():

    start = pbu["Exact Start"]
    end = pbu["Exact End"]

    segment = df[
        (df[TIME_COL] >= start)
        &
        (df[TIME_COL] <= end)
    ]

    if len(segment) < 10:
        continue

    pressure = segment["pressure_smooth"].values
    dpdt = segment["dpdt"].values

    elapsed_hours = (
        segment[TIME_COL]
        -
        segment[TIME_COL].iloc[0]
    ).dt.total_seconds() / 3600

    horner = horner_feature(
        pressure,
        elapsed_hours
    )

    records.append({

        "pbu_index": idx,

        "date": start,

        "duration":
            pbu["Duration (hr)"],

        "gain":
            pbu["Pressure Gain"],

        "start_pressure":
            pressure[0],

        "end_pressure":
            pressure[-1],

        "max_pressure":
            np.max(pressure),

        "mean_dpdt":
            np.mean(dpdt),

        "std_dpdt":
            np.std(dpdt),

        "horner_feature":
            horner

    })

features = pd.DataFrame(records)

#Time Features
features["year"] = (
    features["date"]
    -
    features["date"].min()
).dt.days / 365.25

# =====================================================
# MATCH LABELS
# =====================================================

labels = []

for _, row in features.iterrows():

    label = np.nan

    for _, known in train_pressures.iterrows():

        known_date = known["date"]

        pressure = known["reservoir_pressure"]

        diff_days = abs(
            (row["date"] - known_date).days
        )

        if diff_days <= MATCH_WINDOW_DAYS:

            label = pressure
            break

    labels.append(label)

features["reservoir_pressure"] = labels
print(features[["date","reservoir_pressure"]].dropna())
# =====================================================
# fill missing Horner values FIRST

features["horner_feature"] = (
    features["horner_feature"]
    .fillna(
        features["horner_feature"].median()
    )
)

# training set

train = features.dropna(
    subset=["reservoir_pressure"]
)

# =====================================================
# FEATURES
# =====================================================

feature_cols = [

    "year",
    "end_pressure",
    "horner_feature"

]

X_train = train[feature_cols]
print(X_train.isna().sum())
y_train = train["reservoir_pressure"]
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(
    train[feature_cols]
)

X_all = imputer.transform(
    features[feature_cols]
)
# =====================================================
# MODEL
# =====================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICT
# =====================================================



features["prediction"] = model.predict(
    X_all
)

# =====================================================
# EVALUATION
# =====================================================

# =====================================================
# LEAVE-ONE-OUT CROSS VALIDATION
# =====================================================

loo = LeaveOneOut()

cv_predictions = []
cv_truth = []

for train_idx, test_idx in loo.split(X_train):

    Xtr = X_train[train_idx]
    ytr = y_train.iloc[train_idx]

    Xte = X_train[test_idx]
    yte = y_train.iloc[test_idx]

    m = LinearRegression()

    m.fit(Xtr, ytr)

    pred = m.predict(Xte)[0]

    cv_predictions.append(pred)
    cv_truth.append(yte.iloc[0])

mae = mean_absolute_error(
    cv_truth,
    cv_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        cv_truth,
        cv_predictions
    )
)

r2 = r2_score(
    cv_truth,
    cv_predictions
)

print("\nLOOCV Evaluation")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))
# =====================================================
# FINAL MODEL
# =====================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

features["prediction"] = model.predict(
    X_all
)
# =====================================================
# SAVE
# =====================================================

out = features[[

    "date",
    "duration",
    "gain",
    "horner_feature",
    "prediction",
    "reservoir_pressure"

]]

out.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(out)

print()
print(
    f"Saved to {OUTPUT_FILE}"
)
# CHECKING whihc PBUs are the closest to the found results and what is the problem
for d in train_pressures["date"]:

    kd = pd.to_datetime(d)

    nearest = min(
        features["date"],
        key=lambda x: abs((x-kd).days)
    )

    diff = abs((nearest-kd).days)

    print(
        d,
        nearest,
        diff
    )

# Checking coefficients
print()

for name, coef in zip(
    feature_cols,
    model.coef_
):
    print(name, coef)

print("Intercept:", model.intercept_)