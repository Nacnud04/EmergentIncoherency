
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "SurfaceStats"

# Select the common quantity to plot:
#
# "deviation": point-to-point RMS height difference at 1 m lag
# "height":     RMS height scaled to a nominal 1 m profile
#
PLOT_QUANTITY = "deviation"

MIN_MARS_SAMPLES = 5000

# Source baselines (m)
EARTH_PROFILE_BASELINE = 10e3
MARS_PROFILE_LENGTH = 30e3
MARS_SLOPE_BASELINE = 300.0
MOON_SLOPE_BASELINE = 17.0

TARGET_SCALE = 1.0  # m


# ============================================================
# HELPERS
# ============================================================

def scale_profile_rms_to_unit_length(sigma, H, length):
    """
    Scale profile RMS height to a nominal 1 m profile length.

    Assumes sigma(L) follows a self-affine power law:
        sigma(L) = sigma(L0) * (L / L0)**H

    This is a model-based scale conversion, not an identity
    between profile RMS and point-to-point RMS deviation.
    """
    return sigma / (length / TARGET_SCALE)**H


def scale_slope_to_unit_deviation(slope, H, baseline):
    """
    Scale slope measured at baseline to a height difference
    at TARGET_SCALE.

    Assumes:
        slope(r) proportional to r**(H - 1)

    Returns the height-equivalent deviation in metres.

    The input slope must be expressed as rise/run. For the
    lunar data, the reported median angle is converted to a
    gradient using tan(angle).
    """
    slope_at_target = slope * (
        TARGET_SCALE / baseline
    )**(H - 1)

    return slope_at_target * TARGET_SCALE


def scale_height_difference_to_unit(deviation, H, baseline):
    """
    Scale a point-to-point RMS height difference from a
    reference lag to TARGET_SCALE.

    Assumes:
        deviation(r) proportional to r**H
    """
    return deviation * (TARGET_SCALE / baseline)**H


def read_dat_file(files, keyword):
    """
    Find a data file by filename keyword.
    """
    matches = [
        f for f in files
        if keyword.lower() in os.path.basename(f).lower()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one file matching '{keyword}', "
            f"found: {matches}"
        )

    return matches[0]


def add_metadata(df, body, region_column=None):
    """
    Add consistent body and region metadata.
    """
    df = df.copy()
    df["body"] = body

    if region_column and region_column in df.columns:
        df["Region"] = df[region_column].astype(str)
    elif "Region" not in df.columns:
        df["Region"] = "Unspecified"

    return df


# ============================================================
# FIND INPUT FILES
# ============================================================

files = glob.glob(os.path.join(DATA_DIR, "*.dat"))

if not files:
    raise FileNotFoundError(
        f"No .dat files found in {DATA_DIR!r}"
    )

earth_file = read_dat_file(files, "earth")
europa_file = read_dat_file(files, "europa")
mars_file = read_dat_file(files, "mars")
moon_file = read_dat_file(files, "moon")


# ============================================================
# EARTH
# ============================================================

raw_earth = pd.read_csv(earth_file, skiprows=14)

raw_earth = raw_earth.rename(
    columns={"Type": "Region"}
)

raw_earth["Hurst"] = pd.to_numeric(
    raw_earth["Hurst"], errors="coerce"
)

raw_earth["RMS Height"] = pd.to_numeric(
    raw_earth["RMS Height"], errors="coerce"
)

# Profile RMS height scaled to a nominal 1 m profile.
raw_earth["profile_rms_1m"] = (
    raw_earth["RMS Height"]
    / EARTH_PROFILE_BASELINE**raw_earth["Hurst"]
)

# No point-to-point deviation inferred here.
raw_earth["deviation_1m"] = np.nan

raw_earth = add_metadata(raw_earth, "Earth")


# ============================================================
# EUROPA
# ============================================================

raw_europa = pd.read_csv(europa_file, skiprows=15)

raw_europa = raw_europa.rename(
    columns={"RMSDeviation": "RMSDeviation"}
)

raw_europa["Hurst"] = pd.to_numeric(
    raw_europa["Hurst"], errors="coerce"
)

raw_europa["RMSDeviation"] = pd.to_numeric(
    raw_europa["RMSDeviation"], errors="coerce"
)

# IMPORTANT:
# This assumes RMSDeviation is a point-to-point height
# difference at the baseline specified by the source file.
#
# If the source already reports the value at a 1 m lag,
# leave it unchanged. Otherwise, replace this baseline with
# the actual reference lag.
EUROPA_DEVIATION_BASELINE = 1.0  # m; verify source metadata

raw_europa["deviation_1m"] = scale_height_difference_to_unit(
    raw_europa["RMSDeviation"],
    raw_europa["Hurst"],
    EUROPA_DEVIATION_BASELINE
)

raw_europa["profile_rms_1m"] = np.nan

raw_europa = add_metadata(raw_europa, "Europa")


# ============================================================
# MARS
# ============================================================

raw_mars = pd.read_csv(mars_file, skiprows=17)

raw_mars = raw_mars.rename(
    columns={
        "H": "Hurst",
        "Roughness Unit": "Region"
    }
)

raw_mars["Hurst"] = pd.to_numeric(
    raw_mars["Hurst"], errors="coerce"
)

raw_mars["Number of Samples"] = pd.to_numeric(
    raw_mars["Number of Samples"], errors="coerce"
)

raw_mars["rms_h"] = 10**pd.to_numeric(
    raw_mars["log10 sigma (m)"], errors="coerce"
)

raw_mars["rms_s"] = 10**pd.to_numeric(
    raw_mars["log10 s"], errors="coerce"
)

# Profile RMS height scaled to a nominal 1 m profile.
raw_mars["profile_rms_1m"] = scale_profile_rms_to_unit_length(
    raw_mars["rms_h"],
    raw_mars["Hurst"],
    MARS_PROFILE_LENGTH
)

# RMS slope at 300 m -> RMS slope at 1 m -> height
# difference at 1 m.
raw_mars["deviation_1m"] = scale_slope_to_unit_deviation(
    raw_mars["rms_s"],
    raw_mars["Hurst"],
    MARS_SLOPE_BASELINE
)

# Preserve the original sample-count threshold.
raw_mars = raw_mars[
    raw_mars["Number of Samples"] > MIN_MARS_SAMPLES
].copy()

raw_mars = add_metadata(raw_mars, "Mars")


# ============================================================
# MOON
# ============================================================

raw_moon = pd.read_csv(moon_file, skiprows=16)

raw_moon["Hurst"] = pd.to_numeric(
    raw_moon["Hurst"], errors="coerce"
)

raw_moon["Slope"] = pd.to_numeric(
    raw_moon["Slope"], errors="coerce"
)

# Convert median slope angle to a gradient.
moon_gradient = np.tan(
    np.radians(raw_moon["Slope"])
)

# Scale median gradient to 1 m and convert to a
# height-equivalent difference.
raw_moon["deviation_1m"] = scale_slope_to_unit_deviation(
    moon_gradient,
    raw_moon["Hurst"],
    MOON_SLOPE_BASELINE
)

# No profile RMS height available in this file.
raw_moon["profile_rms_1m"] = np.nan

raw_moon = add_metadata(raw_moon, "Moon")


# ============================================================
# COMBINE DATASETS
# ============================================================

combined = pd.concat(
    [raw_earth, raw_europa, raw_mars, raw_moon],
    ignore_index=True,
    sort=False
)

# Select the common y-axis quantity.
if PLOT_QUANTITY == "deviation":
    y_column = "deviation_1m"
    y_label = "Height difference at 1 m lag [m]"
    title = "Planetary Surface Roughness at 1 m Point-to-Point Lag"

elif PLOT_QUANTITY == "height":
    y_column = "profile_rms_1m"
    y_label = "Profile RMS height at nominal 1 m scale [m]"
    title = "Planetary Profile RMS Height at Nominal 1 m Scale"

else:
    raise ValueError(
        "PLOT_QUANTITY must be 'deviation' or 'height'"
    )


# ============================================================
# PLOT
# ============================================================

body_order = ["Earth", "Europa", "Moon", "Mars"]

colors = {
    "Earth": "blue",
    "Europa": "cyan",
    "Moon": "grey",
    "Mars": "red"
}

markers = {
    "Earth": "o",
    "Europa": "s",
    "Moon": "^",
    "Mars": "D"
}

fig, ax = plt.subplots(figsize=(9, 6))

for body in body_order:
    subset = combined[
        (combined["body"] == body)
        & combined["Hurst"].notna()
        & combined[y_column].notna()
        & np.isfinite(combined[y_column])
    ]

    if subset.empty:
        print(
            f"Warning: no valid {PLOT_QUANTITY} data for {body}"
        )
        continue

    ax.scatter(
        subset["Hurst"],
        subset[y_column],
        s=55,
        alpha=0.8,
        color=colors[body],
        marker=markers[body],
        edgecolor="black",
        linewidth=0.4,
        label=f"{body} (n={len(subset)})"
    )

ax.set_title(title)
ax.set_xlabel("Hurst exponent")
ax.set_ylabel(y_label)

ax.set_xlim(0, 1)
ax.grid(True, alpha=0.25)

# Use a logarithmic y-axis when all plotted values are positive.
positive_values = combined.loc[
    combined[y_column].notna()
    & np.isfinite(combined[y_column])
    & (combined[y_column] > 0),
    y_column
]

if len(positive_values):
    ax.set_yscale("log")

ax.legend(frameon=False)
fig.tight_layout()
plt.show()


# ============================================================
# SUMMARY
# ============================================================

print(f"\nPlotted quantity: {PLOT_QUANTITY}")
print(f"Y-axis column: {y_column}")

for body in body_order:
    subset = combined[
        (combined["body"] == body)
        & combined[y_column].notna()
        & np.isfinite(combined[y_column])
    ]

    print(f"{body}: {len(subset)} plotted points")
