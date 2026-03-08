import pandas as pd
import numpy as np
import os
import sys

# ====================== Configuration Area (Please fill in) ======================
# Set the file paths and parameters here (replace the empty strings with actual paths)
CALIBRATION_FILE_PATH = ""  # Full path of Excel file for Calibration Period
CORRECTION_FILE_PATH = ""  # Full path of Excel file for Correction Period
WET_DAY_THRESHOLD = 0.1  # Wet day threshold (e.g., 0.1 mm)


# =================================================================================

def main():
    print("=" * 50)
    print("Satellite Precipitation Data QM Quantile Mapping Correction Tool")
    print("=" * 50)

    # --- 1. Validate Configuration Parameters ---
    try:
        # Check if file paths are filled
        if not CALIBRATION_FILE_PATH:
            print("Error: Please fill in the CALIBRATION_FILE_PATH in the configuration area.")
            return
        if not CORRECTION_FILE_PATH:
            print("Error: Please fill in the CORRECTION_FILE_PATH in the configuration area.")
            return

        # Check wet day threshold type
        threshold = float(WET_DAY_THRESHOLD)
    except ValueError:
        print("Error: Wet day threshold must be a numeric value.")
        return

    # --- 2. Check File Existence ---
    if not os.path.exists(CALIBRATION_FILE_PATH):
        print(f"Error: Calibration period file not found: {CALIBRATION_FILE_PATH}")
        return
    if not os.path.exists(CORRECTION_FILE_PATH):
        print(f"Error: Correction period file not found: {CORRECTION_FILE_PATH}")
        return

    # --- 3. Read Data ---
    required_cols = ["Station", "Province", "Region", "Year", "Month", "Day", "Date_ID", "E", "L", "F", "D"]

    print("\nReading data...")
    try:
        df_calibration = pd.read_excel(CALIBRATION_FILE_PATH)
        df_correction = pd.read_excel(CORRECTION_FILE_PATH)
    except Exception as e:
        print(f"Failed to read Excel files: {e}")
        return

    # Check if all required columns exist
    for col in required_cols:
        if col not in df_calibration.columns:
            print(f"Error: Calibration period file missing column: {col}")
            return
        if col not in df_correction.columns:
            print(f"Error: Correction period file missing column: {col}")
            return

    # --- 4. Build QM Mapping Relationship (by Station) ---
    print("\nStarting to build QM mapping relationships...")

    # Get all unique stations
    stations = df_calibration["Station"].unique()
    print(f"Total {len(stations)} stations detected")

    # Dictionary to store QM models: model[station][product] = (obs_sorted, tau_obs, sat_sorted, tau_sat)
    qm_models = {}

    for station in stations:
        # Extract calibration period data for current station
        df_sta_calibration = df_calibration[df_calibration["Station"] == station].copy()

        # 1. Process observation data (Column D)
        obs_series = df_sta_calibration["D"].dropna()
        obs_wet = obs_series[obs_series >= threshold].sort_values().values
        n_obs = len(obs_wet)

        # Calculate empirical quantiles for observations (Weibull method: i/(n+1))
        tau_obs = np.arange(1, n_obs + 1) / (n_obs + 1) if n_obs > 0 else np.array([])

        qm_models[station] = {}

        # 2. Process E, L, F products separately
        for prod in ["E", "L", "F"]:
            sat_series = df_sta_calibration[prod].dropna()
            sat_wet = sat_series[sat_series >= threshold].sort_values().values
            n_sat = len(sat_wet)

            # Calculate empirical quantiles for satellite data
            tau_sat = np.arange(1, n_sat + 1) / (n_sat + 1) if n_sat > 0 else np.array([])

            # Save model parameters
            qm_models[station][prod] = (obs_wet, tau_obs, sat_wet, tau_sat)

    print("QM mapping relationships built successfully.")

    # --- 5. Perform Correction ---
    print("\nStarting to correct correction period data...")

    # Initialize result columns
    df_correction["QM-E"] = np.nan
    df_correction["QM-L"] = np.nan
    df_correction["QM-F"] = np.nan

    # Iterate through stations for correction (group by station for higher efficiency)
    for station in stations:
        if station not in qm_models:
            continue

        # Get row indices for current station in correction period data
        idx_correction = df_correction["Station"] == station
        if not idx_correction.any():
            continue

        for prod in ["E", "L", "F"]:
            obs_wet, tau_obs, sat_wet, tau_sat = qm_models[station][prod]

            # Get raw values to be corrected
            raw_vals = df_correction.loc[idx_correction, prod].values

            # Initialize result array (default NaN)
            corrected_vals = np.full_like(raw_vals, np.nan, dtype=np.float64)

            # 1. Process non-NaN values
            not_nan_mask = ~pd.isna(raw_vals)

            # 2. Set dry days (< threshold) to 0 directly
            dry_mask = not_nan_mask & (raw_vals < threshold)
            corrected_vals[dry_mask] = 0.0

            # 3. Perform QM mapping for wet days (>= threshold)
            wet_mask = not_nan_mask & (raw_vals >= threshold)

            if np.any(wet_mask):
                # Check if model is available
                if len(sat_wet) == 0 or len(obs_wet) == 0:
                    # If no wet day samples in calibration period, set to 0
                    corrected_vals[wet_mask] = 0.0
                    # print(f"  Warning: Station {station} Product {prod} has no wet day samples in calibration period, set to 0")
                else:
                    # Core QM logic
                    # Step A: Raw value -> Satellite quantile (Tau)
                    # Linear interpolation, use boundary values for out-of-range
                    tau_vals = np.interp(raw_vals[wet_mask], sat_wet, tau_sat,
                                         left=tau_sat[0], right=tau_sat[-1])

                    # Step B: Satellite quantile -> Observation value
                    final_vals = np.interp(tau_vals, tau_obs, obs_wet,
                                           left=obs_wet[0], right=obs_wet[-1])

                    corrected_vals[wet_mask] = final_vals

            # Write corrected values back to DataFrame
            df_correction.loc[idx_correction, f"QM-{prod}"] = corrected_vals

    # --- 6. Save Results ---
    # Auto-generate output file path
    dir_name = os.path.dirname(CORRECTION_FILE_PATH)
    base_name = os.path.basename(CORRECTION_FILE_PATH)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_QM_Corrected_Results{ext}")

    try:
        print(f"\nSaving results to: {output_path}")
        df_correction.to_excel(output_path, index=False)
        print("=" * 50)
        print("Processing completed successfully!")
        print("=" * 50)
    except Exception as e:
        print(f"Failed to save file. Please check if the file is occupied. Error: {e}")


if __name__ == "__main__":
    main()