import pandas as pd
import numpy as np
from tqdm import tqdm
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# ====================== Configuration Area (Please fill in) ======================
# Set the file paths here (replace empty strings with actual full paths)
TRAIN_DATA_PATH = ""  # Training data file path (e.g., 2001-2015.xlsx)
TEST_DATA_PATH = ""   # Testing data file path (e.g., 2016.xlsx)
# =================================================================================

# ==========================================
# Main Program: Calibrate Precipitation Data using Standard GWR from mgwr
# ==========================================
if __name__ == "__main__":
    # 1. Validate Configuration Parameters
    if not TRAIN_DATA_PATH:
        print("Error: Please fill in TRAIN_DATA_PATH in the configuration area.")
        sys.exit(1)
    if not TEST_DATA_PATH:
        print("Error: Please fill in TEST_DATA_PATH in the configuration area.")
        sys.exit(1)

    # 2. Read Data
    print("Reading data...")
    try:
        train_df = pd.read_excel(TRAIN_DATA_PATH)
        test_df = pd.read_excel(TEST_DATA_PATH)
    except Exception as e:
        print(f"Failed to read Excel files: {e}")
        sys.exit(1)

    # Merge data for unified processing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # 3. Feature Engineering (Critical!)
    print("Building features...")
    all_data['Date'] = pd.to_datetime(all_data['Date'])

    # Extract time features to replace GTWR time kernel
    all_data['month'] = all_data['Date'].dt.month  # Month (1-12)
    all_data['dayofyear'] = all_data['Date'].dt.dayofyear  # Day of year (1-366)

    # Optional advanced technique: Sin/Cos transformation for monthly periodicity
    all_data['month_sin'] = np.sin(2 * np.pi * all_data['month'] / 12)
    all_data['month_cos'] = np.cos(2 * np.pi * all_data['month'] / 12)

    # 4. Define Variables
    y_col = 'D'  # Ground observation precipitation value
    products = ['E', 'L', 'F']  # Satellite precipitation products

    # Auxiliary factors + time features
    # Add month to capture seasonal effects
    aux_features = ['Avg_Temperature', 'Relative_Humidity', 'Elevation', 'month']

    # Clean missing values
    all_cols = [y_col, 'Longitude', 'Latitude', 'is_train'] + products + aux_features
    # Keep only existing columns
    all_cols = [c for c in all_cols if c in all_data.columns]
    all_data = all_data.dropna(subset=all_cols).reset_index(drop=True)

    # 5. Split Training and Testing Sets
    train_idx = all_data['is_train'] == 1
    test_idx = all_data['is_train'] == 0

    # Extract coordinates (mgwr requirement: Longitude first, then Latitude)
    coords = all_data[['Longitude', 'Latitude']].values
    coords_train = coords[train_idx]
    coords_test = coords[test_idx]

    # Process E, L, F products iteratively
    for product in products:
        print(f"\n{'=' * 50}")
        print(f"Processing product: {product}")
        print(f"{'=' * 50}")

        # Build feature matrix X
        # X = [satellite precipitation product, Avg_Temperature, Relative_Humidity, ..., month]
        feature_cols = [product] + aux_features

        # Ensure all features exist
        feature_cols = [f for f in feature_cols if f in all_data.columns]

        X = all_data[feature_cols].values
        y = all_data[y_col].values.reshape((-1, 1))  # mgwr requires y to be 2D (n, 1)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]

        # ---------------------------------------------------------
        # Core Step 1: Automatically select optimal bandwidth (AICc criterion)
        # ---------------------------------------------------------
        print("Searching for optimal spatial bandwidth (Neighbors) using AICc...")
        print("(If this step is too slow, manually set bw=25 since there are only 25 stations)")

        try:
            # Initialize bandwidth selector
            # kernel='gaussian': Gaussian kernel function
            # fixed=False: Adaptive bandwidth (based on neighbor count), suitable for uneven station distribution
            selector = Sel_BW(coords_train, y_train, X_train,
                              kernel='gaussian', fixed=False)

            # Start search (multi=False is faster, multi=True is more accurate but slower)
            optimal_bw = selector.search(criterion='AICc', multi=False)
            print(f"✅ Optimal bandwidth selected: {int(optimal_bw)} neighbors")

        except Exception as e:
            print(f"⚠️ Automatic bandwidth search failed, using default bandwidth. Error: {e}")
            # Set default bandwidth (global regression:25, local regression:10) since only 25 stations exist
            optimal_bw = 15

        # ---------------------------------------------------------
        # Core Step 2: Fit GWR model
        # ---------------------------------------------------------
        print("Fitting GWR model...")
        model = GWR(coords_train, y_train, X_train,
                    bw=optimal_bw, kernel='gaussian', fixed=False)

        results = model.fit()

        # Print model diagnostic report (Important for research/paper writing)
        print("\n=== Model Diagnostic Report ===")
        print(f"R-squared (Adjusted): {results.R2_adj:.4f}")
        print(f"AICc: {results.aicc:.4f}")
        print(f"Sum of Squared Residuals (SSR): {results.SSR:.4f}")
        # Uncomment below to view full coefficient table:
        # print(results.summary())

        # ---------------------------------------------------------
        # Core Step 3: Predict 2016 data
        # ---------------------------------------------------------
        print("\nPredicting 2016 data...")
        pred_results = model.predict(coords_test, X_test)
        predictions = pred_results.predictions.flatten()  # Convert to 1D array

        # ---------------------------------------------------------
        # Save Results
        # ---------------------------------------------------------
        # Extract test set data and merge predictions
        output_df = all_data[test_idx].copy()
        output_df[f'{product}_GWR_Pred'] = predictions

        # Save to Excel
        output_filename = f'MGWR_Result_{product}.xlsx'
        output_df.to_excel(output_filename, index=False)
        print(f"✅ Results saved to: {output_filename}")

    print("\n🎉 All products processed successfully!")