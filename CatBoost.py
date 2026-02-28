# %%
# ====================== 0. Environment and Warning Control ======================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from catboost import CatBoostRegressor
import os

# ====================== Global Random Seed ======================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# ====================== 1. Initialization ======================
print("===== [1/14] Initialization completed, starting core workflow =====")

# Universal relative paths for open source usage
data_path = './data/IMERG-F-train-2001-2015.xlsx'
test_data_path = './data/IMERG-F-test-2016.xlsx'

# ====================== 2. Load Data ======================
print("===== [2/14] Starting data loading =====")
df = pd.read_excel(data_path)
test_df = pd.read_excel(test_data_path)

# ====================== 3. Data Splitting (Time Extrapolation) ======================
train_df = df[df['year'] < 2015]
validation_df = df[df['year'] == 2015]

# ====================== 4. Features and Labels ======================
X_train = train_df.drop(columns=['Bias'])
y_train = train_df['Bias']

X_validation = validation_df.drop(columns=['Bias'])
y_validation = validation_df['Bias']

X_test = test_df.drop(columns=['Bias'])
y_test = test_df['Bias']

# ====================== 5. Hyperopt Search Space ======================
space = {
    'learning_rate': hp.uniform('learning_rate', 0.03, 0.15),
    'depth': hp.quniform('depth', 4, 10, 1),
    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'iterations': hp.quniform('iterations', 200, 600, 50)
}

MAX_EVALS = 100

# ====================== 6. Hyperopt Objective Function ======================
def objective(params):
    current_trial = len(trials.trials) + 1
    print(f"[Hyperopt] Trial {current_trial}/{MAX_EVALS}")

    model = CatBoostRegressor(
        loss_function='RMSE',
        learning_rate=params['learning_rate'],
        depth=int(params['depth']),
        l2_leaf_reg=params['l2_leaf_reg'],
        subsample=params['subsample'],
        iterations=int(params['iterations']),
        random_seed=GLOBAL_SEED,
        verbose=False
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_validation, y_validation),
        use_best_model=True
    )

    y_val_pred = model.predict(X_validation)
    mse = mean_squared_error(y_validation, y_val_pred)

    print(f"[Hyperopt] Trial {current_trial} - Validation MSE: {mse:.6f}")
    return {'loss': mse, 'status': STATUS_OK}

# ====================== 7. Hyperopt Search (With No-Progress Early Stop) ======================
print("===== [7/14] Starting Hyperopt hyperparameter search =====")

trials = Trials()

# >>> EARLY STOP Parameters <<<
patience = 20
min_delta = 1e-5
step = 5

best_loss = np.inf
no_improve_count = 0
current_evals = 0

while current_evals < MAX_EVALS:
    next_evals = min(current_evals + step, MAX_EVALS)

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=next_evals,
        trials=trials,
        rstate=np.random.default_rng(GLOBAL_SEED)
    )

    current_evals = next_evals
    current_best_loss = trials.best_trial['result']['loss']

    if best_loss - current_best_loss > min_delta:
        best_loss = current_best_loss
        no_improve_count = 0
    else:
        no_improve_count += step

    print(f"[EarlyStop Monitor] Best Loss: {best_loss:.6f} | "
          f"No Improve Trials: {no_improve_count}")

    if no_improve_count >= patience:
        print("🚨 Hyperopt early stop triggered (no progress).")
        break

print(f"===== Actual number of Hyperopt trials run: {len(trials.trials)} =====")

# ====================== Print Best Trial Parameters ======================
best_trial = trials.best_trial
best_trial_params = best_trial['misc']['vals']

print("\n" + "="*60)
print("✅ Hyperopt search completed, best trial parameters are as follows:")
print(f"Trial ID: {best_trial['tid'] + 1}")
print(f"Validation MSE: {best_trial['result']['loss']:.6f}")

print("Best Hyperparameters:")
print(f"  learning_rate : {best_trial_params['learning_rate'][0]:.5f}")
print(f"  depth         : {int(best_trial_params['depth'][0])}")
print(f"  l2_leaf_reg   : {best_trial_params['l2_leaf_reg'][0]:.5f}")
print(f"  subsample     : {best_trial_params['subsample'][0]:.5f}")
print(f"  iterations    : {int(best_trial_params['iterations'][0])}")
print("="*60)

# ====================== 8. Optimal Model Parameters ======================
best_params = {
    'loss_function': 'RMSE',
    'learning_rate': best_trial_params['learning_rate'][0],
    'depth': int(best_trial_params['depth'][0]),
    'l2_leaf_reg': best_trial_params['l2_leaf_reg'][0],
    'subsample': best_trial_params['subsample'][0],
    'iterations': int(best_trial_params['iterations'][0]),
    'random_seed': GLOBAL_SEED,
    'verbose': False
}

# ====================== 9. Bootstrap Prediction (500 Iterations) + Residual-Corrected CI ======================
print("===== [9/14] Bootstrap prediction (500 iterations) + residual-corrected CI =====")

n_bootstrap = 500
y_test_preds = []

for i in range(n_bootstrap):
    X_bs, y_bs = resample(X_train, y_train, random_state=GLOBAL_SEED + i)
    model_bs = CatBoostRegressor(**best_params)
    model_bs.fit(X_bs, y_bs, verbose=False)
    y_test_preds.append(model_bs.predict(X_test))

y_test_preds = np.array(y_test_preds)
y_pred_mean = y_test_preds.mean(axis=0)

# ====================== Residual-Corrected 95% Confidence Interval (CI) ======================
best_model = CatBoostRegressor(**best_params)
best_model.fit(X_train, y_train, verbose=False)

train_residuals = y_train - best_model.predict(X_train)
residual_std = np.std(train_residuals)

y_pred_lower = y_pred_mean - 1.96 * residual_std
y_pred_upper = y_pred_mean + 1.96 * residual_std

# ====================== Coverage Rate & CI Width ======================
within_ci = np.mean((y_test >= y_pred_lower) & (y_test <= y_pred_upper)) * 100
ci_width = y_pred_upper - y_pred_lower

print(f"Observations within 95% CI: {within_ci:.2f}%")
print(f"Mean 95% CI width   : {np.mean(ci_width):.4f}")
print(f"Median 95% CI width : {np.median(ci_width):.4f}")

# ====================== 10. Evaluation Metrics ======================
mae = mean_absolute_error(y_test, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")

# ====================== 11. Save Prediction Results ======================
output_dir = './results/catboost/'
os.makedirs(output_dir, exist_ok=True)

test_df['Predicted_Mean'] = y_pred_mean
test_df['Predicted_Lower95'] = y_pred_lower
test_df['Predicted_Upper95'] = y_pred_upper
test_df['CI_Width_95'] = ci_width
test_df.to_csv(os.path.join(output_dir, 'IMERG-F-catboost-bootstrap-residual.csv'), index=False)

# ====================== 12. SHAP Analysis (CatBoost) ======================
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Calculate Mean |SHAP|
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'MeanAbsSHAP': mean_abs_shap
}).sort_values(by='MeanAbsSHAP', ascending=False)

importance_df.to_csv(
    os.path.join(output_dir, 'IMERG-F-catboost-shap_importances.csv'),
    index=False
)

# Select top 10 features
top_features = importance_df['Feature'].head(10).tolist()
top_idx = [X_test.columns.get_loc(f) for f in top_features]

shap_top = shap_values[:, top_idx]
X_test_top = X_test[top_features]
mean_top = importance_df.head(10)['MeanAbsSHAP'].values

# ====================== Side-by-Side Plotting ======================
fig, (ax_left, ax_right) = plt.subplots(
    ncols=2, figsize=(14, 7),
    gridspec_kw={'width_ratios': [1, 2]}
)

# ---- Left: Mean SHAP ----
y_pos = np.arange(len(top_features))
ax_left.barh(y_pos, mean_top, color='#1f77b4')
ax_left.set_yticks(y_pos)
ax_left.set_yticklabels(top_features)
ax_left.invert_yaxis()
ax_left.set_xlabel('Mean |SHAP|')
ax_left.set_title('Mean SHAP Importance')

# Remove redundant borders from left plot
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)

# ---- Right: SHAP Distribution ----
shap.summary_plot(
    shap_top,
    X_test_top,
    show=False,
    plot_size=None
)

ax_right = plt.gca()
ax_right.set_title('SHAP Value Distribution')

# 🔴 Key: Completely disable the right plot's own y-axis system
ax_right.set_yticks([])
ax_right.set_yticklabels([])
ax_right.set_ylabel('')

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, 'F-CatBoost_SHAP_top10_combined.png'),
    dpi=900
)
plt.show()

# ====================== 13. Prediction Comparison Plot (with 95% CI) ======================
plt.figure(figsize=(16, 8))
plt.plot(y_test.reset_index(drop=True), label='Observed', linewidth=2)
plt.plot(y_pred_mean, label='Predicted', linewidth=2)
plt.fill_between(range(len(y_test)), y_pred_lower, y_pred_upper,
                 color='gray', alpha=0.3, label='95% CI')

plt.xlabel('Sample Index')
plt.ylabel('Bias')
plt.title(f'Observed vs Predicted Bias (CatBoost) with 95% CI\nCoverage: {within_ci:.2f}%')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'F-CatBoost_prediction_comparison_CI.png'), dpi=900)
plt.show()

print(f"✅ CatBoost full workflow completed, results saved to: {output_dir}")