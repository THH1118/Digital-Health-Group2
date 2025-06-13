# %%
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import gc
import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
from sklearn.preprocessing import OneHotEncoder

# %% [markdown]
# ### Data Setup

# %%
# Remove the row that have missing vlaues in 'outop' column
def remove_missing_values(df):
    return df.dropna(subset=['outop', 'esrd yrs', 'hd or capd', 'dm', 'htn', 'bw', 'hbs'])

# Remove the row that have the 'Post1wCa' > 50
def remove_outliers(df):
    return df[df['age'] >= 0]

# If 'Post1wCa' > 50, then divide it by 10
def divide_outliers(df):
    df.loc[df['Post1wCa'] > 50, 'Post1wCa'] = df['Post1wCa'] / 10
    return df

# %%
# Import the necessary library
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_excel('/home/jupyter-yuchieh/HBS prediction project/HBS_data_with_Post1wCa.xlsx')
df = remove_missing_values(df)
df = remove_outliers(df)
df = divide_outliers(df)
df = df.drop(columns=['type', 're-op', 'hbs'])

# Use all available numeric features for initial selection
numeric_features = df.select_dtypes(include=['number']).columns.tolist()
numeric_features = [col for col in numeric_features if col != 'Post1wCa']  # Exclude target

# Prepare data for RFECV
X_rfecv = df[numeric_features].copy()  
y = df['Post1wCa']

# Handle categorical features
categorical_features = ['sex', 'bone pain', 'itching', 'weakness', 'calciphylaxis', 
                       'dm', 'htn', 'thymectomy', 'thyroidectomy', 'outop', 'hd or capd']
categorical_features = [col for col in categorical_features if col in df.columns]
for col in categorical_features:
    X_rfecv[col] = df[col].astype('category')  # Now we're working with a proper copy

# Initialize RFECV with a Random Forest estimator
estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
selector = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    scoring='neg_root_mean_squared_error',
    min_features_to_select=5,
    n_jobs=-1,
    verbose=1
)

selector.fit(X_rfecv, y)

importances = selector.estimator_.feature_importances_
support = selector.support_

# Debug and make sure all have the same length
print(f"X_rfecv.columns shape: {X_rfecv.columns.shape}")
print(f"importances shape: {importances.shape}")
print(f"support shape: {support.shape}")

features = X_rfecv.columns
if len(features) != len(importances) or len(features) != len(support):
    features = X_rfecv.columns[support]
    importances = importances[:len(features)]  

# Create a DataFrame with matching lengths
feature_ranks = pd.DataFrame({
    'feature': features,
    'importance': importances[:len(features)],
    'selected': support[:len(features)]
})

# Plot CV scores vs number of features
plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Mean negative RMSE (higher is better)")
plt.plot(
    range(1, len(selector.cv_results_['mean_test_score']) + 1),
    selector.cv_results_['mean_test_score'],
    marker='o'
)
plt.title("RFECV: Feature Selection")
plt.tight_layout()
plt.show()

# Plot feature importances
plt.figure(figsize=(10, 8))
feature_importance_df = feature_ranks.sort_values('importance', ascending=False)
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel('Feature Importance')
plt.title('Features Ranked by Importance')
plt.tight_layout()
plt.show()

# Sort by importance and take top 15
top_15_features = feature_ranks.sort_values('importance', ascending=False).head(15)['feature'].tolist()
print(f"Top 15 features by importance: {top_15_features}")

X = df[top_15_features]


# %% [markdown]
# ### Use CatBoost Regressor

# %%
def objective(trial, X, y, random_states=[42, 101, 123, 456, 789]):
    """
    Optuna objective function that evaluates parameters across multiple random seeds
    to ensure stability of performance.
    """
    # Define the hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.01, log=True),
        'depth': trial.suggest_int('depth', 2, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'border_count': trial.suggest_categorical('border_count', [128, 254]),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.5, 5.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 5, 15)
    }
    
    # Store results for each random seed
    rmse_scores = []
    mape_scores = []
    r2_scores = []
    
    # Evaluate the same parameters across different random seeds
    for seed in random_states:
        # Split data with current seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Select categorical features from current train set
        categorical_features_current = [col for col in categorical_features if col in X_train.columns]
        
        # Convert categorical features to string type (required for CatBoost)
        for col in categorical_features_current:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
        
        # Initialize and train model
        model = CatBoostRegressor(
            **params,
            random_seed=seed,
            cat_features=categorical_features_current,
            verbose=0
        )
        
        # Use early stopping to speed up training
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Store metrics
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mape_scores.append(mape)
        
        # Free memory
        del model
        gc.collect()
    
    # Calculate both average performance and consistency
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    
    # Return a combination of mean and standard deviation to reward both performance and stability
    # A lower standard deviation means more stable results across seeds
    stability_penalty = std_rmse / avg_rmse  # Normalized stability measure
    
    # Combine average performance and stability (lower is better)
    # You can adjust the weight (0.2) to emphasize stability more or less
    final_score = avg_rmse * (1 + 0.2 * stability_penalty)
    
    # Log detailed results for this trial
    trial.set_user_attr('avg_rmse', avg_rmse)
    trial.set_user_attr('std_rmse', std_rmse)
    trial.set_user_attr('avg_mape', np.mean(mape_scores))
    trial.set_user_attr('avg_r2', np.mean(r2_scores))
    
    return final_score

# %%
# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X, y), n_trials=200, show_progress_bar=True)

# Get best parameters
best_params = study.best_params
print("Best parameters:", best_params)
print(f"Best average RMSE: {study.best_trial.user_attrs['avg_rmse']:.4f} ± {study.best_trial.user_attrs['std_rmse']:.4f}")
print(f"Best average MAPE: {study.best_trial.user_attrs['avg_mape']:.4f}")
print(f"Best average R²: {study.best_trial.user_attrs['avg_r2']:.4f}")

# %%
# Visualize parameter importances
fig = plot_param_importances(study)
fig.show()

# Visualize optimization history
fig = plot_optimization_history(study)
fig.show()

# Detailed results across all trials
trial_data = []
for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        params = trial.params.copy()
        params.update({
            'avg_rmse': trial.user_attrs.get('avg_rmse', float('nan')),
            'std_rmse': trial.user_attrs.get('std_rmse', float('nan')),
            'stability': trial.user_attrs.get('std_rmse', float('nan')) / trial.user_attrs.get('avg_rmse', float('nan')),
            'avg_mape': trial.user_attrs.get('avg_mape', float('nan')),
            'avg_r2': trial.user_attrs.get('avg_r2', float('nan')),
        })
        trial_data.append(params)

results_df = pd.DataFrame(trial_data)
results_df = results_df.sort_values('avg_rmse')
print(results_df.head(10))  # Top 10 parameter sets

# %%
# Final evaluation with best parameters across all seeds
random_states = [42, 101, 123, 456, 789]
final_results = {'rmse': [], 'r2': [], 'mape': []}

for seed in random_states:
    print(f"\n--- Evaluating best parameters on seed {seed} ---")
    
    # Split data with current seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Select categorical features from current train set
    categorical_features_current = [col for col in categorical_features if col in X_train.columns]
    
    # Convert categorical features to string type
    for col in categorical_features_current:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # Train model with best parameters
    best_model = CatBoostRegressor(
        **best_params,
        random_seed=seed,
        cat_features=categorical_features_current,
        verbose=0
    )
    best_model.fit(X_train, y_train)
    
    # Predict
    y_pred = best_model.predict(X_test)
    
    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Store results
    final_results['rmse'].append(rmse)
    final_results['r2'].append(r2)
    final_results['mape'].append(mape)
    
    # Print results
    print(f"Seed {seed} - RMSE: {rmse:.4f}")
    print(f"Seed {seed} - R²: {r2:.4f}")
    print(f"Seed {seed} - MAPE: {mape:.4f}")

# Calculate and print average results
print("\n--- Final Results Across All Seeds ---")
print(f'Average RMSE: {np.mean(final_results["rmse"]):.4f} ± {np.std(final_results["rmse"]):.4f}')
print(f'Average R²: {np.mean(final_results["r2"]):.4f} ± {np.std(final_results["r2"]):.4f}')
print(f'Average MAPE: {np.mean(final_results["mape"]):.4f} ± {np.std(final_results["mape"]):.4f}')

# %% [markdown]
# ### XGBoost regressor

# %%
def xgb_objective(trial, X, y, random_states=[42, 101, 123, 456, 789]):
    """
    Optuna objective function for XGBoost that evaluates parameters across multiple random seeds
    to ensure stability of performance.
    """
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0)
    }
    
    # Store results for each random seed
    rmse_scores = []
    mape_scores = []
    r2_scores = []
    
    # Evaluate the same parameters across different random seeds
    for seed in random_states:
        # Split data with current seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Convert categorical features
        X_train_cat = to_category(X_train)
        X_test_cat = to_category(X_test)
        
        # Initialize and train model
        model = XGBRegressor(
            **params,
            random_state=seed,
            enable_categorical=True,
            verbosity=0
        )
        
        # Train with early stopping
        # Fix: Use callbacks for early stopping
        eval_set = [(X_test_cat, y_test)]
        model.fit(
            X_train_cat, y_train
        )
        
        # Make predictions
        y_pred = model.predict(X_test_cat)
        
        # Calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Store metrics
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mape_scores.append(mape)
        
        # Free memory
        del model
        gc.collect()
    
    # Calculate both average performance and consistency
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    
    # Return a combination of mean and standard deviation to reward both performance and stability
    stability_penalty = std_rmse / avg_rmse  # Normalized stability measure
    
    # Combine average performance and stability (lower is better)
    final_score = avg_rmse * (1 + 0.2 * stability_penalty)
    
    # Log detailed results for this trial
    trial.set_user_attr('avg_rmse', avg_rmse)
    trial.set_user_attr('std_rmse', std_rmse)
    trial.set_user_attr('avg_mape', np.mean(mape_scores))
    trial.set_user_attr('avg_r2', np.mean(r2_scores))
    
    return final_score

# %%
# Define helper function if not already available
def to_category(df):
    df_copy = df.copy()
    for col in categorical_features:
        if col in df.columns:
            df_copy[col] = df_copy[col].astype('category')
    return df_copy

# Create a study object and optimize the objective function
xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(lambda trial: xgb_objective(trial, X, y), n_trials=200, show_progress_bar=True, n_jobs=-1)

# Get best parameters
best_xgb_params = xgb_study.best_params
print("Best XGBoost parameters:", best_xgb_params)
print(f"Best average RMSE: {xgb_study.best_trial.user_attrs['avg_rmse']:.4f} ± {xgb_study.best_trial.user_attrs['std_rmse']:.4f}")
print(f"Best average MAPE: {xgb_study.best_trial.user_attrs['avg_mape']:.4f}")
print(f"Best average R²: {xgb_study.best_trial.user_attrs['avg_r2']:.4f}")

# %%
# Visualize parameter importances
fig = plot_param_importances(xgb_study)
fig.show()

# Visualize optimization history
fig = plot_optimization_history(xgb_study)
fig.show()

# Detailed results across all trials
xgb_trial_data = []
for trial in xgb_study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        params = trial.params.copy()
        params.update({
            'avg_rmse': trial.user_attrs.get('avg_rmse', float('nan')),
            'std_rmse': trial.user_attrs.get('std_rmse', float('nan')),
            'stability': trial.user_attrs.get('std_rmse', float('nan')) / trial.user_attrs.get('avg_rmse', float('nan')),
            'avg_mape': trial.user_attrs.get('avg_mape', float('nan')),
            'avg_r2': trial.user_attrs.get('avg_r2', float('nan')),
        })
        xgb_trial_data.append(params)

xgb_results_df = pd.DataFrame(xgb_trial_data)
xgb_results_df = xgb_results_df.sort_values('avg_rmse')
print(xgb_results_df.head(10))  # Top 10 parameter sets

# %%
# Final evaluation with best parameters across all seeds
random_states = [42, 101, 123, 456, 789]
xgb_final_results = {'rmse': [], 'r2': [], 'mape': []}

for seed in random_states:
    print(f"\n--- Evaluating best XGBoost parameters on seed {seed} ---")
    
    # Split data with current seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Convert categorical features
    X_train_cat = to_category(X_train)
    X_test_cat = to_category(X_test)
    
    # Train model with best parameters
    best_xgb_model = XGBRegressor(
        **best_xgb_params,
        random_state=seed,
        enable_categorical=True,
        verbosity=0
    )
    best_xgb_model.fit(X_train_cat, y_train)
    
    # Predict
    y_pred = best_xgb_model.predict(X_test_cat)
    
    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Store results
    xgb_final_results['rmse'].append(rmse)
    xgb_final_results['r2'].append(r2)
    xgb_final_results['mape'].append(mape)
    
    # Print results
    print(f"Seed {seed} - RMSE: {rmse:.4f}")
    print(f"Seed {seed} - R²: {r2:.4f}")
    print(f"Seed {seed} - MAPE: {mape:.4f}")

# Calculate and print average results
print("\n--- Final XGBoost Results Across All Seeds ---")
print(f'Average RMSE: {np.mean(xgb_final_results["rmse"]):.4f} ± {np.std(xgb_final_results["rmse"]):.4f}')
print(f'Average R²: {np.mean(xgb_final_results["r2"]):.4f} ± {np.std(xgb_final_results["r2"]):.4f}')
print(f'Average MAPE: {np.mean(xgb_final_results["mape"]):.4f} ± {np.std(xgb_final_results["mape"]):.4f}')

# %% [markdown]
# ### Use RandomForest Regressor

# %%
def rf_objective(trial, X, y, random_states=[42, 101, 123, 456, 789]):
    """
    Optuna objective function for Random Forest that evaluates parameters across multiple random seeds
    to ensure stability of performance.
    """
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    
    # Store results for each random seed
    rmse_scores = []
    mape_scores = []
    r2_scores = []
    
    # Evaluate the same parameters across different random seeds
    for seed in random_states:
        # Split data with current seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Convert categorical features
        for col in categorical_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype('category')
                X_test[col] = X_test[col].astype('category')
        
        # Initialize and train model
        model = RandomForestRegressor(
            **params,
            random_state=seed,
            n_jobs=-1  # Use all available cores
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Store metrics
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mape_scores.append(mape)
        
        # Free memory
        del model
        gc.collect()
    
    # Calculate both average performance and consistency
    avg_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    
    # Return a combination of mean and standard deviation to reward both performance and stability
    stability_penalty = std_rmse / avg_rmse  # Normalized stability measure
    
    # Combine average performance and stability (lower is better)
    final_score = avg_rmse * (1 + 0.2 * stability_penalty)
    
    # Log detailed results for this trial
    trial.set_user_attr('avg_rmse', avg_rmse)
    trial.set_user_attr('std_rmse', std_rmse)
    trial.set_user_attr('avg_mape', np.mean(mape_scores))
    trial.set_user_attr('avg_r2', np.mean(r2_scores))
    
    return final_score

# %%
# Create a study object and optimize the objective function
rf_study = optuna.create_study(direction='minimize')
rf_study.optimize(lambda trial: rf_objective(trial, X, y), n_trials=200, show_progress_bar=True)

# Get best parameters
best_rf_params = rf_study.best_params
print("Best Random Forest parameters:", best_rf_params)
print(f"Best average RMSE: {rf_study.best_trial.user_attrs['avg_rmse']:.4f} ± {rf_study.best_trial.user_attrs['std_rmse']:.4f}")
print(f"Best average MAPE: {rf_study.best_trial.user_attrs['avg_mape']:.4f}")
print(f"Best average R²: {rf_study.best_trial.user_attrs['avg_r2']:.4f}")

# Visualize parameter importances
fig = plot_param_importances(rf_study)
fig.show()

# Visualize optimization history
fig = plot_optimization_history(rf_study)
fig.show()

# %%
# Final evaluation with best parameters across all seeds
random_states = [42, 101, 123, 456, 789]
rf_final_results = {'rmse': [], 'r2': [], 'mape': []}

for seed in random_states:
    print(f"\n--- Evaluating best Random Forest parameters on seed {seed} ---")
    
    # Split data with current seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Convert categorical features
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    # Train model with best parameters
    best_rf_model = RandomForestRegressor(
        **best_rf_params,
        random_state=seed,
        n_jobs=-1
    )
    best_rf_model.fit(X_train, y_train)
    
    # Predict
    y_pred = best_rf_model.predict(X_test)
    
    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Store results
    rf_final_results['rmse'].append(rmse)
    rf_final_results['r2'].append(r2)
    rf_final_results['mape'].append(mape)
    
    # Print results
    print(f"Seed {seed} - RMSE: {rmse:.4f}")
    print(f"Seed {seed} - R²: {r2:.4f}")
    print(f"Seed {seed} - MAPE: {mape:.4f}")

# Calculate and print average results
print("\n--- Final Random Forest Results Across All Seeds ---")
print(f'Average RMSE: {np.mean(rf_final_results["rmse"]):.4f} ± {np.std(rf_final_results["rmse"]):.4f}')
print(f'Average R²: {np.mean(rf_final_results["r2"]):.4f} ± {np.std(rf_final_results["r2"]):.4f}')
print(f'Average MAPE: {np.mean(rf_final_results["mape"]):.4f} ± {np.std(rf_final_results["mape"]):.4f}')

# %% [markdown]
# ### Try Stacking models

# %%
# Import additional required libraries
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, BayesianRidge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import seaborn as sns


# Helper function to convert features for CatBoost
def to_string(df):
    df_copy = df.copy()
    for col in categorical_features:
        if col in df.columns:
            df_copy[col] = df_copy[col].astype(str)
    return df_copy

# Define helper function if not already available
def to_category(df):
    df_copy = df.copy()
    for col in categorical_features:
        if col in df.columns:
            df_copy[col] = df_copy[col].astype('category')
    return df_copy

# Function to prepare appropriate inputs for each model
def preprocess_for_model(X, model_type):
    if model_type == 'catboost':
        return to_string(X)
    elif model_type in ['xgboost', 'randomforest']:
        return to_category(X)
    else:
        return X

# %%
def stacking_with_multiple_seeds(X, y, random_states=[42, 101, 123, 456, 789]):
    """
    Implement stacking regression with optimized models across multiple random seeds
    
    Parameters:
    -----------
    X : DataFrame - Features
    y : Series - Target variable
    random_states : list - Random seeds to use for data splits
    
    Returns:
    --------
    Dict containing performance metrics across all seeds
    """
    
    # Define the base models with best parameters from optimization
    base_models = {
        'catboost': {
            'model': CatBoostRegressor,
            'params': {'iterations': 1962, 'learning_rate': 0.009647526713336886, 'depth': 2, 'l2_leaf_reg': 9.65613961284073, 'border_count': 128, 'bagging_temperature': 0.21892553675181112, 'random_strength': 4.9470084986713525, 'min_data_in_leaf': 7, 'leaf_estimation_iterations': 7},
            'extra_args': {'cat_features': None, 'verbose': 0},
            'preprocess': 'catboost'
        },
        'xgboost': {
            'model': XGBRegressor,
            'params': {'n_estimators': 881, 'max_depth': 2, 'learning_rate': 0.00699066929802139, 'subsample': 0.7444195982203315, 'colsample_bytree': 0.3158458749432831, 'min_child_weight': 9, 'gamma': 0.0025879069161366397, 'reg_alpha': 0.02178426761143437, 'reg_lambda': 6.500258743742415},
            'extra_args': {'enable_categorical': True, 'verbosity': 0},
            'preprocess': 'xgboost'
        },
        'randomforest': {
            'model': RandomForestRegressor,
            'params': {'n_estimators': 401, 'max_depth': 25, 'min_samples_split': 11, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'bootstrap': True},
            'extra_args': {'n_jobs': -1},
            'preprocess': 'randomforest'
        }
    }
    
    # Define meta-learners to try
    meta_learners = {
        'ridge': Ridge(alpha=1.0),
        'bayesian_ridge': BayesianRidge(),
        'huber': HuberRegressor()
    }
    
    # Store results for each random seed
    seed_results = {
        'individual_models': {
            'catboost': {'rmse': [], 'r2': [], 'mape': []},
            'xgboost': {'rmse': [], 'r2': [], 'mape': []},
            'randomforest': {'rmse': [], 'r2': [], 'mape': []}
        },
        'stacking': {
            meta_name: {'rmse': [], 'r2': [], 'mape': []} for meta_name in meta_learners
        },
        'ensemble_avg': {'rmse': [], 'r2': [], 'mape': []}
    }
    
    # Evaluate across multiple random seeds
    for seed in random_states:
        print(f"\n--- Evaluating stacking with seed {seed} ---")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Setup cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        
        # Create arrays for base models' predictions
        oof_preds = {}  # Out-of-fold predictions on training data
        test_preds = {}  # Predictions on test data
        
        # For each base model, generate out-of-fold predictions
        for model_name, model_config in base_models.items():
            print(f"Training {model_name}...")
            
            # Initialize arrays for this model's predictions
            oof_preds[model_name] = np.zeros(len(X_train))
            test_preds[model_name] = np.zeros(len(X_test))
            
            # Track fold predictions for test data
            fold_test_preds = []
            
            # For each fold in cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                # Split data for this fold
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Preprocess data based on model requirements
                X_tr_proc = preprocess_for_model(X_tr, model_config['preprocess'])
                X_val_proc = preprocess_for_model(X_val, model_config['preprocess'])
                X_test_proc = preprocess_for_model(X_test, model_config['preprocess'])
                
                # Set up categorical features for CatBoost
                extra_args = model_config['extra_args'].copy()
                if model_name == 'catboost':
                    extra_args['cat_features'] = [i for i, col in enumerate(X_tr_proc.columns) 
                                               if col in categorical_features]
                
                # Create and train model
                model = model_config['model'](
                    **model_config['params'],
                    random_state=seed,
                    **extra_args
                )
                
                # Train model
                if model_name == 'catboost':
                    model.fit(
                        X_tr_proc, y_tr,
                        eval_set=[(X_val_proc, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_tr_proc, y_tr)
                
                # Make predictions for this fold
                oof_preds[model_name][val_idx] = model.predict(X_val_proc)
                fold_test_preds.append(model.predict(X_test_proc))
                
                # Free memory
                del model
                gc.collect()
            
            # Average test predictions across folds
            test_preds[model_name] = np.mean(fold_test_preds, axis=0)
            
            # Evaluate individual model performance on test set
            rmse = root_mean_squared_error(y_test, test_preds[model_name])
            r2 = r2_score(y_test, test_preds[model_name])
            mape = mean_absolute_percentage_error(y_test, test_preds[model_name])
            
            seed_results['individual_models'][model_name]['rmse'].append(rmse)
            seed_results['individual_models'][model_name]['r2'].append(r2)
            seed_results['individual_models'][model_name]['mape'].append(mape)
            
            print(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
        
        # Create meta-features for training and testing
        meta_train = np.column_stack([oof_preds[model] for model in base_models.keys()])
        meta_test = np.column_stack([test_preds[model] for model in base_models.keys()])
        
        # Try different meta-learners
        for meta_name, meta_model in meta_learners.items():
            print(f"Training meta-learner: {meta_name}...")
            
            # Train meta-learner
            meta_model.fit(meta_train, y_train)
            
            # Make final predictions
            final_preds = meta_model.predict(meta_test)
            
            # Evaluate
            rmse = root_mean_squared_error(y_test, final_preds)
            r2 = r2_score(y_test, final_preds)
            mape = mean_absolute_percentage_error(y_test, final_preds)
            
            seed_results['stacking'][meta_name]['rmse'].append(rmse)
            seed_results['stacking'][meta_name]['r2'].append(r2)
            seed_results['stacking'][meta_name]['mape'].append(mape)
            
            print(f"Stacking with {meta_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
        
        # Also try simple averaging ensemble
        avg_preds = np.mean([test_preds[model] for model in base_models.keys()], axis=0)
        rmse = root_mean_squared_error(y_test, avg_preds)
        r2 = r2_score(y_test, avg_preds)
        mape = mean_absolute_percentage_error(y_test, avg_preds)
        
        seed_results['ensemble_avg']['rmse'].append(rmse)
        seed_results['ensemble_avg']['r2'].append(r2)
        seed_results['ensemble_avg']['mape'].append(mape)
        
        print(f"Simple averaging ensemble - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
    
    # Print summary results across all seeds
    print("\n=== Summary Results Across All Seeds ===")
    
    # Individual models
    print("\nIndividual Models:")
    for model_name in base_models.keys():
        mean_rmse = np.mean(seed_results['individual_models'][model_name]['rmse'])
        std_rmse = np.std(seed_results['individual_models'][model_name]['rmse'])
        mean_r2 = np.mean(seed_results['individual_models'][model_name]['r2'])
        mean_mape = np.mean(seed_results['individual_models'][model_name]['mape'])
        
        print(f"{model_name.capitalize()} - RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}, "
              f"R²: {mean_r2:.4f}, MAPE: {mean_mape:.4f}")
    
    # Stacking results
    print("\nStacking Results:")
    for meta_name in meta_learners.keys():
        mean_rmse = np.mean(seed_results['stacking'][meta_name]['rmse'])
        std_rmse = np.std(seed_results['stacking'][meta_name]['rmse'])
        mean_r2 = np.mean(seed_results['stacking'][meta_name]['r2'])
        mean_mape = np.mean(seed_results['stacking'][meta_name]['mape'])
        
        print(f"Stacking with {meta_name} - RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}, "
              f"R²: {mean_r2:.4f}, MAPE: {mean_mape:.4f}")
    
    # Simple averaging
    mean_rmse = np.mean(seed_results['ensemble_avg']['rmse'])
    std_rmse = np.std(seed_results['ensemble_avg']['rmse'])
    mean_r2 = np.mean(seed_results['ensemble_avg']['r2'])
    mean_mape = np.mean(seed_results['ensemble_avg']['mape'])
    
    print(f"\nSimple averaging - RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}, "
          f"R²: {mean_r2:.4f}, MAPE: {mean_mape:.4f}")
    
    return seed_results

# %%
# Run stacking with multiple random seeds
random_states = [42, 101, 123, 456, 789]
stacking_results = stacking_with_multiple_seeds(X, y, random_states)

# %%
def plot_model_comparison(results):
    """
    Create a comparison plot of all models and ensembles
    """
    # Prepare data for plotting
    model_names = []
    rmse_means = []
    rmse_stds = []
    r2_means = []
    
    # Add individual models
    for model_name in results['individual_models'].keys():
        model_names.append(model_name.capitalize())
        rmse_values = results['individual_models'][model_name]['rmse']
        rmse_means.append(np.mean(rmse_values))
        rmse_stds.append(np.std(rmse_values))
        r2_means.append(np.mean(results['individual_models'][model_name]['r2']))
    
    # Add stacking results
    for meta_name in results['stacking'].keys():
        model_names.append(f"Stack ({meta_name})")
        rmse_values = results['stacking'][meta_name]['rmse']
        rmse_means.append(np.mean(rmse_values))
        rmse_stds.append(np.std(rmse_values))
        r2_means.append(np.mean(results['stacking'][meta_name]['r2']))
    
    # Add simple averaging
    model_names.append("Simple Avg")
    rmse_values = results['ensemble_avg']['rmse']
    rmse_means.append(np.mean(rmse_values))
    rmse_stds.append(np.std(rmse_values))
    r2_means.append(np.mean(results['ensemble_avg']['r2']))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE plot (lower is better)
    ax1.barh(model_names, rmse_means, xerr=rmse_stds, capsize=5)
    ax1.set_xlabel('RMSE (lower is better)')
    ax1.set_title('Model Comparison: RMSE')
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Highlight best model
    best_idx = np.argmin(rmse_means)
    ax1.get_children()[best_idx].set_color('green')
    
    # R² plot (higher is better)
    ax2.barh(model_names, r2_means)
    ax2.set_xlabel('R² (higher is better)')
    ax2.set_title('Model Comparison: R²')
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Highlight best model
    best_idx = np.argmax(r2_means)
    ax2.get_children()[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

# Plot comparison of all models
comparison_fig = plot_model_comparison(stacking_results)
comparison_fig.show()

# %% [markdown]
# #### Creating figures

# %%
best_params = {'iterations': 1962, 'learning_rate': 0.009647526713336886, 'depth': 2, 'l2_leaf_reg': 9.65613961284073, 'border_count': 128, 'bagging_temperature': 0.21892553675181112, 'random_strength': 4.9470084986713525, 'min_data_in_leaf': 7, 'leaf_estimation_iterations': 7}
best_xgb_params = {'n_estimators': 881, 'max_depth': 2, 'learning_rate': 0.00699066929802139, 'subsample': 0.7444195982203315, 'colsample_bytree': 0.3158458749432831, 'min_child_weight': 9, 'gamma': 0.0025879069161366397, 'reg_alpha': 0.02178426761143437, 'reg_lambda': 6.500258743742415}
best_rf_params = {'n_estimators': 401, 'max_depth': 25, 'min_samples_split': 11, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'bootstrap': True}

# %%
def calculate_simple_average_predictions(X_train, X_test, y_train):
    """Helper function to get simple average ensemble predictions"""
    # Get predictions from each base model
    # CatBoost
    X_train_cat = to_string(X_train)
    X_test_cat = to_string(X_test)
    cat_model = CatBoostRegressor(
        **best_params,
        random_seed=42,
        cat_features=[i for i, col in enumerate(X_train_cat.columns) if col in categorical_features],
        verbose=0
    )
    cat_model.fit(X_train_cat, y_train)
    cat_preds = cat_model.predict(X_test_cat)
    
    # XGBoost
    X_train_xgb = to_category(X_train)
    X_test_xgb = to_category(X_test)
    xgb_model = XGBRegressor(
        **best_xgb_params,
        random_state=42,
        enable_categorical=True,
        verbosity=0
    )
    xgb_model.fit(X_train_xgb, y_train)
    xgb_preds = xgb_model.predict(X_test_xgb)
    
    # RandomForest
    X_train_rf = to_category(X_train)
    X_test_rf = to_category(X_test)
    rf_model = RandomForestRegressor(
        **best_rf_params,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_rf, y_train)
    rf_preds = rf_model.predict(X_test_rf)
    
    # Calculate simple average
    y_pred = (cat_preds + xgb_preds + rf_preds) / 3
    
    return y_pred

# %%
def create_performance_table(X, y, model_type, random_states=[42, 101, 123, 456, 789]):
    """
    Create a comprehensive performance table for the specified model with error breakdowns
    """
    results = {
        'ape_under_5': [],
        'ape_5_to_10': [],
        'ape_10_to_15': [],
        'ape_over_15': [],
        'mae': [],
        'mape': [],
        'r2': [],
        'rmse': [],
        'overestimation': [],
        'underestimation': []
    }
    
    for seed in random_states:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Train and predict with selected model
        if model_type == 'linear_regression':
            # Handle linear regression
            categorical_cols = [col for col in categorical_features if col in X_train.columns]
            numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numeric_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
                ])
            
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        elif model_type == 'catboost':
            X_train_proc = to_string(X_train)
            X_test_proc = to_string(X_test)
            model = CatBoostRegressor(
                **best_params,
                random_seed=seed,
                cat_features=[i for i, col in enumerate(X_train_proc.columns) if col in categorical_features],
                verbose=0
            )
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            
        elif model_type == 'xgboost':
            X_train_proc = to_category(X_train)
            X_test_proc = to_category(X_test)
            model = XGBRegressor(
                **best_xgb_params,
                random_state=seed,
                enable_categorical=True,
                verbosity=0
            )
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            
        elif model_type == 'randomforest':
            X_train_proc = to_category(X_train)
            X_test_proc = to_category(X_test)
            model = RandomForestRegressor(
                **best_rf_params,
                random_state=seed,
                n_jobs=-1
            )
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            
        elif model_type == 'simple_average':
            # For simple average ensemble
            y_pred = calculate_simple_average_predictions(X_train, X_test, y_train)
        
        # Calculate absolute percentage error for each prediction
        ape = np.abs((y_test - y_pred) / y_test) * 100
        
        # Calculate percentage of predictions in each error bracket
        results['ape_under_5'].append(np.mean(ape < 5) * 100)
        results['ape_5_to_10'].append(np.mean((ape >= 5) & (ape < 10)) * 100)
        results['ape_10_to_15'].append(np.mean((ape >= 10) & (ape < 15)) * 100)
        results['ape_over_15'].append(np.mean(ape >= 15) * 100)
        
        # Calculate other metrics
        results['mae'].append(mean_absolute_error(y_test, y_pred))
        results['mape'].append(mean_absolute_percentage_error(y_test, y_pred) * 100)  # Convert to percentage
        results['r2'].append(r2_score(y_test, y_pred))
        results['rmse'].append(root_mean_squared_error(y_test, y_pred))
        
        # Calculate over/underestimation
        errors = y_pred - y_test
        results['overestimation'].append(np.mean(errors > 0) * 100)  # Percentage of overestimations
        results['underestimation'].append(np.mean(errors < 0) * 100)  # Percentage of underestimations
    
    # Create a summary table
    summary = {
        'Metric': [
            'APE < 5%', 'APE 5-10%', 'APE 10-15%', 'APE > 15%',
            'MAE', 'MAPE (%)', 'R² Score', 'RMSE',
            'Overestimation (%)', 'Underestimation (%)'
        ],
        'Mean': [
            f"{np.mean(results['ape_under_5']):.2f}%",
            f"{np.mean(results['ape_5_to_10']):.2f}%",
            f"{np.mean(results['ape_10_to_15']):.2f}%",
            f"{np.mean(results['ape_over_15']):.2f}%",
            f"{np.mean(results['mae']):.4f}",
            f"{np.mean(results['mape']):.2f}%",
            f"{np.mean(results['r2']):.4f}",
            f"{np.mean(results['rmse']):.4f}",
            f"{np.mean(results['overestimation']):.2f}%",
            f"{np.mean(results['underestimation']):.2f}%"
        ],
        'SD': [
            f"{np.std(results['ape_under_5']):.2f}%",
            f"{np.std(results['ape_5_to_10']):.2f}%",
            f"{np.std(results['ape_10_to_15']):.2f}%",
            f"{np.std(results['ape_over_15']):.2f}%",
            f"{np.std(results['mae']):.4f}",
            f"{np.std(results['mape']):.2f}%",
            f"{np.std(results['r2']):.4f}",
            f"{np.std(results['rmse']):.4f}",
            f"{np.std(results['overestimation']):.2f}%",
            f"{np.std(results['underestimation']):.2f}%"
        ]
    }
    
    # Create and display table
    summary_df = pd.DataFrame(summary)
    
    return summary_df, results

# %%
def create_scatter_plot(X, y, model_type, random_state=42):
    """
    Create a scatter plot of actual vs predicted values
    """
    # Split data with a single random state for visualization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Get predictions using the specified model
    if model_type == 'linear_regression':
        # Handle linear regression
        categorical_cols = [col for col in categorical_features if col in X_train.columns]
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ])
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == 'catboost':
        X_train_proc = to_string(X_train)
        X_test_proc = to_string(X_test)
        model = CatBoostRegressor(
            **best_params,
            random_seed=random_state,
            cat_features=[i for i, col in enumerate(X_train_proc.columns) if col in categorical_features],
            verbose=0
        )
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        
    elif model_type == 'xgboost':
        X_train_proc = to_category(X_train)
        X_test_proc = to_category(X_test)
        model = XGBRegressor(
            **best_xgb_params,
            random_state=random_state,
            enable_categorical=True,
            verbosity=0
        )
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        
    elif model_type == 'randomforest':
        X_train_proc = to_category(X_train)
        X_test_proc = to_category(X_test)
        model = RandomForestRegressor(
            **best_rf_params,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        
    elif model_type == 'simple_average':
        # For simple average ensemble
        y_pred = calculate_simple_average_predictions(X_train, X_test, y_train)
    
    # Calculate metrics for this specific split
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter points
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', s=70)
    
    # Plot perfect prediction line
    ax.plot([0, 20], [0, 20], 'r--', lw=2)
    
    # Set x and y axis limits to 0-20
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    # Add labels and title
    ax.set_xlabel('Actual Value', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    
    # Format the model name for the title
    if model_type == 'linear_regression':
        title_model_name = 'Linear Regression (Baseline)'
    else:
        title_model_name = model_type.capitalize()
        
    ax.set_title(f'Actual vs Predicted Values ({title_model_name})', fontsize=14)
    
    # Add metrics annotation
    ax.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                fontsize=12, ha='left', va='top')
    
    # Add grid and improve appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig, (y_test, y_pred)

# %%
def create_bland_altman_plot(y_test, y_pred):
    """
    Create a Bland-Altman plot to assess agreement between actual and predicted values
    
    Parameters:
    -----------
    y_test : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    matplotlib figure
    """
    # Calculate mean and difference
    mean_values = (y_test + y_pred) / 2
    diff_values = y_pred - y_test  # Predicted minus actual
    
    # Calculate mean difference and limits of agreement
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    
    # Create Bland-Altman plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter points
    ax.scatter(mean_values, diff_values, alpha=0.6, edgecolor='k', s=70)
    
    # Add horizontal lines for mean difference and limits of agreement
    ax.axhline(mean_diff, color='k', linestyle='-', lw=2, label=f'Mean difference: {mean_diff:.4f}')
    ax.axhline(upper_limit, color='r', linestyle='--', lw=1.5, 
               label=f'Upper limit of agreement (+1.96 SD): {upper_limit:.4f}')
    ax.axhline(lower_limit, color='r', linestyle='--', lw=1.5,
               label=f'Lower limit of agreement (-1.96 SD): {lower_limit:.4f}')
    
    # Add labels and title
    ax.set_xlabel('Mean of Actual and Predicted Values', fontsize=12)
    ax.set_ylabel('Difference (Predicted - Actual)', fontsize=12)
    ax.set_title('Bland-Altman Plot: Agreement between Actual and Predicted Values', fontsize=14)
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    # Add grid and improve appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

# %%
# Determine your best model type from the stacking results
best_model_type = 'simple_average'  # Change this to the best model from your results

# Create the performance table with multiple random states
summary_df, results = create_performance_table(X, y, best_model_type)

# %%
# Determine your best model type from the stacking results
best_model_type = 'simple_average'  # Change this to the best model from your results

# Create the performance table with multiple random states
summary_df, results = create_performance_table(X, y, best_model_type)
display(summary_df)

# Create scatter plot using a single random state for cleaner visualization
scatter_fig, (y_test, y_pred) = create_scatter_plot(X, y, best_model_type, random_state=42)
scatter_fig.show()

# Create Bland-Altman plot using the same test results
ba_fig = create_bland_altman_plot(y_test, y_pred)
ba_fig.show()

# %%
table = {'Absolute percentage error': {'< 5%': 31.93, '5-10%': 24.2, '10-15%': 19.55, '> 15%': 24.32}, 'MAE (SD), mg/dL': '0.8450 (0.0419)', 'MAPE (SD), %': '11.51 (0.68)', 'R² score': '0.1737', 'RMSE , mg/dL': 1.1457, 'Overestimation, %': 53.18, 'Underestimation, %': 46.82}
def create_horizontal_table_figure(table_dict, title="Model Performance", figsize=(12, 2.5)):
    """
    Create a publication-quality table figure with proper handling of string values
    
    Parameters:
    -----------
    table_dict : dict
        Dictionary containing performance metrics
    title : str
        Title for the figure
    figsize : tuple
        Figure dimensions
        
    Returns:
    --------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Extract APE data for metrics row
    ape_data = table_dict.get('Absolute percentage error', {})
    
    # Define column headers with proper spacing
    headers = [
        "< 5%", "5%–10%", "10%–15%", "> 15%",
        "MAE (SD),mg/dL", "MAPE (SD),%", "R²", "RMSE, mg/dL",
        "Overestimation,%", "Underestimation,%"
    ]
    
    # Prepare data row with proper string handling
    row = [
        f"{ape_data.get('< 5%', 0):.1f}",
        f"{ape_data.get('5-10%', 0):.1f}",
        f"{ape_data.get('10-15%', 0):.1f}",
        f"{ape_data.get('> 15%', 0):.1f}",
        # Handle string values properly - use the string directly
        table_dict.get('MAE (SD),mg/dL', '0.8450 (0.0419)'),  
        table_dict.get('MAPE (SD),%', '11.51 (0.68)'),
        table_dict.get('R² score', '0.1737'),
        # Handle numeric values properly
        f"{float(table_dict.get('RMSE , mg/dL', 1.1457)):.4f}", 
        f"{table_dict.get('Overestimation,%', 53.18):.1f}",
        f"{table_dict.get('Underestimation,%', 46.82):.1f}"
    ]
    
    # Create main table
    main_table = ax.table(
        cellText=[row],
        colLabels=headers,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 0.5]  # Fixed position and size
    )
    
    # Add a header for the APE columns
    ape_header = ax.table(
        cellText=[["Absolute percentage errors"]],
        loc='center',
        bbox=[0, 0.5, 0.4, 0.1],  # Position directly above first 4 columns
        cellLoc='center'
    )
    
    # Style all tables
    for table in [main_table, ape_header]:
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        # Add borders to all cells
        for key, cell in table._cells.items():
            cell.set_edgecolor('black')
    
    # Format main table
    for i, cell in enumerate(main_table._cells.values()):
        # Set all data cells to white background
        cell.set_facecolor('white')
        cell.set_height(0.2)  # Consistent height
        
        # Format header cells
        if i < len(headers):  # Header cells
            cell.set_facecolor('#d6e6d6')
            cell.set_text_props(weight='bold')
    
    # Format APE header
    ape_cell = ape_header[(0, 0)]
    ape_cell.set_facecolor('#d6e6d6')
    ape_cell.set_text_props(weight='bold')
    
    # Add title
    plt.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

# Use the function with your table data ensuring correct keys
table = {'Absolute percentage error': {'< 5%': 31.93, '5-10%': 24.2, '10-15%': 19.55, '> 15%': 24.32}, 
         'MAE (SD),mg/dL': '0.8450 (0.0419)', 
         'MAPE (SD),%': '11.51 (0.68)', 
         'R² score': '0.1737', 
         'RMSE , mg/dL': 1.1457, 
         'Overestimation,%': 53.18, 
         'Underestimation,%': 46.82}

fig = create_horizontal_table_figure(
    table_dict=table,
    title="Simple Average Ensemble Model Performance"
)

plt.show()

# %% [markdown]
# ### Linear Regression

# %%
def evaluate_linear_regression_baseline(X, y, random_states=[42, 101, 123, 456, 789]):
    """
    Evaluate a simple linear regression as a baseline model
    """
    # Store results
    results = {'rmse': [], 'r2': [], 'mape': []}
    
    for seed in random_states:
        print(f"\n--- Evaluating Linear Regression baseline with seed {seed} ---")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Convert categorical features to numeric using one-hot encoding
        # We need to handle categorical features for linear regression
        categorical_cols = [col for col in categorical_features if col in X_train.columns]
        numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ])
        
        # Create and train the linear regression model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # Store results
        results['rmse'].append(rmse)
        results['r2'].append(r2)
        results['mape'].append(mape)
        
        # Print results
        print(f"Linear Regression - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
    
    # Calculate and print average results
    print("\n--- Linear Regression Baseline Results Across All Seeds ---")
    print(f'Average RMSE: {np.mean(results["rmse"]):.4f} ± {np.std(results["rmse"]):.4f}')
    print(f'Average R²: {np.mean(results["r2"]):.4f} ± {np.std(results["r2"]):.4f}')
    print(f'Average MAPE: {np.mean(results["mape"]):.4f} ± {np.std(results["mape"]):.4f}')
    
    return results


