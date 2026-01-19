# Epigenetics Project - Step 4: Training Brain-Specific Epigenetic Clock
# Tissue: Brain

# Install required packages
!pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels adjustText joblib -q

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
import os
from datetime import datetime
import joblib
from collections import Counter

warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import trn_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor

# Plotting configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

print("Installing packages...")
print("Packages loaded")

# Mount Google Drive and set up directories
drive.mount('/content/drive')
print("Setting up Google Drive project structure...")
print("Drive mounted")

# Define project paths
PROJECT_ROOT = '/content/drive/MyDrive/epigenetics_project/'
STEP2_DATA = f'{PROJECT_ROOT}2_data_qc/cleaned_data/'
STEP3_CPGS = f'{PROJECT_ROOT}3_feature_discovery/top_cpgs/'
STEP4_ROOT = f'{PROJECT_ROOT}4_model_training/'
STEP4_FIGURES = f'{STEP4_ROOT}figures/'
STEP4_TABLES = f'{STEP4_ROOT}tables/'
STEP4_MODELS = f'{STEP4_ROOT}models/'
STEP4_REPORTS = f'{STEP4_ROOT}reports/'

# Create directories if they don't exist (incase they have been deleted or cleared)
for folder in [STEP4_FIGURES, STEP4_TABLES, STEP4_MODELS, STEP4_REPORTS]:
    os.makedirs(folder, exist_ok=True)

# Parameters
PARAMS = {
    'n_features': 200,
    'l1_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],
    'alpha': np.logspace(-4, 2, 20),
    'cv_folds': 5,
    'test_size': 0.25,
    'n_bags': 50
}

RANDOM_STATE = 42
TISSUE = 'Brain'

# Utility functions
def print_section(title):
    width = 80
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def print_subsection(title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

def save_figure(filename, dpi=300):
    path = f'{STEP4_FIGURES}{filename}'
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure: {path}")

def save_table(df, filename, desc=""):
    path = f'{STEP4_TABLES}{filename}'
    df.to_csv(path, index=False)
    print(f"Saved table: {path} ({desc})")

def save_model(obj, filename, desc=""):
    path = f'{STEP4_MODELS}{filename}'
    joblib.dump(obj, path)
    print(f"Saved model: {path} ({desc})")

# Data loading function
def load_brain_data():
    print(f"Loading data for {TISSUE}...")

    try:
        # Load top 500 CpGs from Step 3
        cpgs_path = os.path.join(STEP3_CPGS, 'top_500_brain_cpgs.csv')
        if not os.path.exists(cpgs_path):
            cpgs_path = os.path.join(STEP3_CPGS, 'top_500_Brain_cpgs.csv')
        if not os.path.exists(cpgs_path):
            raise FileNotFoundError(f"Top CpGs file not found. Searched: {cpgs_path}")

        print(f"  Loading CpGs from: {cpgs_path}")
        top_cpgs_df = pd.read_csv(cpgs_path)
        cpg_col = None
        for col in top_cpgs_df.columns:
            if 'cpg' in col.lower() or 'CpG' in col:
                cpg_col = col
                break
        if cpg_col is None:
            cpg_col = top_cpgs_df.columns[0]
        top_cpgs = top_cpgs_df[cpg_col].astype(str).tolist()
        print(f"  Found {len(top_cpgs)} top CpGs")

        # Load cleaned methylation data from Step 2
        meth_path = os.path.join(STEP2_DATA, 'cleaned_brain_methylation.csv')
        if not os.path.exists(meth_path):
            meth_path = os.path.join(STEP2_DATA, 'cleaned_Brain_methylation.csv')
        if not os.path.exists(meth_path):
            meth_path = os.path.join(STEP2_DATA, 'brain_methylation_merged.csv')
        if not os.path.exists(meth_path):
            raise FileNotFoundError(f"Methylation file not found. Searched: {meth_path}")

        print(f"  Loading methylation from: {meth_path}")
        meth_data = pd.read_csv(meth_path, index_col=0)
        print(f"  Found methylation data: {meth_data.shape[0]} CpGs x {meth_data.shape[1]} samples")

        # Load cleaned metadata from Step 2
        meta_path = os.path.join(STEP2_DATA, 'cleaned_brain_metadata.csv')
        if not os.path.exists(meta_path):
            meta_path = os.path.join(STEP2_DATA, 'cleaned_Brain_metadata.csv')
        if not os.path.exists(meta_path):
            meta_path = os.path.join(STEP2_DATA, 'brain_metadata_merged.csv')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found. Searched: {meta_path}")

        print(f"  Loading metadata from: {meta_path}")
        meta_data = pd.read_csv(meta_path)
        print(f"  Found metadata: {meta_data.shape[0]} samples x {meta_data.shape[1]} columns")

        # Find age column
        age_col = None
        for col in meta_data.columns:
            if 'age' in col.lower():
                age_col = col
                break
        if age_col is None:
            raise ValueError("No age column found in metadata")
        print(f"  Age column identified: '{age_col}'")

        ages = pd.to_numeric(meta_data[age_col], errors='coerce').values
        valid_age_indices = ~np.isnan(ages)
        print(f"  Found {sum(valid_age_indices)} samples with valid age data")

        # Find sample ID column
        sample_id_col = None
        for col in meta_data.columns:
            if 'sample' in col.lower() or 'id' in col.lower() or 'gsm' in col.lower():
                sample_id_col = col
                break
        if sample_id_col is None:
            sample_id_col = meta_data.columns[0]
        print(f"  Using sample ID column: '{sample_id_col}'")

        # Create age dictionary
        age_dict = {}
        for idx, row in meta_data.iterrows():
            if valid_age_indices[idx]:
                sample_id = str(row[sample_id_col]).strip()
                age = ages[idx]
                clean_id = sample_id.replace(' ', '').replace('\t', '').replace('\n', '').upper()
                age_dict[clean_id] = age

        print(f"  Created age dictionary with {len(age_dict)} samples")

        # Match samples between methylation data and metadata
        common_samples = []
        age_list = []

        for meth_sample in meth_data.columns:
            clean_meth_sample = str(meth_sample).strip().replace(' ', '').replace('\t', '').replace('\n', '').upper()

            if clean_meth_sample in age_dict:
                common_samples.append(meth_sample)
                age_list.append(age_dict[clean_meth_sample])
                continue

            # Try to match by extracting numbers or patterns
            meth_num = ''.join(filter(str.isdigit, clean_meth_sample))
            if meth_num:
                for meta_id in age_dict.keys():
                    meta_num = ''.join(filter(str.isdigit, meta_id))
                    if meta_num and meth_num == meta_num:
                        common_samples.append(meth_sample)
                        age_list.append(age_dict[meta_id])
                        break

        print(f"  Matched {len(common_samples)} samples")

        if len(common_samples) < 50:
            raise ValueError(f"Insufficient samples matched: {len(common_samples)} < 50")

        # Prepare feature matrix
        X = meth_data[common_samples].T
        available_cpgs = [cpg for cpg in top_cpgs if cpg in X.columns]
        print(f"  Found {len(available_cpgs)} available CpGs out of {len(top_cpgs)}")

        if len(available_cpgs) < 100:
            print(f"  Warning: Only {len(available_cpgs)} CpGs available, using all available CpGs")
            available_cpgs = X.columns.tolist()

        X = X[available_cpgs].copy()
        y = np.array(age_list)

        # Apply quality control
        print("  Applying quality control...")

        # Remove samples with >20% missing data
        sample_missing = X.isna().mean(axis=1)
        keep_samples = sample_missing < 0.2
        X = X[keep_samples]
        y = y[keep_samples.values]
        print(f"    Removed {sum(~keep_samples)} samples with >20% missing data")

        # Remove CpGs with >30% missing data
        cpg_missing = X.isna().mean(axis=0)
        keep_cpgs = cpg_missing < 0.3
        X = X.loc[:, keep_cpgs]
        print(f"    Removed {sum(~keep_cpgs)} CpGs with >30% missing data")

        print(f"  Final dataset: {X.shape[0]} samples x {X.shape[1]} CpGs")
        if X.shape[0] == 0:
            raise ValueError("No samples left after quality control")
        if X.shape[1] == 0:
            raise ValueError("No CpGs left after quality control")

        print(f"  Age statistics: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f} ± {y.std():.1f} years")

        # Create age distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('Age (years)')
        plt.ylabel('Number of samples')
        plt.title(f'Brain age distribution (n={len(y)})')
        plt.grid(alpha=0.3)
        save_figure('brain_age_distribution.png')
        plt.show()

        return X, y, X.columns.tolist()

    except Exception as e:
        print(f"Error in load_brain_data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Feature selection function
def select_top_correlated_features(X, y, n_features):
    print(f"  Selecting {n_features} features by correlation...")

    if n_features > X.shape[1]:
        n_features = X.shape[1]

    correlations = []
    for col in X.columns:
        valid = ~X[col].isna()
        if valid.sum() > 10:
            try:
                corr = abs(pearsonr(X.loc[valid, col], y[valid])[0])
                correlations.append(corr if not np.isnan(corr) else 0)
            except:
                correlations.append(0)
        else:
            correlations.append(0)

    correlations = np.array(correlations)
    n_select = min(n_features, (correlations > 0).sum())
    selected_idx = np.argsort(correlations)[-n_select:]
    selected_features = X.columns[selected_idx].tolist()

    print(f"  Selected {len(selected_features)} features")
    if len(selected_features) > 0:
        top_corr = correlations[selected_idx[-1]]
        print(f"  Minimum correlation threshold: {top_corr:.3f}")

    return selected_features

# Hyperparameter optimization function
def find_optimal_hyperparameters(X_train, y_train):
    print("  Finding optimal hyperparameters...")

    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    features = select_top_correlated_features(X_imp, y_train, PARAMS['n_features'])

    if len(features) < 10:
        print("  Not enough features for hyperparameter optimization")
        return 0.01, 0.5

    X_sel = X_imp[features]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_sel)

    model = ElasticNet(max_iter=10000, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        model,
        {'alpha': PARAMS['alpha'], 'l1_ratio': PARAMS['l1_ratio']},
        cv=KFold(n_splits=min(3, len(y_train)//10), shuffle=True, random_state=RANDOM_STATE),
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_scaled, y_train)

    print(f"  Best parameters found:")
    print(f"    Alpha: {grid.best_params_['alpha']:.6f}")
    print(f"    L1 ratio: {grid.best_params_['l1_ratio']:.2f}")

    return grid.best_params_['alpha'], grid.best_params_['l1_ratio']

# Cross-validation with feature stability analysis
def train_cv_model(X_train, y_train, alpha, l1_ratio):
    print("\nPerforming cross-validation with feature stability analysis...")

    if len(y_train) < 20:
        print("  Not enough samples for cross-validation")
        return [], [], [], [], {}

    try:
        bins = pd.qcut(y_train, q=min(5, len(y_train)//10), labels=False, duplicates='drop')
        cv = StratifiedKFold(n_splits=PARAMS['cv_folds'], shuffle=True, random_state=RANDOM_STATE)
        stratify = bins
        print(f"  Using stratified {PARAMS['cv_folds']}-fold cross-validation")
    except:
        cv = KFold(n_splits=PARAMS['cv_folds'], shuffle=True, random_state=RANDOM_STATE)
        stratify = None
        print(f"  Using {PARAMS['cv_folds']}-fold cross-validation")

    maes, r2s, corrs = [], [], []
    selected_features_per_fold = []
    all_selected_features = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, stratify), 1):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]

        if len(y_tr) < 10 or len(y_val) < 5:
            print(f"  Fold {fold}: Skipped due to insufficient samples")
            continue

        imputer = SimpleImputer(strategy='median')
        X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
        X_val_imp = pd.DataFrame(imputer.transform(X_val), index=X_val.index, columns=X_val.columns)

        features = select_top_correlated_features(X_tr_imp, y_tr, PARAMS['n_features'])
        selected_features_per_fold.append(features)
        all_selected_features.extend(features)

        if not features:
            print(f"  Fold {fold}: No features selected")
            continue

        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp[features])
        X_val_sc = scaler.transform(X_val_imp[features])

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=RANDOM_STATE)
        model.fit(X_tr_sc, y_tr)
        pred = model.predict(X_val_sc)

        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        corr = pearsonr(y_val, pred)[0]

        maes.append(mae)
        r2s.append(r2)
        corrs.append(corr)

        print(f"    Fold {fold}: MAE={mae:.2f}, R2={r2:.4f}, r={corr:.3f}, Features={len(features)}")

    if not maes:
        print("  No valid folds completed")
        return [], [], [], [], {}

    # Feature stability analysis
    feature_counts = Counter(all_selected_features)
    total_folds = len(selected_features_per_fold)

    stability_rows = []
    for feat, count in feature_counts.most_common():
        freq = count / total_folds
        stability_rows.append({
            'CpG': feat,
            'Frequency': freq,
            'Count': count,
            'Percentage': freq * 100,
            'Stability_Category': 'Highly stable' if freq >= 0.8 else
                                 'Moderately stable' if freq >= 0.6 else
                                 'Variable'
        })

    stability_df = pd.DataFrame(stability_rows).sort_values('Frequency', ascending=False)
    save_table(stability_df, 'brain_feature_stability.csv', 'Feature selection frequency across CV folds')

    print_subsection("Feature Stability Analysis")
    print(f"  Total unique features selected: {len(stability_df)}")
    print(f"  Features selected in all folds: {(stability_df['Frequency'] == 1.0).sum()}")
    print(f"  Features selected in >=80% folds: {(stability_df['Frequency'] >= 0.8).sum()}")
    print(f"  Features selected in >=60% folds: {(stability_df['Frequency'] >= 0.6).sum()}")

    # Plot feature stability
    if len(stability_df) > 0:
        plt.figure(figsize=(14, 6))

        # Frequency distribution histogram
        plt.subplot(1, 2, 1)
        freq_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        freq_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        binned = pd.cut(stability_df['Frequency'], bins=freq_bins, include_lowest=True, right=True)
        counts = binned.value_counts(sort=False)
        bars = plt.bar(freq_labels, counts.values, color=['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#66b3ff'])
        plt.xlabel('Selection frequency range')
        plt.ylabel('Number of features')
        plt.title('Feature stability distribution')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom')

        # Cumulative plot
        plt.subplot(1, 2, 2)
        thresholds = np.linspace(0, 1, 21)
        cumulative = [(stability_df['Frequency'] >= t).sum() for t in thresholds]
        plt.plot(thresholds * 100, cumulative, 'b-', linewidth=2, marker='o')
        plt.fill_between(thresholds * 100, cumulative, alpha=0.3, color='skyblue')
        plt.xlabel('Minimum selection frequency (%)')
        plt.ylabel('Number of features')
        plt.title('Cumulative feature stability')
        plt.grid(True, alpha=0.3)

        save_figure('brain_feature_stability_analysis.png')
        plt.show()

    # Top stable features
    stable_100 = stability_df[stability_df['Frequency'] == 1.0]
    if len(stable_100) >= 10:
        print("\n  Top 10 most stable features (selected in all folds):")
        for i, cpg in enumerate(stable_100.head(10)['CpG'], 1):
            print(f"    {i:2d}. {cpg}")
    else:
        print("\n  Top 10 most frequent features:")
        for i, row in enumerate(stability_df.head(10).itertuples(), 1):
            print(f"    {i:2d}. {row.CpG} ({row.Frequency*100:.0f}%)")

    # Final feature selection
    stable_candidates = stability_df[stability_df['Frequency'] >= 0.7]['CpG'].tolist()
    if len(stable_candidates) >= PARAMS['n_features']:
        final_features = stable_candidates[:PARAMS['n_features']]
        print(f"\n  Selected {len(final_features)} features with >=70% stability")
    elif stable_candidates:
        final_features = stable_candidates
        print(f"\n  Using all {len(final_features)} features with >=70% stability")
    else:
        final_features = [f[0] for f in feature_counts.most_common(PARAMS['n_features'])]
        print(f"\n  Using top {len(final_features)} features by overall frequency")

    cv_metrics = {
        'mean_mae': np.mean(maes),
        'std_mae': np.std(maes),
        'mean_r2': np.mean(r2s),
        'std_r2': np.std(r2s),
        'mean_corr': np.mean(corrs),
        'std_corr': np.std(corrs),
        'stability_df': stability_df
    }

    print(f"\n  Cross-validation summary ({len(maes)} folds):")
    print(f"    Mean MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"    Mean R2: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    print(f"    Mean correlation: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

    return maes, r2s, corrs, final_features, cv_metrics

# Final model training function
def train_final_model(X_train, y_train, X_test, y_test, alpha, l1_ratio, features):
    print("\nTraining final model...")

    if len(features) == 0:
        print("  No features for model training")
        return None, [], None, None, {}

    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), index=X_test.index, columns=X_test.columns)

    X_train_sel = X_train_imp[features]
    X_test_sel = X_test_imp[features]

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_test_sc = scaler.transform(X_test_sel)

    base_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=RANDOM_STATE)
    model = BaggingRegressor(
        estimator=base_model,
        n_estimators=PARAMS['n_bags'],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )

    print(f"  Training ensemble with {PARAMS['n_bags']} estimators...")
    model.fit(X_train_sc, y_train)

    train_pred = model.predict(X_train_sc)
    test_pred = model.predict(X_test_sc)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_corr = pearsonr(y_train, train_pred)[0]
    test_corr = pearsonr(y_test, test_pred)[0]
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    # Calculate generalization metrics
    mae_gap = train_mae - test_mae
    r2_gap = train_r2 - test_r2
    mae_ratio = train_mae / test_mae if test_mae > 0 else np.inf
    r2_ratio = test_r2 / train_r2 if train_r2 > 0 else np.inf

    # Assess generalization
    if mae_gap > 2.0 or mae_ratio > 1.3 or r2_gap > 0.1:
        generalization = "Poor"
    elif mae_gap > 1.0 or mae_ratio > 1.2 or r2_gap > 0.05:
        generalization = "Moderate"
    elif mae_gap > 0.5 or mae_ratio > 1.1 or r2_gap > 0.02:
        generalization = "Good"
    else:
        generalization = "Excellent"

    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_corr': train_corr,
        'test_corr': test_corr,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'mae_gap': mae_gap,
        'r2_gap': r2_gap,
        'mae_ratio': mae_ratio,
        'r2_ratio': r2_ratio,
        'generalization': generalization
    }

    # Calculate postnatal metrics if applicable
    postnatal_train = y_train >= 0
    if postnatal_train.sum() > 10:
        metrics['postnatal_train_mae'] = mean_absolute_error(y_train[postnatal_train], train_pred[postnatal_train])
        metrics['postnatal_train_r2'] = r2_score(y_train[postnatal_train], train_pred[postnatal_train])

    postnatal_test = y_test >= 0
    if postnatal_test.sum() > 5:
        metrics['postnatal_test_mae'] = mean_absolute_error(y_test[postnatal_test], test_pred[postnatal_test])
        metrics['postnatal_test_r2'] = r2_score(y_test[postnatal_test], test_pred[postnatal_test])

    return model, features, train_pred, test_pred, metrics

# Visualization function
def create_visualizations(y_true, y_pred, split, metrics):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 5:
        print(f"  Not enough data for {split} set visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    fetal = y_true < 0
    postnatal = y_true >= 0

    # Scatter plot
    if fetal.sum() > 0:
        axes[0].scatter(y_true[fetal], y_pred[fetal], color='red', alpha=0.7, s=50, label='Fetal')
    if postnatal.sum() > 0:
        axes[0].scatter(y_true[postnatal], y_pred[postnatal], color='steelblue', alpha=0.6, s=50, label='Postnatal')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
    axes[0].set_xlabel('Actual age (years)', fontweight='bold')
    axes[0].set_ylabel('Predicted age (years)', fontweight='bold')
    axes[0].set_title(f'Brain - {split.capitalize()} set predictions', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Add statistics text
    text = f"MAE = {metrics['test_mae' if split == 'test' else 'train_mae']:.2f} years\n"
    text += f"R² = {metrics['test_r2' if split == 'test' else 'train_r2']:.4f}\n"
    text += f"Correlation = {metrics['test_corr' if split == 'test' else 'train_corr']:.3f}"
    if split == 'test':
        text += f"\n\nGeneralization metrics:\n"
        text += f"MAE gap = {metrics['mae_gap']:.2f} years\n"
        text += f"R² gap = {metrics['r2_gap']:.4f}\n"
        text += f"MAE ratio = {metrics['mae_ratio']:.3f}\n"
        text += f"Assessment = {metrics['generalization']}"

    axes[0].text(0.05, 0.95, text, transform=axes[0].transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Residual plot
    residuals = y_pred - y_true
    if fetal.sum() > 0:
        axes[1].scatter(y_true[fetal], residuals[fetal], color='red', alpha=0.7, s=50)
    if postnatal.sum() > 0:
        axes[1].scatter(y_true[postnatal], residuals[postnatal], color='steelblue', alpha=0.6, s=50)

    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Actual age (years)', fontweight='bold')
    axes[1].set_ylabel('Residual (predicted - actual)', fontweight='bold')
    axes[1].set_title('Residual plot', fontweight='bold')
    axes[1].grid(alpha=0.3)

    save_figure(f'brain_{split}_predictions.png')
    plt.show()

# Main pipeline
def main():
    print_section("Training Brain Epigenetic Clock")

    try:
        # Load data
        print("Step 1: Loading brain data...")
        X, y, all_cpgs = load_brain_data()

        if X.shape[0] == 0 or X.shape[1] == 0:
            print("Error: No data available after loading")
            return

        print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")

        # Split data
        print("\nStep 2: Creating train/test split...")
        try:
            n_bins = min(5, len(y)//20)
            bins = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=PARAMS['test_size'], stratify=bins, random_state=RANDOM_STATE)
            print(f"  Using stratified split with {n_bins} bins")
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=PARAMS['test_size'], random_state=RANDOM_STATE)
            print("  Using random split")

        print(f"  Training samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Training age: {y_train.mean():.1f} ± {y_train.std():.1f} years")
        print(f"  Test age: {y_test.mean():.1f} ± {y_test.std():.1f} years")

        # Hyperparameter optimization
        print_subsection("Hyperparameter Optimization")
        best_alpha, best_l1_ratio = find_optimal_hyperparameters(X_train, y_train)
        print(f"  Optimal alpha: {best_alpha:.6f}")
        print(f"  Optimal L1 ratio: {best_l1_ratio:.2f}")

        # Cross-validation and feature stability
        print_subsection("Cross-Validation and Feature Stability")
        cv_maes, cv_r2s, cv_corrs, final_features, cv_metrics = train_cv_model(X_train, y_train, best_alpha, best_l1_ratio)

        # Final model training
        print_subsection("Final Model Training")
        model, features, train_pred, test_pred, metrics = train_final_model(
            X_train, y_train, X_test, y_test, best_alpha, best_l1_ratio, final_features)

        if model is not None:
            print("\nFinal model performance:")
            print("  All samples:")
            print(f"    Training MAE: {metrics['train_mae']:.2f} years")
            print(f"    Training R²: {metrics['train_r2']:.4f}")
            print(f"    Training correlation: {metrics['train_corr']:.3f}")
            print(f"    Test MAE: {metrics['test_mae']:.2f} years")
            print(f"    Test R²: {metrics['test_r2']:.4f}")
            print(f"    Test correlation: {metrics['test_corr']:.3f}")

            print("\n  Generalization metrics:")
            print(f"    MAE gap (train - test): {metrics['mae_gap']:.2f} years")
            print(f"    R² gap (train - test): {metrics['r2_gap']:.4f}")
            print(f"    MAE ratio (train/test): {metrics['mae_ratio']:.3f}")
            print(f"    R² ratio (test/train): {metrics['r2_ratio']:.3f}")
            print(f"    Generalization assessment: {metrics['generalization']}")

            if 'postnatal_test_mae' in metrics:
                print("\n  Postnatal samples only (age ≥ 0):")
                print(f"    Test MAE: {metrics['postnatal_test_mae']:.2f} years")
                print(f"    Test R²: {metrics['postnatal_test_r2']:.4f}")

            print(f"\n  Features used: {len(features)}")

        # Visualizations
        print_subsection("Visualizations")
        create_visualizations(y_train, train_pred, 'train', metrics)
        create_visualizations(y_test, test_pred, 'test', metrics)

        # Final metrics section
        print_section("Final Performance Metrics")

        print("\n" + "="*80)
        print("FINAL MODEL PERFORMANCE SUMMARY")
        print("="*80)

        print(f"\nDataset Information:")
        print(f"  Total samples: {len(y)}")
        print(f"  Training samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Selected features: {len(features)}")
        print(f"  Age range: {y.min():.1f} to {y.max():.1f} years")
        print(f"  Mean age: {y.mean():.1f} ± {y.std():.1f} years")

        print(f"\nTraining Performance:")
        print(f"  MAE: {metrics['train_mae']:.2f} years")
        print(f"  R²: {metrics['train_r2']:.4f}")
        print(f"  Correlation: {metrics['train_corr']:.3f}")
        print(f"  RMSE: {metrics['train_rmse']:.2f} years")

        print(f"\nTest Performance:")
        print(f"  MAE: {metrics['test_mae']:.2f} years")
        print(f"  R²: {metrics['test_r2']:.4f}")
        print(f"  Correlation: {metrics['test_corr']:.3f}")
        print(f"  RMSE: {metrics['test_rmse']:.2f} years")

        print(f"\nGeneralization Analysis:")
        print(f"  MAE gap (train-test): {metrics['mae_gap']:.2f} years")
        print(f"  R² gap (train-test): {metrics['r2_gap']:.4f}")
        print(f"  MAE ratio (train/test): {metrics['mae_ratio']:.3f}")
        print(f"  R² retention (test/train): {metrics['r2_ratio']:.3f}")
        print(f"  Generalization assessment: {metrics['generalization']}")

        if 'postnatal_test_mae' in metrics:
            print(f"\nPostnatal Samples Only (Age ≥ 0):")
            print(f"  Test MAE: {metrics['postnatal_test_mae']:.2f} years")
            print(f"  Test R²: {metrics['postnatal_test_r2']:.4f}")

        print(f"\nCross-Validation Performance:")
        print(f"  Mean CV MAE: {cv_metrics['mean_mae']:.2f} ± {cv_metrics['std_mae']:.2f}")
        print(f"  Mean CV R²: {cv_metrics['mean_r2']:.4f} ± {cv_metrics['std_r2']:.4f}")
        print(f"  Mean CV correlation: {cv_metrics['mean_corr']:.3f} ± {cv_metrics['std_corr']:.3f}")

        print(f"\nModel Configuration:")
        print(f"  Algorithm: ElasticNet with bagging ensemble")
        print(f"  Ensemble size: {PARAMS['n_bags']} estimators")
        print(f"  Alpha: {best_alpha:.6f}")
        print(f"  L1 ratio: {best_l1_ratio:.2f}")
        print(f"  Features selected: {len(features)}")

        print(f"\nFeature Stability:")
        stability_df = cv_metrics.get('stability_df', pd.DataFrame())
        if not stability_df.empty:
            total_unique = len(stability_df)
            in_all_folds = (stability_df['Frequency'] == 1.0).sum()
            in_80pct_folds = (stability_df['Frequency'] >= 0.8).sum()

            print(f"  Total unique features across folds: {total_unique}")
            print(f"  Features selected in all folds: {in_all_folds}")
            print(f"  Features selected in ≥80% folds: {in_80pct_folds}")

        print(f"\nInterpretation:")
        if metrics['test_r2'] >= 0.9:
            print(f"  Excellent: Test R² of {metrics['test_r2']:.4f} indicates highly accurate age prediction")
        elif metrics['test_r2'] >= 0.8:
            print(f"  Good: Test R² of {metrics['test_r2']:.4f} indicates reliable age prediction")
        elif metrics['test_r2'] >= 0.7:
            print(f"  Moderate: Test R² of {metrics['test_r2']:.4f} indicates acceptable age prediction")
        else:
            print(f"  Needs improvement: Test R² of {metrics['test_r2']:.4f} indicates limited predictive power")

        if metrics['generalization'] == "Excellent":
            print(f"  Excellent generalization: Model performs consistently on training and test data")
        elif metrics['generalization'] == "Good":
            print(f"  Good generalization: Model shows minimal overfitting")
        elif metrics['generalization'] == "Moderate":
            print(f"  Moderate generalization: Some overfitting observed")
        else:
            print(f"  Poor generalization: Significant overfitting observed")

        # Save model and results
        print_subsection("Saving Model and Results")

        # Prepare imputer and scaler
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train[features])

        scaler = RobustScaler()
        scaler.fit(imputer.transform(X_train[features]))

        # Create model artifact
        model_info = {
            'model': model,
            'features': features,
            'imputer': imputer,
            'scaler': scaler,
            'hyperparameters': {
                'alpha': best_alpha,
                'l1_ratio': best_l1_ratio,
                'n_features': len(features),
                'n_estimators': PARAMS['n_bags']
            },
            'metrics': metrics,
            'cv_metrics': cv_metrics,
            'data_info': {
                'n_samples_total': len(y),
                'n_samples_train': len(y_train),
                'n_samples_test': len(y_test),
                'age_range': f"{y.min():.1f}-{y.max():.1f}",
                'mean_age': f"{y.mean():.1f} ± {y.std():.1f}"
            }
        }

        save_model(model_info, 'brain_epigenetic_clock.pkl', 'Final brain epigenetic clock')

        # Save performance summary
        performance_df = pd.DataFrame([{
            'Tissue': TISSUE,
            'N_total_samples': len(y),
            'N_training_samples': len(y_train),
            'N_test_samples': len(y_test),
            'N_features': len(features),
            'Test_MAE': round(metrics['test_mae'], 2),
            'Test_R2': round(metrics['test_r2'], 4),
            'Test_correlation': round(metrics['test_corr'], 3),
            'Test_RMSE': round(metrics['test_rmse'], 2),
            'Training_MAE': round(metrics['train_mae'], 2),
            'Training_R2': round(metrics['train_r2'], 4),
            'Training_correlation': round(metrics['train_corr'], 3),
            'Training_RMSE': round(metrics['train_rmse'], 2),
            'MAE_gap': round(metrics['mae_gap'], 2),
            'R2_gap': round(metrics['r2_gap'], 4),
            'MAE_ratio': round(metrics['mae_ratio'], 3),
            'R2_ratio': round(metrics['r2_ratio'], 3),
            'Generalization_assessment': metrics['generalization'],
            'Alpha': best_alpha,
            'L1_ratio': best_l1_ratio,
            'CV_mean_MAE': round(cv_metrics.get('mean_mae', 0), 2),
            'CV_std_MAE': round(cv_metrics.get('std_mae', 0), 2),
            'CV_mean_R2': round(cv_metrics.get('mean_r2', 0), 4),
            'CV_std_R2': round(cv_metrics.get('std_r2', 0), 4),
            'Postnatal_test_MAE': round(metrics.get('postnatal_test_mae', 0), 2) if 'postnatal_test_mae' in metrics else None,
            'Postnatal_test_R2': round(metrics.get('postnatal_test_r2', 0), 4) if 'postnatal_test_r2' in metrics else None
        }])

        save_table(performance_df, 'brain_clock_performance.csv', 'Brain clock performance summary')

        print_section("Brain Clock Training Complete")

        print("\nSummary:")
        print("-" * 60)
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.2f} years")
        print(f"Generalization: {metrics['generalization']}")
        print(f"Features: {len(features)}")

        print(f"\nOutput files:")
        print(f"  Model: brain_epigenetic_clock.pkl")
        print(f"  Performance: brain_clock_performance.csv")
        print(f"  Feature stability: brain_feature_stability.csv")
        print(f"  Visualizations saved in figures/ directory")

        # Create report
        report = f"""
Brain Epigenetic Clock - Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Information:
- Total samples: {len(y)}
- Training samples: {len(y_train)}
- Test samples: {len(y_test)}
- Selected features: {len(features)}
- Age range: {y.min():.1f} - {y.max():.1f} years
- Mean age: {y.mean():.1f} ± {y.std():.1f} years

Model Performance:
- Test R²: {metrics['test_r2']:.4f}
- Test MAE: {metrics['test_mae']:.2f} years
- Test correlation: {metrics['test_corr']:.3f}
- Generalization assessment: {metrics['generalization']}

Model Details:
- Algorithm: ElasticNet with bagging ensemble
- Ensemble size: {PARAMS['n_bags']} estimators
- Hyperparameters: alpha={best_alpha:.6f}, l1_ratio={best_l1_ratio:.2f}
- Feature selection: Correlation-based with stability analysis

Files generated:
1. brain_epigenetic_clock.pkl - Trained model with preprocessing
2. brain_clock_performance.csv - Performance metrics
3. brain_feature_stability.csv - Feature stability analysis
4. Various visualization files in figures/ directory
"""

        # Save report
        report_path = f'{STEP4_REPORTS}brain_clock_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"  Report: brain_clock_report.txt")

    except Exception as e:
        print(f"\nError in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

        error_report = f"""
Error report - Brain Clock Training
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Error: {str(e)}
"""
        try:
            error_path = f'{STEP4_REPORTS}brain_training_error.txt'
            with open(error_path, 'w') as f:
                f.write(error_report)
            print("  Error report saved")
        except:
            pass

if __name__ == "__main__":
    main()
