import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap


class AdaptiveSMOTE(SMOTE):
    """SMOTE that shrinks k_neighbors when a CV fold is too small."""

    def __init__(self, random_state=42, k_neighbors=5):
        super().__init__(random_state=random_state, k_neighbors=k_neighbors)

    def _fit_resample(self, X, y):
        class_counts = pd.Series(y).value_counts()
        minority_count = int(class_counts.min())

        if minority_count < 2:
            return X, y

        original_k_neighbors = self.k_neighbors
        self.k_neighbors = min(original_k_neighbors, minority_count - 1)
        try:
            return super()._fit_resample(X, y)
        finally:
            self.k_neighbors = original_k_neighbors


def create_pipeline(preprocessor, classifier, use_smote=True, poly_features=False, pca_components=None, smote_class=SMOTE):
    """Create a simple preprocessing/classifier pipeline for notebook experiments."""
    steps = [('preprocessor', preprocessor)]

    if pca_components:
        steps.append(('pca', PCA(n_components=pca_components)))

    if poly_features:
        steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))

    if use_smote:
        steps.append(('smote', smote_class(random_state=42)))

    steps.append(('classifier', classifier))
    return ImbPipeline(steps=steps)


def get_feature_names_from_pipeline(model, X_train=None):
    """Return feature names after preprocessing when a pipeline is used."""
    if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
        try:
            return np.asarray(model.named_steps['preprocessor'].get_feature_names_out())
        except Exception:
            pass
    if X_train is not None:
        return np.asarray(X_train.columns)
    return None


def _unwrap_classifier(model):
    """Return the fitted classifier and, when available, its inner estimator."""
    classifier = model.named_steps['classifier'] if hasattr(model, 'named_steps') and 'classifier' in model.named_steps else model
    inner_classifier = classifier

    calibrated_classifiers = getattr(classifier, 'calibrated_classifiers_', None)
    if calibrated_classifiers:
        inner_classifier = getattr(calibrated_classifiers[0], 'estimator', inner_classifier)

    return classifier, inner_classifier


def _as_dense(matrix):
    """Convert sparse or pandas-backed matrices to a dense ndarray for SHAP."""
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    if hasattr(matrix, 'to_numpy'):
        return matrix.to_numpy()
    return np.asarray(matrix)


def _coefficient_importances(classifier):
    """Return a 1D importance vector for linear models, or None if unsupported."""
    coef = getattr(classifier, 'coef_', None)
    if coef is None:
        return None

    coef = np.asarray(coef)
    if coef.ndim == 1:
        return np.abs(coef)
    return np.mean(np.abs(coef), axis=0)


def _drop_missing_rows(X, y=None):
    """Drop rows with any missing values from X and align y if provided."""
    if not hasattr(X, 'isna'):
        return X, y

    mask = ~X.isna().any(axis=1)
    X_clean = X.loc[mask].copy()
    if y is None:
        return X_clean, None

    y_series = pd.Series(y)
    y_clean = y_series.loc[X_clean.index].copy()
    return X_clean, y_clean


def plot_confusion_matrix_for_model(
    model,
    X_test,
    y_test,
    labels=None,
    display_labels=None,
    title='Confusion Matrix',
):
    y_pred = model.predict(X_test)
    y_true = pd.Series(y_test).dropna()
    y_pred = pd.Series(y_pred)
    observed_labels = np.unique(pd.concat([y_true, y_pred], ignore_index=True))

    if labels is None:
        labels = observed_labels
    else:
        labels = np.asarray(labels)
        if display_labels is None and len(labels) == len(observed_labels):
            expected_encoded = np.array_equal(observed_labels, np.arange(len(observed_labels)))
            provided_human_labels = not np.array_equal(labels, observed_labels)
            if expected_encoded and provided_human_labels:
                display_labels = labels
                labels = observed_labels
        if display_labels is None:
            labels = np.unique(np.concatenate([labels, observed_labels]))
            display_labels = labels

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve_for_model(model, X_test, y_test, title='ROC Curve'):
    if not hasattr(model, 'predict_proba'):
        print('ROC curve skipped: model does not expose predict_proba().')
        return
    unique_classes = np.unique(pd.Series(y_test).dropna())
    if len(unique_classes) != 2:
        print('ROC curve skipped: multiclass targets are not plotted as a single ROC curve.')
        return
    y_score = model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title(f"{title} (AUC={roc_auc_score(y_test, y_score):.3f})")
    plt.tight_layout()
    plt.show()


def plot_model_feature_importance(model, X_train=None, top_n=20, title='Feature Importance'):
    feature_names = get_feature_names_from_pipeline(model, X_train)
    classifier, inner_classifier = _unwrap_classifier(model)

    if hasattr(classifier, 'feature_importances_'):
        importances = np.asarray(classifier.feature_importances_)
    else:
        importances = _coefficient_importances(classifier)
        if importances is None and inner_classifier is not classifier:
            importances = _coefficient_importances(inner_classifier)
    if importances is None:
        print('Feature importance skipped: model does not expose coefficients or feature_importances_.')
        return None

    if feature_names is None:
        feature_names = np.array([f'feature_{i}' for i in range(len(importances))])
    else:
        feature_names = np.asarray(feature_names)[: len(importances)]
        importances = importances[: len(feature_names)]

    fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi = fi.sort_values('importance', ascending=False).head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * len(fi))))
    plt.barh(fi['feature'][::-1], fi['importance'][::-1])
    plt.title(title)
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    return fi


def plot_permutation_importance_for_model(model, X_test, y_test, top_n=20, title='Permutation Importance'):
    feature_names = np.asarray(X_test.columns) if hasattr(X_test, 'columns') else np.array([f'feature_{i}' for i in range(X_test.shape[1])]) if hasattr(X_test, 'shape') else get_feature_names_from_pipeline(model, X_test)
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

    if feature_names is None:
        feature_names = np.array([f'feature_{i}' for i in range(len(result.importances_mean))])
    else:
        feature_names = np.asarray(feature_names)[: len(result.importances_mean)]

    pi = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values('importance_mean', ascending=False).head(top_n)

    plt.figure(figsize=(10, max(4, 0.35 * len(pi))))
    plt.barh(pi['feature'][::-1], pi['importance_mean'][::-1], xerr=pi['importance_std'][::-1])
    plt.title(title)
    plt.xlabel('Permutation Importance')
    plt.tight_layout()
    plt.show()
    return pi


def plot_shap_summary_for_model(model, X_train, X_test, max_display=20, sample_size=500):
    """Generate SHAP summary plots for tree-based or linear models when supported."""
    classifier, inner_classifier = _unwrap_classifier(model)
    feature_names = get_feature_names_from_pipeline(model, X_train)

    X_background = X_train.sample(min(sample_size, len(X_train)), random_state=42) if len(X_train) > sample_size else X_train
    X_shap = X_test.sample(min(sample_size, len(X_test)), random_state=42) if len(X_test) > sample_size else X_test

    try:
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            X_background_transformed = model.named_steps['preprocessor'].transform(X_background)
            X_shap_transformed = model.named_steps['preprocessor'].transform(X_shap)
        else:
            X_background_transformed = X_background
            X_shap_transformed = X_shap

        X_background_transformed = _as_dense(X_background_transformed)
        X_shap_transformed = _as_dense(X_shap_transformed)
        masker = shap.maskers.Independent(X_background_transformed, max_samples=len(X_background_transformed))

        if isinstance(inner_classifier, XGBClassifier) or hasattr(inner_classifier, 'get_booster'):
            explainer = shap.TreeExplainer(
                inner_classifier,
                data=X_background_transformed,
                feature_perturbation='tree_path_dependent',
            )
            shap_values = explainer(X_shap_transformed)
        elif _coefficient_importances(inner_classifier) is not None:
            explainer = shap.LinearExplainer(inner_classifier, masker)
            shap_values = explainer(X_shap_transformed)
        elif hasattr(classifier, 'predict_proba'):
            explainer = shap.Explainer(classifier.predict_proba, masker)
            shap_values = explainer(X_shap_transformed)
        else:
            explainer = shap.Explainer(classifier, masker)
            shap_values = explainer(X_shap_transformed)

        shap.summary_plot(shap_values, X_shap_transformed, feature_names=feature_names, max_display=max_display, show=True)
        return shap_values
    except Exception as exc:
        print(f'SHAP skipped: {exc}')
        return None


def run_interpretability_suite(model, X_train, X_test, y_test, labels=None, top_n=20):
    """Run the standard interpretability outputs."""
    try:
        model.predict(X_test)
    except ValueError as exc:
        message = str(exc)
        if 'NaN' in message or 'missing values' in message:
            print('Interpretability input contains missing values; dropping incomplete rows for this model.')
            X_test, y_test = _drop_missing_rows(X_test, y_test)
            X_train, _ = _drop_missing_rows(X_train)
        else:
            raise

    plot_confusion_matrix_for_model(model, X_test, y_test, labels=labels)
    plot_roc_curve_for_model(model, X_test, y_test)
    plot_model_feature_importance(model, X_train=X_train, top_n=top_n)
    plot_permutation_importance_for_model(model, X_test, y_test, top_n=top_n)
    plot_shap_summary_for_model(model, X_train, X_test, max_display=top_n)


def resolve_ordinal_categories(series, explicit_categories=None):
    """Resolve the encoded category order for an ordinal feature.

    Explicit categories win. Otherwise we accept ordered categoricals, numeric
    dtypes, and numeric-looking string/object codes such as "1", "2", "10".
    """
    if explicit_categories is not None:
        return list(explicit_categories)

    if pd.api.types.is_categorical_dtype(series) and series.cat.ordered:
        return list(series.cat.categories)

    if pd.api.types.is_numeric_dtype(series):
        return sorted(pd.unique(series.dropna()).tolist())

    non_null = series.dropna()
    coerced = pd.to_numeric(non_null, errors='coerce')
    if not non_null.empty and coerced.notna().all():
        ordered = non_null.iloc[np.argsort(coerced.to_numpy())]
        return ordered.drop_duplicates().tolist()

    raise ValueError(
        f"Ordinal feature '{series.name}' needs an explicit order. "
        "Pass ordinal_feature_categories or make the column an ordered "
        "pandas Categorical."
    )


def compute_resilience_score(df):
    item_cols = [f'V2IA{i:02d}' for i in range(1, 26)]
    out = df[['PublicID']].copy()
    out['ResilienceTotalScore'] = df[item_cols].sum(axis=1)
    out['ResilienceLevel'] = out['ResilienceTotalScore'].apply(
        lambda s: 3 if s <= 75 else 2 if s <= 100 else 1
    )
    return out


def compute_stress_average(df):
    cols = [
        'V1EA01', 'V1EA02a', 'V1EA02b', 'V1EA02c', 'V1EA02d', 'V1EA02e',
        'V1EA02f', 'V1EA02g', 'V1EA02h', 'V1EA02i', 'V1EA02j', 'V1EA02k',
        'V1EA02l',
    ]
    out = df[['PublicID']].copy()
    out['stress_average'] = df[cols].mean(axis=1)
    return out


def compute_hassles_uplifts(df):
    hassles_cols = [f'V3JA02{letter}' for letter in 'abcdefghij']
    uplifts_cols = [f'V3JA01{letter}' for letter in 'abcdefghij']
    out = df[['PublicID']].copy()
    out['FrequencyOfHassles'] = df[hassles_cols].gt(0).sum(axis=1)
    out['FrequencyOfUplifts'] = df[uplifts_cols].gt(0).sum(axis=1)
    out['IntensityOfHassles'] = np.where(
        out['FrequencyOfHassles'] > 0,
        df[hassles_cols].sum(axis=1) / out['FrequencyOfHassles'],
        0.0,
    )
    out['IntensityOfUplifts'] = np.where(
        out['FrequencyOfUplifts'] > 0,
        df[uplifts_cols].sum(axis=1) / out['FrequencyOfUplifts'],
        0.0,
    )
    out['HassleUpliftFrequencyRatio'] = np.where(
        out['FrequencyOfUplifts'] > 0,
        out['FrequencyOfHassles'] / out['FrequencyOfUplifts'],
        0.0,
    )
    out['HassleUpliftIntensityRatio'] = np.where(
        out['IntensityOfUplifts'] > 0,
        out['IntensityOfHassles'] / out['IntensityOfUplifts'],
        0.0,
    )
    return out


def compute_stress_level(df):
    reverse_columns = ['V1AH04', 'V1AH05', 'V1AH07', 'V1AH08', 'V3AG04', 'V3AG05', 'V3AG07', 'V3AG08']
    out = df[['PublicID']].copy()
    out['StressTotalScore'] = (6 - df[reverse_columns]).sum(axis=1)
    out['StressLevel'] = out['StressTotalScore'].apply(
        lambda s: 0 if 0 <= s <= 13 else 0.5 if 14 <= s <= 26 else 1 if 27 <= s <= 40 else np.nan
    )
    return out


def compute_edinburgh_scores(df):
    out = df[['PublicID']].copy()
    total = pd.Series(0, index=df.index, dtype='float64')
    for i in range(1, 11):
        col = f'V1CA{i:02d}'
        if i == 10:
            total += (df[col] - 4).abs()
        elif i in {1, 2, 4}:
            total += df[col] - 1
        else:
            total += (df[col] - 4).abs()
    out['TotalEDINScore'] = total
    out['SubEDINScore'] = (out['TotalEDINScore'] >= 10).astype(int)
    return out


def compute_stai_scores(df):
    reverse = {1, 3, 6, 7, 10, 13, 14, 16, 19}
    out = df[['PublicID']].copy()
    total = pd.Series(0, index=df.index, dtype='float64')
    for i in range(1, 21):
        col = f'V1HA{i:02d}'
        if i in reverse and i > 9:
            total += (df[col] - 5).abs()
        else:
            total += df[col]
    out['TotalSTAIScore'] = total
    out['SubSTAIScore'] = (out['TotalSTAIScore'] >= 40).astype(int)
    return out


def build_combined_feature_lists(
    include_maternal_physical=True,
    include_mental_health=True,
    include_paternal=True,
    include_drugs=True,
    include_food=True,
    include_health_knowledge=True,
    include_mother_demo=True,
    include_physical_activity=True,
    include_sleep=True,
):
    """Return the shared feature typing used by the combined outcome notebooks.

    The returned lists reflect the latest typing from the domain-specific notebooks:
    - Maternal health: oDM/acog_PEgHTN are categorical, ChronHTN is binary.
    - Mother demo: Age_at_V1 is numeric, CRace is categorical, V1AF02/V1AF14/V1AF10 are ordinal,
      and has_healthcare is binary.
    - Drugs and health knowledge are binary feature sets.
    - Food includes PRENATALAMOUNT as ordinal.
    """
    numeric_features = []
    categorical_features = []
    ordinal_features = []
    binary_features = []

    if include_maternal_physical:
        categorical_features.extend(['oDM', 'acog_PEgHTN'])
        binary_features.append('ChronHTN')
    if include_mental_health:
        numeric_features.extend([
            'ResilienceTotalScore',
            'stress_average',
            'FrequencyOfHassles',
            'FrequencyOfUplifts',
            'IntensityOfHassles',
            'IntensityOfUplifts',
            'HassleUpliftFrequencyRatio',
            'HassleUpliftIntensityRatio',
            'StressTotalScore',
        ])
        binary_features.extend(['ResilienceLevel', 'StressLevel'])
    if include_paternal:
        numeric_features.extend(['V2AF13', 'V2AF15'])
    if include_food:
        numeric_features.extend([
            'DT_FOLAC',
            'DT_CALC',
            'VITD_MCG',
            'TOTAL_CHOLINE',
            'DT_SODI',
            'PRENATALYEARS',
            'AHEI2010',
            'AHEI_ALCDRKS',
            'AHEI_SODIUM',
            'AHEI_PUFAPCT',
            'AHEI_DHAEPA',
            'AHEI_TRFATPCT',
            'AHEI_RMEATS',
            'AHEI_NUTLEGS',
            'AHEI_SUGBEVS',
            'AHEI_WGRAINS',
            'AHEI_FRUITS',
            'AHEI_VEGS',
            'DT_ALCO',
            'DT_CAFFN',
            'DT_FIBE',
            'DT_SUG_T',
            'DT_CHOL',
            'DT_PFAT',
            'DT_MFAT',
            'DT_SFAT',
            'DT_TFAT',
            'DT_CARB',
            'DT_KCAL',
            'DT_PROT',
            'DT_VITC',
            'DT_VB12',
            'DT_VITB6',
            'DT_NIAC',
            'DT_RIBO',
            'DT_THIA',
            'DT_IRON',
            'DT_TOTN3',
        ])
        ordinal_features.append('PRENATALAMOUNT')
    if include_physical_activity:
        numeric_features.extend(['V2AJ01a2', 'V2AJ01a1'])
    if include_mother_demo:
        numeric_features.append('Age_at_V1')
        categorical_features.append('CRace')
        ordinal_features.extend(['V1AF02', 'V1AF14', 'V1AF10'])
        binary_features.append('has_healthcare')
    if include_drugs:
        binary_features.extend(['V2AH01', 'V2AH02', 'V2AH03', 'V2AH04', 'V2AH05', 'V2AH06'])
    if include_health_knowledge:
        binary_features.extend([
            'V1AD02a',
            'V1AD02b',
            'V1AD02c',
            'V1AD02d',
            'V1AD02e',
            'V1AD02f',
            'V1AD02g',
            'V1AD02h',
            'V1AD02i',
            'V1AD02j',
            'V1AD02k',
        ])
    sleep_features = [
        'rest_dur_avg_all_Mod',
        'rest_sleeptime_avg_all_Mod',
        'sleep_dur_avg_all_Mod',
        'sleep_sleeptime_avg_all_Mod',
        'sleep_Frag_avg_all_Mod',
        'sleep_WASO_avg_all_Mod',
        'sleep_SE_avg_all_Mod',
        'rest_sleeptime_avg_wkday_Mod',
    ] if include_sleep else []
    numeric_features.extend(sleep_features)
    feature_columns = numeric_features + ordinal_features + binary_features + categorical_features
    return {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'ordinal_features': ordinal_features,
        'binary_features': binary_features,
        'feature_columns': feature_columns,
        'sleep_features': sleep_features,
    }


def build_combined_feature_frame(
    data_dir,
    outcome_path=None,
    outcome_column=None,
    outcome_encoding=None,
    outcome_replace_map=None,
    include_outcome=True,
    include_maternal_physical=True,
    include_mental_health=True,
    include_paternal=True,
    include_drugs=True,
    include_food=True,
    include_health_knowledge=True,
    include_mother_demo=True,
    include_physical_activity=True,
    include_sleep=True,
):
    """Load and merge the shared spreadsheet-derived features used by the notebooks."""
    data_dir = Path(data_dir)
    feature_lists = build_combined_feature_lists(
        include_maternal_physical=include_maternal_physical,
        include_mental_health=include_mental_health,
        include_paternal=include_paternal,
        include_drugs=include_drugs,
        include_food=include_food,
        include_health_knowledge=include_health_knowledge,
        include_mother_demo=include_mother_demo,
        include_physical_activity=include_physical_activity,
        include_sleep=include_sleep,
    )

    maternal_physical_features = ['oDM', 'acog_PEgHTN', 'ChronHTN']
    v2i_cols = ['PublicID'] + [f'V2IA{i:02d}' for i in range(1, 26)]
    v1e_cols = [
        'PublicID', 'V1EA01', 'V1EA02a', 'V1EA02b', 'V1EA02c', 'V1EA02d',
        'V1EA02e', 'V1EA02f', 'V1EA02g', 'V1EA02h', 'V1EA02i', 'V1EA02j',
        'V1EA02k', 'V1EA02l',
    ]
    v3j_cols = ['PublicID'] + [f'V3JA01{letter}' for letter in 'abcdefghij'] + [f'V3JA02{letter}' for letter in 'abcdefghij']
    v1a_stress_cols = ['PublicID', 'V1AH04', 'V1AH05', 'V1AH07', 'V1AH08']
    v3a_stress_cols = ['PublicID', 'V3AG04', 'V3AG05', 'V3AG07', 'V3AG08']
    paternal_features = ['V2AF13', 'V2AF15']
    drug_features = ['V2AH01', 'V2AH02', 'V2AH03', 'V2AH04', 'V2AH05', 'V2AH06']
    food_features = [
        'DT_FOLAC', 'DT_CALC', 'VITD_MCG', 'TOTAL_CHOLINE', 'DT_SODI',
        'PRENATALYEARS', 'PRENATALAMOUNT', 'AHEI2010', 'AHEI_ALCDRKS',
        'AHEI_SODIUM', 'AHEI_PUFAPCT', 'AHEI_DHAEPA', 'AHEI_TRFATPCT',
        'AHEI_RMEATS', 'AHEI_NUTLEGS', 'AHEI_SUGBEVS', 'AHEI_WGRAINS',
        'AHEI_FRUITS', 'AHEI_VEGS', 'DT_ALCO', 'DT_CAFFN', 'DT_FIBE',
        'DT_SUG_T', 'DT_CHOL', 'DT_PFAT', 'DT_MFAT', 'DT_SFAT', 'DT_TFAT',
        'DT_CARB', 'DT_KCAL', 'DT_PROT', 'DT_VITC', 'DT_VB12', 'DT_VITB6',
        'DT_NIAC', 'DT_RIBO', 'DT_THIA', 'DT_IRON', 'DT_TOTN3',
    ]
    health_knowledge_features = [
        'V1AD02a', 'V1AD02b', 'V1AD02c', 'V1AD02d', 'V1AD02e', 'V1AD02f',
        'V1AD02g', 'V1AD02h', 'V1AD02i', 'V1AD02j', 'V1AD02k',
    ]
    mother_numeric_features = ['Age_at_V1', 'V1AF02', 'V1AF10']
    mother_categorical_features = ['CRace', 'V1AF14']
    healthcare_response_columns = ['V1AF15a', 'V1AF15b', 'V1AF15c', 'V1AF15d', 'V1AF15e', 'V1AF15f', 'V1AF15g']
    physical_activity_features = ['V2AJ01a2', 'V2AJ01a1']
    sleep_features = feature_lists['sleep_features']

    df_maternal_physical = None
    if include_maternal_physical:
        df_maternal_physical = pd.read_csv(
            data_dir / 'PREGNANCY_OUTCOMES.csv',
            usecols=maternal_physical_features + ['PublicID'],
        )
    df_v2i = None
    df_v1e = None
    df_v3j = None
    df_v1a_stress = None
    df_v3a_stress = None
    if include_mental_health:
        df_v2i = pd.read_csv(data_dir / 'V2I.csv', usecols=v2i_cols)
        df_v1e = pd.read_csv(data_dir / 'V1E.CSV', usecols=v1e_cols, encoding='ISO-8859-1')
        df_v3j = pd.read_csv(data_dir / 'V3J.csv', usecols=v3j_cols)
        df_v1a_stress = pd.read_csv(data_dir / 'V1A.CSV', usecols=v1a_stress_cols)
        df_v3a_stress = pd.read_csv(data_dir / 'V3A.CSV', usecols=v3a_stress_cols)

    df_paternal = None
    if include_paternal:
        df_paternal = pd.read_csv(data_dir / 'V2A.csv', usecols=paternal_features + ['PublicID'], low_memory=False)
        df_paternal[paternal_features] = df_paternal[paternal_features].replace({'R': np.nan, 'D': np.nan})
        df_paternal[paternal_features] = df_paternal[paternal_features].apply(pd.to_numeric, errors='coerce')

    df_drugs = None
    if include_drugs:
        df_drugs = pd.read_csv(data_dir / 'V2A.csv', usecols=drug_features + ['PublicID'], low_memory=False)
        df_drugs = df_drugs.replace({'R': np.nan, 'D': np.nan})
        df_drugs[drug_features] = df_drugs[drug_features].apply(pd.to_numeric, errors='coerce')

    df_food = None
    if include_food:
        df_food = pd.read_csv(data_dir / 'FOOD_FREQUENCY_ANALYSIS.csv', usecols=food_features + ['PublicID'])
        df_food[food_features] = df_food[food_features].apply(pd.to_numeric, errors='coerce')

    df_health_knowledge = None
    if include_health_knowledge:
        df_health_knowledge = pd.read_csv(data_dir / 'V1A.csv', usecols=health_knowledge_features + ['PublicID'])
        df_health_knowledge[health_knowledge_features] = df_health_knowledge[health_knowledge_features].apply(pd.to_numeric, errors='coerce')

    df_mother_demo = None
    if include_mother_demo:
        df_mother_num = pd.read_csv(
            data_dir / 'V1A.CSV',
            usecols=['V1AF02', 'V1AF14', 'V1AF10', 'PublicID'],
            encoding='latin-1',
        )
        df_mother_demo = pd.read_csv(
            data_dir / 'DEMOGRAPHICS.CSV',
            usecols=['Age_at_V1', 'PublicID', 'CRace'],
            encoding='latin-1',
        )
        df_mother_demo = pd.merge(df_mother_demo, df_mother_num, on='PublicID', how='inner')
        df_mother_demo[mother_numeric_features] = df_mother_demo[mother_numeric_features].apply(pd.to_numeric, errors='coerce')
        df_healthcare = pd.read_csv(data_dir / 'V1A.CSV', usecols=healthcare_response_columns + ['PublicID'])
        df_healthcare['has_healthcare'] = 0
        for index, row in df_healthcare.iterrows():
            for column in healthcare_response_columns:
                if row[column] == 1:
                    df_healthcare.at[index, 'has_healthcare'] = 1
                    break
                if row[column] == 0:
                    df_healthcare.at[index, 'has_healthcare'] = 0
                    break
        df_mother_demo = pd.merge(df_mother_demo, df_healthcare[['PublicID', 'has_healthcare']], on='PublicID', how='inner')
        df_mother_demo = df_mother_demo[~df_mother_demo['V1AF14'].isin(['D', 'R'])]

    df_physical_activity = None
    if include_physical_activity:
        df_physical_activity = pd.read_csv(
            data_dir / 'V2A.csv',
            usecols=physical_activity_features + ['PublicID'],
            low_memory=False,
        )
        df_physical_activity[physical_activity_features] = df_physical_activity[physical_activity_features].apply(pd.to_numeric, errors='coerce')

    df_sleep = None
    if include_sleep:
        df_sleep = pd.read_csv(
            data_dir / 'modified/SLEEP_ACTIGRAPHY_MODIFIED.CSV',
            usecols=sleep_features + ['PublicID'],
        )
        df_sleep[sleep_features] = df_sleep[sleep_features].apply(pd.to_numeric, errors='coerce')

    outcome_df = None
    if include_outcome:
        if outcome_path is None or outcome_column is None:
            raise ValueError('outcome_path and outcome_column are required when include_outcome=True')
        outcome_df = pd.read_csv(data_dir / outcome_path, usecols=['PublicID', outcome_column], encoding=outcome_encoding)
        if outcome_replace_map is not None:
            outcome_df[outcome_column] = outcome_df[outcome_column].replace(outcome_replace_map)
        outcome_df[outcome_column] = pd.to_numeric(outcome_df[outcome_column], errors='coerce')
        outcome_df = outcome_df.drop_duplicates(subset=['PublicID'])

    combined_df = None
    if include_maternal_physical:
        combined_df = df_maternal_physical
    if include_mental_health:
        mental_health_features = compute_resilience_score(df_v2i)
        mental_health_features = mental_health_features.merge(compute_stress_average(df_v1e), on='PublicID', how='outer')
        mental_health_features = mental_health_features.merge(compute_hassles_uplifts(df_v3j), on='PublicID', how='outer')
        mental_health_features = mental_health_features.merge(
            compute_stress_level(pd.merge(df_v1a_stress, df_v3a_stress, on='PublicID', how='inner')),
            on='PublicID',
            how='outer',
        )
        combined_df = mental_health_features if combined_df is None else pd.merge(combined_df, mental_health_features, on='PublicID', how='left')
    if include_paternal:
        combined_df = df_paternal if combined_df is None else combined_df.merge(df_paternal, on='PublicID', how='left')
    if include_drugs:
        combined_df = df_drugs if combined_df is None else combined_df.merge(df_drugs, on='PublicID', how='left')
    if include_food:
        combined_df = df_food if combined_df is None else combined_df.merge(df_food, on='PublicID', how='left')
    if include_health_knowledge:
        combined_df = df_health_knowledge if combined_df is None else combined_df.merge(df_health_knowledge, on='PublicID', how='left')
    if include_mother_demo:
        combined_df = df_mother_demo if combined_df is None else combined_df.merge(df_mother_demo, on='PublicID', how='left')
    if include_physical_activity:
        combined_df = df_physical_activity if combined_df is None else combined_df.merge(df_physical_activity, on='PublicID', how='left')
    if include_sleep:
        combined_df = combined_df.merge(df_sleep, on='PublicID', how='left')
    if include_outcome:
        combined_df = combined_df.merge(outcome_df, on='PublicID', how='inner')
        combined_df = combined_df.dropna(subset=[outcome_column]).copy()

    return {
        'combined_df': combined_df,
        'outcome_df': outcome_df,
        'feature_lists': feature_lists,
        'maternal_physical_features': maternal_physical_features,
        'paternal_features': paternal_features,
        'drug_features': drug_features,
        'food_features': food_features,
        'health_knowledge_features': health_knowledge_features,
        'mother_numeric_features': mother_numeric_features,
        'mother_categorical_features': mother_categorical_features,
        'physical_activity_features': physical_activity_features,
        'sleep_features': sleep_features,
    }


def make_master_split_ids(df, target_column='MH_outcome', id_column='PublicID', test_size=0.2, random_state=42):
    """Create a single stratified subject split that can be reused across domains."""
    split_frame = df[[id_column, target_column]].dropna().drop_duplicates(subset=[id_column])
    train_ids, test_ids = train_test_split(
        split_frame[id_column],
        test_size=test_size,
        random_state=random_state,
        stratify=split_frame[target_column],
    )
    return train_ids, test_ids


def load_or_create_master_split_ids(
    df,
    split_path,
    target_column='MH_outcome',
    id_column='PublicID',
    test_size=0.2,
    random_state=42,
):
    """Load a persisted split if it exists, otherwise create and save one."""
    split_path = Path(split_path)
    if split_path.exists():
        split_df = pd.read_csv(split_path)
        train_ids = split_df.loc[split_df['split'] == 'train', id_column].tolist()
        test_ids = split_df.loc[split_df['split'] == 'test', id_column].tolist()
        return train_ids, test_ids

    train_ids, test_ids = make_master_split_ids(
        df,
        target_column=target_column,
        id_column=id_column,
        test_size=test_size,
        random_state=random_state,
    )
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        id_column: list(train_ids) + list(test_ids),
        'split': ['train'] * len(train_ids) + ['test'] * len(test_ids),
    }).to_csv(split_path, index=False)
    return train_ids, test_ids


def make_preprocessor(
    numeric_features,
    ordinal_features=None,
    ordinal_feature_categories=None,
    binary_features=None,
    categorical_features=None,
    impute=False,
):
    """Build preprocessing for numeric, ordinal, binary, and categorical features.

    By default, missing values are passed through without imputation so that
    downstream estimators that can handle missingness can do so directly.
    Set ``impute=True`` to enable median/mode imputation.
    """
    transformers = []
    if numeric_features:
        numeric_steps = []
        if impute:
            numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
        numeric_steps.append(('scaler', StandardScaler()))
        numeric_pipe = SkPipeline(steps=numeric_steps)
        transformers.append(('num', numeric_pipe, numeric_features))
    if ordinal_features:
        if not ordinal_feature_categories:
            raise ValueError(
                'ordinal_feature_categories must be provided for ordinal_features '
                'so the intended order is preserved.'
            )
        ordinal_steps = []
        if impute:
            ordinal_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        ordinal_categories = [ordinal_feature_categories[feature] for feature in ordinal_features]
        ordinal_steps.append(
            (
                'ordinal',
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    dtype=np.float64,
                ),
            )
        )
        ordinal_pipe = SkPipeline(steps=ordinal_steps)
        transformers.append(('ord', ordinal_pipe, ordinal_features))
    if binary_features:
        if impute:
            binary_pipe = SkPipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent'))]
            )
            transformers.append(('bin', binary_pipe, binary_features))
        else:
            transformers.append(('bin', 'passthrough', binary_features))
    if categorical_features:
        categorical_steps = []
        if impute:
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
        categorical_pipe = SkPipeline(steps=categorical_steps)
        transformers.append(('cat', categorical_pipe, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder='drop')


def make_model_and_grid(model_name, random_state=42):
    """Return the estimator and hyperparameter grid for a supported model."""
    model_name = model_name.lower()
    if model_name == 'lr':
        estimator = LogisticRegression(
            random_state=random_state,
            max_iter=5000,
            solver='saga',
        )
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    elif model_name == 'rf':
        estimator = RandomForestClassifier(random_state=random_state)
        param_grid = {
            'classifier__n_estimators': [500, 600, 700],
            'classifier__max_depth': [20, 18, 15],
            'classifier__min_samples_split': [3, 5, 7],
            'classifier__min_samples_leaf': [1, 2, 4],
        }
    elif model_name == 'xgb':
        estimator = XGBClassifier(random_state=random_state, eval_metric='logloss')
        param_grid = {
            'classifier__learning_rate': [0.01, 0.05, 0.001],
            'classifier__n_estimators': [100, 80, 60],
            'classifier__max_depth': [7, 4, 6],
            'classifier__subsample': [0.8, 0.7, 0.5],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
        }
    elif model_name == 'svm':
        estimator = CalibratedClassifierCV(estimator=SVC(), ensemble=False)
        param_grid = [
            {
                'classifier__estimator__kernel': ['linear'],
                'classifier__estimator__C': [0.1, 1, 10, 100],
            },
            {
                'classifier__estimator__kernel': ['rbf'],
                'classifier__estimator__C': [0.1, 1, 10, 100],
                'classifier__estimator__gamma': ['scale', 'auto', 0.01, 0.1],
            },
        ]
    else:
        raise ValueError(f'Unsupported model_name: {model_name}')
    return estimator, param_grid


def run_model_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    numeric_features=None,
    categorical_features=None,
    ordinal_features=None,
    ordinal_feature_categories=None,
    binary_features=None,
    impute=False,
    scoring='auto',
    cv=5,
    verbose=1,
    n_jobs=-1,
):
    """Run the shared preprocessing, SMOTE, grid search, and evaluation flow."""
    estimator, param_grid = make_model_and_grid(model_name)

    feature_columns = []
    if numeric_features:
        feature_columns.extend(numeric_features)
    if ordinal_features:
        feature_columns.extend(ordinal_features)
    if binary_features:
        feature_columns.extend(binary_features)
    if categorical_features:
        feature_columns.extend(categorical_features)

    if numeric_features:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[numeric_features] = X_train[numeric_features].apply(pd.to_numeric, errors='coerce')
        X_test[numeric_features] = X_test[numeric_features].apply(pd.to_numeric, errors='coerce')
    if binary_features:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[binary_features] = X_train[binary_features].apply(pd.to_numeric, errors='coerce')
        X_test[binary_features] = X_test[binary_features].apply(pd.to_numeric, errors='coerce')

    resolved_ordinal_categories = None
    if ordinal_features:
        resolved_ordinal_categories = {}
        for feature in ordinal_features:
            series = X_train[feature]
            explicit_categories = None
            if ordinal_feature_categories and feature in ordinal_feature_categories:
                explicit_categories = ordinal_feature_categories[feature]
            resolved_ordinal_categories[feature] = resolve_ordinal_categories(
                series,
                explicit_categories=explicit_categories,
            )

    if not impute and feature_columns:
        train_mask = X_train[feature_columns].notna().all(axis=1) & y_train.notna()
        test_mask = X_test[feature_columns].notna().all(axis=1) & y_test.notna()
        dropped_train = int((~train_mask).sum())
        dropped_test = int((~test_mask).sum())
        if dropped_train or dropped_test:
            print(
                f"Dropping rows with missing values because impute=False "
                f"(train: {dropped_train}, test: {dropped_test})."
            )
        X_train = X_train.loc[train_mask].copy()
        y_train = y_train.loc[X_train.index].copy()
        X_test = X_test.loc[test_mask].copy()
        y_test = y_test.loc[X_test.index].copy()

    label_encoder = None
    if model_name.lower() == 'xgb':
        label_encoder = LabelEncoder()
        y_train = pd.Series(
            label_encoder.fit_transform(y_train),
            index=y_train.index,
            name=y_train.name,
        )
        y_test = pd.Series(
            label_encoder.transform(y_test),
            index=y_test.index,
            name=y_test.name,
        )

    unique_classes = np.unique(pd.Series(y_train).dropna())
    is_binary = len(unique_classes) == 2
    if scoring in (None, 'auto'):
        # Binary tasks use standard F1; multiclass tasks use macro F1.
        scoring = 'f1' if is_binary else 'f1_macro'
    positive_label = sorted(unique_classes.tolist())[-1] if is_binary else None

    print(
        f"Final dataset sizes for {model_name.upper()} "
        f"(impute={impute}): train={len(X_train)}, test={len(X_test)}"
    )

    preprocessor = make_preprocessor(
        numeric_features,
        ordinal_features=ordinal_features,
        ordinal_feature_categories=resolved_ordinal_categories,
        binary_features=binary_features,
        categorical_features=categorical_features,
        impute=impute,
    )
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', AdaptiveSMOTE(random_state=42)),
        ('classifier', estimator),
    ])
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {'accuracy': accuracy_score(y_test, y_pred)}
    if is_binary:
        metrics['precision'] = precision_score(
            y_test,
            y_pred,
            pos_label=positive_label,
            zero_division=0,
        )
        metrics['recall'] = recall_score(
            y_test,
            y_pred,
            pos_label=positive_label,
            zero_division=0,
        )
        metrics['f1'] = f1_score(
            y_test,
            y_pred,
            pos_label=positive_label,
            zero_division=0,
        )
        metrics['macro_precision'] = precision_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0,
        )
        metrics['macro_recall'] = recall_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0,
        )
        metrics['macro_f1'] = f1_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0,
        )
    else:
        metrics['precision'] = precision_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0,
        )
        metrics['recall'] = recall_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0,
        )
        metrics['f1'] = f1_score(
            y_test,
            y_pred,
            average='macro',
            zero_division=0,
        )
        metrics['weighted_f1'] = f1_score(
            y_test,
            y_pred,
            average='weighted',
            zero_division=0,
        )

    y_score = best_model.predict_proba(X_test)
    if len(unique_classes) == 2 and y_score.shape[1] >= 2:
        metrics['roc_auc'] = roc_auc_score(y_test, y_score[:, 1])
    elif y_score.shape[1] > 2:
        metrics['roc_auc_ovr_macro'] = roc_auc_score(
            y_test,
            y_score,
            multi_class='ovr',
            average='macro',
            labels=best_model.named_steps['classifier'].classes_,
        )
    else:
        print('ROC AUC skipped: unable to compute a stable score for this target.')

    if len(unique_classes) > 2:
        if label_encoder is not None:
            labels = list(range(len(label_encoder.classes_)))
            target_names = [str(label) for label in label_encoder.classes_]
        else:
            labels = sorted(unique_classes.tolist())
            target_names = [str(label) for label in labels]
        print('Per-class metrics:')
        print(
            classification_report(
                y_test,
                y_pred,
                labels=labels,
                target_names=target_names,
                zero_division=0,
            )
        )

    print('Best parameters found:', grid_search.best_params_)
    print(f"Best CV Score ({scoring}): {grid_search.best_score_:.4f}")

    classifier = best_model.named_steps['classifier']
    _, inner_classifier = _unwrap_classifier(best_model)
    try:
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        feature_names = []
        if numeric_features:
            feature_names.extend(numeric_features)
        if categorical_features:
            feature_names.extend(categorical_features)
    feature_names = np.asarray(feature_names)
    if model_name.lower() == 'svm':
        svm_coef = getattr(inner_classifier, 'coef_', None)
        if svm_coef is not None:
            svm_coef = np.asarray(svm_coef)
            if svm_coef.ndim == 1 or svm_coef.shape[0] == 1:
                coef_vector = svm_coef.ravel()
                print('Model Coefficients (inner linear SVM):')
            else:
                coef_vector = np.mean(np.abs(svm_coef), axis=0)
                print('Model Coefficients (mean absolute across classes for linear SVM):')
            for feature, coef in zip(feature_names[: len(coef_vector)], coef_vector):
                print(f"{feature}: {coef}")
        else:
            print('Skipping feature-level SVM output: non-linear SVM has no coefficients.')
    elif hasattr(classifier, 'coef_'):
        print('Model Coefficients:')
        for feature, coef in zip(feature_names[: len(classifier.coef_[0])], classifier.coef_[0]):
            print(f"{feature}: {coef}")
    elif hasattr(classifier, 'feature_importances_'):
        print('Feature Importances:')
        for feature, importance in zip(feature_names[: len(classifier.feature_importances_)], classifier.feature_importances_):
            print(f"{feature}: {importance}")

    print(f"Evaluation Metrics for {model_name.upper()} with shared preprocessing and adaptive CV scoring:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if is_binary:
        print(f"Precision (positive class): {metrics['precision']:.4f}")
        print(f"Recall (positive class): {metrics['recall']:.4f}")
        print(f"F1 (positive class): {metrics['f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
    else:
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Macro F1: {metrics['f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    if 'roc_auc_ovr_macro' in metrics:
        print(f"ROC AUC (ovr macro): {metrics['roc_auc_ovr_macro']:.4f}")

    return best_model, y_pred, metrics
