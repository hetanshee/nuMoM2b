import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


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


def make_preprocessor(numeric_features, categorical_features=None, impute=False):
    """Build preprocessing for numeric and categorical features.

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
    impute=False,
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1,
):
    """Run the shared preprocessing, SMOTE, grid search, and evaluation flow."""
    estimator, param_grid = make_model_and_grid(model_name)

    feature_columns = []
    if numeric_features:
        feature_columns.extend(numeric_features)
    if categorical_features:
        feature_columns.extend(categorical_features)

    if numeric_features:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[numeric_features] = X_train[numeric_features].apply(pd.to_numeric, errors='coerce')
        X_test[numeric_features] = X_test[numeric_features].apply(pd.to_numeric, errors='coerce')

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

    print(
        f"Final dataset sizes for {model_name.upper()} "
        f"(impute={impute}): train={len(X_train)}, test={len(X_test)}"
    )

    preprocessor = make_preprocessor(
        numeric_features,
        categorical_features=categorical_features,
        impute=impute,
    )
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
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

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro'),
    }

    y_score = best_model.predict_proba(X_test)
    unique_classes = np.unique(pd.Series(y_test).dropna())
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

    print('Best parameters found:', grid_search.best_params_)
    print(f"Best Macro F1 Score: {grid_search.best_score_:.4f}")

    classifier = best_model.named_steps['classifier']
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
        print('Skipping feature-level SVM output to keep notebook output compact.')
    elif hasattr(classifier, 'coef_'):
        print('Model Coefficients:')
        for feature, coef in zip(feature_names[: len(classifier.coef_[0])], classifier.coef_[0]):
            print(f"{feature}: {coef}")
    elif hasattr(classifier, 'feature_importances_'):
        print('Feature Importances:')
        for feature, importance in zip(feature_names[: len(classifier.feature_importances_)], classifier.feature_importances_):
            print(f"{feature}: {importance}")

    print(f"Evaluation Metrics for {model_name.upper()} with shared preprocessing and macro F1 grid search:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    if 'roc_auc_ovr_macro' in metrics:
        print(f"ROC AUC (ovr macro): {metrics['roc_auc_ovr_macro']:.4f}")

    return best_model, y_pred, metrics
