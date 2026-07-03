import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


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


def make_preprocessor(numeric_features, categorical_features=None):
    """Impute and scale numeric features, and optionally encode categorical ones."""
    transformers = []
    if numeric_features:
        numeric_pipe = SkPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = SkPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ])
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
    scoring='f1_macro',
    cv=5,
    verbose=1,
    n_jobs=-1,
):
    """Run the shared preprocessing, SMOTE, grid search, and evaluation flow."""
    estimator, param_grid = make_model_and_grid(model_name)
    preprocessor = make_preprocessor(numeric_features, categorical_features=categorical_features)
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

    y_score = best_model.predict_proba(X_test)[:, 1]
    metrics['roc_auc'] = roc_auc_score(y_test, y_score)

    print('Best parameters found:', grid_search.best_params_)
    print(f"Best Macro F1 Score: {grid_search.best_score_:.4f}")

    classifier = best_model.named_steps['classifier']
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    if model_name.lower() == 'svm':
        print('Skipping feature-level SVM output to keep notebook output compact.')
    elif hasattr(classifier, 'coef_'):
        print('Model Coefficients:')
        for feature, coef in zip(feature_names, classifier.coef_[0]):
            print(f"{feature}: {coef}")
    elif hasattr(classifier, 'feature_importances_'):
        print('Feature Importances:')
        for feature, importance in zip(feature_names, classifier.feature_importances_):
            print(f"{feature}: {importance}")

    print(f"Evaluation Metrics for {model_name.upper()} with shared preprocessing and macro F1 grid search:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    return best_model, y_pred, metrics
