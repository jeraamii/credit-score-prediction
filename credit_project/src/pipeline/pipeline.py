"""
src/pipeline/pipeline.py
Credit Score Classification Pipeline — OOP Style
Classes: Preprocessor, BaseTrainer (+ concrete subclasses), Evaluator
"""

import re
import os
import warnings
import numpy as np
import pandas as pd
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "C.csv"
MODEL_DIR = BASE_DIR / "models"
MLFLOW_DB = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
EXPERIMENT_NAME = "credit_score_classification"

TARGET = "Credit_Score"
DROP_COLS = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN", "Month"]


# 1.  PREPROCESSOR
class Preprocessor:
    """
    Handles all data cleaning, feature engineering, and transformation.
    Wraps a sklearn ColumnTransformer (numeric + categorical branches).
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_pipeline: ColumnTransformer | None = None
        self.num_features: list[str] = []
        self.cat_features: list[str] = []
        self.is_fitted: bool = False

    # Static cleaning helpers
    @staticmethod
    def _clean_numeric_str(series: pd.Series) -> pd.Series:
        """Strip trailing/embedded non-numeric chars then cast to float."""
        return pd.to_numeric(
            series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce",
        )

    @staticmethod
    def _parse_credit_history_age(series: pd.Series) -> pd.Series:
        """Convert 'X Years and Y Months' → total months (float)."""
        def _to_months(val):
            if pd.isna(val):
                return np.nan
            y = re.search(r"(\d+)\s*Year", str(val))
            m = re.search(r"(\d+)\s*Month", str(val))
            return (int(y.group(1)) if y else 0) * 12 + (int(m.group(1)) if m else 0)
        return series.apply(_to_months)

    # Public API
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop irrelevant columns and clean messy values."""
        df = df.copy()
        df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

        messy_numeric = [
            "Age", "Annual_Income", "Num_of_Loan",
            "Num_of_Delayed_Payment", "Changed_Credit_Limit",
            "Outstanding_Debt", "Amount_invested_monthly",
        ]
        for col in messy_numeric:
            if col in df.columns:
                df[col] = self._clean_numeric_str(df[col])

        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = self._parse_credit_history_age(df["Credit_History_Age"])

        # Remove biologically impossible ages
        if "Age" in df.columns:
            df = df[(df["Age"].isna()) | (df["Age"].between(18, 100))]

        if TARGET in df.columns:
            df = df[df[TARGET].notna()]

        return df.reset_index(drop=True)

    def _build_feature_pipeline(self, X: pd.DataFrame):
        self.num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_features  = X.select_dtypes(include=["object", "category"]).columns.tolist()

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        self.feature_pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.num_features),
                ("cat", cat_pipe, self.cat_features),
            ],
            remainder="drop",
        )

    def fit_transform(self, df: pd.DataFrame):
        """Clean, fit, and transform. Returns (X_array, y_array)."""
        df = self.clean(df)
        X  = df.drop(columns=[TARGET])
        y  = self.label_encoder.fit_transform(df[TARGET])
        self._build_feature_pipeline(X)
        X_out = self.feature_pipeline.fit_transform(X)
        self.is_fitted = True
        return X_out, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a new dataframe (inference-time)."""
        df = self.clean(df)
        if TARGET in df.columns:
            df = df.drop(columns=[TARGET])
        return self.feature_pipeline.transform(df)

    def save(self, path: str | Path):
        joblib.dump(self, path)
        print(f"  [Preprocessor] saved → {path}")

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)


# 2.  BASE TRAINER
class BaseTrainer:
    """Abstract base class for all model trainers."""

    model_name: str = "base"

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None

    def build(self):
        raise NotImplementedError

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model = self.build()
        self.model.fit(X_train, y_train)
        return self.model

    def save(self, path: str | Path):
        joblib.dump(self.model, path)
        print(f"  [{self.model_name}] saved → {path}")

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)


# 3.  CONCRETE TRAINERS
class LogisticRegressionTrainer(BaseTrainer):
    model_name = "logistic_regression"

    def build(self):
        return LogisticRegression(
            C=self.hyperparams.get("C", 1.0),
            max_iter=self.hyperparams.get("max_iter", 500),
            solver=self.hyperparams.get("solver", "lbfgs"),
            random_state=42,
        )


class DecisionTreeTrainer(BaseTrainer):
    model_name = "decision_tree"

    def build(self):
        return DecisionTreeClassifier(
            max_depth=self.hyperparams.get("max_depth", 10),
            min_samples_split=self.hyperparams.get("min_samples_split", 5),
            random_state=42,
        )


class RandomForestTrainer(BaseTrainer):
    model_name = "random_forest"

    def build(self):
        return RandomForestClassifier(
            n_estimators=self.hyperparams.get("n_estimators", 200),
            max_depth=self.hyperparams.get("max_depth", None),
            n_jobs=-1,
            random_state=42,
        )


class GradientBoostingTrainer(BaseTrainer):
    model_name = "gradient_boosting"

    def build(self):
        return GradientBoostingClassifier(
            n_estimators=self.hyperparams.get("n_estimators", 150),
            learning_rate=self.hyperparams.get("learning_rate", 0.1),
            max_depth=self.hyperparams.get("max_depth", 5),
            random_state=42,
        )


class AdaBoostTrainer(BaseTrainer):
    model_name = "adaboost"

    def build(self):
        return AdaBoostClassifier(
            n_estimators=self.hyperparams.get("n_estimators", 100),
            learning_rate=self.hyperparams.get("learning_rate", 0.5),
            random_state=42,
        )


# 4.  EVALUATOR
class Evaluator:
    """Computes classification metrics and returns a dict for MLflow logging."""

    def __init__(self, label_encoder: LabelEncoder):
        self.label_encoder = label_encoder

    def evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred  = model.predict(X_test)
        classes = self.label_encoder.classes_

        acc       = accuracy_score(y_test, y_pred)
        f1_w      = f1_score(y_test, y_pred, average="weighted")
        f1_mac    = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall    = recall_score(y_test, y_pred, average="weighted")

        print(f"\n  Accuracy    : {acc:.4f}")
        print(f"  F1 Weighted : {f1_w:.4f}")
        print(f"  F1 Macro    : {f1_mac:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=classes)}")

        return {
            "accuracy":  round(acc, 4),
            "f1":        round(f1_w, 4),
            "f1_macro":  round(f1_mac, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
        }


# 5.  PIPELINE RUNNER
def run_pipeline():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Credit Score Classification Pipeline")
    print("=" * 60)

    # Load & preprocess
    print("\n[1/4] Loading data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df)
    classes = preprocessor.label_encoder.classes_
    print(f"  Features after preprocessing: {X.shape[1]}")
    print(f"  Classes: {list(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # Save preprocessor
    preprocessor.save(MODEL_DIR / "preprocessor.pkl")

    # Define experiments
    experiments = [
        LogisticRegressionTrainer(C=1.0, max_iter=500, solver="lbfgs"),
        DecisionTreeTrainer(max_depth=10, min_samples_split=5),
        RandomForestTrainer(n_estimators=200),
        GradientBoostingTrainer(n_estimators=150, learning_rate=0.1, max_depth=5),
        AdaBoostTrainer(n_estimators=100, learning_rate=0.5),
    ]

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment(EXPERIMENT_NAME)
    evaluator  = Evaluator(preprocessor.label_encoder)

    best_model  = None
    best_score  = -1.0
    best_name   = ""
    all_metrics = []

    print("\n[2/4] Training models ...")
    for trainer in experiments:
        print(f"\n  ── {trainer.model_name} ──")
        with mlflow.start_run(run_name=trainer.model_name):
            mlflow.set_tags({
                "model_type": trainer.model_name,
                "task": "classification",
            })
            mlflow.log_params(trainer.hyperparams)

            model   = trainer.train(X_train, y_train)
            metrics = evaluator.evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            all_metrics.append({"model": trainer.model_name, **metrics})

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = model
                best_name  = trainer.model_name

    # Summary table
    print("\n[3/4] Results Summary")
    results_df = (
        pd.DataFrame(all_metrics)
        .set_index("model")
        .sort_values("f1", ascending=False)
    )
    print(results_df.to_string())

    # Save best model
    print(f"\n[4/4] Best model: {best_name}  (F1={best_score:.4f})")
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    (MODEL_DIR / "best_model_name.txt").write_text(best_name)

    print(f"\n  Artifacts saved to: {MODEL_DIR}")
    print("=" * 60)
    return best_model, preprocessor


if __name__ == "__main__":
    run_pipeline()
