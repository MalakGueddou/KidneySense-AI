import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1. DÉFINITION DES VARIABLES DU MODÈLE 1
# ==============================================================================

NUM_FEATURES = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
BIN_FEATURES = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
ALL_FEATURES = NUM_FEATURES + BIN_FEATURES
TARGET_COL = "class"


def resolve_clean_uci_path() -> Path:
    candidates = [
        Path("data/clean_kidney_disease.csv"), 
        Path("clean_kidney_disease.csv"),
        Path("data/clean/clean_kidney_disease.csv"),
        Path("data/clean/uci_ckd_clean.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Fichier introuvable : placez 'clean_kidney_disease.csv' à la racine ou dans 'data/clean/'."
    )


def build_preprocessor() -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_FEATURES),
            ("bin", bin_pipe, BIN_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_on_test(model_name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall_score(y_test, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "sensitivity_recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def main():
    # 1. Chargement des données propres
    data_path = resolve_clean_uci_path()
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET_COL]).copy()

    X = df[ALL_FEATURES]
    y = df[TARGET_COL].astype(int)

    print(f"Dataset chargé : {data_path} ({len(df)} patients | CKD=1: {y.sum()}, Non-CKD=0: {(1-y).sum()})")

    # 2. Split Train / Test stratifié (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Candidats Modèle 1
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_pipelines = {}

    print("\n--- VALIDATION CROISÉE (5-FOLD SUR TRAIN) & TEST SET ---")
    for name, clf in candidates.items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])

        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["roc_auc", "recall", "f1", "neg_brier_score"],
            n_jobs=-1
        )

        # Entraînement final sur tout X_train
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        test_metrics = evaluate_on_test(name, pipe, X_test, y_test)
        test_metrics["cv_roc_auc_mean"] = round(float(cv_scores["test_roc_auc"].mean()), 4)
        test_metrics["cv_recall_mean"] = round(float(cv_scores["test_recall"].mean()), 4)
        test_metrics["cv_brier_mean"] = round(float(-cv_scores["test_neg_brier_score"].mean()), 4)
        results.append(test_metrics)

        print(
            f"[{name:<18}] CV ROC-AUC={test_metrics['cv_roc_auc_mean']:.4f} | "
            f"Test ROC-AUC={test_metrics['roc_auc']:.4f} | "
            f"Sensibilité={test_metrics['sensitivity_recall']:.4f} | "
            f"Spécificité={test_metrics['specificity']:.4f} | "
            f"Brier={test_metrics['brier_score']:.4f} | "
            f"FN={test_metrics['confusion_matrix']['fn']}"
        )

    # 4. Sélection du meilleur modèle (priorité : ROC-AUC, puis sensibilité, puis Brier bas)
    best_res = sorted(
        results,
        key=lambda r: (r["cv_roc_auc_mean"], r["sensitivity_recall"], -r["brier_score"]),
        reverse=True,
    )[0]
    best_name = best_res["model"]
    best_pipe = trained_pipelines[best_name]

    print(f"\n>>> Meilleur Modèle 1 sélectionné : {best_name}")
    y_test_pred = best_pipe.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=["Non-CKD (0)", "CKD (1)"]))

    # 5. Importance des variables (si RandomForest / GradientBoosting ou Coefs LogisticRegression)
    clf_step = best_pipe.named_steps["classifier"]
    print("--- Top 10 Variables les plus importantes ---")
    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_
    else:
        importances = np.abs(clf_step.coef_[0])

    feat_imp = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
    for feat, val in feat_imp.head(10).items():
        print(f"  - {feat:<10}: {val:.4f}")

    # 6. Sauvegarde de l'artefact & métadonnées pour l'API / Orchestrateur
    art_dir = Path("artifacts")
    art_dir.mkdir(exist_ok=True)

    model_path = art_dir / "model1_uci_pipeline.joblib"
    joblib.dump(best_pipe, model_path)

    meta = {
        "best_model": best_name,
        "features_order": ALL_FEATURES,
        "num_features": NUM_FEATURES,
        "bin_features": BIN_FEATURES,
        "all_candidates_metrics": results,
    }
    meta_path = art_dir / "model1_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nModèle 1 sauvegardé : {model_path}")
    print(f"Métadonnées sauvegardées : {meta_path}")


if __name__ == "__main__":
    main()