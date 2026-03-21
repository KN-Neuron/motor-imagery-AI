"""
CSP + classical ML pipeline.

Includes the pairwise MultiClassCSP transformer and the full
grid-search pipeline over 7 classifiers (LDA, SVM, RF, KNN, GB, LR, MLP).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MultiClassCSP(BaseEstimator, TransformerMixin):
    """
    Pairwise CSP for multiclass (3-class) problems.

    Trains a separate binary CSP for each pair of classes,
    then concatenates all features into one vector.
    """

    def __init__(self, n_components: int = 4, reg: str = "ledoit_wolf"):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        classes = np.unique(y)
        self.pairs_ = [(c1, c2) for i, c1 in enumerate(classes) for c2 in classes[i + 1 :]]
        self.csps_ = []
        for c1, c2 in self.pairs_:
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=True,
                norm_trace=False,
            )
            mask = (y == c1) | (y == c2)
            csp.fit(X[mask], y[mask])
            self.csps_.append(csp)
        return self

    def transform(self, X):
        features = [csp.transform(X) for csp in self.csps_]
        return np.concatenate(features, axis=1)


def get_ml_models(task_mode: str = "ternary") -> dict[str, dict[str, Any]]:
    """
    Return the model definitions + param grids for CSP+ML grid search.

    Parameters
    ----------
    task_mode : "binary" or "ternary"
        Binary uses direct CSP (no pairwise), ternary uses MultiClassCSP.
        Also adjusts ``csp__n_components`` search range.

    Returns
    -------
    dict of model configs
    """
    csp_components = [4, 6, 8] if task_mode == "binary" else [4, 6]

    models = {
        "LDA": {
            "model": LinearDiscriminantAnalysis(),
            "params": [
                {
                    "classifier__solver": ["svd"],
                    "classifier__shrinkage": [None],
                    "csp__n_components": csp_components,
                },
                {
                    "classifier__solver": ["lsqr"],
                    "classifier__shrinkage": [None, "auto"],
                    "csp__n_components": csp_components,
                },
            ],
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42, class_weight="balanced"),
            "params": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
                "csp__n_components": csp_components,
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 10, None],
                "classifier__min_samples_leaf": [1, 5],
                "csp__n_components": csp_components,
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
                "csp__n_components": csp_components,
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__max_depth": [3, 5],
                "csp__n_components": csp_components,
            },
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
                "classifier__solver": ["saga"],
                "csp__n_components": csp_components,
            },
        },
        "MLP": {
            "model": MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
            "params": {
                "classifier__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "classifier__alpha": [0.0001, 0.001],
                "classifier__learning_rate_init": [0.001, 0.01],
                "csp__n_components": csp_components,
            },
        },
    }
    return models


def run_csp_ml_grid(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    task_mode: str = "ternary",
    n_splits: int = 5,
    model_names: list[str] | None = None,
    csp_reg: str = "ledoit_wolf",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run CSP + ML grid search with subject-based CV.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timepoints)
    y : (n_epochs,)
    subjects : (n_epochs,)
    task_mode : "binary" or "ternary"
    n_splits : int
    model_names : list of str, optional
        Subset of models to evaluate (default: all).
    csp_reg : str
        CSP regularization.

    Returns
    -------
    list of dicts with keys: model, best_cv_acc, best_params, time_s, grid_obj
    """
    models = get_ml_models(task_mode)
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}

    gkf = GroupKFold(n_splits=n_splits)
    all_results = []

    for name, config in models.items():
        if verbose:
            print(f"\n{'#'*50}\n  {name}\n{'#'*50}")

        if task_mode == "binary":
            csp_step = CSP(n_components=4, reg=csp_reg, log=True, norm_trace=False)
        else:
            csp_step = MultiClassCSP(n_components=4, reg=csp_reg)

        pipe = Pipeline([
            ("csp", csp_step),
            ("scaler", StandardScaler()),
            ("classifier", config["model"]),
        ])

        grid = GridSearchCV(
            pipe,
            param_grid=config["params"],
            cv=gkf,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            error_score="raise",
        )

        start = time.time()
        grid.fit(X, y, groups=subjects)
        elapsed = time.time() - start

        if verbose:
            print(f"Best params: {grid.best_params_}")
            print(f"Best CV accuracy: {grid.best_score_:.4f}")
            print(f"Time: {elapsed:.1f}s")

        all_results.append({
            "model": name,
            "best_cv_acc": grid.best_score_,
            "best_params": grid.best_params_,
            "time_s": elapsed,
            "grid_obj": grid,
        })

    return all_results
