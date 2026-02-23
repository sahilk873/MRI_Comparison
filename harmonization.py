"""
ComBat harmonization for radiomic features (feature-domain batch correction).

Matches the manuscript: "ComBat harmonization to the classical radiomic features only;
batch variable is protocol (LF, 3T_low, 3T_high), and subject ID is included as a covariate
so that within-subject biological variation is preserved while protocol-specific location
and scale effects are estimated and removed."

Reference: Johnson et al., Biostatistics 2007; Fortin et al., NeuroImage 2017.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def combat_radiomics(
    data: np.ndarray,
    batch: np.ndarray,
    covariate: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    ComBat batch-effect correction (empirical Bayes) for radiomics.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        Feature matrix; each row is one sample (e.g. one subject-condition).
    batch : np.ndarray, shape (n_samples,), dtype str or int
        Batch (protocol) label per sample, e.g. 'LF', '3T_low', '3T_high'.
    covariate : np.ndarray, shape (n_samples,) or (n_samples, n_cov), optional
        Covariate(s) to preserve (e.g. subject ID encoded as dummy or int).
        If 1D, treated as categorical and expanded to dummies (excluding reference).

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)
        Harmonized feature matrix.
    """
    n_samples, n_features = data.shape
    batch_labels = np.asarray(batch).ravel()
    batches = np.unique(batch_labels)
    n_batches = len(batches)

    # Design matrix: intercept + covariate dummies (preserve biological variation)
    if covariate is not None:
        cov = np.asarray(covariate).ravel()
        if cov.ndim == 1 and cov.size == n_samples:
            # Categorical covariate -> dummy encoding (drop first to avoid collinearity)
            cov_vals = pd.Categorical(cov)
            dummies = pd.get_dummies(cov_vals, drop_first=True).values.astype(np.float64)
            X = np.column_stack([np.ones(n_samples), dummies])
        else:
            X = np.column_stack([np.ones(n_samples), np.asarray(covariate)])
    else:
        X = np.ones((n_samples, 1))

    # Standardize: remove covariate effect and global mean/std per feature
    # Solve X @ gamma = data (OLS per feature)
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    gamma = (XtX_inv @ X.T) @ data  # (n_covariates, n_features)
    data_std = data - X @ gamma
    grand_mean = np.mean(data_std, axis=0, keepdims=True)
    grand_std = np.std(data_std, axis=0, keepdims=True) + 1e-10
    data_std = (data_std - grand_mean) / grand_std

    # Per-batch location and scale (method-of-moments)
    batch_delta = np.zeros((n_batches, n_features))
    batch_lambda = np.zeros((n_batches, n_features))
    for i, b in enumerate(batches):
        idx = batch_labels == b
        n_b = np.sum(idx)
        batch_delta[i] = np.mean(data_std[idx], axis=0)
        batch_lambda[i] = np.std(data_std[idx], axis=0) + 1e-10

    # Empirical Bayes: shrink batch effects toward zero
    # Prior: delta_g ~ N(0, tau_sq), lambda_g^2 ~ InvGamma
    # We use a simple shrinkage toward global mean of batch params
    mean_delta = np.mean(batch_delta, axis=0)
    var_delta = np.var(batch_delta, axis=0) + 1e-10
    mean_lambda_sq = np.mean(batch_lambda ** 2, axis=0)
    var_lambda_sq = np.var(batch_lambda ** 2, axis=0) + 1e-10

    # Shrinkage weights (simplified EB)
    for i in range(n_batches):
        n_b = np.sum(batch_labels == batches[i])
        # Shrink delta
        batch_delta[i] = (var_delta * batch_delta[i] + (mean_delta / (n_batches + 1))) / (var_delta + 1.0 / (n_batches + 1))
        # Shrink lambda (keep scale adjustment moderate)
        batch_lambda_sq = batch_lambda[i] ** 2
        batch_lambda[i] = np.sqrt(
            (var_lambda_sq * batch_lambda_sq + mean_lambda_sq / (n_batches + 1))
            / (var_lambda_sq + 1.0 / (n_batches + 1))
        )

    # Adjust: remove batch location, scale by inverse batch scale
    out = data_std.copy()
    for i, b in enumerate(batches):
        idx = batch_labels == b
        out[idx] = (out[idx] - batch_delta[i]) / batch_lambda[i]

    # Back to original scale (add covariate effect and un-standardize)
    out = out * grand_std + grand_mean
    out = out + X @ gamma
    return out.astype(np.float64)


def similarity_and_fdi(
    feature_matrix: np.ndarray,
    meta: pd.DataFrame,
    subject_col: str = "subject",
    condition_col: str = "condition",
    condition_order: Tuple[str, str, str] = ("64mT", "3T_lowres", "3T_highres"),
) -> pd.DataFrame:
    """
    Compute paired cosine similarity, L2 distance, and FDI from a feature matrix.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_rows, n_features)
        One row per sample; order must match meta.
    meta : pd.DataFrame
        Must have subject_col and condition_col; one row per row of feature_matrix.
    subject_col, condition_col : str
        Column names.
    condition_order : tuple of three condition names
        (LF, 3T_low, 3T_high) for FDI denominator.

    Returns
    -------
    pd.DataFrame with columns subject, mod (if present), cos_lf_3Tlow, cos_3Tlow_3Thigh,
    l2_lf_3Tlow, l2_3Tlow_3Thigh, FDI_cos, FDI_l2.
    """
    from scipy.spatial.distance import cosine as scipy_cosine

    meta = meta.copy()
    meta["_idx"] = np.arange(len(meta))
    c1, c2, c3 = condition_order
    rows = []
    for sub, grp in meta.groupby(subject_col):
        grp = grp.drop_duplicates(subset=[condition_col])
        idx_map = grp.set_index(condition_col)["_idx"].to_dict()
        if not all(k in idx_map for k in (c1, c2, c3)):
            continue
        i1, i2, i3 = idx_map[c1], idx_map[c2], idx_map[c3]
        v1 = np.asarray(feature_matrix[i1], dtype=np.float64).copy()
        v2 = np.asarray(feature_matrix[i2], dtype=np.float64).copy()
        v3 = np.asarray(feature_matrix[i3], dtype=np.float64).copy()
        if np.any(np.isnan(v1)) or np.any(np.isnan(v2)) or np.any(np.isnan(v3)):
            continue
        # Z-score within subject (manuscript: "z-scored prior to cosine" for radiomics)
        for v in (v1, v2, v3):
            v -= np.mean(v)
            s = np.std(v)
            if s > 0:
                v /= s
        cos_12 = 1.0 - float(scipy_cosine(v1, v2))
        cos_23 = 1.0 - float(scipy_cosine(v2, v3))
        l2_12 = float(np.linalg.norm(v1 - v2))
        l2_23 = float(np.linalg.norm(v2 - v3))
        d_cos_12 = 1.0 - cos_12
        d_cos_23 = 1.0 - cos_23
        eps = 1e-12
        fdi_cos = d_cos_12 / (d_cos_12 + d_cos_23 + eps)
        fdi_l2 = l2_12 / (l2_12 + l2_23 + eps)
        row = {
            subject_col: sub,
            "cos_lf_3Tlow": cos_12,
            "cos_3Tlow_3Thigh": cos_23,
            "l2_lf_3Tlow": l2_12,
            "l2_3Tlow_3Thigh": l2_23,
            "FDI_cos": fdi_cos,
            "FDI_l2": fdi_l2,
        }
        if "modality" in grp.columns:
            row["modality"] = grp["modality"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def run_combat_on_radiomics_table(
    features_path: str,
    meta_columns: list,
    feature_columns: list,
    batch_col: str = "condition",
    subject_col: str = "subject",
    condition_order: Tuple[str, str, str] = ("64mT", "3T_lowres", "3T_highres"),
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load radiomics table (long format: one row per subject-condition), run ComBat,
    and return before/after similarity+FDI tables and harmonized matrix.

    Parameters
    ----------
    features_path : str
        Path to CSV with columns = meta_columns + feature_columns.
    meta_columns : list
        e.g. ['subject', 'modality', 'condition'].
    feature_columns : list
        Column names that are radiomic features.
    batch_col, subject_col : str
        Used for ComBat (batch) and covariate (subject).
    condition_order : tuple
        For FDI naming.

    Returns
    -------
    before_df : pd.DataFrame
        similarity_and_fdi on raw features.
    after_df : pd.DataFrame
        similarity_and_fdi on harmonized features.
    harmonized_matrix : np.ndarray
        ComBat-corrected feature matrix (same order as table).
    """
    df = pd.read_csv(features_path)
    X = df[feature_columns].values.astype(np.float64)
    batch = df[batch_col].values
    cov = df[subject_col].values
    meta = df[meta_columns].copy()
    meta["condition"] = df[batch_col]

    before_df = similarity_and_fdi(
        X, meta, subject_col=subject_col, condition_col="condition", condition_order=condition_order
    )
    X_combat = combat_radiomics(X, batch, covariate=cov)
    after_df = similarity_and_fdi(
        X_combat, meta, subject_col=subject_col, condition_col="condition", condition_order=condition_order
    )
    return before_df, after_df, X_combat
