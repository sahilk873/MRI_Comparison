#!/usr/bin/env python3
"""
Run ComBat harmonization on radiomics and print before/after similarity and FDI.

Reproduces the manuscript ComBat baseline (Table 5): cosine and L2 for LF vs 3T_low
and 3T_low vs 3T_high, before and after ComBat, plus optional FDI.

Usage (after running the notebook to generate features_classical_radiomics.csv):
  python run_combat_radiomics.py --features-csv path/to/features_classical_radiomics.csv
  python run_combat_radiomics.py --features-csv path/to/features.csv --meta-cols subject modality condition
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from harmonization import combat_radiomics, run_combat_on_radiomics_table, similarity_and_fdi


def main():
    ap = argparse.ArgumentParser(description="ComBat on radiomics, before/after similarity and FDI")
    ap.add_argument("--features-csv", required=True, help="Path to radiomics CSV (subject, condition, feature columns)")
    ap.add_argument("--meta-cols", nargs="+", default=["subject", "modality", "condition"],
                    help="Meta columns (default: subject modality condition)")
    ap.add_argument("--condition-order", nargs=3, default=["64mT", "3T_lowres", "3T_highres"],
                    help="Condition names for FDI order (LF, 3T_low, 3T_high)")
    ap.add_argument("--out-csv", default=None, help="Optional: save before/after summary to CSV")
    args = ap.parse_args()

    path = Path(args.features_csv)
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    meta_cols = [c for c in args.meta_cols if c in df.columns]
    if "condition" not in df.columns and "protocol" in df.columns:
        df = df.rename(columns={"protocol": "condition"})
    if "condition" not in df.columns:
        print("Error: need 'condition' (or 'protocol') column.", file=sys.stderr)
        sys.exit(1)
    if "subject" not in df.columns:
        print("Error: need 'subject' column.", file=sys.stderr)
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    if not feature_cols:
        print("Error: no numeric feature columns found.", file=sys.stderr)
        sys.exit(1)

    meta_cols = [c for c in meta_cols if c in df.columns]
    if "condition" not in meta_cols:
        meta_cols.append("condition")

    before_df, after_df, _ = run_combat_on_radiomics_table(
        str(path),
        meta_columns=meta_cols,
        feature_columns=feature_cols,
        batch_col="condition",
        subject_col="subject",
        condition_order=tuple(args.condition_order),
    )

    cols = ["cos_lf_3Tlow", "cos_3Tlow_3Thigh", "l2_lf_3Tlow", "l2_3Tlow_3Thigh"]
    print("Radiomics stability (paired cosine and L2) — BEFORE ComBat")
    print("-" * 50)
    print(before_df[cols].describe().loc[["mean", "std"]])
    print()
    print("Radiomics stability — AFTER ComBat")
    print("-" * 50)
    print(after_df[cols].describe().loc[["mean", "std"]])
    print()
    print("Before ComBat  mean cos(LF, 3T_low):", before_df["cos_lf_3Tlow"].mean())
    print("After ComBat   mean cos(LF, 3T_low):", after_df["cos_lf_3Tlow"].mean())
    print("Before ComBat  mean L2(LF, 3T_low):", before_df["l2_lf_3Tlow"].mean())
    print("After ComBat   mean L2(LF, 3T_low):", after_df["l2_lf_3Tlow"].mean())

    if args.out_csv:
        summary = pd.DataFrame({
            "metric": ["cos_lf_3Tlow", "cos_3Tlow_3Thigh", "l2_lf_3Tlow", "l2_3Tlow_3Thigh"],
            "before_mean": [
                before_df["cos_lf_3Tlow"].mean(), before_df["cos_3Tlow_3Thigh"].mean(),
                before_df["l2_lf_3Tlow"].mean(), before_df["l2_3Tlow_3Thigh"].mean(),
            ],
            "after_mean": [
                after_df["cos_lf_3Tlow"].mean(), after_df["cos_3Tlow_3Thigh"].mean(),
                after_df["l2_lf_3Tlow"].mean(), after_df["l2_3Tlow_3Thigh"].mean(),
            ],
        })
        summary.to_csv(args.out_csv, index=False)
        print(f"Saved summary to {args.out_csv}")


if __name__ == "__main__":
    main()
