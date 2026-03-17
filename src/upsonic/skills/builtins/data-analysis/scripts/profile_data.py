#!/usr/bin/env python3
"""Quick data profiling script.

Usage:
    python profile_data.py <file_path> [--top N] [--output FORMAT]

Reads a CSV, Excel, or JSON file and prints a comprehensive data profile
including shape, types, missing values, descriptive stats, and value distributions.

Arguments:
    file_path   Path to the data file (.csv, .xlsx, .json, .tsv)
    --top N     Number of top values to show per column (default: 5)
    --output    Output format: text (default) or json
"""

import argparse
import json
import sys
from pathlib import Path


def load_data(file_path):
    """Load data from various file formats."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas is required. Install with: pip install pandas", file=sys.stderr)
        sys.exit(1)

    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    elif suffix == ".json":
        return pd.read_json(file_path)
    else:
        print(f"Error: Unsupported file format: {suffix}", file=sys.stderr)
        sys.exit(1)


def profile(df, top_n=5):
    """Generate a data profile dictionary."""
    result = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": {},
        "missing_summary": {},
    }

    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].count()),
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().sum() / len(df) * 100, 2),
            "unique_count": int(df[col].nunique()),
        }

        if df[col].dtype in ("int64", "float64", "int32", "float32"):
            desc = df[col].describe()
            col_info["stats"] = {
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
            }
        else:
            top_vals = df[col].value_counts().head(top_n)
            col_info["top_values"] = {
                str(k): int(v) for k, v in top_vals.items()
            }

        result["columns"][col] = col_info

        if col_info["null_pct"] > 0:
            result["missing_summary"][col] = col_info["null_pct"]

    return result


def print_text_report(report):
    """Print a human-readable text report."""
    shape = report["shape"]
    print(f"Dataset: {shape['rows']} rows x {shape['columns']} columns\n")

    print("=" * 60)
    print("COLUMN PROFILES")
    print("=" * 60)

    for col, info in report["columns"].items():
        print(f"\n--- {col} ---")
        print(f"  Type: {info['dtype']}")
        print(f"  Non-null: {info['non_null_count']} | Null: {info['null_count']} ({info['null_pct']}%)")
        print(f"  Unique values: {info['unique_count']}")

        if "stats" in info:
            s = info["stats"]
            print(f"  Mean: {s['mean']} | Std: {s['std']}")
            print(f"  Min: {s['min']} | 25%: {s['25%']} | 50%: {s['50%']} | 75%: {s['75%']} | Max: {s['max']}")

        if "top_values" in info:
            print("  Top values:")
            for val, count in info["top_values"].items():
                print(f"    {val}: {count}")

    if report["missing_summary"]:
        print("\n" + "=" * 60)
        print("MISSING DATA SUMMARY")
        print("=" * 60)
        for col, pct in sorted(report["missing_summary"].items(), key=lambda x: -x[1]):
            print(f"  {col}: {pct}% missing")


def main():
    parser = argparse.ArgumentParser(description="Profile a data file")
    parser.add_argument("file_path", help="Path to data file")
    parser.add_argument("--top", type=int, default=5, help="Top N values per column")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    df = load_data(args.file_path)
    report = profile(df, top_n=args.top)

    if args.output == "json":
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)


if __name__ == "__main__":
    main()
