import os
import argparse
import yaml
import torch
import numpy as np
import csv
from glob import glob

import sys

sys.path.insert(0, os.getcwd())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = config["paths"]["results_dir"]
    result_files = sorted(glob(os.path.join(results_dir, f"{args.dataset}_*.pt")))

    if not result_files:
        print(f"No result files found for {args.dataset}")
        return

    summary_rows = []
    for result_file in result_files:
        result = torch.load(result_file, weights_only=False)
        filename = os.path.basename(result_file).replace(".pt", "")

        if args.dataset == "iemocap":
            row = {
                "filename": filename,
                "avg_uar": f"{result['avg_uar']:.2f}",
                "std_uar": f"{result['std_uar']:.2f}",
                "avg_f1": f"{result['avg_f1']:.2f}",
                "std_f1": f"{result['std_f1']:.2f}",
            }

            for fold_idx, fold_result in enumerate(result["fold_results"]):
                row[f"fold{fold_idx}_uar"] = f"{fold_result['uar']:.2f}"
                row[f"fold{fold_idx}_f1"] = f"{fold_result['macro_f1']:.2f}"

            summary_rows.append(row)

            cm_text_path = os.path.join(results_dir, f"{filename}_confusion_matrix.txt")
            with open(cm_text_path, "w") as f:
                f.write(f"Confusion Matrix for {filename}\n")
                for fold_idx, fold_result in enumerate(result["fold_results"]):
                    cm = fold_result["confusion_matrix"]
                    f.write(f"\nFold {fold_idx}:\n")
                    labels = config["emotion_labels"]
                    f.write(f"  {'':>10}")
                    for l in labels:
                        f.write(f"{l:>10}")
                    f.write("\n")
                    for i, l in enumerate(labels):
                        f.write(f"  {l:>10}")
                        for j in range(len(labels)):
                            f.write(f"{cm[i][j]:>10}")
                        f.write("\n")
        else:
            test_metrics = result.get("test_metrics", {})
            row = {
                "filename": filename,
                "uar": f"{test_metrics.get('uar', 0):.2f}",
                "macro_f1": f"{test_metrics.get('macro_f1', 0):.2f}",
            }
            summary_rows.append(row)

    csv_path = os.path.join(results_dir, f"{args.dataset}_summary.csv")
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Summary saved to {csv_path}")
    print("\nResults:")
    for row in summary_rows:
        print(f"  {row['filename']}: {row}")


if __name__ == "__main__":
    main()
