"""
Display Fed-LLM / TA-FedX-CPS output figures inside Google Colab.

Use after running:
    !python src/run_full_pipeline.py

Then run this in a normal Colab code cell:
    %run src/show_figures_colab.py
"""

from pathlib import Path
import argparse

try:
    from IPython.display import Image, Markdown, display
except ImportError:
    Image = Markdown = display = None

OUTPUT_DIR_NAME = "fedx_had_outputs"

FIGURES = [
    ("Proposed TA-FedX-IDS Confusion Matrix", "confusion_matrix_ta_fedx_ids.png"),
    ("Adaptive CDAW Confusion Matrix", "confusion_matrix_adaptive_cdaw.png"),
    ("ROC Comparison", "roc_comparison.png"),
    ("Adaptive CDAW Score Distribution", "cdaw_score_distribution.png"),
    ("SHAP Explanation for Ambiguous Sample", "shap_ambiguous_sample.png"),
]

TABLES = [
    ("Final Metrics", "metrics.csv"),
    ("Ablation Summary", "ablation_summary.csv"),
    ("Client Trust History", "trust_history.csv"),
]


def show_markdown(text):
    if display and Markdown:
        display(Markdown(text))
    else:
        print(text)


def show_figures(output_dir_name=OUTPUT_DIR_NAME):
    output_dir = find_output_dir(output_dir_name)
    if output_dir is None:
        raise FileNotFoundError(
            f"{output_dir_name} folder not found. Run the detector or comparison script first."
        )

    show_markdown(f"# Fed-LLM / TA-FedX-CPS Figures\n`{output_dir}`")
    for title, filename in FIGURES:
        path = output_dir / filename
        if path.exists():
            show_markdown(f"## {title}")
            if display and Image:
                display(Image(filename=str(path)))
            else:
                print(path)
        else:
            show_markdown(f"**Missing:** `{path}`")

    show_markdown("# Output Tables")
    for title, filename in TABLES:
        path = output_dir / filename
        if path.exists():
            show_markdown(f"## {title}\n`{path}`")
            try:
                import pandas as pd

                display(pd.read_csv(path).head(20))
            except Exception:
                print(path)


def find_output_dir(output_dir_name=OUTPUT_DIR_NAME):
    candidates = [
        Path.cwd() / output_dir_name,
        Path.cwd().parent / output_dir_name,
        Path.cwd().parent.parent / output_dir_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(Path.cwd().glob(f"**/{output_dir_name}"))
    return matches[0] if matches else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display TA-FedX-CPS output figures in Colab")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_NAME)
    args = parser.parse_args()
    show_figures(args.output_dir)
