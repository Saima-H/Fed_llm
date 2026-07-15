"""
Display Fed-LLM / TA-FedX-CPS output figures inside Google Colab.

Use after running:
    !python src/run_full_pipeline.py

Then run this in a normal Colab code cell:
    %run src/show_figures_colab.py
"""

from pathlib import Path

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


def show_figures():
    output_dir = find_output_dir()
    if output_dir is None:
        raise FileNotFoundError(
            "fedx_had_outputs folder not found. Run: !python src/run_full_pipeline.py"
        )

    show_markdown("# Fed-LLM / TA-FedX-CPS Figures")
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


def find_output_dir():
    candidates = [
        Path.cwd() / OUTPUT_DIR_NAME,
        Path.cwd().parent / OUTPUT_DIR_NAME,
        Path.cwd().parent.parent / OUTPUT_DIR_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(Path.cwd().glob(f"**/{OUTPUT_DIR_NAME}"))
    return matches[0] if matches else None


if __name__ == "__main__":
    show_figures()
