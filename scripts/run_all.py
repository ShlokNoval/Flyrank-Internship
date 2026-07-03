from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ml_utils import OUTPUT_DIR, RAW_PATH, ROOT, read_json


def run_command(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(
            f"Starter data not found: {RAW_PATH}\n"
            "The anonymized starter CSV ships with this repo. "
            "Restore it from git (`git checkout -- data/raw/content_refresh_anonymized.csv`).\n"
            "No BigQuery export is needed here — this repo runs entirely on the bundled sample."
        )

    run_command([sys.executable, str(ROOT / "scripts" / "01_prepare_features.py")])
    run_command([sys.executable, str(ROOT / "scripts" / "02_baseline_score.py")])
    run_command([sys.executable, str(ROOT / "scripts" / "03_train_model.py")])
    run_command([sys.executable, str(ROOT / "scripts" / "04_evaluate_and_export.py")])
    run_command([sys.executable, str(ROOT / "scripts" / "05_build_pdf_report.py")])

    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        print("\nPipeline complete")
        print(f"Rows scored: {summary['rows_scored']:,}")
        print(f"Best model: {summary['best_model']}")
        print(f"Queue: {summary['queue_output']}")
        print(f"Report: {summary['report_output']}")
        print(f"PDF: {OUTPUT_DIR / 'flyrank_refresh_model_results.pdf'}")


if __name__ == "__main__":
    main()
