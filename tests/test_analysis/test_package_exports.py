import subprocess
import sys
from pathlib import Path


def test_qa_backend_import_does_not_cycle_through_analysis_exports():
    project_root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import qa.answer_generator; "
                "from analysis import GroqBackend, PaperAnalyzer"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
