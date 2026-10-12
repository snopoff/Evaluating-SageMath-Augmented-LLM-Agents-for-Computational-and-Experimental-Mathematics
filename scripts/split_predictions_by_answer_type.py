from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.split_predictions_by_answer_type import main  # noqa: E402


if __name__ == "__main__":
    main()
