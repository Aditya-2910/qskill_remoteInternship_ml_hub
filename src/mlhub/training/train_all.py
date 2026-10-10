"""Train every model: ``python -m mlhub.training.train_all``."""

from __future__ import annotations

import argparse
import logging
import os

from mlhub.config import get_settings
from mlhub.training import train_house, train_iris, train_spam

TASKS = {"iris": train_iris, "spam": train_spam, "house": train_house}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks", nargs="*", metavar="TASK", help=f"which models to train, any of {', '.join(TASKS)} (default: all)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="never download data; use synthetic stand-ins for spam and house"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    args = parser.parse_args()
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        parser.error(f"unknown task(s): {', '.join(unknown)}; choose from {', '.join(TASKS)}")
    tasks = args.tasks or list(TASKS)
    if args.offline:
        os.environ["MLHUB_OFFLINE"] = "1"
    if args.no_mlflow:
        os.environ["MLHUB_MLFLOW"] = "0"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    print(f"Training {', '.join(tasks)} into {settings.models_dir}")
    for name in tasks:
        result = TASKS[name].train(settings)
        print(
            f"  {name:6s} best={result.best_model:24s} "
            + " ".join(f"{k}={v:.3f}" for k, v in result.test_metrics.items())
        )


if __name__ == "__main__":
    main()
