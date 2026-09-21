"""Resolve the external data folder used by the CS5228 project."""

from __future__ import annotations

import os
from pathlib import Path


ENV_DATA_ROOT = "CS5228_DATA_ROOT"
DEFAULT_BUNDLE_NAME = "CS5228 Project Data"


def resolve_data_root(project_root: Path, explicit: Path | None = None) -> Path:
    """Return a data root containing ``train.csv``.

    Resolution order is: an explicit CLI value, ``CS5228_DATA_ROOT``, a legacy
    in-repository ``data`` folder, then the standard sibling Drive bundle.
    """

    project_root = Path(project_root).expanduser().resolve()
    candidates: list[tuple[str, Path]] = []
    if explicit is not None:
        candidates.append(("--data-root", Path(explicit).expanduser()))

    environment_value = os.environ.get(ENV_DATA_ROOT)
    if environment_value:
        candidates.append((ENV_DATA_ROOT, Path(environment_value).expanduser()))

    candidates.extend(
        [
            ("legacy project data folder", project_root / "data"),
            (
                "standard sibling data bundle",
                project_root.parent / DEFAULT_BUNDLE_NAME / "data",
            ),
        ]
    )

    attempted = []
    for label, candidate in candidates:
        candidate = candidate.resolve()
        attempted.append(f"- {label}: {candidate}")
        if (candidate / "train.csv").is_file():
            return candidate

    attempted_text = "\n".join(attempted)
    raise FileNotFoundError(
        "Could not locate the CS5228 data root (a folder containing train.csv).\n"
        f"Checked:\n{attempted_text}\n"
        f"Set {ENV_DATA_ROOT} to the external bundle's data folder or pass "
        "--data-root explicitly."
    )
