"""User-adjudicated shot/N input exclusions shared by production sorters.

The repository registry deliberately persists until a corrected data set has
been reviewed. Raw loaders remain usable for investigating invalid inputs.
"""

from dataclasses import dataclass
import csv
import hashlib
import io
from pathlib import Path


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs/known_invalid_inputs.csv"
INPUT_VALIDITY_VERSION = "known-invalid-inputs-v1"
KNOWN_INVALID_INPUT = "KNOWN_INVALID_INPUT"
FIELDS = ["shot", "ntor", "issue_id", "reason", "reviewer", "review_date", "evidence"]


@dataclass(frozen=True)
class InputValidityRegistry:
    sha256: str
    issues: dict[tuple[str, int], dict[str, str]]

    def diagnostic(self, shot: str, ntor: int) -> str | None:
        issue = self.issues.get((shot, ntor))
        if issue is None:
            return None
        return (
            f"{issue['issue_id']}: {issue['reason']} "
            f"Reviewer={issue['reviewer']}; date={issue['review_date']}; "
            f"evidence={issue['evidence']}; "
            f"{INPUT_VALIDITY_VERSION} registry_sha256={self.sha256}"
        )


def load_input_validity_registry(path: Path = REGISTRY_PATH) -> InputValidityRegistry:
    """Read one strict snapshot per run; missing or malformed policy aborts."""
    content = path.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    if reader.fieldnames != FIELDS:
        raise ValueError(f"{path}: expected registry columns {FIELDS}")
    issues = {}
    for line, row in enumerate(reader, 2):
        if set(row) != set(FIELDS) or any(
            not value or not value.strip() for value in row.values()
        ):
            raise ValueError(f"{path}:{line}: every registry field must be nonempty")
        shot = row["shot"]
        if shot in {".", ".."} or "/" in shot or "\\" in shot or shot != shot.strip():
            raise ValueError(f"{path}:{line}: shot must be an exact directory basename")
        try:
            ntor = int(row["ntor"])
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: ntor must be a positive integer") from exc
        if ntor < 1:
            raise ValueError(f"{path}:{line}: ntor must be a positive integer")
        key = (shot, ntor)
        if key in issues:
            raise ValueError(f"{path}:{line}: duplicate shot/ntor exclusion {key}")
        issues[key] = row
    return InputValidityRegistry(hashlib.sha256(content).hexdigest(), issues)
