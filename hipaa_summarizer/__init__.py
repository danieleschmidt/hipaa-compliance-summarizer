from pathlib import Path
from typing import List, Optional


def parse_requirements(path: Optional[str] = "requirements.txt") -> List[str]:
    """Return a list of requirements from a file."""
    if path is None:
        path = "requirements.txt"
    req_file = Path(path)
    if not req_file.exists():
        return []
    return [line.strip() for line in req_file.read_text().splitlines() if line.strip()]
