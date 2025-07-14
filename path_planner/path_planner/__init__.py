import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
ROOT_DIR = FILE_DIR.parent
RESOURCE_DIR = ROOT_DIR / "resource"

sys.path.append(FILE_DIR.as_posix())
