import sys
from pathlib import Path
import importlib.util

CURRENT_DIR = Path(__file__).resolve().parent  # backend/backend
BACKEND_DIR = CURRENT_DIR.parent               # backend
ROOT_DIR = BACKEND_DIR.parent                  # project root

for p in [str(ROOT_DIR), str(ROOT_DIR / "src"), str(BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

spec = importlib.util.spec_from_file_location("outer_main", str(BACKEND_DIR / "main.py"))
outer_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(outer_mod)
app = outer_mod.app
