import sys
from pathlib import Path

# Let tests import the project as `src.<module>` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
