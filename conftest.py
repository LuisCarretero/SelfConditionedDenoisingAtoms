import sys
from pathlib import Path

# Repo has no package layout (no top-level __init__.py); make sibling
# top-level imports (`from models.callbacks import …`) work from tests/.
sys.path.insert(0, str(Path(__file__).parent))
