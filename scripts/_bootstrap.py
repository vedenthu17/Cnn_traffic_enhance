"""Make `import roadlens` work when scripts are run as `python scripts/<name>.py`."""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# imagecorruptions imports the deprecated pkg_resources
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
