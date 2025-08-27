__version__ = "1.0.0"
__title__ = "NexusForge Quantum Finance Simulator"
__description__ = "Ultimate AI-Powered Quantum Finance Framework"
__author__ = "NexusForge Team"
__email__ = "team@nexusforge.io"
__license__ = "MIT"
__url__ = "https://github.com/yourusernamehere/nexusforge-quantum-finance"

from src.core.config import settings
from src.core.logging import setup_logging

setup_logging()

__all__ = [
    "settings",
    "__version__",
    "__title__",
    "__description__",
    "__author__",
    "__email__",
    "__license__",
    "__url__",
]
