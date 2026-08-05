"""
File-based asset store keyed by identifier.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_UNSAFE_CHARS_RE = re.compile(r'[\/\\:*?"<>|]')


class AssetManager:
    """
    Manage assets in a centralized directory.

    Parameters
    ----------
    base_dir : str | Path
        Root directory for assets.
    extension : str
        File extension to append (e.g. ".pdf").
    """

    def __init__(self, base_dir: str | Path, extension: str = ".pdf") -> None:
        self.base_dir = Path(base_dir)
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize(identifier: str) -> str:
        """Replace filesystem-unsafe characters with '_'."""
        return _UNSAFE_CHARS_RE.sub("_", identifier)

    def get_path(self, identifier: str) -> Path:
        return self.base_dir / (self._sanitize(identifier) + self.extension)

    def is_available(self, identifier: str) -> bool:
        return self.get_path(identifier).is_file()

    def read(self, identifier: str) -> bytes:
        """Read and return the raw binary content."""
        path = self.get_path(identifier)
        if not path.is_file():
            raise FileNotFoundError(f"Asset not found for '{identifier}' at {path}")
        return path.read_bytes()

    def save(self, identifier: str, data: bytes) -> Path:
        """Write binary data to disk. Returns the path written to."""
        path = self.get_path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.info("Saved asset '%s' (%d bytes) -> %s", identifier, len(data), path)
        return path
