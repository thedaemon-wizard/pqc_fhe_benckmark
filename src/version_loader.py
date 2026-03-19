"""
Dynamic version loader for PQC-FHE modules.
Reads version from version.json at the project root.
"""

import json
import os
from pathlib import Path

_CACHED_VERSION = None


def get_version(module_key: str = 'core') -> str:
    """Load version from version.json, with caching.

    Args:
        module_key: Key in the 'modules' dict of version.json.
                    Falls back to top-level 'version' if key not found.

    Returns:
        Version string (e.g., '3.2.0').
    """
    global _CACHED_VERSION
    if _CACHED_VERSION is not None:
        return _CACHED_VERSION

    root = Path(__file__).resolve().parent.parent
    vf = root / 'version.json'
    try:
        with open(vf, 'r') as f:
            data = json.load(f)
        version = data.get('modules', {}).get(module_key, data.get('version', '3.2.0'))
        _CACHED_VERSION = version
        return version
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return "3.2.0"
