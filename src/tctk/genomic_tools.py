"""Backwards-compatibility shim — use ``tctk.genomic.hail`` instead."""

import importlib as _importlib

__all__ = ["scan_interval", "find_variant"]

def __getattr__(name):
    _map = {
        "scan_interval": ("tctk.genomic.hail", "scan_interval"),
        "find_variant": ("tctk.genomic.hail", "find_variant"),
    }
    if name in _map:
        mod_path, attr = _map[name]
        mod = _importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'tctk.genomic_tools' has no attribute {name!r}")
