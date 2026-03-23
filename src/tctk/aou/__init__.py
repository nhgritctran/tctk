import importlib as _importlib

__all__ = ["Dsub", "SocioEconomicStatus", "Demographic", "GWAS"]

# Lazy imports — these modules depend on the All of Us workbench
# environment (OWNER_EMAIL, WORKSPACE_CDR, etc.), so we defer
# importing until the classes are actually used.

def __getattr__(name):
    _map = {
        "Dsub": ("tctk.aou.dsub", "Dsub"),
        "SocioEconomicStatus": ("tctk.aou.ses", "SocioEconomicStatus"),
        "Demographic": ("tctk.aou.demographic", "Demographic"),
        "GWAS": ("tctk.aou.gwas", "GWAS"),
    }
    if name in _map:
        mod_path, attr = _map[name]
        mod = _importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'tctk.aou' has no attribute {name!r}")
