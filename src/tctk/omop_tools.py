"""Backwards-compatibility shim — use ``tctk.omop`` instead."""

import warnings as _warnings

_warnings.warn(
    "tctk.omop_tools is deprecated; use tctk.omop instead.",
    DeprecationWarning,
    stacklevel=2,
)

from tctk.omop import Condition2ICD, Condition2SNOMED, ConditionMapperBase

__all__ = ["Condition2ICD", "Condition2SNOMED", "ConditionMapperBase"]
