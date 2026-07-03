"""Sheaf geometry: restriction maps, the sheaf Laplacian, and LoRA modulation."""

from __future__ import annotations

from .base import SheafGeometry
from .fixed import FixedGeometry
from .lora import LoRAGeometry, create_lora_geometry, create_sudoku_lora_geometry
from .restriction_maps import (
    DirectionalRestrictionMaps,
    SharedRestrictionMap,
    SudokuRestrictionMaps,
    build_directional_restriction_maps,
    build_shared_restriction_maps,
    build_sudoku_restriction_maps,
    compute_direction_index,
    get_direction_names,
    make_rm_initializer,
    make_sudoku_rm_initializer,
    normalize_rm_sharing,
)

__all__ = [
    "SheafGeometry",
    "FixedGeometry",
    "LoRAGeometry",
    "create_lora_geometry",
    "create_sudoku_lora_geometry",
    "DirectionalRestrictionMaps",
    "SharedRestrictionMap",
    "SudokuRestrictionMaps",
    "build_directional_restriction_maps",
    "build_shared_restriction_maps",
    "build_sudoku_restriction_maps",
    "compute_direction_index",
    "get_direction_names",
    "make_rm_initializer",
    "make_sudoku_rm_initializer",
    "normalize_rm_sharing",
]
