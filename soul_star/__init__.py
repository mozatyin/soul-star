"""Soul Star Map — reusable generation package."""
from ._api import generate_soul_star, generate_cinema_soul_star, generate_cosmos_soul_star
from .defaults import CINEMA_P_V9, COSMOS_P_V9
__all__ = [
    'generate_soul_star',
    'generate_cinema_soul_star',
    'generate_cosmos_soul_star',
    'CINEMA_P_V9',
    'COSMOS_P_V9',
]
