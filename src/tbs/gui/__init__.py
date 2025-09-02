"""GUI package for Trading Bot Simulator."""

from .streamlit_app import main as streamlit_main
from .dash_app import app as dash_app

__all__ = ["streamlit_main", "dash_app"]
