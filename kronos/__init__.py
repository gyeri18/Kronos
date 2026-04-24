"""
Kronos - Time Series Prediction Library
========================================

A fork of shiyu-coder/Kronos providing tools for stock market
time series forecasting with support for Chinese and global markets.

Usage:
    from kronos import KronosModel
    from kronos.data import prepare_data
    from kronos.utils import generate_future_dates
"""

__version__ = "0.2.0"
__author__ = "Kronos Contributors"
__license__ = "MIT"

from kronos.model import KronosModel
from kronos.data import prepare_data, load_csv_data
from kronos.utils import generate_future_dates, calculate_prediction_parameters

__all__ = [
    "KronosModel",
    "prepare_data",
    "load_csv_data",
    "generate_future_dates",
    "calculate_prediction_parameters",
]
