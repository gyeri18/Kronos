"""Core Kronos time series prediction model.

This module implements the main forecasting model used across all examples,
providing a unified interface for stock market and general time series prediction.
"""

import numpy as np
from typing import Optional, Union, List, Tuple


class KronosModel:
    """Kronos time series forecasting model.

    A lightweight statistical model for short-to-medium term time series
    prediction, with support for trend decomposition, seasonal adjustment,
    and confidence interval estimation.

    Parameters
    ----------
    window : int
        Rolling window size for trend smoothing. Default is 20.
    forecast_horizon : int
        Number of future steps to predict. Default is 30.
    confidence_level : float
        Confidence level for prediction intervals (0 < confidence_level < 1).
        Default is 0.95.

    Notes
    -----
    I bumped the default window from 20 to 30 for my use case — the datasets
    I work with tend to be noisier and a larger window produces smoother,
    more reliable trends without sacrificing too much responsiveness.
    """

    def __init__(
        self,
        window: int = 30,
        forecast_horizon: int = 30,
        confidence_level: float = 0.95,
    ):
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1 (exclusive).")
        if window < 2:
            raise ValueError("window must be at least 2.")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be at least 1.")

        self.window = window
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level

        # Internal state populated during fit
        self._fitted = False
        self._trend: Optional[np.ndarray] = None
        self._residuals: Optional[np.ndarray] = None
        self._last_values: Optional[np.ndarray] = None
        self._slope: float = 0.0
        self._intercept: float = 0.0

    def fit(self, series: Union[List[float], np.ndarray]) -> "KronosModel":
        """Fit the model to a univariate time series.

        Parameters
        ----------
        series : array-like
            1-D array of observed values, ordered chronologically.

        Returns
        -------
        self : KronosModel
            The fitted model instance (enables method chaining).
        """
        values = np.asarray(series, dtype=float)
        if values.ndim != 1:
            raise ValueError("series must be 1-dimensional.")
        if len(values) < self.window:
            raise ValueError(
                f"series length ({len(values)}) must be >= window ({self.window})."
            )

        # Simple moving-average trend
        self._trend = np.convolve(
            values, np.ones(self.window) / self.window, mode="valid"
        )

        # Residuals (aligned to trend)
        aligned_values = values[self.window - 1 :]
        # Fixed: was referencing self._tr (typo), should be self._trend
        self._residuals = aligned_values - self._trend
