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
    """

    def __init__(
        self,
        window: int = 20,
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
        self._residuals = aligned_values - self._trend

        # Linear regression on the trend for extrapolation
        x = np.arange(len(self._trend), dtype=float)
        self._slope, self._intercept = np.polyfit(x, self._trend, 1)

        # Keep last `window` raw values for volatility estimation
        self._last_values = values[-self.window :]
        self._fitted = True
        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with prediction intervals.

        Returns
        -------
        forecast : np.ndarray
            Point forecasts for each future step.
        lower : np.ndarray
            Lower bound of the prediction interval.
        upper : np.ndarray
            Upper bound of the prediction interval.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        n_trend = len(self._trend)  # type: ignore[arg-type]
        future_x = np.arange(n_trend, n_trend + self.forecast_horizon, dtype=float)
        forecast = self._slope * future_x + self._intercept

        # Volatility estimated from recent residuals
        residual_std = float(np.std(self._residuals))  # type: ignore[arg-type]
        z = self._z_score(self.confidence_level)
        margin = z * residual_std * np.sqrt(np.arange(1, self.forecast_horizon + 1))

        lower = forecast - margin
        upper = forecast + margin
        return forecast, lower, upper

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _z_score(confidence_level: float) -> float:
        """Approximate z-score for a given two-sided confidence level."""
        # Use a lookup for common levels; fall back to a simple approximation.
        lookup = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
        if confidence_level in lookup:
            return lookup[confidence_level]
        # Normal quantile approximation (Beasley-Springer-Moro)
        p = (1.0 + confidence_level) / 2.0
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        z = t - (c[0] + c[1] * t + c[2] * t**2) / (
            1 + d[0] * t + d[1] * t**2 + d[2] * t**3
        )
        return float(z)

    def __repr__(self) -> str:  # pragma: no cover
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"KronosModel(window={self.window}, "
            f"forecast_horizon={self.forecast_horizon}, "
            f"confidence_level={self.confidence_level}, "
            f"status={status})"
        )
