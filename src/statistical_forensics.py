#!/usr/bin/env python3
import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

# Set up module logger
logger = logging.getLogger(__name__)


class StatisticalForensics:
    """Class for statistical forensic analysis to detect data manipulation."""

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        """Initialize with a DataFrame.

        Args:
            df: DataFrame to analyze
        """
        self.df = df

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set the DataFrame to analyze.

        Args:
            df: DataFrame to analyze
        """
        self.df = df

    def analyze_terminal_digits(
        self, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze terminal digits of numeric values for non-random patterns.

        Args:
            columns: Optional list of numeric columns to analyze.
                     If None, analyze all numeric columns.

        Returns:
            Dict[str, Any]: Analysis results for each column
        """
        if self.df is None:
            return {"error": "No dataset loaded"}

        results = {}

        # Determine which columns to analyze
        numeric_cols = []
        if columns is None:
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    numeric_cols.append(col)
        else:
            numeric_cols = [
                col
                for col in columns
                if col in self.df.columns
                and pd.api.types.is_numeric_dtype(self.df[col])
            ]

        # For each numeric column, analyze terminal digits
        for col in numeric_cols:
            # Skip columns with too few values
            if self.df[col].count() < 20:
                continue

            # Extract terminal digits
            terminal_digits = self._extract_terminal_digits(self.df[col])

            if not terminal_digits:
                continue

            # Count frequencies of each terminal digit
            digit_counts = Counter(terminal_digits)

            # Expected frequency under uniform distribution
            n_digits = len(terminal_digits)
            expected_freq = n_digits / 10  # 10 possible digits (0-9)

            # Calculate chi-square test for uniformity
            observed = [digit_counts.get(d, 0) for d in range(10)]
            expected = [expected_freq] * 10

            try:
                chi2, p_value = stats.chisquare(observed, expected)

                # Calculate entropy as a measure of randomness
                # Higher entropy = more random = more uniform distribution
                entropy = self._calculate_entropy(observed, n_digits)

                # Determine suspicious level based on p-value and entropy
                if p_value < 0.001:
                    suspicion = "high"
                    suspicion_score = 9
                elif p_value < 0.01:
                    suspicion = "medium"
                    suspicion_score = 7
                elif p_value < 0.05:
                    suspicion = "low"
                    suspicion_score = 5
                else:
                    suspicion = "none"
                    suspicion_score = 3

                # Check for specific patterns
                most_common = digit_counts.most_common(1)[0][0]
                least_common = min(digit_counts.items(), key=lambda x: x[1])[0]

                # Detect if even digits are over-represented
                even_count = sum(digit_counts.get(d, 0) for d in [0, 2, 4, 6, 8])
                even_ratio = even_count / n_digits

                patterns = []

                if even_ratio > 0.65:
                    patterns.append("even_digits_overrepresented")

                if digit_counts.get(5, 0) / n_digits > 0.2:
                    patterns.append("digit_5_overrepresented")

                if digit_counts.get(0, 0) / n_digits > 0.2:
                    patterns.append("digit_0_overrepresented")

                # Store results
                results[col] = {
                    "terminal_digit_counts": {
                        str(d): c for d, c in digit_counts.items()
                    },
                    "chi_square": float(chi2),
                    "p_value": float(p_value),
                    "entropy": float(entropy),
                    "suspicion": suspicion,
                    "suspicion_score": suspicion_score,
                    "most_common_digit": int(most_common),
                    "least_common_digit": int(least_common),
                    "patterns_detected": patterns,
                    "sample_size": n_digits,
                }

            except Exception as e:
                results[col] = {
                    "error": f"Error analyzing terminal digits: {str(e)}",
                    "sample_size": len(terminal_digits),
                }

        return results

    def _extract_terminal_digits(self, series: pd.Series) -> List[int]:
        """Extract terminal digits from numeric values.

        Args:
            series: Pandas Series with numeric values

        Returns:
            List[int]: List of terminal digits
        """
        terminal_digits = []

        for value in series.dropna():
            # Skip non-numeric values
            if not isinstance(value, (int, float)) or math.isnan(value):
                continue

            # Convert to string and extract last digit
            str_value = str(abs(value))

            # Find the position of the decimal point
            decimal_pos = str_value.find(".")

            if decimal_pos == -1:  # Integer
                if len(str_value) > 0:
                    terminal_digits.append(int(str_value[-1]))
            else:  # Float
                # Get the digit right before the decimal point
                if decimal_pos > 0:
                    terminal_digits.append(int(str_value[decimal_pos - 1]))

        return terminal_digits

    def _calculate_entropy(self, counts: List[int], total: int) -> float:
        """Calculate Shannon entropy for a distribution.

        Higher entropy indicates a more uniform (random) distribution.

        Args:
            counts: List of counts for each category
            total: Total number of observations

        Returns:
            float: Entropy value
        """
        entropy = 0.0

        for count in counts:
            if count == 0:
                continue

            p = count / total
            entropy -= p * math.log2(p)

        # Normalize by maximum entropy (uniform distribution)
        max_entropy = math.log2(len(counts))

        if max_entropy > 0:
            return entropy / max_entropy
        else:
            return 0.0

    def detect_multimodality(
        self, columns: Optional[List[str]] = None, by_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect multimodality in distributions that might indicate manipulation.

        Args:
            columns: Optional list of numeric columns to analyze.
                    If None, analyze all numeric columns.
            by_group: Optional column to group by for separate analysis

        Returns:
            Dict[str, Any]: Analysis results for each column
        """
        if self.df is None:
            return {"error": "No dataset loaded"}

        results = {}

        # Determine which columns to analyze
        numeric_cols = []
        if columns is None:
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    numeric_cols.append(col)
        else:
            numeric_cols = [
                col
                for col in columns
                if col in self.df.columns
                and pd.api.types.is_numeric_dtype(self.df[col])
            ]

        # If grouping is requested, check if the column exists
        if by_group and by_group not in self.df.columns:
            return {"error": f"Grouping column '{by_group}' not found in dataset"}

        # For each numeric column, analyze distribution
        for col in numeric_cols:
            # Skip columns with too few values
            if self.df[col].count() < 20:
                continue

            # Analyze either grouped or overall
            if by_group:
                col_results = {}

                # For each group, analyze separately
                for group_name, group_df in self.df.groupby(by_group):
                    # Skip groups with too few values
                    if len(group_df) < 20:
                        continue

                    values = group_df[col].dropna()

                    # Skip if not enough values
                    if len(values) < 20:
                        continue

                    group_result = self._analyze_distribution(values)

                    if group_result:
                        col_results[str(group_name)] = group_result

                results[col] = col_results
            else:
                # Analyze overall distribution
                values = self.df[col].dropna()

                if len(values) >= 20:
                    results[col] = self._analyze_distribution(values)

        return results

    def _analyze_distribution(self, values: pd.Series) -> Dict[str, Any]:
        """Analyze distribution for multimodality and other anomalies.

        Args:
            values: Series of numeric values to analyze

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Basic statistics
            mean = float(values.mean())
            median = float(values.median())
            std_dev = float(values.std())
            min_val = float(values.min())
            max_val = float(values.max())
            n = len(values)

            # Calculate skewness and kurtosis
            skewness = float(stats.skew(values))
            kurtosis = float(stats.kurtosis(values))

            # Perform Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = (
                stats.shapiro(values) if n <= 5000 else (np.nan, np.nan)
            )

            # Check for bimodality using Hartigan's dip test
            # Approximation: kurtosis < 0 often indicates bimodality
            potential_bimodal = kurtosis < -0.5

            # Check for suspicious clustering using kernel density estimation
            kde = stats.gaussian_kde(values)
            x = np.linspace(min_val, max_val, 100)
            density = kde(x)

            # Find peaks in density
            peaks = []
            for i in range(1, len(density) - 1):
                if density[i] > density[i - 1] and density[i] > density[i + 1]:
                    peaks.append((x[i], density[i]))

            # Sort peaks by density (highest first)
            peaks.sort(key=lambda p: p[1], reverse=True)

            # Determine suspicion level
            suspicion_score = 0

            # Multiple strong modes is suspicious
            if len(peaks) >= 2 and peaks[1][1] > 0.7 * peaks[0][1]:
                suspicion_score += 3

            # Very low p-value on normality test is suspicious
            if shapiro_p is not None and shapiro_p < 0.001:
                suspicion_score += 2

            # Extreme kurtosis is suspicious
            if abs(kurtosis) > 2:
                suspicion_score += 2

            # Check for unusual gaps in the distribution
            sorted_values = sorted(values)
            gaps = [
                sorted_values[i + 1] - sorted_values[i]
                for i in range(len(sorted_values) - 1)
            ]
            mean_gap = np.mean(gaps)
            max_gap = np.max(gaps)

            # Check if the max gap is significantly larger than the mean
            if max_gap > 5 * mean_gap:
                suspicion_score += 2

            # Determine overall suspicion level
            if suspicion_score >= 5:
                suspicion = "high"
            elif suspicion_score >= 3:
                suspicion = "medium"
            elif suspicion_score >= 1:
                suspicion = "low"
            else:
                suspicion = "none"

            # Return results
            return {
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "min": min_val,
                "max": max_val,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "shapiro_p": float(shapiro_p) if shapiro_p is not None else None,
                "potential_multimodal": potential_bimodal,
                "peak_count": len(peaks),
                "peak_values": [float(p[0]) for p in peaks[:3]],  # Top 3 peaks
                "suspicious_gaps": max_gap > 5 * mean_gap,
                "max_gap_size": float(max_gap),
                "suspicion": suspicion,
                "suspicion_score": suspicion_score,
                "sample_size": n,
            }

        except Exception as e:
            logger.error(f"Error analyzing distribution: {str(e)}")
            return {
                "error": f"Error analyzing distribution: {str(e)}",
                "sample_size": len(values),
            }

    def check_inlier_anomalies(
        self, columns: Optional[List[str]] = None, by_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check for abnormal clustering of values near means (inlier anomalies).

        Too many values clustered unnaturally close to means can indicate fabrication.

        Args:
            columns: Optional list of numeric columns to analyze.
                    If None, analyze all numeric columns.
            by_group: Optional column to group by for separate analysis

        Returns:
            Dict[str, Any]: Analysis results for each column
        """
        if self.df is None:
            return {"error": "No dataset loaded"}

        results = {}

        # Determine which columns to analyze
        numeric_cols = []
        if columns is None:
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    numeric_cols.append(col)
        else:
            numeric_cols = [
                col
                for col in columns
                if col in self.df.columns
                and pd.api.types.is_numeric_dtype(self.df[col])
            ]

        # If grouping is requested, check if the column exists
        if by_group and by_group not in self.df.columns:
            return {"error": f"Grouping column '{by_group}' not found in dataset"}

        # For each numeric column, analyze distribution
        for col in numeric_cols:
            # Skip columns with too few values
            if self.df[col].count() < 20:
                continue

            # Analyze either grouped or overall
            if by_group:
                col_results = {}

                # For each group, analyze separately
                for group_name, group_df in self.df.groupby(by_group):
                    # Skip groups with too few values
                    if len(group_df) < 20:
                        continue

                    values = group_df[col].dropna()

                    # Skip if not enough values
                    if len(values) < 20:
                        continue

                    group_result = self._analyze_inliers(values)

                    if group_result:
                        col_results[str(group_name)] = group_result

                results[col] = col_results
            else:
                # Analyze overall distribution
                values = self.df[col].dropna()

                if len(values) >= 20:
                    results[col] = self._analyze_inliers(values)

        return results

    def _analyze_inliers(self, values: pd.Series) -> Dict[str, Any]:
        """Analyze distribution for unusual clustering near the mean.

        Args:
            values: Series of numeric values to analyze

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Calculate mean and standard deviation
            mean = values.mean()
            std_dev = values.std()
            n = len(values)

            if std_dev == 0:
                return {
                    "error": "Zero standard deviation",
                    "sample_size": n,
                }

            # Calculate z-scores
            z_scores = [
                (value - mean) / std_dev for value in values if not pd.isna(value)
            ]

            # Count values in different z-score ranges
            very_close_count = sum(1 for z in z_scores if abs(z) < 0.5)
            close_count = sum(1 for z in z_scores if 0.5 <= abs(z) < 1.0)
            moderate_count = sum(1 for z in z_scores if 1.0 <= abs(z) < 2.0)
            far_count = sum(1 for z in z_scores if abs(z) >= 2.0)

            # Calculate percentages
            very_close_pct = very_close_count / n
            close_pct = close_count / n
            moderate_pct = moderate_count / n
            far_pct = far_count / n

            # Expected percentages under normal distribution
            expected_very_close = 0.383  # ~38.3% within 0.5 std devs
            expected_close = 0.348  # ~34.8% between 0.5 and 1.0 std devs
            expected_moderate = 0.236  # ~23.6% between 1.0 and 2.0 std devs
            expected_far = 0.033  # ~3.3% beyond 2.0 std devs

            # Calculate ratios of observed to expected
            very_close_ratio = very_close_pct / expected_very_close
            close_ratio = close_pct / expected_close
            moderate_ratio = moderate_pct / expected_moderate
            far_ratio = far_pct / expected_far

            # Determine suspicion level
            inlier_suspicion = 0

            # Too many values very close to the mean
            if very_close_ratio > 1.5 and very_close_count >= 10:
                inlier_suspicion += 3
            elif very_close_ratio > 1.2 and very_close_count >= 10:
                inlier_suspicion += 1

            # Too few values far from the mean
            if far_ratio < 0.5 and n >= 50:
                inlier_suspicion += 2

            # Imbalance between close and moderate ranges
            if close_ratio > 1.3 and moderate_ratio < 0.7:
                inlier_suspicion += 2

            # Determine overall suspicion level
            if inlier_suspicion >= 5:
                suspicion = "high"
                suspicion_score = 9
            elif inlier_suspicion >= 3:
                suspicion = "medium"
                suspicion_score = 7
            elif inlier_suspicion >= 1:
                suspicion = "low"
                suspicion_score = 5
            else:
                suspicion = "none"
                suspicion_score = 3

            # Return results
            return {
                "mean": float(mean),
                "std_dev": float(std_dev),
                "very_close_count": very_close_count,
                "close_count": close_count,
                "moderate_count": moderate_count,
                "far_count": far_count,
                "very_close_pct": float(very_close_pct),
                "close_pct": float(close_pct),
                "moderate_pct": float(moderate_pct),
                "far_pct": float(far_pct),
                "very_close_ratio": float(very_close_ratio),
                "close_ratio": float(close_ratio),
                "moderate_ratio": float(moderate_ratio),
                "far_ratio": float(far_ratio),
                "suspicion": suspicion,
                "suspicion_score": suspicion_score,
                "sample_size": n,
            }

        except Exception as e:
            logger.error(f"Error analyzing inliers: {str(e)}")
            return {
                "error": f"Error analyzing inliers: {str(e)}",
                "sample_size": len(values),
            }

    def detect_linear_sequences(
        self,
        columns: Optional[List[str]] = None,
        by_group: Optional[str] = None,
        threshold: float = 0.99,
    ) -> Dict[str, Any]:
        """Detect suspiciously perfect linear sequences in data.

        Args:
            columns: Optional list of numeric columns to analyze.
                    If None, analyze all numeric columns.
            by_group: Optional column to group by for separate analysis
            threshold: R-squared threshold for suspicion (default: 0.99)

        Returns:
            Dict[str, Any]: Analysis results for each column
        """
        if self.df is None:
            return {"error": "No dataset loaded"}

        results = {}

        # Determine which columns to analyze
        numeric_cols = []
        if columns is None:
            for col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    numeric_cols.append(col)
        else:
            numeric_cols = [
                col
                for col in columns
                if col in self.df.columns
                and pd.api.types.is_numeric_dtype(self.df[col])
            ]

        # If grouping is requested, check if the column exists
        if by_group and by_group not in self.df.columns:
            return {"error": f"Grouping column '{by_group}' not found in dataset"}

        # For each numeric column, analyze distribution
        for col in numeric_cols:
            # Skip columns with too few values
            if self.df[col].count() < 10:
                continue

            # Analyze either grouped or overall
            if by_group:
                col_results = {}

                # For each group, analyze separately
                for group_name, group_df in self.df.groupby(by_group):
                    # Skip groups with too few values
                    if len(group_df) < 10:
                        continue

                    values = group_df[col].dropna()

                    # Skip if not enough values
                    if len(values) < 10:
                        continue

                    group_result = self._analyze_linearity(values, threshold)

                    if group_result:
                        col_results[str(group_name)] = group_result

                results[col] = col_results
            else:
                # Analyze overall distribution
                values = self.df[col].dropna()

                if len(values) >= 10:
                    results[col] = self._analyze_linearity(values, threshold)

        return results

    def _analyze_linearity(
        self, values: pd.Series, threshold: float = 0.99
    ) -> Dict[str, Any]:
        """Analyze a sequence for suspicious linearity patterns.

        Args:
            values: Series of numeric values to analyze
            threshold: R-squared threshold for suspicion

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Sort values to check for linearity in the sequence
            sorted_values = values.sort_values().reset_index(drop=True)
            n = len(sorted_values)

            # Create X as the sequence index
            X = np.arange(n).reshape(-1, 1)
            y = sorted_values.values

            # Fit linear regression
            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)

            # Calculate R-squared
            y_pred = model.predict(X)
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

            # Calculate mean absolute percentage error
            mape = np.mean(np.abs((y - y_pred) / y)) * 100 if np.all(y != 0) else np.nan

            # Determine suspicion level
            if r_squared >= threshold and n >= 15:
                suspicion = "high"
                suspicion_score = 9
            elif r_squared >= threshold * 0.98 and n >= 10:
                suspicion = "medium"
                suspicion_score = 7
            elif r_squared >= threshold * 0.95:
                suspicion = "low"
                suspicion_score = 5
            else:
                suspicion = "none"
                suspicion_score = 3

            # Return results
            return {
                "r_squared": float(r_squared),
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "mape": float(mape) if not np.isnan(mape) else None,
                "suspicion": suspicion,
                "suspicion_score": suspicion_score,
                "sample_size": n,
            }

        except Exception as e:
            logger.error(f"Error analyzing linearity: {str(e)}")
            return {
                "error": f"Error analyzing linearity: {str(e)}",
                "sample_size": len(values),
            }
