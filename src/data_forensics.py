#!/usr/bin/env python3
import json
import logging
import os
import re
import shutil
import tempfile
import traceback
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anthropic import Anthropic

# Set up module logger
logger = logging.getLogger(__name__)


class DataForensics:
    """Class for detecting potential data manipulation in research datasets."""

    def __init__(self) -> None:
        """Initialize the DataForensics object."""
        self.findings: List[Dict[str, Any]] = []
        self.plots: List[Dict[str, Any]] = []
        self.df: Optional[pd.DataFrame] = None
        self.excel_metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def analyze_dataset(self, filepath: str, id_col: Optional[str] = None, 
                      sort_cols: Optional[Union[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """Analyze a dataset for potential manipulation.
        
        Args:
            filepath: Path to the dataset file
            id_col: Column name for IDs
            sort_cols: Column name(s) for sorting/grouping
            
        Returns:
            List of findings as dictionaries
        """
        self.findings = []
        self.plots = []

        # Determine file type and read data
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".xlsx":
            self.df = pd.read_excel(filepath)
            self.excel_metadata = self.extract_excel_metadata(filepath)
        elif ext == ".csv":
            self.df = pd.read_csv(filepath)
        elif ext == ".dta":
            self.df = pd.read_stata(filepath)
        elif ext == ".sav":
            try:
                import pyreadstat

                # Try different encodings
                encodings = [
                    "latin1",
                    "cp1252",
                    None,
                ]  # None lets pyreadstat try to detect encoding
                read_success = False

                for encoding in encodings:
                    try:
                        self.df, meta = pyreadstat.read_sav(filepath, encoding=encoding)
                        read_success = True
                        break
                    except Exception as e:
                        last_error = str(e)
                        continue

                if not read_success:
                    raise ValueError(
                        f"Could not read SPSS file with any encoding: {last_error}"
                    )
            except ImportError:
                raise ValueError(
                    "The pyreadstat package is required to read SPSS files"
                )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Analyze sorting anomalies if sort columns are provided
        if sort_cols and id_col:
            sorting_issues = self.check_sorting_anomalies(id_col, sort_cols)
            if sorting_issues:
                self.findings.append(
                    {"type": "sorting_anomaly", "details": sorting_issues}
                )

        # Check for duplicate IDs
        if id_col:
            duplicates = self.check_duplicate_ids(id_col)
            if duplicates:
                self.findings.append({"type": "duplicate_ids", "details": duplicates})

        # Statistical anomalies
        self.analyze_statistical_anomalies()

        # Check for data fabrication patterns
        self.detect_fabrication_patterns()

        # Check for terminal digit anomalies
        self.analyze_terminal_digits()

        # Check for variance anomalies between groups
        if sort_cols:
            group_col = sort_cols[0] if isinstance(sort_cols, list) else sort_cols
            self.analyze_variance_anomalies(group_col)

        # Check for inlier patterns (too many values near means)
        self.detect_inlier_patterns()

        return self.findings

    def check_sorting_anomalies(self, id_col: str, sort_cols: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Check for anomalies in sorting order that might indicate manipulation.
        
        Args:
            id_col: Column name for IDs
            sort_cols: Column name(s) for sorting/grouping
            
        Returns:
            List of sorting anomalies found
        """
        anomalies = []

        # For each sorting level
        if isinstance(sort_cols, str):
            sort_cols = [sort_cols]

        # Get the unique values for each sort column
        for col in sort_cols:
            unique_vals = self.df[col].unique()

            for val in unique_vals:
                # Get rows for this value
                subset = self.df[self.df[col] == val]

                # Check if IDs are in sequence
                ids = subset[id_col].values

                # Find out-of-sequence IDs
                for i in range(1, len(ids)):
                    if ids[i] < ids[i - 1]:
                        # Found an out-of-sequence ID
                        row_idx = subset.iloc[i].name
                        anomalies.append(
                            {
                                "row_index": int(row_idx),
                                "id": int(ids[i])
                                if pd.api.types.is_integer_dtype(type(ids[i]))
                                else ids[i],
                                "previous_id": int(ids[i - 1])
                                if pd.api.types.is_integer_dtype(type(ids[i - 1]))
                                else ids[i - 1],
                                "sort_column": col,
                                "sort_value": val
                                if not pd.api.types.is_integer_dtype(type(val))
                                else int(val),
                            }
                        )

        return anomalies

    def check_duplicate_ids(self, id_col: str) -> List[Dict[str, Any]]:
        """Check for duplicate ID values that might indicate manipulation.
        
        Args:
            id_col: Column name for IDs
            
        Returns:
            List of duplicate ID findings
        """
        id_counts = self.df[id_col].value_counts()
        duplicates = id_counts[id_counts > 1].index.tolist()

        duplicate_details = []
        for dup_id in duplicates:
            rows = self.df[self.df[id_col] == dup_id]
            duplicate_details.append(
                {
                    "id": int(dup_id)
                    if pd.api.types.is_integer_dtype(type(dup_id))
                    else dup_id,
                    "count": len(rows),
                    "row_indices": [int(idx) for idx in rows.index.tolist()],
                }
            )

        return duplicate_details

    def extract_excel_metadata(self, excel_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract metadata from Excel file including calcChain information.
        
        Args:
            excel_file: Path to the Excel file
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Metadata extracted from the Excel file
        """
        metadata = {"calc_chain": []}

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Extract the Excel file (which is a zip file)
            with zipfile.ZipFile(excel_file, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Look for calcChain.xml
            calc_chain_path = os.path.join(tmpdirname, "xl", "calcChain.xml")
            if os.path.exists(calc_chain_path):
                # Parse the XML
                tree = ET.parse(calc_chain_path)
                root = tree.getroot()

                # Extract calculation order
                for c in root.findall(".//{*}c"):
                    cell_ref = c.get("r")
                    if cell_ref:
                        # Convert Excel cell reference to row and column
                        col_name = "".join(filter(str.isalpha, cell_ref))
                        row_num = int("".join(filter(str.isdigit, cell_ref)))

                        metadata["calc_chain"].append(
                            {"cell_ref": cell_ref, "column": col_name, "row": row_num}
                        )

        return metadata

    def analyze_calc_chain(self, suspect_rows: List[int]) -> List[Dict[str, Any]]:
        """Analyze the calcChain to detect row movement.
        
        Args:
            suspect_rows: List of row indices to check for movement evidence
            
        Returns:
            List[Dict[str, Any]]: Findings about row movements
        """
        if not hasattr(self, "excel_metadata") or not self.excel_metadata.get(
            "calc_chain"
        ):
            return []

        findings = []
        calc_chain = self.excel_metadata["calc_chain"]

        # Group by row number
        rows_in_chain = {}
        for item in calc_chain:
            row = item["row"]
            if row not in rows_in_chain:
                rows_in_chain[row] = []
            rows_in_chain[row].append(item)

        # Check sequence for suspect rows
        for row in suspect_rows:
            if row in rows_in_chain:
                # Find all cells in this row
                cells = rows_in_chain[row]

                # Find preceding and following cells in calc chain
                for i, cell in enumerate(
                    [
                        c
                        for c in calc_chain
                        if c["cell_ref"] in [x["cell_ref"] for x in cells]
                    ]
                ):
                    if i > 0 and i < len(calc_chain) - 1:
                        prev_cell = calc_chain[i - 1]
                        next_cell = calc_chain[i + 1]

                        # If previous and next cells are from different rows, it might indicate movement
                        if prev_cell["row"] != row and next_cell["row"] != row:
                            if abs(prev_cell["row"] - next_cell["row"]) == 1:
                                # This suggests the row was moved from between these consecutive rows
                                findings.append(
                                    {
                                        "row": row,
                                        "likely_original_position": f"between rows {prev_cell['row']} and {next_cell['row']}",
                                        "evidence": f"Cell {cell['cell_ref']} calculation order suggests movement",
                                    }
                                )

        return findings

    def analyze_statistical_anomalies(self) -> None:
        """Look for statistical anomalies in the data.
        
        Identifies outliers in numeric columns and adds findings to self.findings.
        """
        # If there are fewer than 5 columns, skip this analysis
        if len(self.df.columns) < 5:
            return

        # Focus on numeric columns
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        # For each numeric column, look for outliers and check their distribution
        for col in numeric_cols:
            # Skip ID columns or columns with few unique values
            if (
                self.df[col].nunique() < 5
                or self.df[col].nunique() > len(self.df) * 0.5
            ):
                continue

            # Look for outliers using Z-score
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outliers = [int(idx) for idx in self.df[z_scores > 3].index.tolist()]

            if outliers:
                self.findings.append(
                    {
                        "type": "statistical_anomaly",
                        "column": col,
                        "outlier_rows": outliers,
                        "details": f"Found {len(outliers)} outliers (z-score > 3)",
                    }
                )

                # Create visualization
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.df[col])
                plt.title(f"Distribution of {col} with Outliers")

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    self.plots.append({"column": col, "plot_path": tmp.name})

    def analyze_suspicious_observations(self, suspicious_rows: List[int], group_col: str, 
                                  outcome_vars: List[str]) -> Dict[str, Any]:
        """Analyze whether suspicious observations show a strong effect in the expected direction.
        
        Args:
            suspicious_rows: List of row indices considered suspicious
            group_col: Column name for grouping/conditions
            outcome_vars: List of outcome variable column names
            
        Returns:
            Dict[str, Any]: Analysis results for each outcome variable
        """
        results = {}

        # Get the groups
        groups = self.df[group_col].unique()
        if len(groups) < 2:
            return {"error": "Need at least two groups to analyze effects"}

        for var in outcome_vars:
            try:
                # Skip if this variable is not numeric
                if not pd.api.types.is_numeric_dtype(self.df[var]):
                    continue

                # Calculate effect size for suspicious observations only
                suspicious_df = self.df.loc[suspicious_rows]
                if len(suspicious_df) < 2:
                    continue

                # Check if we have at least one data point per group
                has_all_groups = all(
                    g in suspicious_df[group_col].unique() for g in groups
                )
                if not has_all_groups:
                    continue

                group_means = suspicious_df.groupby(group_col)[var].mean()
                effect_size_suspicious = group_means.max() - group_means.min()

                # Calculate effect size for the rest of the data
                non_suspicious_df = self.df.drop(suspicious_rows)
                if len(non_suspicious_df) < 2:
                    continue

                group_means_non = non_suspicious_df.groupby(group_col)[var].mean()
                effect_size_non_suspicious = (
                    group_means_non.max() - group_means_non.min()
                )

                # Calculate t-test for suspicious observations
                from scipy import stats

                group_values = [
                    suspicious_df[suspicious_df[group_col] == g][var].values
                    for g in groups
                ]

                # Only proceed with t-test if we have enough data in each group
                if all(len(values) > 1 for values in group_values):
                    # Handle variance issues safely
                    try:
                        t_stat, p_value = stats.ttest_ind(
                            group_values[0], group_values[1], equal_var=False
                        )
                    except (ValueError, ZeroDivisionError):
                        t_stat, p_value = float("nan"), float("nan")
                else:
                    t_stat, p_value = float("nan"), float("nan")

                # Calculate ratio safely
                if effect_size_non_suspicious != 0:
                    ratio = effect_size_suspicious / effect_size_non_suspicious
                else:
                    ratio = (
                        float("inf") if effect_size_suspicious != 0 else float("nan")
                    )

                results[var] = {
                    "effect_size_suspicious": float(effect_size_suspicious),
                    "effect_size_non_suspicious": float(effect_size_non_suspicious),
                    "ratio": float(ratio),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                }

                # Create visualization
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                sns.boxplot(x=group_col, y=var, data=suspicious_df)
                plt.title(f"{var} - Suspicious Observations")

                plt.subplot(1, 2, 2)
                sns.boxplot(x=group_col, y=var, data=non_suspicious_df)
                plt.title(f"{var} - Other Observations")

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    results[var]["plot_path"] = tmp.name

            except Exception as e:
                results[var] = {"error": str(e)}

        return results

    def detect_fabrication_patterns(self) -> None:
        """
        Detect possible data fabrication by looking for unusual patterns in the data.

        This includes:
        - Excessive digit repetition
        - Unusual uniformity in significant digits
        - Lack of expected natural distributions
        - Strong linear progressions
        - Idiosyncratic response clusters (e.g., identical wrong answers)
        
        Adds findings to self.findings if fabrication patterns are detected.
        """
        # Analyze both numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Skip columns with few unique values or ID columns
        analyzed_numeric_cols = []
        for col in numeric_cols:
            # Skip columns with too few unique values or too many (like IDs)
            if (
                self.df[col].nunique() < 5
                or self.df[col].nunique() > len(self.df) * 0.5
            ):
                continue

            analyzed_numeric_cols.append(col)

        # For categorical columns, find those with reasonable cardinality
        analyzed_categorical_cols = []
        for col in categorical_cols:
            # Include columns with more than 2 but fewer than 50% unique values
            if (
                self.df[col].nunique() >= 2
                and self.df[col].nunique() < len(self.df) * 0.5
            ):
                analyzed_categorical_cols.append(col)

        if not analyzed_numeric_cols and not analyzed_categorical_cols:
            return  # No suitable columns for analysis

        fabrication_findings = []

        # Check for excessive digit repetition
        for col in analyzed_numeric_cols:
            try:
                # Convert to string and analyze digit patterns
                digits = self.df[col].astype(str).str.replace(".", "")

                # Count repeating digits (like 333, 555)
                repeat_counts = []
                total_valid = 0

                for val in digits:
                    if not isinstance(val, str) or not val.isdigit():
                        continue

                    total_valid += 1

                    # Count repeating sequences
                    for repeat_len in range(3, min(6, len(val))):
                        for i in range(len(val) - repeat_len + 1):
                            if len(set(val[i : i + repeat_len])) == 1:  # All same digit
                                repeat_counts.append(val[i : i + repeat_len])
                                break

                # If more than 5% of values have repeating digits, flag it
                if total_valid > 0 and len(repeat_counts) / total_valid > 0.05:
                    fabrication_findings.append(
                        {
                            "column": col,
                            "issue": "excessive_repetition",
                            "details": f"Found {len(repeat_counts)} values with unusual digit repetition patterns",
                            "examples": repeat_counts[:5],
                        }
                    )

                # Check for unusual distributions of last digits
                # Extract last digit using string operations to avoid pattern issue
                last_digits = []
                for val in self.df[col].astype(str):
                    if val and re.search(r"\d$", val):
                        last_digits.append(val[-1])

                # Convert to Series
                last_digit_series = pd.Series(last_digits)
                last_digit_counts = last_digit_series.value_counts().to_dict()

                # For natural data, last digits should be roughly uniform
                if last_digit_counts:
                    digit_counts = np.array(list(last_digit_counts.values()))
                    total_digits = sum(digit_counts)

                    if total_digits > 50:  # Only analyze if we have enough data points
                        expected = (
                            total_digits / 10
                        )  # Expected uniform count for each digit
                        chi2_stat = np.sum((digit_counts - expected) ** 2 / expected)

                        # Chi-square critical value for 9 df at p=0.001 is about 27.88
                        if chi2_stat > 27.88:
                            most_common = last_digit_series.value_counts().index[0]
                            fabrication_findings.append(
                                {
                                    "column": col,
                                    "issue": "unusual_last_digits",
                                    "details": f"Last digits are not uniformly distributed (chi²={chi2_stat:.2f})",
                                    "most_common_digit": most_common,
                                }
                            )
            except Exception as e:
                logger.error(f"Error analyzing digit patterns in {col}: {e}")
                continue

        # Check for unusual linear progressions
        for col in analyzed_numeric_cols:
            try:
                # Sort values and look for unusually perfect linear sequences
                sorted_vals = self.df[col].sort_values().values
                diffs = np.diff(sorted_vals)

                # Check if differences are unusually consistent
                if len(diffs) > 10:  # Only analyze if enough data points
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs)

                    # If standard deviation is very small relative to mean, it suggests perfect progressions
                    if std_diff > 0 and mean_diff > 0 and std_diff / mean_diff < 0.05:
                        # This may indicate artificially created sequences
                        fabrication_findings.append(
                            {
                                "column": col,
                                "issue": "artificial_sequence",
                                "details": "Values show unusually perfect linear progression",
                                "sequence_diff_mean": float(mean_diff),
                                "sequence_diff_std": float(std_diff),
                            }
                        )
            except Exception as e:
                logger.error(f"Error analyzing sequences in {col}: {e}")
                continue

        # Check for idiosyncratic response clusters in categorical columns
        for col in analyzed_categorical_cols:
            try:
                # Get value counts
                value_counts = self.df[col].value_counts()

                # Look for unusual clusters of identical unusual responses
                unusual_values = []

                # Check for values that are abnormally clustered
                if (
                    len(value_counts) >= 5
                ):  # Only meaningful if there are enough distinct values
                    # Find values with suspiciously high frequency
                    q75 = value_counts.quantile(0.75)
                    iqr = value_counts.quantile(0.75) - value_counts.quantile(0.25)
                    threshold = q75 + 1.5 * iqr

                    # Get outlier frequencies (unusually common responses)
                    outliers = value_counts[value_counts > threshold]

                    if not outliers.empty:
                        # For each unusually common response, check if they're clustered in rows
                        for value, count in outliers.items():
                            # Skip if value is empty/null
                            if pd.isna(value) or value == "":
                                continue

                            # Find rows with this value
                            row_indices = self.df[self.df[col] == value].index.tolist()

                            # Check if these rows appear in sequence or clusters
                            if (
                                len(row_indices) >= 5
                            ):  # Only check if enough occurrences
                                # Calculate gaps between consecutive indices
                                gaps = np.diff(sorted(row_indices))

                                # If many are adjacent (gap=1) or uniformly spaced, that's suspicious
                                adjacent_count = sum(gaps == 1)
                                if adjacent_count / len(gaps) > 0.7:
                                    unusual_values.append(
                                        {
                                            "value": str(value),
                                            "count": int(count),
                                            "issue": "sequential_cluster",
                                            "details": f"Found {adjacent_count} adjacent occurrences out of {len(gaps)} gaps",
                                        }
                                    )

                                # Check for uniform spacing
                                if len(gaps) > 5:
                                    gap_mean = np.mean(gaps)
                                    gap_std = np.std(gaps)
                                    if (
                                        gap_std / gap_mean < 0.3
                                    ):  # Suspiciously consistent spacing
                                        unusual_values.append(
                                            {
                                                "value": str(value),
                                                "count": int(count),
                                                "issue": "uniform_spacing",
                                                "details": f"Occurrences are uniformly spaced (mean={gap_mean:.1f}, std={gap_std:.1f})",
                                            }
                                        )

                # If we found unusual value clusters, add to findings
                if unusual_values:
                    fabrication_findings.append(
                        {
                            "column": col,
                            "issue": "idiosyncratic_clusters",
                            "details": f"Found unusual clusters of identical responses",
                            "unusual_values": unusual_values,
                        }
                    )
            except Exception as e:
                logger.error(f"Error analyzing categorical patterns in {col}: {e}")
                continue

        if fabrication_findings:
            self.findings.append(
                {"type": "data_fabrication", "details": fabrication_findings}
            )

            # Create visualization
            self.create_digit_distribution_plot(analyzed_numeric_cols)

    def create_digit_distribution_plot(self, columns: List[str]) -> None:
        """Create plot showing distribution of last digits for each column.
        
        Args:
            columns: List of column names to analyze for digit distribution
        """
        if not columns:
            return

        # Take at most 4 columns to avoid overcrowded plots
        cols_to_plot = columns[: min(4, len(columns))]

        fig, axes = plt.subplots(
            len(cols_to_plot), 1, figsize=(10, 4 * len(cols_to_plot))
        )

        # If only one column, axes is not an array
        if len(cols_to_plot) == 1:
            axes = [axes]

        for i, col in enumerate(cols_to_plot):
            try:
                # Extract last digits using string operations
                last_digits = []
                for val in self.df[col].astype(str):
                    if val and re.search(r"\d$", val):
                        last_digits.append(val[-1])

                # Convert to Series and count frequencies
                last_digit_series = pd.Series(last_digits)

                # Create digit counts for all digits 0-9
                all_digits = pd.Series(range(10)).astype(str)
                digit_counts = pd.Series(0, index=all_digits)

                # Update with actual counts
                actual_counts = last_digit_series.value_counts()
                for digit, count in actual_counts.items():
                    if digit in digit_counts.index:
                        digit_counts[digit] = count

                # Sort by digit
                digit_counts = digit_counts.sort_index()

                # Plot
                sns.barplot(x=digit_counts.index, y=digit_counts.values, ax=axes[i])
                axes[i].set_title(f"Last Digit Distribution for {col}")
                axes[i].set_xlabel("Last Digit")
                axes[i].set_ylabel("Frequency")

                # Add reference line for expected uniform distribution
                expected = len(last_digits) / 10 if last_digits else 0
                if expected > 0:
                    axes[i].axhline(
                        y=expected,
                        color="r",
                        linestyle="--",
                        alpha=0.7,
                        label=f"Expected ({expected:.1f})",
                    )
                    axes[i].legend()

            except Exception as e:
                logger.error(f"Error creating plot for {col}: {e}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error creating plot: {str(e)}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )

        plt.tight_layout()

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.savefig(tmp.name)
            plt.close()
            self.plots.append({"column": "last_digits", "plot_path": tmp.name})

    def analyze_terminal_digits(self) -> None:
        """
        Analyze terminal (last) digits of numeric values for anomalies.

        In naturally occurring numerical data, terminal digits should follow a uniform distribution.
        Data manipulation or fabrication often results in non-uniform distribution of terminal digits.
        
        Adds findings to self.findings if anomalies are detected.
        """
        # Focus on numeric columns
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_cols:
            return  # No numeric columns to analyze

        # Skip columns with integer values or few unique values
        analyzed_cols = []
        for col in numeric_cols:
            # Skip columns with integer values or too few unique values
            if self.df[col].nunique() < 10 or np.all(
                self.df[col] == self.df[col].round()
            ):
                continue

            analyzed_cols.append(col)

        if not analyzed_cols:
            return  # No suitable columns for analysis

        anomalies = []

        for col in analyzed_cols:
            try:
                # Get decimal part of values using string operations
                decimal_parts = []
                for val in self.df[col].astype(str):
                    parts = val.split(".")
                    if len(parts) > 1 and parts[1]:
                        decimal_parts.append(parts[1][0])  # First decimal digit

                if len(decimal_parts) < 30:  # Need sufficient sample size
                    continue

                # Convert to Series
                first_decimal = pd.Series(decimal_parts)

                if len(first_decimal) < 30:  # Too few valid values
                    continue

                # Count frequencies
                digit_counts = pd.Series(0, index=[str(i) for i in range(10)])
                for digit, count in first_decimal.value_counts().items():
                    if digit in digit_counts.index:
                        digit_counts[digit] = count

                # Chi-square test against uniform distribution
                observed = digit_counts.values
                expected = np.ones(10) * len(first_decimal) / 10

                # Skip if expected count is too small
                if np.min(expected) < 1:
                    continue

                from scipy.stats import chisquare

                chi2, p_value = chisquare(observed, expected)

                # Highly significant deviation from uniform distribution
                if p_value < 0.001 and len(first_decimal) >= 100:
                    # Find the most over-represented digits
                    ratios = observed / expected
                    overrepresented = digit_counts.index[ratios > 1.5].tolist()
                    underrepresented = digit_counts.index[ratios < 0.5].tolist()

                    anomalies.append(
                        {
                            "column": col,
                            "chi_square": float(chi2),
                            "p_value": float(p_value),
                            "overrepresented_digits": overrepresented,
                            "underrepresented_digits": underrepresented,
                            "sample_size": len(first_decimal),
                        }
                    )
            except Exception as e:
                logger.error(f"Error analyzing terminal digits for {col}: {e}")
                continue

        if anomalies:
            self.findings.append(
                {"type": "terminal_digit_anomaly", "details": anomalies}
            )

            # Create visualization
            self.create_terminal_digit_plot(anomalies)

    def create_terminal_digit_plot(self, anomalies: List[Dict[str, Any]]) -> None:
        """Create visualization of anomalous terminal digit distributions.
        
        Args:
            anomalies: List of anomaly dictionaries containing terminal digit findings
        """
        if not anomalies:
            return

        # Take at most 3 columns with the strongest anomalies (lowest p-values)
        anomalies_sorted = sorted(anomalies, key=lambda x: x["p_value"])[
            : min(3, len(anomalies))
        ]

        fig, axes = plt.subplots(
            len(anomalies_sorted), 1, figsize=(10, 5 * len(anomalies_sorted))
        )

        # Ensure axes is a list even for a single plot
        if len(anomalies_sorted) == 1:
            axes = [axes]

        for i, anomaly in enumerate(anomalies_sorted):
            try:
                col = anomaly["column"]

                # Extract first decimal digit again
                decimal_parts = []
                for val in self.df[col].astype(str):
                    parts = val.split(".")
                    if len(parts) > 1 and parts[1]:
                        decimal_parts.append(parts[1][0])  # First decimal digit

                # Convert to Series and get counts for each digit 0-9
                first_decimal = pd.Series(decimal_parts)
                digit_counts = pd.Series(0, index=[str(i) for i in range(10)])

                for digit, count in first_decimal.value_counts().items():
                    if digit in digit_counts.index:
                        digit_counts[digit] = count

                # Create bar plot
                bars = axes[i].bar(digit_counts.index, digit_counts.values)

                # Add expected line
                expected = len(first_decimal) / 10
                axes[i].axhline(
                    y=expected,
                    color="r",
                    linestyle="--",
                    label=f"Expected frequency ({expected:.1f})",
                )

                # Color bars based on over/under-representation
                for j, bar in enumerate(bars):
                    digit = str(j)
                    if digit in anomaly["overrepresented_digits"]:
                        bar.set_color("red")
                    elif digit in anomaly["underrepresented_digits"]:
                        bar.set_color("blue")

                axes[i].set_title(
                    f"Terminal Digit Distribution for {col}\nχ²={anomaly['chi_square']:.2f}, p={anomaly['p_value']:.6f}"
                )
                axes[i].set_xlabel("Terminal Digit")
                axes[i].set_ylabel("Frequency")
                axes[i].set_xticks(range(10))
                axes[i].legend()

            except Exception as e:
                logger.error(f"Error creating terminal digit plot: {e}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error creating plot: {str(e)}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )

        plt.tight_layout()

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.savefig(tmp.name)
            plt.close()
            self.plots.append({"column": "terminal_digits", "plot_path": tmp.name})

    def analyze_variance_anomalies(self, group_col: str) -> None:
        """
        Analyze whether variance is unusually low in some groups compared to others.

        This can indicate data manipulation where some groups have had their
        values adjusted to produce a desired effect.
        
        Args:
            group_col: Column name for grouping/condition variable
            
        Adds findings to self.findings if anomalies are detected.
        """
        # Check if group column exists
        if group_col not in self.df.columns:
            return

        # Get numeric columns for analysis
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        # Skip columns with too few unique values
        analyzed_cols = []
        for col in numeric_cols:
            if (
                col == group_col
                or self.df[col].nunique() < 5
                or self.df[col].nunique() > len(self.df) * 0.5
            ):
                continue

            analyzed_cols.append(col)

        if not analyzed_cols:
            return  # No suitable columns to analyze

        # Get unique groups
        groups = self.df[group_col].unique()
        if len(groups) < 2:
            return  # Need at least two groups

        anomalies = []

        for col in analyzed_cols:
            try:
                # Calculate variance for each group
                group_vars = {}
                for group in groups:
                    group_data = self.df[self.df[group_col] == group][col]
                    if len(group_data) < 5:  # Skip groups with too few observations
                        continue
                    group_vars[group] = float(group_data.var())

                if len(group_vars) < 2:
                    continue

                # Calculate variance ratio (max/min)
                max_var = max(group_vars.values())
                min_var = (
                    min(group_vars.values())
                    if min(group_vars.values()) > 0
                    else float("inf")
                )
                var_ratio = max_var / min_var if min_var > 0 else float("inf")

                # If one group has much lower variance than others, it's suspicious
                # Using var ratio > 5 as a heuristic threshold
                if var_ratio > 5:
                    min_var_group = min(group_vars.items(), key=lambda x: x[1])[0]
                    max_var_group = max(group_vars.items(), key=lambda x: x[1])[0]

                    anomalies.append(
                        {
                            "column": col,
                            "variance_ratio": float(var_ratio),
                            "low_variance_group": str(min_var_group),
                            "low_variance_value": float(group_vars[min_var_group]),
                            "high_variance_group": str(max_var_group),
                            "high_variance_value": float(group_vars[max_var_group]),
                        }
                    )
            except Exception as e:
                logger.error(f"Error analyzing variance in {col}: {e}")
                continue

        if anomalies:
            self.findings.append({"type": "variance_anomaly", "details": anomalies})

            # Create visualization
            self.create_variance_comparison_plot(anomalies, group_col)

    def create_variance_comparison_plot(self, anomalies: List[Dict[str, Any]], group_col: str) -> None:
        """Create plots comparing distributions across groups with variance anomalies.
        
        Args:
            anomalies: List of anomaly dictionaries containing variance findings
            group_col: Column name for grouping variable
        """
        if not anomalies:
            return

        # Take at most 2 columns with the largest variance ratios
        anomalies_sorted = sorted(
            anomalies, key=lambda x: x["variance_ratio"], reverse=True
        )[: min(2, len(anomalies))]

        for anomaly in anomalies_sorted:
            try:
                col = anomaly["column"]
                low_var_group = anomaly["low_variance_group"]
                high_var_group = anomaly["high_variance_group"]

                plt.figure(figsize=(12, 6))

                # Create boxplots for each group
                ax = sns.boxplot(x=group_col, y=col, data=self.df)

                # Highlight the suspicious groups
                for i, group in enumerate(ax.get_xticklabels()):
                    if group.get_text() == str(low_var_group):
                        ax.get_xticklabels()[i].set_color("red")
                    elif group.get_text() == str(high_var_group):
                        ax.get_xticklabels()[i].set_color("blue")

                plt.title(
                    f"Variance Comparison for {col} across {group_col}\n"
                    + f"Variance ratio: {anomaly['variance_ratio']:.2f}"
                )

                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    plt.savefig(tmp.name)
                    plt.close()
                    self.plots.append(
                        {"column": f"variance_{col}", "plot_path": tmp.name}
                    )

            except Exception as e:
                logger.error(f"Error creating variance comparison plot for {col}: {e}")

    def detect_inlier_patterns(self) -> None:
        """
        Detect whether there are too many values clustered near the means.

        In fabricated data, values are often created to be close to desired
        means, resulting in an unusual lack of outliers and too many "inliers".
        
        Adds findings to self.findings if anomalies are detected.
        """
        # Get numeric columns for analysis
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        # Skip columns with too few unique values
        analyzed_cols = []
        for col in numeric_cols:
            if (
                self.df[col].nunique() < 10
                or self.df[col].nunique() > len(self.df) * 0.5
            ):
                continue

            analyzed_cols.append(col)

        if not analyzed_cols:
            return  # No suitable columns to analyze

        anomalies = []

        for col in analyzed_cols:
            try:
                # Get data with no missing values
                data = self.df[col].dropna()

                if len(data) < 30:  # Need sufficient sample size
                    continue

                # Calculate mean and standard deviation
                mean = data.mean()
                std = data.std()

                if std == 0:  # Skip if standard deviation is zero
                    continue

                # Calculate z-scores
                z_scores = np.abs((data - mean) / std)

                # Count values within 0.5 standard deviations of the mean
                inlier_count = sum(z_scores < 0.5)
                inlier_percent = inlier_count / len(data)

                # For normal distribution, about 38% of values should be within 0.5 std
                # If over 50% are within this range, it's suspiciously clustered
                if inlier_percent > 0.5:
                    anomalies.append(
                        {
                            "column": col,
                            "inlier_percentage": float(inlier_percent),
                            "expected_percentage": 0.383,  # Theoretical value for normal distribution
                            "z_threshold": 0.5,
                        }
                    )
            except Exception as e:
                logger.error(f"Error analyzing inlier patterns in {col}: {e}")
                continue

        if anomalies:
            self.findings.append({"type": "inlier_anomaly", "details": anomalies})

            # Create visualization
            self.create_inlier_distribution_plot(anomalies)

    def create_inlier_distribution_plot(self, anomalies: List[Dict[str, Any]]) -> None:
        """Create distribution plots for columns with inlier anomalies.
        
        Args:
            anomalies: List of anomaly dictionaries containing inlier pattern findings
        """
        if not anomalies:
            return

        # Take at most 2 columns with the highest inlier percentages
        anomalies_sorted = sorted(
            anomalies, key=lambda x: x["inlier_percentage"], reverse=True
        )[: min(2, len(anomalies))]

        fig, axes = plt.subplots(
            len(anomalies_sorted), 1, figsize=(10, 5 * len(anomalies_sorted))
        )

        # Ensure axes is a list even for a single plot
        if len(anomalies_sorted) == 1:
            axes = [axes]

        for i, anomaly in enumerate(anomalies_sorted):
            try:
                col = anomaly["column"]
                inlier_pct = anomaly["inlier_percentage"]
                expected_pct = anomaly["expected_percentage"]

                # Get column data
                data = self.df[col].dropna()

                # Calculate mean and std
                mean = data.mean()
                std = data.std()

                # Plot histogram with normal distribution overlay
                sns.histplot(data, kde=True, ax=axes[i])

                # Add vertical lines for mean ± 0.5 SD
                axes[i].axvline(
                    mean - 0.5 * std,
                    color="r",
                    linestyle="--",
                    label=f"Mean ± 0.5 SD ({inlier_pct:.1%} within vs {expected_pct:.1%} expected)",
                )
                axes[i].axvline(mean + 0.5 * std, color="r", linestyle="--")

                # Shade the "inlier" region
                axes[i].axvspan(
                    mean - 0.5 * std, mean + 0.5 * std, alpha=0.2, color="red"
                )

                axes[i].set_title(
                    f"Distribution of {col}\nUnusually many values near mean (possible fabrication)"
                )
                axes[i].legend()

            except Exception as e:
                logger.error(f"Error creating inlier distribution plot for {col}: {e}")
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error creating plot: {str(e)}",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )

        plt.tight_layout()

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            plt.savefig(tmp.name)
            plt.close()
            self.plots.append({"column": "inlier_distribution", "plot_path": tmp.name})

    def segment_and_analyze_with_claude(self, client: Anthropic, max_rows_per_chunk: int = 300) -> List[Dict[str, Any]]:
        """
        Segment dataset into manageable chunks and use Claude to detect anomalies in each chunk.

        Args:
            client: Claude API client
            max_rows_per_chunk: Maximum number of rows per chunk

        Returns:
            List[Dict[str, Any]]: A list of findings from Claude's analysis, 
            each finding is a dictionary that may contain anomaly results or error information
        """
        if not hasattr(self, "df"):
            return [{"error": "DataFrame not initialized in DataForensics object", "hint": "Make sure to set forensics.df before calling this method"}]
            
        if self.df is None:
            return [{"error": "DataFrame is None in DataForensics object", "hint": "Make sure forensics.df is assigned a valid DataFrame"}]
            
        if len(self.df) == 0:
            return [{"error": "DataFrame is empty (0 rows)", "hint": "The dataset must contain at least one row of data"}]

        claude_findings: List[Dict[str, Any]] = []

        # Determine number of chunks needed
        total_rows: int = len(self.df)
        chunk_count: int = (
            total_rows + max_rows_per_chunk - 1
        ) // max_rows_per_chunk  # Ceiling division

        logger.info(
            f"Segmenting dataset with {total_rows} rows into {chunk_count} chunks"
        )

        # Process each chunk
        for i in range(chunk_count):
            start_idx = i * max_rows_per_chunk
            end_idx = min((i + 1) * max_rows_per_chunk, total_rows)

            # Extract chunk
            chunk_df = self.df.iloc[start_idx:end_idx].reset_index(drop=True)

            # Skip empty chunks
            if len(chunk_df) == 0:
                continue

            logger.info(
                f"Analyzing chunk {i + 1}/{chunk_count} with rows {start_idx} to {end_idx - 1}"
            )

            # Include Excel metadata for the chunk's rows if available
            excel_metadata = {}
            if (
                hasattr(self, "excel_metadata")
                and self.excel_metadata
                and "calc_chain" in self.excel_metadata
            ):
                # Filter calc chain entries for rows in this chunk
                chunk_calc_chain = [
                    entry
                    for entry in self.excel_metadata["calc_chain"]
                    if entry["row"] >= start_idx and entry["row"] < end_idx
                ]
                if chunk_calc_chain:
                    excel_metadata["calc_chain"] = chunk_calc_chain
                    logger.info(
                        f"Including {len(chunk_calc_chain)} calcChain entries for chunk {i + 1}"
                    )

            # Prepare data summary for Claude
            numeric_cols = chunk_df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = chunk_df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Basic statistics for numeric columns
            numeric_stats = {}
            for col in numeric_cols:
                if (
                    len(chunk_df[col].dropna()) > 0
                ):  # Only include if we have non-NA values
                    numeric_stats[col] = {
                        "mean": float(chunk_df[col].mean()),
                        "median": float(chunk_df[col].median()),
                        "std": float(chunk_df[col].std()),
                        "min": float(chunk_df[col].min()),
                        "max": float(chunk_df[col].max()),
                        "unique_count": int(chunk_df[col].nunique()),
                    }

            # Basic statistics for categorical columns
            categorical_stats = {}
            for col in categorical_cols:
                if (
                    len(chunk_df[col].dropna()) > 0
                ):  # Only include if we have non-NA values
                    value_counts = chunk_df[col].value_counts()
                    top_values = value_counts.head(5).to_dict()  # Get top 5 values
                    categorical_stats[col] = {
                        "unique_count": int(chunk_df[col].nunique()),
                        "top_values": {str(k): int(v) for k, v in top_values.items()},
                    }

            # Validate and clean the data before sending to Claude
            # Ensure we're not sending NaN or other problematic values that might cause JSON issues
            clean_chunk_df = chunk_df.copy()
            for col in clean_chunk_df.columns:
                if pd.api.types.is_numeric_dtype(clean_chunk_df[col]):
                    # Replace NaN, Inf, -Inf with null for JSON compatibility
                    clean_chunk_df[col] = clean_chunk_df[col].replace([np.inf, -np.inf, np.nan], None)
            
            # Ensure both numeric and categorical stats don't have NaN or infinity values
            for col, stats in numeric_stats.items():
                for key, value in stats.items():
                    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        numeric_stats[col][key] = None
            
            # Convert the dataframe to string with error handling
            try:
                df_string = clean_chunk_df.to_string()
                # Truncate if too large (to avoid token limits)
                if len(df_string) > 200000:  # ~200KB should be safe
                    logger.warning(f"Chunk {i+1} data too large, truncating for Claude analysis")
                    df_string = df_string[:200000] + "\n[... truncated ...]"
            except Exception as df_err:
                logger.error(f"Error converting dataframe to string: {df_err}")
                df_string = f"[Error rendering dataframe: {df_err}]"
            
            # Create prompt for Claude
            prompt = f"""Analyze this dataset chunk for potential data manipulation or anomalies.

Dataset Information:
- Rows: {len(chunk_df)} (from row {start_idx} to {end_idx - 1} in the original dataset)
- Columns: {len(chunk_df.columns)}

Numeric Columns Statistics:
{json.dumps(numeric_stats, indent=2)}

Categorical Columns Statistics:
{json.dumps(categorical_stats, indent=2)}

All Data in this Chunk:
{df_string}

{
                f'''
Excel Metadata (calcChain entries for these rows):
{json.dumps(excel_metadata, indent=2)}

Note: The calcChain contains formula calculation order information from Excel. 
Patterns where a cell is calculated out of sequence (between cells from different rows) 
may indicate row movement or manipulation in Excel.
'''
                if excel_metadata
                else ""
            }

Look for the following potential anomalies:
1. Statistical anomalies (unusual patterns, outliers, distributions)
2. Suspicious repeated values or patterns
3. Unnaturally perfect distributions
4. Evidence of data fabrication
5. Terminal digit anomalies
6. Uniform spacing patterns
7. Implausible correlations
8. Unusual clustering of values
9. Values that should not logically be present in a column
10. Any other potential signs of manipulation

Return your findings in the following JSON format (ONLY respond with valid JSON):
{{
  "anomalies_detected": true/false,
  "confidence": 1-10,
  "findings": [
    {{
      "type": "anomaly_type",
      "description": "Detailed description of anomaly",
      "columns_involved": ["col1", "col2"],
      "row_indices": [list of row indices in the chunk, if applicable],
      "severity": 1-10
    }}
  ],
  "explanation": "Overall assessment of the data chunk"
}}
"""
            try:
                # Call Claude to analyze the chunk
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract JSON from response
                content: str = response.content[0].text
                json_start: int = content.find("{")
                json_end: int = content.rfind("}") + 1
                
                # Check if we actually found valid JSON markers
                if json_start == -1 or json_end <= json_start:
                    logger.error(f"Could not find valid JSON markers in Claude's response for chunk {i+1}")
                    logger.error(f"Claude response: {content[:200]}...")
                    
                    # Create an error finding
                    claude_findings.append({
                        "error": "Invalid JSON format in Claude response",
                        "chunk": i + 1,
                        "rows": f"{start_idx}-{end_idx - 1}",
                        "raw_response": content[:500] if len(content) > 500 else content
                    })
                    continue  # Skip to next chunk
                
                json_str: str = content[json_start:json_end]

                # Handle JSON parsing with error checking
                try:
                    chunk_findings: Any = json.loads(json_str)

                    # Verify that chunk_findings is a dictionary before using .get()
                    if not isinstance(chunk_findings, dict):
                        logger.warning(
                            f"Claude returned non-dictionary result for chunk {i + 1}: {chunk_findings}"
                        )
                        chunk_findings: Dict[str, Any] = {
                            "error": "Invalid response format",
                            "raw_response": str(chunk_findings),
                        }

                    # Update row indices to reference original dataset
                    if (
                        isinstance(chunk_findings, dict)
                        and "findings" in chunk_findings
                    ):
                        for finding in chunk_findings["findings"]:
                            if "row_indices" in finding:
                                finding["row_indices"] = [
                                    start_idx + idx for idx in finding["row_indices"]
                                ]
                                finding["chunk"] = (
                                    i + 1
                                )  # Store chunk number for reference

                    # Safety check before adding to findings - we should already have a dict here, but double-check
                    if isinstance(chunk_findings, dict):
                        claude_findings.append(chunk_findings)
                    else:
                        logger.warning(
                            f"After validation, chunk_findings is still not a dictionary: {type(chunk_findings)}"
                        )
                        error_dict = {
                            "error": f"Invalid finding type after validation: {type(chunk_findings)}",
                            "chunk": i + 1,
                            "rows": f"{start_idx}-{end_idx - 1}",
                            "raw_data": str(chunk_findings)[:200]
                            if isinstance(chunk_findings, str)
                            else "Non-string type",
                        }
                        claude_findings.append(error_dict)

                    # If significant anomalies found, add to the object's findings
                    if (
                        isinstance(chunk_findings, dict)
                        and isinstance(chunk_findings.get("anomalies_detected"), bool)
                        and chunk_findings.get("anomalies_detected", False)
                        and isinstance(chunk_findings.get("confidence"), (int, float))
                        and chunk_findings.get("confidence", 0) >= 7
                    ):
                        if "findings" in chunk_findings:
                            for finding in chunk_findings["findings"]:
                                if (
                                    isinstance(finding, dict)
                                    and finding.get("severity", 0) >= 7
                                ):  # Only include high severity findings
                                    self.findings.append(
                                        {
                                            "type": "claude_detected_anomaly",
                                            "details": finding,
                                            "chunk": i + 1,
                                            "rows": f"{start_idx}-{end_idx - 1}",
                                        }
                                    )
                except json.JSONDecodeError as e:
                    error_details = traceback.format_exc()
                    logger.error(f"JSON parse error in chunk {i + 1}: {e}")
                    logger.error(f"Error details: {error_details}")
                    
                    # More comprehensive error analysis
                    json_error_position = e.pos if hasattr(e, 'pos') else -1
                    json_context = ''
                    if json_error_position > 0 and json_error_position < len(json_str):
                        start_pos = max(0, json_error_position - 20)
                        end_pos = min(len(json_str), json_error_position + 20)
                        error_marker = '→ERROR→'
                        json_context = (
                            json_str[start_pos:json_error_position] + 
                            error_marker + 
                            json_str[json_error_position:end_pos]
                        )
                        
                    logger.error(f"JSON error context: {json_context}")
                    logger.error(f"Problematic JSON string: {json_str[:200]}...")
                    
                    chunk_findings = {
                        "error": f"JSON parse error: {str(e)}",
                        "chunk": i + 1,
                        "rows": f"{start_idx}-{end_idx - 1}",
                        "json_start": json_str[:200] if len(json_str) > 200 else json_str,
                        "json_error_position": json_error_position,
                        "json_context": json_context
                    }
                    claude_findings.append(chunk_findings)

            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Failed processing chunk {i + 1}: {e}")
                logger.error(f"Error details: {error_details}")
                
                # Try to determine if it was a Claude API error or another type
                error_type = "API" if "anthropic" in str(e).lower() or "claude" in str(e).lower() else "Processing"
                error_hint = ""
                
                # Specific handling for common error types
                if "context window" in str(e).lower() or "token limit" in str(e).lower():
                    error_hint = "The data chunk may be too large to process. Try reducing max_rows_per_chunk."
                elif "rate limit" in str(e).lower():
                    error_hint = "API rate limit exceeded. Try again after waiting a few minutes."
                elif "timeout" in str(e).lower():
                    error_hint = "Request timed out. The API may be experiencing high load."
                
                error_data = {
                    "error": f"{error_type} error: {str(e)}",
                    "hint": error_hint,
                    "traceback": error_details[:500],  # Include part of traceback but not too long
                    "chunk": i + 1,
                    "rows": f"{start_idx}-{end_idx - 1}",
                    "rows_count": end_idx - start_idx,
                    # Include information on chunk size to help diagnose if the chunk is too large
                    "chunk_size_bytes": len(chunk_df.to_string()) if 'chunk_df' in locals() else -1
                }
                claude_findings.append(error_data)

        return claude_findings

    def generate_report(self) -> str:
        """Generate an HTML report of the findings.
        
        Returns:
            str: HTML report of findings
        """
        if not self.findings:
            return "No anomalies detected in the data."

        report = "<html><head><title>Data Forensics Report</title>"
        report += "<style>body{font-family:Arial; margin:20px;} "
        report += ".finding{margin:20px 0; padding:15px; border:1px solid #ddd; border-radius:5px;} "
        report += ".important{color:red; font-weight:bold;}"
        report += ".sub-finding{margin:10px 0 10px 20px; padding:10px; border-left:3px solid #eee;}"
        report += "</style></head><body>"
        report += "<h1>Data Forensics Report</h1>"

        # Summarize findings
        report += "<h2>Summary of Findings</h2>"
        report += f"<p>Found {len(self.findings)} potential issues in the data.</p>"

        # Detailed findings
        report += "<h2>Detailed Findings</h2>"

        for i, finding in enumerate(self.findings):
            report += "<div class='finding'>"
            report += (
                f"<h3>Finding {i + 1}: {finding['type'].replace('_', ' ').title()}</h3>"
            )

            if finding["type"] == "sorting_anomaly":
                report += "<p class='important'>Detected rows that are out of sequence, which may indicate manual manipulation.</p>"
                report += "<ul>"
                for anomaly in finding["details"]:
                    report += f"<li>Row {anomaly['row_index']}: ID {anomaly['id']} comes after ID {anomaly['previous_id']} in {anomaly['sort_column']}={anomaly['sort_value']}</li>"
                report += "</ul>"

            elif finding["type"] == "duplicate_ids":
                report += "<p class='important'>Detected duplicate ID values, which may indicate duplication or manipulation.</p>"
                report += "<ul>"
                for dup in finding["details"]:
                    report += f"<li>ID {dup['id']} appears {dup['count']} times in rows {', '.join(map(str, dup['row_indices']))}</li>"
                report += "</ul>"

            elif finding["type"] == "statistical_anomaly":
                report += (
                    f"<p>Detected unusual values in column '{finding['column']}'.</p>"
                )
                report += f"<p>{finding['details']}</p>"

                # Add plot if available
                for plot in self.plots:
                    if plot["column"] == finding["column"]:
                        report += f"<img src='{plot['plot_path']}' width='600'/>"

            elif finding["type"] == "data_fabrication":
                report += "<p class='important'>Detected patterns that suggest possible data fabrication.</p>"
                report += "<ul>"
                for issue in finding["details"]:
                    report += f"<li><strong>Column {issue['column']}</strong>: "

                    if issue["issue"] == "excessive_repetition":
                        report += f"Excessive Digit Repetition - {issue['details']}"
                        if "examples" in issue:
                            report += f" (e.g., {', '.join(issue['examples'][:3])})"

                    elif issue["issue"] == "unusual_last_digits":
                        report += f"Non-uniform Terminal Digits - {issue['details']}"
                        if "most_common_digit" in issue:
                            report += (
                                f", most common digit: {issue['most_common_digit']}"
                            )

                    elif issue["issue"] == "artificial_sequence":
                        report += f"Artificial Sequence - {issue['details']}"
                        if "sequence_diff_mean" in issue:
                            report += (
                                f", mean difference: {issue['sequence_diff_mean']:.3f}"
                            )

                    elif issue["issue"] == "idiosyncratic_clusters":
                        report += (
                            f"Idiosyncratic Response Clusters - {issue['details']}"
                        )

                        # Add detail for each unusual value
                        if "unusual_values" in issue and issue["unusual_values"]:
                            report += "<div class='sub-finding'>"
                            for val_info in issue["unusual_values"]:
                                report += f'<p>Value <strong>"{val_info["value"]}"</strong> appears {val_info["count"]} times'
                                if val_info["issue"] == "sequential_cluster":
                                    report += f" in sequential clusters ({val_info['details']})"
                                elif val_info["issue"] == "uniform_spacing":
                                    report += (
                                        f" with uniform spacing ({val_info['details']})"
                                    )
                                report += "</p>"
                            report += "</div>"

                    report += "</li>"
                report += "</ul>"

                # Add plots if available
                for plot in self.plots:
                    if plot["column"] == "last_digits":
                        report += f"<img src='{plot['plot_path']}' width='600'/>"

            elif finding["type"] == "terminal_digit_anomaly":
                report += "<p class='important'>Detected anomalies in terminal digits that suggest data manipulation.</p>"
                report += "<ul>"
                for anomaly in finding["details"]:
                    report += f"<li>Column {anomaly['column']}: Terminal digits are not uniformly distributed (χ²={anomaly['chi_square']:.2f}, p={anomaly['p_value']:.6f})"

                    if anomaly.get("overrepresented_digits"):
                        report += f"<br>Overrepresented digits: {', '.join(anomaly['overrepresented_digits'])}"

                    if anomaly.get("underrepresented_digits"):
                        report += f"<br>Underrepresented digits: {', '.join(anomaly['underrepresented_digits'])}"

                    report += "</li>"
                report += "</ul>"

                # Add plot if available
                for plot in self.plots:
                    if plot["column"] == "terminal_digits":
                        report += f"<img src='{plot['plot_path']}' width='600'/>"

            elif finding["type"] == "variance_anomaly":
                report += "<p class='important'>Detected suspicious differences in variance between groups.</p>"
                report += "<ul>"
                for anomaly in finding["details"]:
                    report += f"<li>Column {anomaly['column']}: Variance ratio between groups is {anomaly['variance_ratio']:.2f} (suspicious if >5)"
                    report += f"<br>Group {anomaly['low_variance_group']} has unusually low variance ({anomaly['low_variance_value']:.2f}) compared to Group {anomaly['high_variance_group']} ({anomaly['high_variance_value']:.2f})"
                    report += "</li>"
                report += "</ul>"

                # Add plots if available
                for plot in self.plots:
                    if "variance_" in plot["column"]:
                        report += f"<img src='{plot['plot_path']}' width='600'/>"

            elif finding["type"] == "inlier_anomaly":
                report += "<p class='important'>Detected too many values clustered near means, suggesting possible data fabrication.</p>"
                report += "<ul>"
                for anomaly in finding["details"]:
                    report += f"<li>Column {anomaly['column']}: {anomaly['inlier_percentage'] * 100:.1f}% of values are within 0.5 SD of the mean"
                    report += f" (only {anomaly['expected_percentage'] * 100:.1f}% expected in natural data)</li>"
                report += "</ul>"

                # Add plot if available
                for plot in self.plots:
                    if plot["column"] == "inlier_distribution":
                        report += f"<img src='{plot['plot_path']}' width='600'/>"

            report += "</div>"

        report += "</body></html>"
        return report


class ExcelForensics:
    """Specialized class for forensic analysis of Excel files."""

    def __init__(self, excel_file: str) -> None:
        """Initialize with an Excel file path.
        
        Args:
            excel_file: Path to the Excel file to analyze
        """
        self.excel_file: str = excel_file
        self.temp_dir: Optional[str] = None
        self.extracted: bool = False
        self.calc_chain: List[Dict[str, Any]] = []

    def __enter__(self) -> 'ExcelForensics':
        """Context manager entry - extract the Excel file.
        
        Returns:
            ExcelForensics: Self for use with context manager
        """
        self.temp_dir = tempfile.mkdtemp()
        self.extract_excel()
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], 
                exc_tb: Optional[Any]) -> None:
        """Context manager exit - clean up temporary files.
        
        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def extract_excel(self) -> None:
        """Extract Excel file contents into a temporary directory."""
        with zipfile.ZipFile(self.excel_file, "r") as zip_ref:
            zip_ref.extractall(self.temp_dir)
        self.extracted = True

        # Parse calculation chain
        self.parse_calc_chain()

    def parse_calc_chain(self) -> None:
        """Parse Excel calculation chain XML file to extract calculation ordering information."""
        if not self.extracted:
            self.extract_excel()

        calc_chain_path = os.path.join(self.temp_dir, "xl", "calcChain.xml")
        if os.path.exists(calc_chain_path):
            # Parse XML
            tree = ET.parse(calc_chain_path)
            root = tree.getroot()

            # Process calculation chain entries
            for c in root.findall(".//{*}c"):
                ref = c.get("r")
                if ref:
                    # Convert Excel cell reference to row and column
                    col_str = "".join(filter(str.isalpha, ref))
                    row_num = int("".join(filter(str.isdigit, ref)))

                    self.calc_chain.append({"ref": ref, "row": row_num, "col": col_str})
        else:
            logger.info("No calculation chain found in this Excel file.")

    def analyze_row_movement(self, row_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Analyze the calculation chain to detect evidence of row movement.

        Args:
            row_numbers: List of row numbers to check for movement

        Returns:
            List[Dict[str, Any]]: Findings about row movements, each with details about the evidence
        """
        findings = []

        # Group calc chain entries by row
        row_entries = {}
        for entry in self.calc_chain:
            row = entry["row"]
            if row not in row_entries:
                row_entries[row] = []
            row_entries[row].append(entry)

        # Find the position of each row in the calculation chain
        for row in row_numbers:
            if row not in row_entries:
                continue

            entries = row_entries[row]

            # For each entry in this row, find adjacent entries in the chain
            for entry in entries:
                idx = self.calc_chain.index(entry)

                # Check entries before and after
                if idx > 0 and idx < len(self.calc_chain) - 1:
                    prev_entry = self.calc_chain[idx - 1]
                    next_entry = self.calc_chain[idx + 1]

                    # If adjacent entries are from different rows, it might indicate movement
                    if prev_entry["row"] != row and next_entry["row"] != row:
                        if abs(prev_entry["row"] - next_entry["row"]) == 1:
                            findings.append(
                                {
                                    "row": row,
                                    "evidence": f"Cell {entry['ref']} calculation is between rows {prev_entry['row']} and {next_entry['row']}",
                                    "likely_original_position": f"between rows {prev_entry['row']} and {next_entry['row']}",
                                }
                            )

        return findings
