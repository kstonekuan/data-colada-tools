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
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
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
        self.df_pl: Optional[pl.DataFrame] = None  # Polars DataFrame
        self.df: Optional[pd.DataFrame] = (
            None  # Also keep pandas DataFrame for compatibility
        )
        self.excel_metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def analyze_dataset(
        self,
        filepath: str,
        id_col: Optional[str] = None,
        sort_cols: Optional[Union[str, List[str]]] = None,
        user_suspicions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze a dataset for potential manipulation.

        Args:
            filepath: Path to the dataset file
            id_col: Column name for IDs
            sort_cols: Column name(s) for sorting/grouping
            user_suspicions: Optional dictionary with user-specified suspicions to check,
                            format: {
                                "focus_columns": List[str],  # Columns to prioritize checking
                                "potential_issues": List[str],  # e.g., "sorting", "out_of_order", "duplicates"
                                "treatment_columns": List[str],  # Potential treatment indicator columns
                                "outcome_columns": List[str],  # Outcome variables to analyze
                                "suspicious_rows": List[int],  # Specific rows to check more carefully
                                "suspect_grouping": str,  # Potential column to check for group-based manipulation
                                "description": str  # Free-text description of suspicions
                            }

        Returns:
            List of findings as dictionaries
        """
        self.findings = []
        self.plots = []

        # Initialize user suspicions with defaults if not provided
        self.user_suspicions = user_suspicions or {}

        # Add a entry for the suspicions to the findings list so it's recorded
        if self.user_suspicions:
            self.findings.append(
                {
                    "type": "user_suspicions",
                    "details": self.user_suspicions,
                    "note": "User-provided analysis guidance (used as supplementary information only)",
                }
            )

        # Determine file type and read data
        ext = os.path.splitext(filepath)[1].lower()

        # Load data with Polars when possible, fallback to pandas for special formats
        try:
            if ext == ".csv":
                # Use Polars for CSV files
                self.df_pl = pl.read_csv(filepath)
                # Create pandas dataframe for compatibility with existing code
                self.df = self.df_pl.to_pandas()
            elif ext == ".xlsx":
                # For Excel files, first read with pandas for metadata
                self.df = pd.read_excel(filepath)
                # Then convert to polars
                self.df_pl = pl.from_pandas(self.df)
                self.excel_metadata = self.extract_excel_metadata(filepath)
            elif ext == ".parquet":
                # Direct polars support for parquet
                self.df_pl = pl.read_parquet(filepath)
                self.df = self.df_pl.to_pandas()
            elif ext == ".dta":
                # Use pandas for Stata files, then convert
                self.df = pd.read_stata(filepath)
                self.df_pl = pl.from_pandas(self.df)
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
                            self.df, meta = pyreadstat.read_sav(
                                filepath, encoding=encoding
                            )
                            read_success = True
                            # Convert to polars
                            self.df_pl = pl.from_pandas(self.df)
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

            # Ensure we have both pandas and polars dataframes available
            if self.df is None and self.df_pl is not None:
                self.df = self.df_pl.to_pandas()
            elif self.df_pl is None and self.df is not None:
                self.df_pl = pl.from_pandas(self.df)

        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {str(e)}")

        # Analyze sorting anomalies if sort columns are provided
        if sort_cols and id_col:
            # Check if user has suspicions related to sorting or out-of-order values
            additional_columns_to_check = []
            prioritize_out_of_order = False

            if "potential_issues" in self.user_suspicions:
                potential_issues = [
                    issue.lower()
                    for issue in self.user_suspicions.get("potential_issues", [])
                ]
                prioritize_out_of_order = any(
                    issue in ["out of order", "out-of-order", "out_of_order", "sorting"]
                    for issue in potential_issues
                )

            if "focus_columns" in self.user_suspicions:
                additional_columns_to_check = self.user_suspicions.get(
                    "focus_columns", []
                )

            if "outcome_columns" in self.user_suspicions:
                additional_columns_to_check.extend(
                    self.user_suspicions.get("outcome_columns", [])
                )

            # Pass these additional parameters to the sorting anomaly check
            sorting_issues = self.check_sorting_anomalies(
                id_col,
                sort_cols,
                check_dependent_vars=True,
                prioritize_columns=additional_columns_to_check,
                prioritize_out_of_order=prioritize_out_of_order,
            )

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

    def analyze_column_unique_values(
        self, client: Anthropic, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze unique values in each column using Claude.

        Args:
            client: Claude API client
            columns: Optional list of columns to analyze. If None, analyze all columns.

        Returns:
            Dictionary mapping column names to analysis results
        """
        if self.df_pl is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")

        # Use all columns if none specified
        available_columns = self.df_pl.columns
        if columns is None:
            columns = available_columns
        elif not all(col in available_columns for col in columns):
            missing = [col for col in columns if col not in available_columns]
            raise ValueError(f"Columns not found in dataset: {missing}")

        results = {}

        for column in columns:
            # Use Polars to get unique values - much faster than pandas
            unique_values_pl = self.df_pl.select(pl.col(column)).unique()
            unique_values = unique_values_pl.to_series().to_numpy()
            unique_count = len(unique_values)

            # Limit to 100 values if there are too many
            if unique_count > 100:
                logger.info(
                    f"Column {column} has {unique_count} unique values. Sampling 100 for analysis."
                )
                # Take a representative sample including first and last values
                # This is more efficient than concatenating arrays
                if len(unique_values) > 0:
                    first_values = unique_values[:50]
                    last_values = unique_values[-50:]
                    values_for_analysis = np.concatenate([first_values, last_values])
                    sampling_note = f"NOTE: This column has {unique_count} unique values. Only showing a sample of 100."
                else:
                    values_for_analysis = unique_values
                    sampling_note = "Column has no values."
            else:
                values_for_analysis = unique_values
                sampling_note = f"Complete set of {unique_count} unique values."

            # Convert to string representations when possible
            values_str = []
            for val in values_for_analysis:
                if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                    values_str.append("NA/NULL")
                elif isinstance(val, (float, int, str, bool)):
                    values_str.append(str(val))
                else:
                    values_str.append(repr(val))

            # Get data type using Polars
            col_dtype = self.df_pl.schema[column]

            # Create a prompt for Claude to analyze this column's unique values
            prompt = f"""Analyze the following unique values from the column '{column}' in a research dataset.

Column: {column}
Data type: {col_dtype}
{sampling_note}

Unique values:
{json.dumps(values_str, indent=2)}

Please analyze these values and identify any potential anomalies, patterns, or issues that might indicate data manipulation.
Things to look for:
1. Unusual patterns or sequences
2. Gaps or clusters in numeric data 
3. Unexplained outliers
4. Non-random distribution of digits
5. Unusual formatting or inconsistencies
6. Values that seem out of context for this type of data
7. Any other suspicious patterns

Be concise but thorough in your analysis. If you don't see any issues, say so.
Rate the suspiciousness of this column's values on a scale of 1-10, where 10 is highly suspicious.
Format this exactly as: "SUSPICION_RATING: [1-10]"
"""

            try:
                # Send prompt to Claude for analysis
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract the response
                analysis = response.content[0].text

                # Parse the suspicion rating
                suspicion_rating = None
                match = re.search(r"SUSPICION_RATING:\s*(\d+)", analysis)
                if match:
                    suspicion_rating = int(match.group(1))

                results[column] = {
                    "analysis": analysis,
                    "unique_count": unique_count,
                    "suspicion_rating": suspicion_rating,
                    "sample_analyzed": len(values_for_analysis) < unique_count,
                }

            except Exception as e:
                logger.error(f"Error analyzing column {column}: {e}")
                results[column] = {"error": str(e), "unique_count": unique_count}

        return results

    def check_sorting_anomalies(
        self,
        id_col: str,
        sort_cols: Union[str, List[str]],
        check_dependent_vars: bool = True,
        prioritize_columns: Optional[List[str]] = None,
        prioritize_out_of_order: bool = False,
    ) -> List[Dict[str, Any]]:
        """Check for anomalies in sorting order that might indicate manipulation.

        Args:
            id_col: Column name for IDs
            sort_cols: Column name(s) for sorting/grouping
            check_dependent_vars: Whether to look for dependent variables that might be sorted
            prioritize_columns: Specific columns to prioritize in the out-of-order analysis
            prioritize_out_of_order: Whether to use more sensitive thresholds for out-of-order detection

        Returns:
            List of sorting anomalies found with enhanced out-of-order analysis
        """
        if self.df_pl is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")

        anomalies = []

        # Identify potential dependent variables (numeric columns that aren't used for sorting)
        dependent_vars = []
        prioritized_vars = []

        if check_dependent_vars:
            sort_cols_list = [sort_cols] if isinstance(sort_cols, str) else sort_cols

            # Get numeric columns using Polars
            numeric_dtypes = [
                pl.Float32,
                pl.Float64,
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]

            # Find numeric columns
            schema = self.df_pl.schema
            numeric_cols = [
                col_name
                for col_name, dtype in schema.items()
                if any(isinstance(dtype, t) for t in numeric_dtypes)
            ]

            # Get all potential dependent variables
            dependent_vars = [
                col
                for col in numeric_cols
                if col != id_col and col not in sort_cols_list
            ]

            # If user provided specific columns to prioritize, check those first
            if prioritize_columns:
                # Filter to ensure we only include columns that exist and are numeric
                available_columns = self.df_pl.columns
                prioritized_vars = [
                    col
                    for col in prioritize_columns
                    if col in available_columns
                    and col in numeric_cols
                    and col != id_col
                    and col not in sort_cols_list
                ]

                # If we're prioritizing, put these columns at the front of the list
                if prioritized_vars:
                    # Remove prioritized vars from dependent_vars to avoid duplicates
                    dependent_vars = [
                        col for col in dependent_vars if col not in prioritized_vars
                    ]
                    # Combine lists with prioritized vars first
                    dependent_vars = prioritized_vars + dependent_vars

        # For each sorting level
        if isinstance(sort_cols, str):
            sort_cols = [sort_cols]

        # Process each sort column - this is significantly faster with Polars
        for col in sort_cols:
            # Get unique values efficiently with Polars
            unique_vals = self.df_pl.select(pl.col(col)).unique().to_series().to_list()

            for val in unique_vals:
                # Get rows for this value using Polars filter
                subset_pl = self.df_pl.filter(pl.col(col) == val)

                # Add row indices for better tracking
                subset_pl = subset_pl.with_row_count("_row_idx")

                # Sort by ID to make it easier to find anomalies
                sorted_subset = subset_pl.sort(id_col)

                # Get the ID and row index columns
                id_vals = sorted_subset.select([id_col, "_row_idx"]).to_pandas()

                # Find out-of-sequence IDs
                if len(id_vals) > 1:
                    # Calculate differences between consecutive IDs
                    id_vals["id_diff"] = id_vals[id_col].diff()

                    # Negative diffs indicate out-of-sequence IDs
                    anomaly_rows = id_vals[id_vals["id_diff"] < 0]

                    for _, row in anomaly_rows.iterrows():
                        row_idx = int(row["_row_idx"])
                        current_id = row[id_col]

                        # Get previous ID value
                        prev_idx = id_vals.index[id_vals.index.get_loc(row.name) - 1]
                        prev_id = id_vals.loc[prev_idx, id_col]

                        # Convert to int if it's an integer type
                        if isinstance(current_id, (int, np.integer)):
                            current_id = int(current_id)
                        if isinstance(prev_id, (int, np.integer)):
                            prev_id = int(prev_id)

                        # Basic anomaly info
                        anomaly = {
                            "row_index": row_idx,
                            "id": current_id,
                            "previous_id": prev_id,
                            "sort_column": col,
                            "sort_value": int(val)
                            if isinstance(val, (int, np.integer))
                            else val,
                        }

                        # Convert subset_pl to pandas DataFrame for compatibility with existing analysis function
                        # In the future, this function could be rewritten to use Polars as well
                        subset_pd = subset_pl.to_pandas()

                        # Check if dependent variables also follow an unusual pattern within this group
                        if dependent_vars:
                            # Use the original analysis function which works with pandas DataFrame
                            out_of_order_analysis = (
                                self._analyze_out_of_order_dependent_vars(
                                    subset_pd,
                                    row_idx,
                                    dependent_vars,
                                    prioritize_out_of_order=prioritize_out_of_order,
                                    prioritized_columns=prioritized_vars
                                    if prioritized_vars
                                    else None,
                                )
                            )
                            if out_of_order_analysis:
                                anomaly["out_of_order_analysis"] = out_of_order_analysis

                        anomalies.append(anomaly)

        return anomalies

    def _analyze_out_of_order_dependent_vars(
        self,
        subset: pd.DataFrame,
        row_idx: int,
        dependent_vars: List[str],
        prioritize_out_of_order: bool = False,
        prioritized_columns: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyzes potential out-of-order observations for dependent variables.

        This looks for values in dependent variables that break a strong sorting pattern,
        which could indicate manual manipulation of values to achieve desired results.

        Args:
            subset: DataFrame subset for the current group
            row_idx: Index of the row that's out of order
            dependent_vars: List of potential dependent variables to check
            prioritize_out_of_order: Whether to use more sensitive thresholds for detecting out-of-order patterns
            prioritized_columns: Specific columns to prioritize and analyze more thoroughly

        Returns:
            Dict with out-of-order analysis if found, None otherwise
        """
        # We need enough rows to have a meaningful pattern
        # Use a smaller threshold if user suspects out-of-order observations
        min_rows = 3 if prioritize_out_of_order else 5
        if len(subset) < min_rows:
            return None

        # Get the row that's out of order
        row_position = subset.index.get_loc(row_idx)

        # Check each dependent variable - prioritize the ones specified by the user
        for var in dependent_vars:
            # Skip if the column doesn't have numeric values
            if not pd.api.types.is_numeric_dtype(subset[var]):
                continue

            # Is this a prioritized column?
            is_priority_column = prioritized_columns and var in prioritized_columns

            # Create a series with all values for this variable
            values = subset[var].copy()

            # Adjust tolerance based on whether this is a priority column
            # For priority columns or when prioritizing out-of-order detection,
            # we're more lenient in determining if a pattern exists
            if prioritize_out_of_order or is_priority_column:
                # For prioritized analysis, allow a small percentage of exceptions
                allowed_exceptions = max(
                    1, int(len(values) * 0.1)
                )  # Allow up to 10% exceptions

                # Check if variable is mostly sorted (with some tolerance)
                exceptions_asc = sum(
                    1
                    for i in range(len(values) - 1)
                    if values.iloc[i] > values.iloc[i + 1]
                    and i != row_position - 1
                    and i != row_position
                )

                exceptions_desc = sum(
                    1
                    for i in range(len(values) - 1)
                    if values.iloc[i] < values.iloc[i + 1]
                    and i != row_position - 1
                    and i != row_position
                )

                is_sorted_asc = exceptions_asc <= allowed_exceptions
                is_sorted_desc = exceptions_desc <= allowed_exceptions
            else:
                # Standard strict check - must be perfectly sorted
                is_sorted_asc = all(
                    values.iloc[i] <= values.iloc[i + 1]
                    for i in range(len(values) - 1)
                    if i != row_position - 1 and i != row_position
                )
                is_sorted_desc = all(
                    values.iloc[i] >= values.iloc[i + 1]
                    for i in range(len(values) - 1)
                    if i != row_position - 1 and i != row_position
                )

            # If values are sorted except around our anomalous row, this is suspicious
            if is_sorted_asc or is_sorted_desc:
                # Get the current value
                current_value = values.iloc[row_position]

                # Figure out what the value should be to maintain the pattern
                if row_position > 0 and row_position < len(values) - 1:
                    prev_value = values.iloc[row_position - 1]
                    next_value = values.iloc[row_position + 1]

                    # For ascending pattern
                    if is_sorted_asc:
                        # Value should be between prev and next in sorted order
                        if current_value > next_value or current_value < prev_value:
                            likely_original = (prev_value + next_value) / 2

                            # Calculate statistical impact - does this anomaly create/strengthen a correlation?
                            # Get the mean for this variable when grouped by a boolean representation of the sorting column
                            variable_difference = self._calculate_statistical_impact(
                                subset, var, current_value, likely_original
                            )

                            return {
                                "sorted_by": [var],
                                "breaking_pattern": f"value {current_value} breaks ascending pattern",
                                "imputed_original_values": [
                                    {
                                        "row_index": int(row_idx),
                                        "column": var,
                                        "current": float(current_value),
                                        "likely_original": float(likely_original),
                                    }
                                ],
                                "statistical_impact": variable_difference,
                            }

                    # For descending pattern
                    elif is_sorted_desc:
                        # Value should be between prev and next in sorted order
                        if current_value < next_value or current_value > prev_value:
                            likely_original = (prev_value + next_value) / 2

                            # Calculate statistical impact
                            variable_difference = self._calculate_statistical_impact(
                                subset, var, current_value, likely_original
                            )

                            return {
                                "sorted_by": [var],
                                "breaking_pattern": f"value {current_value} breaks descending pattern",
                                "imputed_original_values": [
                                    {
                                        "row_index": int(row_idx),
                                        "column": var,
                                        "current": float(current_value),
                                        "likely_original": float(likely_original),
                                    }
                                ],
                                "statistical_impact": variable_difference,
                            }

        # No patterns found
        return None

    def _calculate_statistical_impact(
        self,
        subset: pd.DataFrame,
        var: str,
        current_value: float,
        likely_original: float,
    ) -> str:
        """Calculate the statistical impact of an out-of-order value.

        Args:
            subset: DataFrame subset for the current group
            var: Variable name
            current_value: Current value in the dataset
            likely_original: Likely original value before manipulation

        Returns:
            String describing the statistical impact
        """
        try:
            # Make a copy to avoid modifying the original data
            subset_copy = subset.copy()

            # Find the row with the current value
            row_with_value = subset_copy[subset_copy[var] == current_value]

            # If we don't find the row, we can't calculate impact
            if len(row_with_value) == 0:
                return "Impact cannot be calculated"

            # Get original mean and standard deviation
            original_mean = subset_copy[var].mean()
            original_std = subset_copy[var].std()

            # Identify a categorical column that might be a treatment indicator
            categorical_cols = subset_copy.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Also include binary numeric columns (0/1)
            for col in subset_copy.select_dtypes(include=["number"]).columns:
                if set(subset_copy[col].unique()).issubset({0, 1}):
                    categorical_cols.append(col)

            # If we don't have any treatment indicators, just report the impact on mean
            if not categorical_cols:
                # Replace the value
                row_idx = row_with_value.index[0]
                subset_copy.at[row_idx, var] = likely_original

                # Calculate new statistics
                new_mean = subset_copy[var].mean()
                new_std = subset_copy[var].std()

                mean_change = ((new_mean - original_mean) / original_mean) * 100
                std_change = ((new_std - original_std) / original_std) * 100

                return (
                    f"Mean would change by {mean_change:.1f}%, SD by {std_change:.1f}%"
                )

            # Try each categorical column as a potential treatment indicator
            impacts = []
            for cat_col in categorical_cols:
                if subset_copy[cat_col].nunique() != 2:
                    continue

                # Get the two groups
                subset_copy[cat_col].unique()

                # Calculate original difference between groups
                group_means = subset_copy.groupby(cat_col)[var].mean()
                original_diff = abs(group_means.max() - group_means.min())

                # Replace the value
                row_idx = row_with_value.index[0]
                subset_copy.loc[row_idx, cat_col]
                subset_copy.at[row_idx, var] = likely_original

                # Calculate new difference
                new_group_means = subset_copy.groupby(cat_col)[var].mean()
                new_diff = abs(new_group_means.max() - new_group_means.min())

                # Calculate percent change in difference
                if original_diff > 0:
                    pct_change = ((original_diff - new_diff) / original_diff) * 100
                    impacts.append(
                        f"Difference between groups in {cat_col} would decrease by {pct_change:.1f}%"
                    )

            if impacts:
                return "; ".join(impacts)
            else:
                return "No significant impact on group differences"

        except Exception as e:
            return f"Error calculating impact: {str(e)}"

    def check_duplicate_ids(self, id_col: str) -> List[Dict[str, Any]]:
        """Check for duplicate ID values that might indicate manipulation.

        Args:
            id_col: Column name for IDs

        Returns:
            List of duplicate ID findings
        """
        # Use Polars for much faster duplicate detection
        if self.df_pl is None:
            raise ValueError("No dataset loaded")

        # Group by ID and count occurrences
        counts = (
            self.df_pl.select(pl.col(id_col))
            .group_by(id_col)
            .agg(pl.count().alias("count"))
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
        )

        duplicate_details = []

        # For each duplicate ID, get details
        for row in counts.iter_rows(named=True):
            dup_id = row[id_col]
            count = row["count"]

            # Find row indices with this duplicate ID
            # Convert to integers list for serialization
            rows_df = self.df_pl.filter(pl.col(id_col) == dup_id)
            try:
                # For Polars >= 0.19.0
                row_indices = [int(idx) for idx in rows_df.row_nums()]
            except AttributeError:
                # Fallback for older Polars versions
                # Create a temporary dataframe with row indices
                temp_df = self.df_pl.with_row_count("_row_idx")
                filtered = temp_df.filter(pl.col(id_col) == dup_id)
                row_indices = [
                    int(idx) for idx in filtered.select("_row_idx").to_series()
                ]

            # Convert ID to int if it looks like an integer
            if isinstance(dup_id, (int, np.integer)):
                dup_id = int(dup_id)

            duplicate_details.append(
                {"id": dup_id, "count": int(count), "row_indices": row_indices}
            )

        return duplicate_details

    def extract_excel_metadata(
        self, excel_file: str
    ) -> Dict[str, List[Dict[str, Any]]]:
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

    def analyze_suspicious_observations(
        self, suspicious_rows: List[int], group_col: str, outcome_vars: List[str]
    ) -> Dict[str, Any]:
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
                            "details": "Found unusual clusters of identical responses",
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

    def create_variance_comparison_plot(
        self, anomalies: List[Dict[str, Any]], group_col: str
    ) -> None:
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

    def segment_and_analyze_with_claude(
        self,
        client: Anthropic,
        max_rows_per_chunk: int = 300,
        user_suspicions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Segment dataset into manageable chunks and use Claude to detect anomalies in each chunk.

        Args:
            client: Claude API client
            max_rows_per_chunk: Maximum number of rows per chunk
            user_suspicions: Optional dictionary with user-specified suspicions to guide analysis
                            but not bias the results excessively

        Returns:
            List[Dict[str, Any]]: A list of findings from Claude's analysis,
            each finding is a dictionary that may contain anomaly results or error information
        """
        # If user_suspicions is not provided, use the one initialized in analyze_dataset
        if user_suspicions is None and hasattr(self, "user_suspicions"):
            user_suspicions = self.user_suspicions
        if not hasattr(self, "df"):
            return [
                {
                    "error": "DataFrame not initialized in DataForensics object",
                    "hint": "Make sure to set forensics.df before calling this method",
                }
            ]

        if self.df is None:
            return [
                {
                    "error": "DataFrame is None in DataForensics object",
                    "hint": "Make sure forensics.df is assigned a valid DataFrame",
                }
            ]

        if len(self.df) == 0:
            return [
                {
                    "error": "DataFrame is empty (0 rows)",
                    "hint": "The dataset must contain at least one row of data",
                }
            ]

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
                    clean_chunk_df[col] = clean_chunk_df[col].replace(
                        [np.inf, -np.inf, np.nan], None
                    )

            # Ensure both numeric and categorical stats don't have NaN or infinity values
            for col, stats in numeric_stats.items():
                for key, value in stats.items():
                    if isinstance(value, float) and (
                        np.isnan(value) or np.isinf(value)
                    ):
                        numeric_stats[col][key] = None

            # Convert the dataframe to string with error handling
            try:
                df_string = clean_chunk_df.to_string()
                # Truncate if too large (to avoid token limits)
                if len(df_string) > 200000:  # ~200KB should be safe
                    logger.warning(
                        f"Chunk {i + 1} data too large, truncating for Claude analysis"
                    )
                    df_string = df_string[:200000] + "\n[... truncated ...]"
            except Exception as df_err:
                logger.error(f"Error converting dataframe to string: {df_err}")
                df_string = f"[Error rendering dataframe: {df_err}]"

            # Create prompt for Claude
            # Prepare section for user suspicions if provided
            user_suspicion_section = ""
            if user_suspicions and isinstance(user_suspicions, dict):
                # Format the user suspicions in a helpful way, but emphasize being objective
                user_suspicion_section = "\nUser has raised the following areas to explore (analyze these objectively, but do not let these suggestions bias your findings):\n"

                if "description" in user_suspicions and user_suspicions["description"]:
                    user_suspicion_section += (
                        f"- Description: {user_suspicions['description']}\n"
                    )

                if (
                    "focus_columns" in user_suspicions
                    and user_suspicions["focus_columns"]
                ):
                    focus_cols = ", ".join(user_suspicions["focus_columns"])
                    user_suspicion_section += f"- Columns of interest: {focus_cols}\n"

                if (
                    "potential_issues" in user_suspicions
                    and user_suspicions["potential_issues"]
                ):
                    issues = ", ".join(user_suspicions["potential_issues"])
                    user_suspicion_section += f"- Potential issues to check: {issues}\n"

                if (
                    "treatment_columns" in user_suspicions
                    and user_suspicions["treatment_columns"]
                ):
                    treatment_cols = ", ".join(user_suspicions["treatment_columns"])
                    user_suspicion_section += (
                        f"- Potential treatment indicators: {treatment_cols}\n"
                    )

                if (
                    "outcome_columns" in user_suspicions
                    and user_suspicions["outcome_columns"]
                ):
                    outcome_cols = ", ".join(user_suspicions["outcome_columns"])
                    user_suspicion_section += f"- Outcome variables: {outcome_cols}\n"

                if (
                    "suspicious_rows" in user_suspicions
                    and user_suspicions["suspicious_rows"]
                ):
                    # Format row ranges for readability
                    rows = [str(r) for r in user_suspicions["suspicious_rows"]]
                    if len(rows) > 10:
                        rows_str = (
                            f"{', '.join(rows[:10])}... (and {len(rows) - 10} more)"
                        )
                    else:
                        rows_str = ", ".join(rows)
                    user_suspicion_section += (
                        f"- Specific rows to examine: {rows_str}\n"
                    )

                if (
                    "suspect_grouping" in user_suspicions
                    and user_suspicions["suspect_grouping"]
                ):
                    user_suspicion_section += f"- Check for group-based patterns using: {user_suspicions['suspect_grouping']}\n"

                # Add important note to avoid bias
                user_suspicion_section += "\nIMPORTANT: While considering these areas, maintain objectivity and report what the data actually shows, not what is expected. Do not let these suggestions narrow your analysis or bias your findings. Thoroughly analyze all patterns in the data.\n"

            prompt = f"""Analyze this dataset chunk for potential data manipulation or anomalies.{
                user_suspicion_section
            }

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

1. Out-of-Order Observations:
   - Check if the data appears to be sorted by one or more columns but has values that are suspiciously out of sequence
   - Look for rows that break an otherwise perfect sorting pattern (these often indicate manual manipulation)
   - When values are out of order in an otherwise sorted dataset, reconstruct what the original values likely were
   - Pay attention to dependent variables that show perfect sorting within groups except for a few "convenient" outliers
   - Test whether restoring the suspected original values would eliminate statistical significance
   - Example: If rows are sorted by group (0/1) then by response count, but some values in group 1 break the pattern

2. Statistical anomalies (unusual patterns, outliers, distributions)
3. Suspicious repeated values or patterns
4. Unnaturally perfect distributions
5. Evidence of data fabrication
6. Terminal digit anomalies
7. Uniform spacing patterns
8. Implausible correlations
9. Unusual clustering of values
10. Values that should not logically be present in a column
11. Any other potential signs of manipulation

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
      "severity": 1-10,
      "out_of_order_analysis": {{
        "sorted_by": ["column names that appear to be sorted"],
        "breaking_pattern": "description of how the pattern is broken",
        "imputed_original_values": [{{"row_index": 0, "column": "col", "current": 0, "likely_original": 0}}],
        "statistical_impact": "how this affects statistical significance"
      }}
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
                    logger.error(
                        f"Could not find valid JSON markers in Claude's response for chunk {i + 1}"
                    )
                    logger.error(f"Claude response: {content[:200]}...")

                    # Create an error finding
                    claude_findings.append(
                        {
                            "error": "Invalid JSON format in Claude response",
                            "chunk": i + 1,
                            "rows": f"{start_idx}-{end_idx - 1}",
                            "raw_response": content[:500]
                            if len(content) > 500
                            else content,
                        }
                    )
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
                            # Add chunk number reference
                            finding["chunk"] = i + 1

                            # Update row indices with offsets
                            if "row_indices" in finding:
                                try:
                                    # Convert and normalize row_indices
                                    # Could be a list of integers, a list of strings, or a single value
                                    row_indices = finding["row_indices"]

                                    # If it's not a list, convert it to one
                                    if not isinstance(row_indices, list):
                                        row_indices = [row_indices]

                                    # Convert all indices to integers, then add offset
                                    finding["row_indices"] = []
                                    for idx in row_indices:
                                        try:
                                            if isinstance(idx, str):
                                                # Try to convert string to int
                                                idx = int(idx)
                                            # Add the offset to translate to original dataset
                                            finding["row_indices"].append(
                                                start_idx + idx
                                            )
                                        except (ValueError, TypeError) as e:
                                            # If a specific index can't be converted, log the error but continue processing
                                            logger.warning(
                                                f"Could not process row index '{idx}': {e}"
                                            )

                                except Exception as e:
                                    # Log error but preserve the finding
                                    logger.error(f"Error updating row indices: {e}")
                                    # Add a diagnostic field
                                    finding["row_indices_error"] = (
                                        f"Original format could not be processed: {str(e)}"
                                    )

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
                    json_error_position = e.pos if hasattr(e, "pos") else -1
                    json_context = ""
                    if json_error_position > 0 and json_error_position < len(json_str):
                        start_pos = max(0, json_error_position - 20)
                        end_pos = min(len(json_str), json_error_position + 20)
                        error_marker = "→ERROR→"
                        json_context = (
                            json_str[start_pos:json_error_position]
                            + error_marker
                            + json_str[json_error_position:end_pos]
                        )

                    logger.error(f"JSON error context: {json_context}")
                    logger.error(f"Problematic JSON string: {json_str[:200]}...")

                    chunk_findings = {
                        "error": f"JSON parse error: {str(e)}",
                        "chunk": i + 1,
                        "rows": f"{start_idx}-{end_idx - 1}",
                        "json_start": json_str[:200]
                        if len(json_str) > 200
                        else json_str,
                        "json_error_position": json_error_position,
                        "json_context": json_context,
                    }
                    claude_findings.append(chunk_findings)

            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Failed processing chunk {i + 1}: {e}")
                logger.error(f"Error details: {error_details}")

                # Try to determine if it was a Claude API error or another type
                error_type = (
                    "API"
                    if "anthropic" in str(e).lower() or "claude" in str(e).lower()
                    else "Processing"
                )
                error_hint = ""

                # Specific handling for common error types
                if (
                    "context window" in str(e).lower()
                    or "token limit" in str(e).lower()
                ):
                    error_hint = "The data chunk may be too large to process. Try reducing max_rows_per_chunk."
                elif "rate limit" in str(e).lower():
                    error_hint = "API rate limit exceeded. Try again after waiting a few minutes."
                elif "timeout" in str(e).lower():
                    error_hint = (
                        "Request timed out. The API may be experiencing high load."
                    )

                error_data = {
                    "error": f"{error_type} error: {str(e)}",
                    "hint": error_hint,
                    "traceback": error_details[
                        :500
                    ],  # Include part of traceback but not too long
                    "chunk": i + 1,
                    "rows": f"{start_idx}-{end_idx - 1}",
                    "rows_count": end_idx - start_idx,
                    # Include information on chunk size to help diagnose if the chunk is too large
                    "chunk_size_bytes": len(chunk_df.to_string())
                    if "chunk_df" in locals()
                    else -1,
                }
                claude_findings.append(error_data)

        return claude_findings

    def compare_with_without_suspicious_rows(
        self,
        suspicious_rows: List[int],
        group_column: Optional[str] = None,
        outcome_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare analysis results with and without suspicious rows to see how they impact the results.

        Args:
            suspicious_rows: List of row indices considered suspicious
            group_column: Name of the column containing group/treatment information
            outcome_columns: Names of outcome/dependent variable columns to analyze

        Returns:
            Dict[str, Any]: Dictionary containing comparison results
        """
        if not hasattr(self, "df") or self.df is None:
            return {
                "error": "DataFrame not initialized or is None",
                "hint": "Make sure to set forensics.df before calling this method",
            }

        if len(suspicious_rows) == 0:
            return {
                "error": "No suspicious rows provided",
                "hint": "Provide a list of row indices to compare",
            }

        if len(self.df) == 0:
            return {
                "error": "DataFrame is empty (0 rows)",
                "hint": "The dataset must contain at least one row of data",
            }

        # Create a DataFrame without the suspicious rows
        df_without_suspicious = self.df.drop(suspicious_rows).reset_index(drop=True)

        # If no outcome columns are provided, try to automatically detect numeric columns
        if outcome_columns is None:
            # Exclude the group column from outcome columns if provided
            numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
            if group_column and group_column in numeric_cols:
                numeric_cols.remove(group_column)
            outcome_columns = numeric_cols[:5]  # Use first 5 numeric columns as default

        results = {
            "suspicious_rows_count": len(suspicious_rows),
            "dataset_size": len(self.df),
            "dataset_size_without_suspicious": len(df_without_suspicious),
            "outcome_columns": outcome_columns,
            "comparison_results": [],
        }

        # If group column is provided, compare differences between groups
        if group_column and group_column in self.df.columns:
            # Check that the group column has at least 2 unique values
            unique_groups = self.df[group_column].unique()
            if len(unique_groups) >= 2:
                results["group_column"] = group_column
                results["groups"] = [str(g) for g in unique_groups]

                # For each outcome column, calculate statistics with and without suspicious rows
                for col in outcome_columns:
                    if col in self.df.columns and pd.api.types.is_numeric_dtype(
                        self.df[col]
                    ):
                        try:
                            # Calculate statistics for full dataset
                            full_stats = self._calculate_group_statistics(
                                self.df, group_column, col
                            )

                            # Calculate statistics without suspicious rows
                            clean_stats = self._calculate_group_statistics(
                                df_without_suspicious, group_column, col
                            )

                            # Calculate effect sizes
                            full_effect_size = self._calculate_effect_size(full_stats)
                            clean_effect_size = self._calculate_effect_size(clean_stats)

                            # Calculate percent change in effect size
                            effect_size_change = 0
                            if clean_effect_size != 0:
                                effect_size_change = (
                                    (full_effect_size - clean_effect_size)
                                    / abs(clean_effect_size)
                                ) * 100

                            # Hypothesis testing (t-test) with and without suspicious rows
                            full_ttest = self._perform_ttest(self.df, group_column, col)
                            clean_ttest = self._perform_ttest(
                                df_without_suspicious, group_column, col
                            )

                            # Check if significance changes
                            significance_changed = (
                                full_ttest["p_value"] < 0.05
                                and clean_ttest["p_value"] >= 0.05
                            ) or (
                                full_ttest["p_value"] >= 0.05
                                and clean_ttest["p_value"] < 0.05
                            )

                            results["comparison_results"].append(
                                {
                                    "column": col,
                                    "with_suspicious": {
                                        "group_stats": full_stats,
                                        "effect_size": full_effect_size,
                                        "ttest": full_ttest,
                                    },
                                    "without_suspicious": {
                                        "group_stats": clean_stats,
                                        "effect_size": clean_effect_size,
                                        "ttest": clean_ttest,
                                    },
                                    "effect_size_change_percent": effect_size_change,
                                    "significance_changed": significance_changed,
                                    "significance_change_description": self._describe_significance_change(
                                        full_ttest["p_value"], clean_ttest["p_value"]
                                    ),
                                }
                            )
                        except Exception as e:
                            results["comparison_results"].append(
                                {"column": col, "error": str(e)}
                            )
            else:
                results["warning"] = (
                    f"Group column '{group_column}' has fewer than 2 unique values"
                )
        else:
            # Without a group column, compare overall statistics
            for col in outcome_columns:
                if col in self.df.columns and pd.api.types.is_numeric_dtype(
                    self.df[col]
                ):
                    try:
                        # Calculate statistics for full dataset
                        full_mean = float(self.df[col].mean())
                        full_std = float(self.df[col].std())

                        # Calculate statistics without suspicious rows
                        clean_mean = float(df_without_suspicious[col].mean())
                        clean_std = float(df_without_suspicious[col].std())

                        # Calculate percent changes
                        mean_change = 0
                        if clean_mean != 0:
                            mean_change = (
                                (full_mean - clean_mean) / abs(clean_mean)
                            ) * 100

                        std_change = 0
                        if clean_std != 0:
                            std_change = ((full_std - clean_std) / abs(clean_std)) * 100

                        results["comparison_results"].append(
                            {
                                "column": col,
                                "with_suspicious": {"mean": full_mean, "std": full_std},
                                "without_suspicious": {
                                    "mean": clean_mean,
                                    "std": clean_std,
                                },
                                "mean_change_percent": mean_change,
                                "std_change_percent": std_change,
                            }
                        )
                    except Exception as e:
                        results["comparison_results"].append(
                            {"column": col, "error": str(e)}
                        )

        return results

    def _calculate_group_statistics(
        self, df: pd.DataFrame, group_col: str, outcome_col: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each group in the dataset."""
        group_stats = {}
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][outcome_col]
            group_stats[str(group)] = {
                "count": len(group_data),
                "mean": float(group_data.mean())
                if len(group_data) > 0
                else float("nan"),
                "std": float(group_data.std()) if len(group_data) > 1 else float("nan"),
                "min": float(group_data.min()) if len(group_data) > 0 else float("nan"),
                "max": float(group_data.max()) if len(group_data) > 0 else float("nan"),
            }
        return group_stats

    def _calculate_effect_size(self, group_stats: Dict[str, Dict[str, float]]) -> float:
        """Calculate simple effect size (difference between max and min group means)."""
        if len(group_stats) < 2:
            return 0.0

        # Extract means, handling potential NaN values
        means = [stats.get("mean", float("nan")) for stats in group_stats.values()]
        valid_means = [m for m in means if not np.isnan(m)]

        if len(valid_means) < 2:
            return 0.0

        return float(max(valid_means) - min(valid_means))

    def _perform_ttest(
        self, df: pd.DataFrame, group_col: str, outcome_col: str
    ) -> Dict[str, float]:
        """Perform t-test between the first two groups in the dataset."""
        from scipy import stats

        groups = df[group_col].unique()
        if len(groups) < 2:
            return {"t_statistic": float("nan"), "p_value": float("nan")}

        # Use the first two groups for the t-test
        group1_data = df[df[group_col] == groups[0]][outcome_col].dropna()
        group2_data = df[df[group_col] == groups[1]][outcome_col].dropna()

        if len(group1_data) < 2 or len(group2_data) < 2:
            return {"t_statistic": float("nan"), "p_value": float("nan")}

        try:
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            return {"t_statistic": float(t_stat), "p_value": float(p_val)}
        except Exception:
            return {"t_statistic": float("nan"), "p_value": float("nan")}

    def _describe_significance_change(
        self, p_val_with: float, p_val_without: float
    ) -> str:
        """Create a description of how statistical significance changed."""
        if np.isnan(p_val_with) or np.isnan(p_val_without):
            return "Could not assess significance change due to insufficient data"

        if p_val_with < 0.05 and p_val_without >= 0.05:
            return "Result is significant WITH suspicious rows, but NOT significant without them"
        elif p_val_with >= 0.05 and p_val_without < 0.05:
            return "Result is NOT significant WITH suspicious rows, but IS significant without them"
        elif p_val_with < 0.05 and p_val_without < 0.05:
            p_change = (
                abs(p_val_with - p_val_without) / max(p_val_with, p_val_without) * 100
            )
            if p_change > 50:
                return f"Result remains significant but p-value changes substantially ({p_change:.1f}% change)"
            else:
                return "Result remains significant with or without suspicious rows"
        else:
            return "Result remains non-significant with or without suspicious rows"

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

    def __enter__(self) -> "ExcelForensics":
        """Context manager entry - extract the Excel file.

        Returns:
            ExcelForensics: Self for use with context manager
        """
        self.temp_dir = tempfile.mkdtemp()
        self.extract_excel()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
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
