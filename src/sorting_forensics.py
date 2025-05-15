#!/usr/bin/env python3
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Set up module logger
logger = logging.getLogger(__name__)


class SortingForensics:
    """Class for detecting sorting anomalies that might indicate data manipulation."""

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

    def check_sorting_anomalies(
        self,
        id_col: str,
        sort_cols: Union[str, List[str]],
        check_dependent_vars: bool = True,
        prioritize_columns: Optional[List[str]] = None,
        prioritize_out_of_order: bool = False,
    ) -> List[Dict[str, Any]]:
        """Check for anomalies in sorting order that might indicate manipulation.

        Specifically designed to detect patterns where IDs are not in sequence within
        condition groups in a way that suggests manual manipulation (e.g., rows moved
        by hand or IDs altered).

        Args:
            id_col: Column name for IDs
            sort_cols: Column name(s) for sorting/grouping (e.g., Condition, Block)
            check_dependent_vars: Whether to analyze numeric columns for suspicious patterns
            prioritize_columns: Optional list of columns to check with higher priority
            prioritize_out_of_order: Whether to prioritize out-of-order ID findings

        Returns:
            List[Dict[str, Any]]: Anomalies found in the data
        """
        if self.df is None:
            return [{"error": "No dataset loaded", "type": "data_error"}]

        # Ensure id_col is in the DataFrame
        if id_col not in self.df.columns:
            return [
                {
                    "error": f"ID column '{id_col}' not found in dataset",
                    "type": "column_error",
                }
            ]

        # Convert sort_cols to a list if it's a string
        if isinstance(sort_cols, str):
            sort_cols = [sort_cols]

        # Validate sort columns
        valid_sort_cols = [col for col in sort_cols if col in self.df.columns]
        if not valid_sort_cols:
            return [
                {"error": "Sort columns not found in dataset", "type": "column_error"}
            ]

        try:
            # Find all numeric columns that could be dependent variables
            numeric_cols = []
            if check_dependent_vars:
                for col in self.df.columns:
                    if col != id_col and col not in sort_cols:
                        if pd.api.types.is_numeric_dtype(self.df[col]):
                            numeric_cols.append(col)

            # Prioritize specified columns
            if prioritize_columns:
                # Move prioritized columns to the front of numeric_cols
                for col in reversed(prioritize_columns):
                    if col in numeric_cols:
                        numeric_cols.remove(col)
                        numeric_cols.insert(0, col)

            # Check for duplicate IDs first
            duplicates = self._check_duplicate_ids(id_col, sort_cols)

            # Then check for out-of-sequence IDs within group clusters
            out_of_sequence = self._check_out_of_sequence_ids(id_col, sort_cols)

            # Combine all findings
            all_anomalies = duplicates + out_of_sequence

            # If no anomalies found, return empty list
            if not all_anomalies:
                return []

            # Filter anomalies to reduce false positives
            filtered_anomalies = self._filter_anomalies(
                all_anomalies, id_col, sort_cols
            )

            # For the filtered anomalies, check if they show statistical anomalies
            if check_dependent_vars and numeric_cols and filtered_anomalies:
                # Analyze suspicious rows for dependent variable anomalies
                row_indices = [
                    int(anomaly["row_index"]) for anomaly in filtered_anomalies
                ]
                self._analyze_dependent_vars(
                    filtered_anomalies, row_indices, numeric_cols
                )

            # Sort anomalies by severity (highest first) and then by row index
            filtered_anomalies.sort(
                key=lambda x: (-x.get("severity", 0), x.get("row_index", 0))
            )

            return filtered_anomalies

        except Exception as e:
            logger.error(f"Error checking for sorting anomalies: {str(e)}")
            return [
                {
                    "error": f"Error in sorting analysis: {str(e)}",
                    "type": "analysis_error",
                }
            ]

    def _check_duplicate_ids(
        self, id_col: str, sort_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Check for duplicate IDs within different conditions.

        Args:
            id_col: Column name for IDs
            sort_cols: Column names for grouping

        Returns:
            List[Dict[str, Any]]: Duplicate ID anomalies
        """
        findings = []

        # Focus on IDs that appear multiple times
        id_counts = self.df[id_col].value_counts()
        repeated_ids = id_counts[id_counts > 1].index.tolist()

        for dup_id in repeated_ids:
            dup_rows = self.df[self.df[id_col] == dup_id]

            # Only flag as abnormal if the ID appears in different condition groups
            condition_count = len(dup_rows.drop_duplicates(subset=sort_cols))

            if condition_count > 1:
                # This is suspicious - same ID in multiple condition groups
                for idx, row in dup_rows.iterrows():
                    # Add information about other instances
                    other_indices = dup_rows.index.tolist()
                    other_indices.remove(idx)

                    # Create a unique finding for each duplicate row
                    finding = {
                        "type": "duplicate_id",
                        "id_value": str(dup_id),
                        "row_index": int(idx),
                        "condition_values": {col: str(row[col]) for col in sort_cols},
                        "other_instances": other_indices,
                        "duplicate_count": len(dup_rows),
                        "different_conditions": condition_count,
                        "severity": 9,  # Duplicates across conditions are highly suspicious
                        "description": f"ID {dup_id} appears in {condition_count} different conditions",
                        "previous_id": int(self.df.iloc[max(0, idx - 1)][id_col])
                        if idx > 0
                        else None,
                    }

                    findings.append(finding)

        return findings

    def _check_out_of_sequence_ids(
        self, id_col: str, sort_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Check for IDs that are out of sequence within condition groups.

        Args:
            id_col: Column name for IDs
            sort_cols: Column names for grouping

        Returns:
            List[Dict[str, Any]]: Out-of-sequence ID anomalies
        """
        findings = []

        # Determine if ID column is numeric
        is_numeric_id = pd.api.types.is_numeric_dtype(self.df[id_col])

        # For each combination of sort columns, check ID sequence
        for name, group in self.df.groupby(sort_cols):
            # Skip groups with just one row
            if len(group) <= 1:
                continue

            # Get expected sequence based on sort order
            sorted_group = group.sort_values(id_col)

            # For numeric IDs, check sequence gaps and distances
            if is_numeric_id:
                # Calculate differences between consecutive IDs
                sorted_ids = sorted_group[id_col].values
                expected_positions = dict(zip(sorted_ids, range(len(sorted_ids))))

                # For each row in original order, check how far it is from expected position
                for idx, row in group.iterrows():
                    id_value = row[id_col]
                    actual_position = group.index.get_loc(idx)
                    expected_position = expected_positions.get(id_value)

                    # Calculate position difference (how many rows out of place)
                    position_diff = (
                        abs(actual_position - expected_position)
                        if expected_position is not None
                        else 0
                    )

                    # If significantly out of place, add a finding
                    # Require a larger position difference to be considered suspicious
                    if position_diff >= 4:
                        # Get surrounding context for IDs
                        surrounding_ids = self._get_surrounding_ids(
                            group, actual_position, id_col, window=2
                        )

                        finding = {
                            "type": "out_of_sequence_id",
                            "id_value": str(id_value),
                            "row_index": int(idx),
                            "condition_values": {
                                col: str(row[col]) for col in sort_cols
                            },
                            "actual_position": actual_position,
                            "expected_position": expected_position,
                            "position_difference": position_diff,
                            "surrounding_ids": surrounding_ids,
                            "severity": min(
                                5 + position_diff, 9
                            ),  # Higher severity for larger differences
                            "description": f"ID {id_value} is {position_diff} positions out of sequence",
                            "previous_id": int(self.df.iloc[max(0, idx - 1)][id_col])
                            if idx > 0
                            else None,
                        }

                        findings.append(finding)

            # For non-numeric IDs, just check if the order matches sorted order
            else:
                # Compare original order with sorted order
                original_ids = group[id_col].tolist()
                sorted_ids = sorted_group[id_col].tolist()

                # If order differs, identify out-of-sequence IDs
                if original_ids != sorted_ids:
                    # Map each ID to its sorted position
                    expected_positions = {
                        id_val: i for i, id_val in enumerate(sorted_ids)
                    }

                    # Check each ID's position
                    for i, (idx, row) in enumerate(group.iterrows()):
                        id_value = row[id_col]
                        expected_position = expected_positions.get(id_value, i)

                        # If position differs by at least 2 places, flag it
                        if abs(i - expected_position) >= 2:
                            # Get surrounding context for IDs
                            surrounding_ids = self._get_surrounding_ids(
                                group, i, id_col, window=2
                            )

                            finding = {
                                "type": "out_of_sequence_id",
                                "id_value": str(id_value),
                                "row_index": int(idx),
                                "condition_values": {
                                    col: str(row[col]) for col in sort_cols
                                },
                                "actual_position": i,
                                "expected_position": expected_position,
                                "position_difference": abs(i - expected_position),
                                "surrounding_ids": surrounding_ids,
                                "severity": 7,  # Non-numeric IDs out of sequence are suspicious
                                "description": f"ID {id_value} is out of sequence",
                                "previous_id": str(original_ids[i - 1])
                                if i > 0
                                else None,
                            }

                            findings.append(finding)

        return findings

    def _get_surrounding_ids(
        self, group: pd.DataFrame, position: int, id_col: str, window: int = 2
    ) -> Dict[str, Any]:
        """Get IDs surrounding a specific position in a group.

        Args:
            group: DataFrame group to analyze
            position: Position in the group
            id_col: Column name for IDs
            window: Number of rows to include before and after

        Returns:
            Dict[str, Any]: Surrounding ID information
        """
        # Get IDs before and after the position
        start = max(0, position - window)
        end = min(len(group), position + window + 1)

        # Get the rows in this window
        window_rows = group.iloc[start:end]

        # Extract position and ID for each row in window
        surrounding = {
            f"position_{i}": {
                "relative_position": i - position,
                "id_value": str(row[id_col]),
            }
            for i, (_, row) in enumerate(window_rows.iterrows(), start=start)
        }

        return surrounding

    def _filter_anomalies(
        self, anomalies: List[Dict[str, Any]], id_col: str, sort_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter anomalies to reduce false positives.

        Args:
            anomalies: List of all anomalies found
            id_col: Column name for IDs
            sort_cols: Column names for grouping

        Returns:
            List[Dict[str, Any]]: Filtered anomalies
        """
        # Always keep duplicate IDs - they're highly suspicious
        duplicates = [a for a in anomalies if a["type"] == "duplicate_id"]

        # For out-of-sequence IDs, filter to keep only clusters
        out_of_sequence = [a for a in anomalies if a["type"] == "out_of_sequence_id"]

        # Group anomalies by condition values
        condition_groups = {}
        for anomaly in out_of_sequence:
            # Create a key from condition values
            key = tuple(
                str(anomaly["condition_values"].get(col, "")) for col in sort_cols
            )

            if key not in condition_groups:
                condition_groups[key] = []

            condition_groups[key].append(anomaly)

        # Require a larger cluster size to reduce false positives
        filtered_out_of_sequence = []
        for condition, group_anomalies in condition_groups.items():
            if (
                len(group_anomalies) >= 5
            ):  # Changed from 3 to 5 to reduce false positives
                # This is a cluster of anomalies - keep them all
                filtered_out_of_sequence.extend(group_anomalies)

                # Increase the severity to reflect the cluster
                for anomaly in group_anomalies:
                    # Add cluster information
                    anomaly["cluster_size"] = len(group_anomalies)
                    anomaly["severity"] = min(anomaly["severity"] + 1, 9)

                    # Enhance the description
                    condition_str = ", ".join(
                        [f"{col}={val}" for col, val in zip(sort_cols, condition)]
                    )
                    anomaly["description"] += (
                        f" (Part of a cluster of {len(group_anomalies)} anomalies in {condition_str})"
                    )
            else:
                # If fewer than 5 anomalies, only keep those with very high position difference
                for anomaly in group_anomalies:
                    if (
                        anomaly.get("position_difference", 0) >= 7
                    ):  # Changed from 4 to 7
                        # This is still significant even if isolated
                        filtered_out_of_sequence.append(anomaly)

        # Combine filtered anomalies
        return duplicates + filtered_out_of_sequence

    def _analyze_dependent_vars(
        self,
        anomalies: List[Dict[str, Any]],
        row_indices: List[int],
        numeric_cols: List[str],
    ) -> None:
        """Analyze dependent variables for suspicious patterns.

        Args:
            anomalies: List of anomalies to analyze
            row_indices: List of row indices for anomalous observations
            numeric_cols: List of numeric columns to analyze
        """
        if self.df is None or not row_indices or not numeric_cols:
            return

        try:
            # Create masks for anomalous and non-anomalous rows
            anomalous_mask = self.df.index.isin(row_indices)

            # For each numeric column, compare anomalous vs. non-anomalous
            for col in numeric_cols:
                if col not in self.df.columns:
                    continue

                # Get values for anomalous and non-anomalous rows
                anomalous_values = self.df.loc[anomalous_mask, col].dropna()
                non_anomalous_values = self.df.loc[~anomalous_mask, col].dropna()

                # Skip if not enough data
                if len(anomalous_values) < 2 or len(non_anomalous_values) < 2:
                    continue

                # Calculate basic statistics
                anomalous_mean = anomalous_values.mean()
                non_anomalous_mean = non_anomalous_values.mean()
                anomalous_std = anomalous_values.std()
                non_anomalous_std = non_anomalous_values.std()

                # Calculate effect size (Cohen's d)
                if non_anomalous_std > 0:
                    effect_size = (
                        anomalous_mean - non_anomalous_mean
                    ) / non_anomalous_std
                else:
                    effect_size = 0

                # Record significant differences
                if abs(effect_size) >= 0.5:  # Medium or larger effect
                    effect_direction = "higher" if effect_size > 0 else "lower"

                    # Update anomalies with this information
                    for anomaly in anomalies:
                        if "dependent_var_effects" not in anomaly:
                            anomaly["dependent_var_effects"] = []

                        anomaly["dependent_var_effects"].append(
                            {
                                "column": col,
                                "effect_size": float(effect_size),
                                "direction": effect_direction,
                                "anomalous_mean": float(anomalous_mean),
                                "non_anomalous_mean": float(non_anomalous_mean),
                                "anomalous_std": float(anomalous_std),
                                "non_anomalous_std": float(non_anomalous_std),
                            }
                        )

                        # Increase severity if large effects found
                        if abs(effect_size) >= 0.8:  # Large effect
                            anomaly["severity"] = min(anomaly["severity"] + 1, 9)

        except Exception as e:
            logger.error(f"Error analyzing dependent variables: {str(e)}")
            # Continue without analysis rather than failing
