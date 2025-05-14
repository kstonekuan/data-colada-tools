#!/usr/bin/env python3
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from anthropic import Anthropic

# Set up module logger
logger = logging.getLogger(__name__)


class DataForensics:
    """Base class for forensic analysis of tabular datasets to detect manipulation."""

    def __init__(self) -> None:
        """Initialize a new DataForensics instance."""
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.file_stats: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}
        self.excel_meta: Dict[str, Any] = {}
        self.column_analysis: Dict[str, Any] = {}

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
                              "suspicious_rows": List[int],  # Specific rows to check
                              "suspect_grouping": str,  # Potential column to check for group-based manipulation
                              "description": str  # Free-text description of suspicions
                           }

        Returns:
            List[Dict[str, Any]]: List of findings
        """
        # Store the filepath for future reference
        self.file_path = filepath
        file_ext = os.path.splitext(filepath)[1].lower()

        # Load the file based on extension
        try:
            # Try different approaches to load the file
            if file_ext == ".xlsx" or file_ext == ".xls":
                # Load Excel file
                self.df = pd.read_excel(filepath)
                logger.info(f"Loaded Excel file: {filepath} ({len(self.df)} rows)")
                self.file_stats["format"] = "excel"
                self.file_stats["extension"] = file_ext
                self.file_stats["rows"] = len(self.df)
                self.file_stats["columns"] = len(self.df.columns)

            elif file_ext == ".csv":
                # Try different encodings for CSV
                try:
                    self.df = pd.read_csv(filepath)
                except UnicodeDecodeError:
                    # Try with different encoding if default fails
                    self.df = pd.read_csv(filepath, encoding="latin1")

                logger.info(f"Loaded CSV file: {filepath} ({len(self.df)} rows)")
                self.file_stats["format"] = "csv"
                self.file_stats["extension"] = file_ext
                self.file_stats["rows"] = len(self.df)
                self.file_stats["columns"] = len(self.df.columns)

            elif file_ext == ".txt" or file_ext == ".tsv":
                # Try different delimiters for text files
                try:
                    # First try tab delimiter
                    self.df = pd.read_csv(filepath, sep="\t")
                except Exception:
                    # Then try autodetect separator
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        first_line = f.readline()
                        if "," in first_line:
                            sep = ","
                        elif ";" in first_line:
                            sep = ";"
                        else:
                            sep = r"\s+"  # Any whitespace

                    self.df = pd.read_csv(filepath, sep=sep)

                logger.info(f"Loaded text file: {filepath} ({len(self.df)} rows)")
                self.file_stats["format"] = "text"
                self.file_stats["extension"] = file_ext
                self.file_stats["rows"] = len(self.df)
                self.file_stats["columns"] = len(self.df.columns)

            else:
                # Unsupported file type
                error_msg = f"Unsupported file type: {file_ext}"
                logger.error(error_msg)
                return [{"error": error_msg, "type": "file_error"}]

            # Basic initial analysis
            findings = []

            # Collect metadata
            self.meta["file_path"] = filepath
            self.meta["file_size"] = os.path.getsize(filepath)
            self.meta["modified_date"] = os.path.getmtime(filepath)
            self.meta["row_count"] = len(self.df)
            self.meta["column_count"] = len(self.df.columns)
            self.meta["has_headers"] = (
                True  # Assuming pandas correctly detected headers
            )
            self.meta["column_names"] = list(self.df.columns)
            self.meta["column_types"] = {
                col: str(self.df[col].dtype) for col in self.df.columns
            }

            # Return findings
            return findings

        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg, "type": "loading_error"}]

    def check_duplicate_ids(self, id_col: str) -> List[Dict[str, Any]]:
        """Check for duplicate IDs in the dataset.

        Args:
            id_col: Column name for IDs

        Returns:
            List[Dict[str, Any]]: List of findings about duplicate IDs
        """
        if self.df is None:
            return [{"error": "No dataset loaded", "type": "data_error"}]

        if id_col not in self.df.columns:
            return [
                {
                    "error": f"ID column '{id_col}' not found in dataset",
                    "type": "column_error",
                }
            ]

        try:
            # Check for duplicate IDs
            duplicated = self.df[self.df.duplicated(id_col, keep=False)]
            if len(duplicated) == 0:
                return []

            # Create findings for each set of duplicates
            findings = []
            for dup_id in duplicated[id_col].unique():
                dup_rows = duplicated[duplicated[id_col] == dup_id]
                findings.append(
                    {
                        "id_value": str(dup_id),
                        "count": len(dup_rows),
                        "rows": dup_rows.index.tolist(),
                        "severity": 9,  # Duplicate IDs are serious issues
                        "description": f"ID {dup_id} appears {len(dup_rows)} times",
                        "type": "duplicate_id",
                    }
                )

            return findings

        except Exception as e:
            error_msg = f"Error checking duplicate IDs: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg, "type": "analysis_error"}]

    def analyze_column_unique_values(
        self, client: Anthropic, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze unique values in each column using Claude.

        Args:
            client: Claude API client
            columns: Optional list of columns to analyze. If None, analyze all columns.

        Returns:
            Dict[str, Any]: Dictionary mapping column names to analysis results
        """
        if self.df is None:
            return {"error": "No dataset loaded"}

        results = {}

        # Determine which columns to analyze
        if columns is None:
            columns_to_analyze = self.df.columns
        else:
            # Filter out columns that don't exist in the dataset
            columns_to_analyze = [col for col in columns if col in self.df.columns]
            if len(columns_to_analyze) != len(columns):
                missing = set(columns) - set(columns_to_analyze)
                logger.warning(f"Columns not found in dataset: {missing}")

        for column in columns_to_analyze:
            # Basic statistics
            unique_values = self.df[column].unique()
            unique_count = len(unique_values)
            non_null_count = self.df[column].count()
            is_numeric = pd.api.types.is_numeric_dtype(self.df[column])

            # Create column info
            column_info = {
                "unique_count": unique_count,
                "total_count": len(self.df),
                "non_null_count": non_null_count,
                "is_numeric": is_numeric,
                "dtype": str(self.df[column].dtype),
            }

            # Calculate uniqueness ratio
            if non_null_count > 0:
                column_info["uniqueness_ratio"] = unique_count / non_null_count

            # For numeric columns, add more statistics
            if is_numeric:
                column_info["min"] = float(self.df[column].min())
                column_info["max"] = float(self.df[column].max())
                column_info["mean"] = float(self.df[column].mean())
                column_info["median"] = float(self.df[column].median())
                column_info["std"] = float(self.df[column].std())

            # Sample of unique values (up to 20)
            if unique_count <= 20:
                # Convert all values to strings to ensure JSON serializability
                column_info["unique_values"] = [str(v) for v in unique_values]
            else:
                # Sample values
                sample_size = min(20, unique_count)
                sampled_values = np.random.choice(
                    unique_values, sample_size, replace=False
                )
                column_info["sampled_values"] = [str(v) for v in sampled_values]
                column_info["is_sampled"] = True

            # Analyze with Claude if the column has appropriate characteristics
            if (
                5 <= unique_count <= 1000
            ):  # Only analyze columns with reasonable unique counts
                prompt = self._generate_column_analysis_prompt(column, column_info)

                try:
                    response = client.messages.create(
                        model="claude-3-7-sonnet-latest",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": prompt}],
                    )

                    # Extract response content
                    content = response.content[0].text

                    # Try to parse JSON from the response
                    try:
                        # Find JSON block
                        json_start = content.find("{")
                        json_end = content.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = content[json_start:json_end]
                            claude_analysis = json.loads(json_str)
                            column_info.update(claude_analysis)
                    except json.JSONDecodeError:
                        # If not valid JSON, use the raw response
                        column_info["claude_analysis"] = content

                except Exception as e:
                    column_info["claude_error"] = str(e)

            # Store results
            results[column] = column_info

        return results

    def _generate_column_analysis_prompt(
        self, column_name: str, column_info: Dict[str, Any]
    ) -> str:
        """Generate a prompt for Claude to analyze column values.

        Args:
            column_name: The name of the column being analyzed
            column_info: Dictionary with information about the column

        Returns:
            str: Prompt for Claude
        """
        # Extract values to display
        if "unique_values" in column_info:
            values_sample = column_info["unique_values"]
            is_complete = True
        else:
            values_sample = column_info.get("sampled_values", [])
            is_complete = False

        sample_str = "\n".join([f"- {v}" for v in values_sample])

        # Create prompt template
        prompt = f"""Analyze this column from a research dataset for potential data manipulation or unusual patterns:

Column Name: {column_name}
Data Type: {column_info["dtype"]}
Number of Unique Values: {column_info["unique_count"]} out of {column_info["total_count"]} rows
{"Complete list" if is_complete else "Sample"} of unique values:
{sample_str}

For numeric columns:
- Look for suspicious patterns like too many round numbers
- Check for unusual clusters or gaps
- Note any anomalous spacing between values
- Identify any unusual sequences or patterns

For categorical columns:
- Look for unusual coding patterns
- Check for inconsistent category names (slight misspellings)
- Identify any unexpected categories

Return your analysis as JSON with these fields:
{{
  "suspicion_rating": 1-10 scale where 10 is highly suspicious,
  "patterns_identified": ["list", "of", "patterns"],
  "explanation": "Brief explanation of your findings"
}}

IMPORTANT: Only return valid JSON. Do not include any other text or explanation outside the JSON object."""

        return prompt

    def compare_with_without_suspicious_rows(
        self,
        suspicious_rows: List[int],
        group_col: str,
        outcome_cols: List[str],
    ) -> Dict[str, Any]:
        """Compare statistical tests with and without suspicious observations.

        Args:
            suspicious_rows: List of suspicious row indices
            group_col: Column name for grouping
            outcome_cols: List of outcome column names

        Returns:
            Dict[str, Any]: Comparison results
        """
        if self.df is None:
            return {"error": "No dataset loaded"}

        if group_col not in self.df.columns:
            return {"error": f"Group column '{group_col}' not found in dataset"}

        # Validate outcome columns
        valid_outcome_cols = [col for col in outcome_cols if col in self.df.columns]
        if not valid_outcome_cols:
            return {"error": "No valid outcome columns found"}

        try:
            # Create a mask for suspicious rows
            suspicious_mask = self.df.index.isin(suspicious_rows)

            # Filter original data and suspicious rows
            df_without_suspicious = self.df[~suspicious_mask].copy()
            df_suspicious_only = self.df[suspicious_mask].copy()

            # Get unique groups
            groups = self.df[group_col].unique()
            if len(groups) < 2:
                return {"error": "Need at least two groups for comparison"}

            results = {
                "full_dataset": {},
                "without_suspicious": {},
                "suspicious_only": {},
            }

            # Get descriptive statistics and p-values for each outcome
            for outcome in valid_outcome_cols:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(self.df[outcome]):
                    continue

                # Get t-test p-values for each pair of groups
                group_pairs = []
                for i, g1 in enumerate(groups):
                    for g2 in groups[i + 1 :]:
                        pair = f"{g1} vs {g2}"

                        # Full dataset
                        g1_data_full = self.df[self.df[group_col] == g1][
                            outcome
                        ].dropna()
                        g2_data_full = self.df[self.df[group_col] == g2][
                            outcome
                        ].dropna()

                        # Without suspicious
                        g1_data_without = df_without_suspicious[
                            df_without_suspicious[group_col] == g1
                        ][outcome].dropna()
                        g2_data_without = df_without_suspicious[
                            df_without_suspicious[group_col] == g2
                        ][outcome].dropna()

                        # Only suspicious (if any in each group)
                        g1_data_suspicious = df_suspicious_only[
                            df_suspicious_only[group_col] == g1
                        ][outcome].dropna()
                        g2_data_suspicious = df_suspicious_only[
                            df_suspicious_only[group_col] == g2
                        ][outcome].dropna()

                        # Calculate statistics for each dataset
                        pair_results = {
                            "full_dataset": self._calculate_comparison_stats(
                                g1_data_full, g2_data_full, g1, g2
                            ),
                            "without_suspicious": self._calculate_comparison_stats(
                                g1_data_without, g2_data_without, g1, g2
                            ),
                        }

                        # Only add suspicious stats if there are enough data points
                        if len(g1_data_suspicious) > 1 and len(g2_data_suspicious) > 1:
                            pair_results["suspicious_only"] = (
                                self._calculate_comparison_stats(
                                    g1_data_suspicious, g2_data_suspicious, g1, g2
                                )
                            )

                        group_pairs.append({"pair": pair, "results": pair_results})

                results[outcome] = {
                    "group_pairs": group_pairs,
                    "descriptive_stats": {
                        "full_dataset": self._calculate_descriptive_stats(
                            self.df, outcome, group_col
                        ),
                        "without_suspicious": self._calculate_descriptive_stats(
                            df_without_suspicious, outcome, group_col
                        ),
                        "suspicious_only": self._calculate_descriptive_stats(
                            df_suspicious_only, outcome, group_col
                        ),
                    },
                }

            return results

        except Exception as e:
            error_msg = f"Error comparing with/without suspicious rows: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    def _calculate_comparison_stats(
        self,
        group1_data: pd.Series,
        group2_data: pd.Series,
        group1_name: Any,
        group2_name: Any,
    ) -> Dict[str, Any]:
        """Calculate comparison statistics between two groups.

        Args:
            group1_data: Data for first group
            group2_data: Data for second group
            group1_name: Name of first group
            group2_name: Name of second group

        Returns:
            Dict[str, Any]: Comparison statistics
        """
        # Skip if not enough data
        if len(group1_data) < 2 or len(group2_data) < 2:
            return {
                "error": "Not enough data points for comparison",
                "n1": len(group1_data),
                "n2": len(group2_data),
            }

        try:
            # Calculate t-test
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

            # Calculate Cohen's d effect size
            mean1, mean2 = group1_data.mean(), group2_data.mean()
            std1, std2 = group1_data.std(), group2_data.std()
            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((len(group1_data) - 1) * std1**2 + (len(group2_data) - 1) * std2**2)
                / (len(group1_data) + len(group2_data) - 2)
            )
            cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

            return {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "mean_difference": float(mean2 - mean1),
                "cohens_d": float(cohens_d),
                "n1": int(len(group1_data)),
                "n2": int(len(group2_data)),
                "effect_size_interpretation": self._interpret_effect_size(cohens_d),
                "significant": bool(p_value < 0.05),
            }

        except Exception as e:
            return {"error": str(e), "n1": len(group1_data), "n2": len(group2_data)}

    def _calculate_descriptive_stats(
        self, df: pd.DataFrame, outcome_col: str, group_col: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for each group.

        Args:
            df: DataFrame with data
            outcome_col: Outcome column name
            group_col: Grouping column name

        Returns:
            Dict[str, Dict[str, float]]: Descriptive statistics by group
        """
        results = {}

        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][outcome_col].dropna()

            if len(group_data) > 0:
                results[str(group)] = {
                    "mean": float(group_data.mean()),
                    "median": float(group_data.median()),
                    "std": float(group_data.std()) if len(group_data) > 1 else 0,
                    "min": float(group_data.min()),
                    "max": float(group_data.max()),
                    "n": int(len(group_data)),
                }

        return results

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            cohens_d: Cohen's d value

        Returns:
            str: Effect size interpretation
        """
        cohens_d = abs(cohens_d)  # Use absolute value for interpretation

        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"

    def analyze_suspicious_observations(
        self,
        suspicious_rows: List[int],
        group_col: str,
        outcome_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Analyze suspicious observations to see if they show a strong effect.

        Args:
            suspicious_rows: List of suspicious row indices
            group_col: Column name for grouping
            outcome_cols: List of outcome column names

        Returns:
            List[Dict[str, Any]]: Analysis of suspicious observations
        """
        if self.df is None:
            return [{"error": "No dataset loaded", "type": "data_error"}]

        if group_col not in self.df.columns:
            return [
                {
                    "error": f"Group column '{group_col}' not found in dataset",
                    "type": "column_error",
                }
            ]

        # Validate outcome columns
        valid_outcome_cols = [col for col in outcome_cols if col in self.df.columns]
        if not valid_outcome_cols:
            return [{"error": "No valid outcome columns found", "type": "column_error"}]

        try:
            # Create a mask for suspicious rows
            suspicious_mask = self.df.index.isin(suspicious_rows)

            # Get suspicious rows
            suspicious_df = self.df.loc[suspicious_mask]

            # Analyze each outcome variable
            findings = []

            for outcome in valid_outcome_cols:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(self.df[outcome]):
                    continue

                # Get group means for each condition
                group_means = self.df.groupby(group_col)[outcome].mean()

                # For each suspicious row, compare with its group mean
                for idx, row in suspicious_df.iterrows():
                    # Get row's group
                    row_group = row[group_col]

                    # Skip if group or outcome value is missing
                    if pd.isna(row_group) or pd.isna(row[outcome]):
                        continue

                    # Get group mean
                    group_mean = group_means.get(row_group)

                    # Skip if no group mean available
                    if pd.isna(group_mean):
                        continue

                    # Calculate how far the row's value is from its group mean
                    distance = row[outcome] - group_mean

                    # Calculate standard deviation for the group (excluding this row)
                    group_data = self.df[
                        (self.df[group_col] == row_group) & (~self.df.index.isin([idx]))
                    ][outcome]
                    group_std = group_data.std()

                    # Calculate z-score (if possible)
                    if not pd.isna(group_std) and group_std > 0:
                        z_score = distance / group_std

                        # Only add finding if z-score is significant
                        if abs(z_score) > 1.5:  # Threshold for "unusual" values
                            findings.append(
                                {
                                    "row_index": int(idx),
                                    "group": str(row_group),
                                    "outcome": outcome,
                                    "value": float(row[outcome]),
                                    "group_mean": float(group_mean),
                                    "difference": float(distance),
                                    "z_score": float(z_score),
                                    "effect_direction": "stronger"
                                    if (distance > 0)
                                    else "weaker",
                                    "magnitude": self._interpret_z_score(z_score),
                                    "type": "suspicious_effect",
                                }
                            )

            return findings

        except Exception as e:
            error_msg = f"Error analyzing suspicious observations: {str(e)}"
            logger.error(error_msg)
            return [{"error": error_msg, "type": "analysis_error"}]

    def _interpret_z_score(self, z_score: float) -> str:
        """Interpret the magnitude of a z-score.

        Args:
            z_score: Z-score value

        Returns:
            str: Interpretation
        """
        z_abs = abs(z_score)

        if z_abs < 1.5:
            return "typical"
        elif z_abs < 2:
            return "somewhat unusual"
        elif z_abs < 3:
            return "unusual"
        elif z_abs < 4:
            return "very unusual"
        else:
            return "extreme"

    def segment_and_analyze_with_claude(
        self, client: Anthropic, user_suspicions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Segment the dataset and analyze each segment with Claude for anomalies.

        Args:
            client: Claude API client
            user_suspicions: Optional dictionary with user-specified suspicions to check

        Returns:
            List[Dict[str, Any]]: List of findings for each segment
        """
        if self.df is None:
            return [{"error": "No dataset loaded"}]

        # Configure segmentation
        max_rows_per_chunk = 500  # Maximum rows to include in each chunk
        max_chunks = (
            10  # Maximum number of chunks to analyze to prevent API cost overrun
        )

        # Determine total rows
        total_rows = len(self.df)

        # Calculate chunk size based on total rows (smaller datasets get larger chunks)
        if total_rows <= 1000:
            chunk_size = min(max_rows_per_chunk, total_rows)
        else:
            chunk_size = min(max_rows_per_chunk, total_rows // 10)

        # Apply minimum and maximum constraints
        chunk_size = max(50, min(chunk_size, max_rows_per_chunk))

        # Calculate number of chunks
        num_chunks = min(max_chunks, (total_rows + chunk_size - 1) // chunk_size)

        # Prepare chunks
        logger.info(
            f"Segmenting dataset into {num_chunks} chunks of ~{chunk_size} rows each"
        )
        results = []

        # Build suspicion context for the prompt
        suspicion_context = ""
        if user_suspicions:
            suspicion_parts = []

            if "focus_columns" in user_suspicions and user_suspicions["focus_columns"]:
                columns = ", ".join(user_suspicions["focus_columns"])
                suspicion_parts.append(f"Pay special attention to columns: {columns}")

            if (
                "potential_issues" in user_suspicions
                and user_suspicions["potential_issues"]
            ):
                issues = ", ".join(user_suspicions["potential_issues"])
                suspicion_parts.append(f"Look for these specific issues: {issues}")

            if "description" in user_suspicions and user_suspicions["description"]:
                suspicion_parts.append(f"Context: {user_suspicions['description']}")

            if suspicion_parts:
                suspicion_context = "The user suspect issues in this data: " + " ".join(
                    suspicion_parts
                )

        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * chunk_size
            end_row = min(start_row + chunk_size, total_rows)

            logger.info(
                f"Analyzing chunk {chunk_idx + 1}/{num_chunks} (rows {start_row}-{end_row})"
            )

            # Get the chunk
            chunk_df = self.df.iloc[start_row:end_row].copy()

            # Generate prompt for Claude
            prompt = self._generate_segment_analysis_prompt(
                chunk_df, chunk_idx, suspicion_context
            )

            try:
                # Call Claude with the prompt
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )

                # Process the response
                content = response.content[0].text

                # Try to extract JSON
                result = self._extract_analysis_json(content)

                # Add metadata to the result
                result.update(
                    {
                        "chunk": chunk_idx + 1,
                        "rows": f"{start_row}-{end_row}",
                        "row_indices": list(range(start_row, end_row)),
                    }
                )

                # Add result to the list
                results.append(result)

            except Exception as e:
                # Log the error and add an error result
                logger.error(f"Error analyzing chunk {chunk_idx + 1}: {str(e)}")
                results.append(
                    {
                        "error": str(e),
                        "chunk": chunk_idx + 1,
                        "rows": f"{start_row}-{end_row}",
                        "row_indices": list(range(start_row, end_row)),
                        "anomalies_detected": False,
                        "traceback": traceback.format_exc(),
                    }
                )

        return results

    def _generate_segment_analysis_prompt(
        self, df: pd.DataFrame, chunk_idx: int, suspicion_context: str = ""
    ) -> str:
        """Generate a prompt for Claude to analyze a data segment.

        Args:
            df: DataFrame segment to analyze
            chunk_idx: Index of the current chunk
            suspicion_context: Optional context about user suspicions

        Returns:
            str: Prompt for Claude
        """
        # Convert DataFrame to string representation for the prompt
        data_str = df.to_string(index=True, max_rows=500, max_cols=20)

        # Get column types as a string
        column_types = "\n".join([f"{col}: {df[col].dtype}" for col in df.columns])

        # Calculate some basic statistics
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        stats_str = ""
        if numeric_columns:
            stats = df[numeric_columns].describe().to_string()
            stats_str = f"\nStatistics for numeric columns:\n{stats}\n"

        # Build the prompt
        prompt = f"""Analyze this segment of research data to detect potential data manipulation.

Data Segment (Chunk {chunk_idx + 1}):
{data_str}

Column types:
{column_types}
{stats_str}

{suspicion_context}

Data manipulation in research often shows these patterns:
1. Sorting anomalies - IDs out of sequence within experimental conditions
2. Duplicate IDs or near-duplicates with small changes
3. Statistical anomalies like non-random terminal digits
4. Unusual distributions or clusters of values
5. Values that too perfectly match hypotheses
6. Unrealistic patterns in numeric data
7. Too many round numbers
8. Excessive similarity or regularity in random values
9. Observations that appear to have been moved between conditions

Return your analysis as JSON with these fields (in valid JSON format):
{{
  "anomalies_detected": true/false,
  "confidence": 1-10 scale where 10 is very confident in findings,
  "explanation": "Summary of what you found or didn't find",
  "findings": [
    {{
      "type": "anomaly type (e.g., sorting_issue, statistical_anomaly)",
      "description": "Detailed description of the issue",
      "severity": 1-10 scale where 10 is very serious,
      "row_indices": [list of row indices involved],
      "columns_involved": [list of column names]
    }}
  ]
}}

IMPORTANT: Only return valid JSON. Do not include any other text or explanation outside the JSON object."""

        return prompt

    def _extract_analysis_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response text.

        Args:
            text: Text response from Claude

        Returns:
            Dict[str, Any]: Extracted JSON, or error object
        """
        try:
            # Try to find JSON block
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                # Fallback: try to parse the whole text as JSON
                return json.loads(text)

        except json.JSONDecodeError:
            # Return error with the raw text
            return {
                "error": "Failed to parse JSON from Claude's response",
                "raw_response": text[:1000] + "..." if len(text) > 1000 else text,
                "anomalies_detected": False,
            }
