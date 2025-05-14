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

import numpy as np
import pandas as pd

# Set up module logger
logger = logging.getLogger(__name__)

class DataForensics:
    """Class for forensic analysis of tabular datasets to detect manipulation."""

    def __init__(self) -> None:
        """Initialize a new DataForensics instance."""
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.file_stats: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}
        self.excel_meta: Dict[str, Any] = {}
        self.column_analysis: Dict[str, Any] = {}
    
    def analyze_dataset(self, file_path: str) -> None:
        """Load and analyze a dataset for initial statistics and metadata.
        
        Args:
            file_path: Path to the dataset file
            
        Raises:
            ValueError: If the file format is not supported
        """
        self.file_path = file_path
        
        # Get file statistics
        try:
            file_stat = os.stat(file_path)
            self.file_stats = {
                "size_bytes": file_stat.st_size,
                "modified_time": file_stat.st_mtime,
                "created_time": file_stat.st_ctime,
            }
        except Exception as e:
            logger.error(f"Error getting file stats: {str(e)}")
            self.file_stats = {"error": str(e)}
        
        # Determine file type and read data
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Load data based on file extension
        if file_ext == ".csv":
            self.df = pd.read_csv(file_path)
            self.meta["format"] = "CSV"
            
        elif file_ext == ".xlsx" or file_ext == ".xls":
            self.df = pd.read_excel(file_path)
            self.meta["format"] = "Excel"
            
            # Extract Excel metadata if it's an XLSX file
            if file_ext == ".xlsx":
                try:
                    self.excel_meta = self.extract_excel_metadata(file_path)
                except Exception as e:
                    logger.error(f"Error extracting Excel metadata: {str(e)}")
                    self.excel_meta = {"error": str(e)}
                    
        elif file_ext == ".txt" or file_ext == ".tsv":
            self.df = pd.read_csv(file_path, sep="\t")
            self.meta["format"] = "Text/TSV"
            
        elif file_ext == ".sav":
            # Import pyreadstat lazily to improve startup time
            from src.lazy_imports import get_pyreadstat
            pyreadstat = get_pyreadstat()
            
            # Read SPSS file
            self.df, meta = pyreadstat.read_sav(file_path)
            self.meta = {
                "format": "SPSS",
                "label": meta.file_label,
                "num_cases": meta.number_cases,
                "num_vars": meta.number_of_variables,
                "var_labels": meta.variable_labels,
                "value_labels": meta.value_labels,
            }
            
        elif file_ext == ".dta":
            # Read Stata file
            self.df = pd.read_stata(file_path)
            self.meta["format"] = "Stata"
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        # Basic dataset analysis
        if self.df is not None:
            self.meta.update({
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "column_names": list(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                "missing_values": {col: int(self.df[col].isna().sum()) for col in self.df.columns},
                "unique_counts": {col: int(self.df[col].nunique()) for col in self.df.columns},
            })
            
        logger.info(f"Loaded dataset with {self.meta.get('rows', 0)} rows and {self.meta.get('columns', 0)} columns")
    
    def extract_excel_metadata(self, excel_path: str) -> Dict[str, Any]:
        """Extract metadata from an Excel file.
        
        Args:
            excel_path: Path to the Excel file
            
        Returns:
            Dict with metadata extraction results
        """
        # Use the ExcelForensics class for detailed examination
        with ExcelForensics(excel_path) as ef:
            # Basic Excel metadata
            metadata = {
                "has_calc_chain": len(ef.calc_chain) > 0,
                "calc_chain_entries": len(ef.calc_chain),
            }
            
            # Additional sheet info if available
            try:
                xl = pd.ExcelFile(excel_path)
                metadata["sheets"] = xl.sheet_names
                metadata["sheet_count"] = len(xl.sheet_names)
                
                # Get row counts for each sheet
                sheet_rows = {}
                for sheet in xl.sheet_names:
                    df = pd.read_excel(excel_path, sheet_name=sheet)
                    sheet_rows[sheet] = len(df)
                metadata["sheet_rows"] = sheet_rows
                
            except Exception as e:
                logger.error(f"Error extracting sheet information: {str(e)}")
                metadata["sheet_error"] = str(e)
                
            return metadata
    
    def analyze_column_unique_values(
        self, client: Any, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze unique values in each column using Claude.
        
        This can detect suspicious patterns in the data distribution.
        
        Args:
            client: Claude API client
            columns: Optional list of columns to analyze. If None, analyze all columns.
            
        Returns:
            Dict mapping column names to analysis results
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")
            
        # Determine which columns to analyze
        cols_to_analyze = columns if columns else list(self.df.columns)
        
        # Filter out non-existent columns
        valid_cols = [col for col in cols_to_analyze if col in self.df.columns]
        if len(valid_cols) < len(cols_to_analyze):
            logger.warning(f"Some columns not found in dataset. Analyzing only: {valid_cols}")
        
        # Results storage
        results = {}
        
        # For each column
        for col in valid_cols:
            logger.info(f"Analyzing unique values for column: {col}")
            
            # Get value counts
            values = self.df[col].value_counts().reset_index()
            values.columns = ['value', 'count']
            
            # Handle different data types
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # For numeric columns, add descriptive statistics
                stats = self.df[col].describe()
                
                # Get all unique values as a sorted list (handle potential non-sortable elements)
                try:
                    unique_values = sorted(self.df[col].dropna().unique())
                    
                    # Convert numpy types to Python types for JSON serialization
                    unique_values = [
                        float(val) if isinstance(val, np.floating) else
                        int(val) if isinstance(val, np.integer) else val
                        for val in unique_values
                    ]
                except:
                    # If values can't be sorted, just use them as is
                    unique_values = list(self.df[col].dropna().unique())
                
                # Handle large number of unique values
                if len(unique_values) > 50:
                    sample_size = min(50, len(unique_values))
                    # Take some from start, middle and end
                    start = unique_values[:sample_size//3]
                    middle_start = len(unique_values)//2 - sample_size//6
                    middle = unique_values[middle_start:middle_start + sample_size//3]
                    end = unique_values[-sample_size//3:]
                    
                    sampled_values = start + middle + end
                    value_info = f"{len(unique_values)} unique values (showing {len(sampled_values)} samples): {sampled_values}"
                else:
                    value_info = f"{len(unique_values)} unique values: {unique_values}"
                
                # Numeric column prompt tailored for statistical anomalies
                prompt = f"""Analyze this column's unique values for potential data manipulation:

Column: {col}
Data type: {self.df[col].dtype}
Number of rows: {len(self.df)}
Unique values: {len(unique_values)}
Statistics:
- Mean: {stats['mean']}
- Std Dev: {stats['std']}
- Min: {stats['min']}
- 25th percentile: {stats['25%']}
- Median: {stats['50%']}
- 75th percentile: {stats['75%']}
- Max: {stats['max']}

Value distribution (sorted by frequency):
{values.head(20).to_string(index=False)}

{value_info}

Analyze this data and identify any suspicious patterns that could indicate manipulation.
Look for:
1. Terminal digit anomalies (non-random distribution of final digits)
2. Unusual clustering of values
3. Gaps in otherwise continuous distributions
4. Too many inliers (values suspiciously close to means)
5. Perfect sequences or progressions
6. Fabricated-looking patterns

Respond with JSON in this format:
{{
  "suspicion_rating": 1-10,
  "explanations": ["Reason 1", "Reason 2", ...],
  "unusual_patterns": ["Pattern 1", "Pattern 2", ...],
  "statistical_anomalies": ["Anomaly 1", "Anomaly 2", ...],
  "manipulation_indicators": ["Indicator 1", "Indicator 2", ...]
}}

The suspicion_rating should be 1-3 for normal data, 4-6 for somewhat unusual patterns, and 7-10 for likely manipulation."""

            else:
                # For categorical/text columns
                most_common = values.head(10).to_dict(orient='records')
                total_counts = values['count'].sum()
                
                # Format the most common values 
                common_formatted = []
                for record in most_common:
                    value = record['value']
                    count = record['count']
                    percent = (count / total_counts) * 100
                    common_formatted.append(f"{value}: {count} ({percent:.1f}%)")
                
                # Non-numeric column prompt tailored for categorical manipulation
                prompt = f"""Analyze this column's unique values for potential data manipulation:

Column: {col}
Data type: {self.df[col].dtype}
Number of rows: {len(self.df)}
Unique value count: {self.df[col].nunique()}

Most common values:
{', '.join(common_formatted)}

Analyze this data and identify any suspicious patterns that could indicate manipulation.
Look for:
1. Unnaturally uniform or non-uniform distributions
2. Suspicious patterns in text data (like similar prefixes/suffixes with variations)
3. Evidence of artificial categorization
4. Anomalous frequency distributions compared to natural data
5. Fabricated-looking patterns

Respond with JSON in this format:
{{
  "suspicion_rating": 1-10,
  "explanations": ["Reason 1", "Reason 2", ...],
  "unusual_patterns": ["Pattern 1", "Pattern 2", ...],
  "categorical_anomalies": ["Anomaly 1", "Anomaly 2", ...],
  "manipulation_indicators": ["Indicator 1", "Indicator 2", ...]
}}

The suspicion_rating should be 1-3 for normal data, 4-6 for somewhat unusual patterns, and 7-10 for likely manipulation."""
            
            # Call Claude
            try:
                logger.info("Sending prompt to Claude for analysis")
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                # Extract JSON response
                content = response.content[0].text
                
                # Try to find and parse JSON
                import re
                json_pattern = r"\{.*\}"
                matches = re.findall(json_pattern, content, re.DOTALL)
                
                if matches:
                    # Use the longest match as it's likely the full JSON
                    json_str = max(matches, key=len)
                    analysis = json.loads(json_str)
                    
                    # Add metadata
                    analysis["unique_count"] = self.df[col].nunique()
                    analysis["total_rows"] = len(self.df)
                    analysis["null_count"] = int(self.df[col].isna().sum())
                    
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        # Add statistical measures for numeric columns
                        analysis["statistics"] = {
                            "mean": float(stats["mean"]),
                            "std": float(stats["std"]),
                            "min": float(stats["min"]),
                            "25%": float(stats["25%"]),
                            "50%": float(stats["50%"]),
                            "75%": float(stats["75%"]),
                            "max": float(stats["max"]),
                        }
                    
                    results[col] = analysis
                else:
                    logger.warning(f"Could not extract JSON from Claude's response for column {col}")
                    results[col] = {
                        "error": "Could not parse Claude response",
                        "suspicion_rating": 1,
                        "unique_count": self.df[col].nunique(),
                    }
            except Exception as e:
                logger.error(f"Error analyzing column {col}: {str(e)}")
                results[col] = {
                    "error": str(e),
                    "suspicion_rating": 1,
                    "unique_count": self.df[col].nunique(),
                }
        
        # Save the results to the object for future reference
        self.column_analysis = results
        
        return results

    def check_duplicate_ids(self, id_col: str) -> List[Dict[str, Any]]:
        """Check for duplicate IDs that could indicate manipulation.
        
        Args:
            id_col: Column name for IDs
            
        Returns:
            List of duplicate ID findings
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")
            
        if id_col not in self.df.columns:
            logger.error(f"ID column '{id_col}' not found in the dataframe")
            return []
        
        # Find duplicate IDs
        duplicate_mask = self.df[id_col].duplicated(keep=False)
        duplicates = self.df[duplicate_mask].copy()
        
        if len(duplicates) == 0:
            logger.info(f"No duplicate IDs found in column {id_col}")
            return []
            
        # Group by ID to analyze each set of duplicates
        findings = []
        
        for id_value, group in duplicates.groupby(id_col):
            # Skip if there's only one row (shouldn't happen with our filter)
            if len(group) <= 1:
                continue
                
            # Convert value for JSON serialization
            if isinstance(id_value, (np.integer, np.floating)):
                id_value = int(id_value) if isinstance(id_value, np.integer) else float(id_value)
                
            # Get the row indices for these duplicates
            row_indices = group.index.tolist()
            
            finding = {
                "id": id_value,
                "count": len(group),
                "row_indices": [int(idx) for idx in row_indices],
            }
            
            # Find differences between the duplicate rows
            if len(group) > 1:
                differences = {}
                first_row = group.iloc[0]
                
                for col in group.columns:
                    if col == id_col:
                        continue
                        
                    # Check if values are different
                    if not group[col].equals(first_row[col]):
                        # Store the distinct values for this column
                        distinct_values = group[col].unique()
                        
                        # Convert numpy types for JSON serialization
                        if any(isinstance(val, (np.integer, np.floating, np.bool_)) for val in distinct_values):
                            distinct_values = [
                                int(val) if isinstance(val, np.integer)
                                else float(val) if isinstance(val, np.floating)
                                else bool(val) if isinstance(val, np.bool_)
                                else val
                                for val in distinct_values
                            ]
                        
                        differences[col] = list(distinct_values)
                
                finding["differences"] = differences
                        
            findings.append(finding)
        
        logger.info(f"Found {len(findings)} sets of duplicate IDs in column {id_col}")
        return findings

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
            sort_cols: Column name(s) for sorting/grouping
            check_dependent_vars: Whether to look for dependent variables that might be sorted
            prioritize_columns: Specific columns to prioritize in the out-of-order analysis
            prioritize_out_of_order: Whether to use more sensitive thresholds for out-of-order detection

        Returns:
            List of sorting anomalies found with enhanced out-of-order analysis
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")

        anomalies = []
        logger.info(f"Starting sorting anomaly detection for ID column '{id_col}'")

        # Ensure ID column exists
        if id_col not in self.df.columns:
            logger.error(f"ID column '{id_col}' not found in dataframe")
            return []

        # Identify potential dependent variables (numeric columns that aren't used for sorting)
        dependent_vars = []
        prioritized_vars = []

        # Convert sort_cols to list if it's a string
        if isinstance(sort_cols, str):
            sort_cols_list = [sort_cols]
        else:
            sort_cols_list = sort_cols.copy() if sort_cols else []

        # Ensure sort columns exist
        valid_sort_cols = [col for col in sort_cols_list if col in self.df.columns]
        if not valid_sort_cols:
            logger.error(f"None of the sort columns {sort_cols_list} found in dataframe")
            return []
        
        if len(valid_sort_cols) < len(sort_cols_list):
            logger.warning(f"Some sort columns not found. Using only: {valid_sort_cols}")
        
        sort_cols_list = valid_sort_cols

        if check_dependent_vars:
            # Find numeric columns using pandas
            numeric_cols = list(self.df.select_dtypes(include=['number']).columns)
            logger.info(f"Found {len(numeric_cols)} numeric columns")

            # Get all potential dependent variables
            dependent_vars = [
                col
                for col in numeric_cols
                if col != id_col and col not in sort_cols_list
            ]
            logger.info(f"Found {len(dependent_vars)} potential dependent variables")

            # If user provided specific columns to prioritize, check those first
            if prioritize_columns:
                # Filter to ensure we only include columns that exist and are numeric
                available_columns = list(self.df.columns)
                prioritized_vars = [
                    col
                    for col in prioritize_columns
                    if col in available_columns
                    and col in numeric_cols
                    and col != id_col
                    and col not in sort_cols_list
                ]
                logger.info(f"Will prioritize {len(prioritized_vars)} columns: {prioritized_vars}")

        # Process each sort column (condition column)
        for col in sort_cols_list:
            logger.info(f"Analyzing sort column: {col}")
            
            # Get unique values in the sorting column (e.g., different conditions)
            unique_vals = self.df[col].dropna().unique()
            logger.info(f"Found {len(unique_vals)} unique values in sort column {col}")

            # For each unique value in the sorting column (e.g., each condition)
            for val in unique_vals:
                # Get all rows with this value
                subset = self.df[self.df[col] == val].copy()
                
                if len(subset) <= 3:  # Need several rows to detect meaningful ordering issues
                    continue  
                
                logger.info(f"Analyzing {len(subset)} rows for {col}={val}")

                # First check: look for exact duplicate IDs (completely identical values)
                duplicate_ids = subset[id_col].duplicated(keep=False)
                if duplicate_ids.any():
                    dup_indices = subset[duplicate_ids].index.tolist()
                    for idx in dup_indices:
                        dup_id = subset.loc[idx, id_col]
                        # Find the first occurrence of this ID
                        first_idx = subset[subset[id_col] == dup_id].index[0]
                        
                        # Only report the duplicate if it's not the first occurrence
                        if idx != first_idx:
                            anomaly = {
                                "id": dup_id if not isinstance(dup_id, (int, np.integer)) else int(dup_id),
                                "previous_id": dup_id if not isinstance(dup_id, (int, np.integer)) else int(dup_id),
                                "row_index": int(idx),
                                "first_occurrence_row": int(first_idx),
                                "sort_column": col,
                                "sort_value": val if not isinstance(val, (int, np.integer)) else int(val),
                                "anomaly_type": "duplicate_id",
                                "description": f"Duplicate ID {dup_id} in {col}={val}"
                            }
                            anomalies.append(anomaly)
                
                # Second check: IDs out of sequence in ways that suggest manual manipulation
                
                # Convert to numeric if possible for better comparison
                id_values = subset[id_col]
                is_numeric = pd.api.types.is_numeric_dtype(id_values)
                
                # Create a sorted version of the dataset to compare against
                # Exclude duplicate IDs (already handled above)
                unique_subset = subset.drop_duplicates(subset=[id_col])
                sorted_subset = unique_subset.sort_values(id_col)
                
                # If already perfectly sorted, skip further checks
                if sorted_subset.index.equals(unique_subset.index):
                    continue
                
                # Look for IDs that are out of sequence
                sorted_ids = sorted_subset[id_col].values
                original_ids = unique_subset[id_col].values
                
                # Track clusters of out-of-sequence IDs
                # We're looking for specific patterns like small clusters of IDs that are out of order
                out_of_seq_indices = []
                
                # For numeric IDs, we can use a more specific algorithm
                if is_numeric:
                    # Calculate how out of order each ID is (distance from sorted position)
                    id_positions = {id_val: i for i, id_val in enumerate(sorted_ids)}
                    original_positions = np.array([id_positions[id_val] for id_val in original_ids])
                    expected_positions = np.arange(len(original_ids))
                    position_diff = np.abs(original_positions - expected_positions)
                    
                    # Filter to significant displacements (moved by more than 2 positions)
                    # This helps ignore minor reorderings that might be coincidental
                    significant_shifts = position_diff > 2
                    if np.sum(significant_shifts) > 0:
                        # Get the indices of rows with significantly out of order IDs
                        shifted_indices = np.where(significant_shifts)[0]
                        
                        # Convert to original dataframe indices
                        for idx_in_subset in shifted_indices:
                            out_of_seq_indices.append(unique_subset.index[idx_in_subset])
                else:
                    # For non-numeric IDs, we need a different approach
                    # Check for cases where the order is clearly wrong
                    for i in range(1, len(original_ids)):
                        curr_id = original_ids[i]
                        prev_id = original_ids[i-1]
                        
                        # Find positions in sorted array
                        curr_pos = np.where(sorted_ids == curr_id)[0][0]
                        prev_pos = np.where(sorted_ids == prev_id)[0][0]
                        
                        # If current ID should come before previous ID, it's out of sequence
                        if curr_pos < prev_pos:
                            out_of_seq_indices.append(unique_subset.index[i])
                
                # Process out-of-sequence rows (create anomaly records)
                for idx in out_of_seq_indices:
                    row = subset.loc[idx]
                    row_id = row[id_col]
                    
                    # Find the previous row in the original dataset (needed for UI)
                    prev_idx = None
                    prev_id = None
                    
                    # Get the index of this row in the original subset
                    row_pos_in_subset = unique_subset.index.get_loc(idx)
                    
                    # If not the first row, get previous row info
                    if row_pos_in_subset > 0:
                        prev_idx = unique_subset.index[row_pos_in_subset - 1]
                        prev_id = unique_subset.loc[prev_idx, id_col]
                    else:
                        # If it's the first row but still flagged, find the closest ID that should come before it
                        prev_id = sorted_ids[0]
                    
                    # Get the row's expected position in sorted order
                    expected_pos = np.where(sorted_ids == row_id)[0][0]
                    
                    anomaly = {
                        "id": row_id if not isinstance(row_id, (int, np.integer)) else int(row_id),
                        "previous_id": prev_id if not isinstance(prev_id, (int, np.integer)) else int(prev_id),
                        "row_index": int(idx),
                        "sort_column": col,
                        "sort_value": val if not isinstance(val, (int, np.integer)) else int(val),
                        "expected_position": int(expected_pos),
                        "actual_position": int(row_pos_in_subset),
                        "anomaly_type": "out_of_sequence",
                        "description": f"ID {row_id} out of sequence in {col}={val}"
                    }
                    
                    # Check for unusual patterns in dependent variables
                    if dependent_vars:
                        out_of_order_analysis = self._analyze_out_of_order_dependent_vars(
                            subset,
                            idx,
                            dependent_vars,
                            prioritize_out_of_order=prioritize_out_of_order,
                            prioritized_columns=prioritized_vars if prioritized_vars else None,
                        )
                        
                        if out_of_order_analysis:
                            anomaly["out_of_order_analysis"] = out_of_order_analysis
                    
                    anomalies.append(anomaly)
        
        # Final filtering to reduce false positives
        # Only keep anomalies that are likely to be actual manipulation
        
        # Group anomalies by sort value to look for clusters
        anomalies_by_sort = {}
        for anomaly in anomalies:
            sort_val = anomaly["sort_value"]
            if sort_val not in anomalies_by_sort:
                anomalies_by_sort[sort_val] = []
            anomalies_by_sort[sort_val].append(anomaly)
        
        # Only keep duplicate IDs and clusters of out-of-sequence IDs
        filtered_anomalies = []
        for sort_val, group in anomalies_by_sort.items():
            # Always keep duplicate IDs
            duplicates = [a for a in group if a.get("anomaly_type") == "duplicate_id"]
            filtered_anomalies.extend(duplicates)
            
            # Only keep clusters of out-of-sequence IDs
            out_of_seq = [a for a in group if a.get("anomaly_type") == "out_of_sequence"]
            if len(out_of_seq) >= 2:  # Need at least two to form a cluster
                filtered_anomalies.extend(out_of_seq)
        
        logger.info(f"Found {len(filtered_anomalies)} sorting anomalies after filtering")
        return filtered_anomalies
            
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
        # Find the position where subset.index equals row_idx
        try:
            row_position = subset.index.tolist().index(row_idx)
        except ValueError:
            logger.warning(f"Row index {row_idx} not found in subset indices")
            return None

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
                                "imputed_original_values": [likely_original],
                                "effect_direction": "increases" if current_value > likely_original else "decreases",
                                "statistical_impact": variable_difference,
                            }
                    # For descending pattern
                    else:
                        # Value should be between prev and next in reverse sorted order
                        if current_value < next_value or current_value > prev_value:
                            likely_original = (prev_value + next_value) / 2

                            # Calculate statistical impact
                            variable_difference = self._calculate_statistical_impact(
                                subset, var, current_value, likely_original
                            )

                            return {
                                "sorted_by": [var],
                                "breaking_pattern": f"value {current_value} breaks descending pattern",
                                "imputed_original_values": [likely_original],
                                "effect_direction": "increases" if current_value > likely_original else "decreases",
                                "statistical_impact": variable_difference,
                            }

        # No clear pattern found
        return None

    def _calculate_statistical_impact(
        self, df: pd.DataFrame, var: str, current_value: float, likely_original: float
    ) -> str:
        """Calculate the statistical impact of a value change on group differences.

        Args:
            df: DataFrame for the current subset
            var: Variable name to analyze
            current_value: Current (potentially manipulated) value
            likely_original: Likely original (non-manipulated) value based on pattern

        Returns:
            String describing the statistical impact
        """
        try:
            # Need at least one group variable to assess impact
            if len(df.columns) <= 2:  # Just the ID and this variable
                return "Insufficient data to calculate impact"

            # Find a potential grouping variable (first categorical with 2-5 unique values)
            group_vars = []
            for col in df.columns:
                if col != var and df[col].nunique() >= 2 and df[col].nunique() <= 5:
                    group_vars.append(col)

            if not group_vars:
                return "No suitable grouping variables found"

            # Use the first potential grouping variable
            group_var = group_vars[0]

            # Create a copy with the manipulated value
            df_manipulated = df.copy()

            # Create another copy with the original value
            df_original = df.copy()

            # Find the row with our current value
            # This is complex because we need to handle potentially identical values
            mask = (df[var] == current_value)
            if mask.sum() > 1:
                # If multiple rows match, we need more constraints
                # We'll use the first row that matches, which isn't ideal but should work in most cases
                row_idx = mask.idxmax()
            else:
                row_idx = mask.idxmax()

            # Set the original value
            df_original.loc[row_idx, var] = likely_original

            # Calculate means by group for both datasets
            manip_means = df_manipulated.groupby(group_var)[var].mean()
            orig_means = df_original.groupby(group_var)[var].mean()

            # Calculate the absolute difference between groups
            manip_diff = manip_means.max() - manip_means.min()
            orig_diff = orig_means.max() - orig_means.min()

            # Calculate the percentage change in difference
            perc_change = ((manip_diff - orig_diff) / orig_diff) * 100 if orig_diff != 0 else float('inf')

            # Determine the impact
            if abs(perc_change) <= 5:
                return f"Minimal impact on group differences ({perc_change:.1f}% change)"
            elif perc_change > 5:
                return f"Increases group differences by {perc_change:.1f}%"
            else:
                return f"Decreases group differences by {abs(perc_change):.1f}%"

        except Exception as e:
            logger.error(f"Error calculating statistical impact: {str(e)}")
            return "Error calculating statistical impact"

    def analyze_suspicious_observations(
        self, suspicious_rows: List[int], group_col: str, outcome_cols: List[str]
    ) -> Dict[str, Any]:
        """Analyze effect sizes of suspicious observations.

        Args:
            suspicious_rows: List of row indices for suspicious observations
            group_col: Column name for grouping variable
            outcome_cols: List of outcome variable column names

        Returns:
            Dict with analysis results
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")

        results = {
            "suspicious_rows": len(suspicious_rows),
            "variables_analyzed": outcome_cols,
            "effects": {},
        }

        try:
            # Create masks for suspicious and non-suspicious observations
            suspicious_mask = self.df.index.isin(suspicious_rows)
            
            # Create a copy of the dataframe with a flag for suspicious observations
            df_copy = self.df.copy()
            df_copy["_suspicious"] = suspicious_mask
            
            # For each outcome column, calculate the effect
            for col in outcome_cols:
                # Skip if not a numeric column
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    continue
                    
                # Get suspicious and non-suspicious means
                mean_suspicious = df_copy[df_copy["_suspicious"]][col].mean()
                mean_non_suspicious = df_copy[~df_copy["_suspicious"]][col].mean()
                
                # Get means by group for suspicious observations
                try:
                    susp_by_group = df_copy[df_copy["_suspicious"]].groupby(group_col)[col].mean()
                    non_susp_by_group = df_copy[~df_copy["_suspicious"]].groupby(group_col)[col].mean()
                    
                    # Format the results for this variable
                    results["effects"][col] = {
                        "mean_suspicious": mean_suspicious if not np.isnan(mean_suspicious) else None,
                        "mean_non_suspicious": mean_non_suspicious if not np.isnan(mean_non_suspicious) else None,
                        "diff_vs_non_suspicious": mean_suspicious - mean_non_suspicious if not np.isnan(mean_suspicious) and not np.isnan(mean_non_suspicious) else None,
                        "means_by_group_suspicious": {
                            str(group): float(mean) 
                            for group, mean in susp_by_group.items()
                        },
                        "means_by_group_non_suspicious": {
                            str(group): float(mean) 
                            for group, mean in non_susp_by_group.items()
                        },
                    }
                    
                    # Calculate the group difference for suspicious observations
                    if len(susp_by_group) >= 2:
                        results["effects"][col]["susp_group_diff"] = float(susp_by_group.max() - susp_by_group.min())
                        
                    # Calculate the group difference for non-suspicious observations
                    if len(non_susp_by_group) >= 2:
                        results["effects"][col]["non_susp_group_diff"] = float(non_susp_by_group.max() - non_susp_by_group.min())
                        
                except Exception as e:
                    logger.error(f"Error calculating group means for {col}: {str(e)}")
                    results["effects"][col] = {
                        "error": f"Error calculating group means: {str(e)}",
                        "mean_suspicious": mean_suspicious if not np.isnan(mean_suspicious) else None,
                        "mean_non_suspicious": mean_non_suspicious if not np.isnan(mean_non_suspicious) else None,
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing suspicious observations: {str(e)}")
            return {"error": str(e)}

    def compare_with_without_suspicious_rows(
        self, suspicious_rows: List[int], group_col: str, outcome_cols: List[str]
    ) -> Dict[str, Any]:
        """Compare statistical results with and without suspicious observations.

        Args:
            suspicious_rows: List of row indices for suspicious observations
            group_col: Column name for grouping variable
            outcome_cols: List of outcome variable column names

        Returns:
            Dict with comparison results
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")
            
        # Create a mask for suspicious rows
        suspicious_mask = self.df.index.isin(suspicious_rows)
        
        # Create a filtered dataset without suspicious rows
        df_filtered = self.df[~suspicious_mask].copy()
        
        results = {
            "variables_analyzed": outcome_cols,
            "comparisons": {},
        }
        
        # For each outcome column, compare statistics and p-values
        for col in outcome_cols:
            # Skip if not a numeric column
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
                
            # Calculate statistics with all data
            try:
                # Check if we can do a meaningful statistical test
                if self.df[group_col].nunique() < 2:
                    logger.warning(f"Not enough groups in {group_col} for statistical test")
                    continue
                    
                # T-test between groups with all data
                groups = self.df[group_col].unique()
                
                if len(groups) == 2:
                    # For two groups, use t-test
                    from scipy import stats
                    
                    group1 = self.df[self.df[group_col] == groups[0]][col].dropna()
                    group2 = self.df[self.df[group_col] == groups[1]][col].dropna()
                    
                    if len(group1) < 2 or len(group2) < 2:
                        logger.warning(f"Not enough data in groups for t-test with {col}")
                        continue
                        
                    t_full, p_full = stats.ttest_ind(group1, group2, equal_var=False)
                    
                    # Also calculate means and differences
                    mean1_full = group1.mean()
                    mean2_full = group2.mean()
                    diff_full = mean1_full - mean2_full
                    
                    # T-test with filtered data
                    group1_filtered = df_filtered[df_filtered[group_col] == groups[0]][col].dropna()
                    group2_filtered = df_filtered[df_filtered[group_col] == groups[1]][col].dropna()
                    
                    if len(group1_filtered) < 2 or len(group2_filtered) < 2:
                        logger.warning(f"Not enough data in filtered groups for t-test with {col}")
                        t_filtered, p_filtered = np.nan, np.nan
                        mean1_filtered, mean2_filtered, diff_filtered = np.nan, np.nan, np.nan
                    else:
                        t_filtered, p_filtered = stats.ttest_ind(group1_filtered, group2_filtered, equal_var=False)
                        mean1_filtered = group1_filtered.mean()
                        mean2_filtered = group2_filtered.mean()
                        diff_filtered = mean1_filtered - mean2_filtered
                        
                    # Store the results
                    results["comparisons"][col] = {
                        "test_type": "t-test",
                        "full_data": {
                            "statistic": float(t_full),
                            "p_value": float(p_full),
                            "sig_0.05": bool(p_full < 0.05),
                            "mean_group1": float(mean1_full),
                            "mean_group2": float(mean2_full),
                            "diff": float(diff_full),
                        },
                        "without_suspicious": {
                            "statistic": float(t_filtered) if not np.isnan(t_filtered) else None,
                            "p_value": float(p_filtered) if not np.isnan(p_filtered) else None,
                            "sig_0.05": bool(p_filtered < 0.05) if not np.isnan(p_filtered) else None,
                            "mean_group1": float(mean1_filtered) if not np.isnan(mean1_filtered) else None,
                            "mean_group2": float(mean2_filtered) if not np.isnan(mean2_filtered) else None,
                            "diff": float(diff_filtered) if not np.isnan(diff_filtered) else None,
                        },
                        "change": {
                            "diff_change": float(diff_full - diff_filtered) if not np.isnan(diff_filtered) else None,
                            "diff_percent_change": float(((diff_full - diff_filtered) / diff_filtered) * 100) if diff_filtered != 0 and not np.isnan(diff_filtered) else None,
                            "sig_changed": bool(p_full < 0.05 and p_filtered >= 0.05 or p_full >= 0.05 and p_filtered < 0.05) if not np.isnan(p_filtered) else None,
                        }
                    }
                
                else:
                    # For more than two groups, use ANOVA
                    from scipy import stats
                    
                    # Full data ANOVA
                    anova_data_full = [self.df[self.df[group_col] == group][col].dropna() for group in groups]
                    
                    # Check if we have enough data
                    if any(len(group_data) < 2 for group_data in anova_data_full):
                        logger.warning(f"Not enough data in some groups for ANOVA with {col}")
                        continue
                        
                    f_full, p_full = stats.f_oneway(*anova_data_full)
                    
                    # Calculate means and max difference
                    means_full = {str(group): float(self.df[self.df[group_col] == group][col].mean()) for group in groups}
                    max_diff_full = max(means_full.values()) - min(means_full.values())
                    
                    # Filtered data ANOVA
                    anova_data_filtered = [df_filtered[df_filtered[group_col] == group][col].dropna() for group in groups]
                    
                    # Check if we have enough data in filtered version
                    if any(len(group_data) < 2 for group_data in anova_data_filtered):
                        logger.warning(f"Not enough data in some filtered groups for ANOVA with {col}")
                        f_filtered, p_filtered = np.nan, np.nan
                        means_filtered = {str(group): np.nan for group in groups}
                        max_diff_filtered = np.nan
                    else:
                        f_filtered, p_filtered = stats.f_oneway(*anova_data_filtered)
                        means_filtered = {str(group): float(df_filtered[df_filtered[group_col] == group][col].mean()) for group in groups}
                        max_diff_filtered = max(means_filtered.values()) - min(means_filtered.values())
                        
                    # Store the results
                    results["comparisons"][col] = {
                        "test_type": "ANOVA",
                        "full_data": {
                            "statistic": float(f_full),
                            "p_value": float(p_full),
                            "sig_0.05": bool(p_full < 0.05),
                            "means_by_group": means_full,
                            "max_diff": float(max_diff_full),
                        },
                        "without_suspicious": {
                            "statistic": float(f_filtered) if not np.isnan(f_filtered) else None,
                            "p_value": float(p_filtered) if not np.isnan(p_filtered) else None,
                            "sig_0.05": bool(p_filtered < 0.05) if not np.isnan(p_filtered) else None,
                            "means_by_group": means_filtered if not np.isnan(list(means_filtered.values())[0]) else None,
                            "max_diff": float(max_diff_filtered) if not np.isnan(max_diff_filtered) else None,
                        },
                        "change": {
                            "diff_change": float(max_diff_full - max_diff_filtered) if not np.isnan(max_diff_filtered) else None,
                            "diff_percent_change": float(((max_diff_full - max_diff_filtered) / max_diff_filtered) * 100) if max_diff_filtered != 0 and not np.isnan(max_diff_filtered) else None,
                            "sig_changed": bool(p_full < 0.05 and p_filtered >= 0.05 or p_full >= 0.05 and p_filtered < 0.05) if not np.isnan(p_filtered) else None,
                        }
                    }
            
            except Exception as e:
                logger.error(f"Error comparing statistics for {col}: {str(e)}")
                results["comparisons"][col] = {"error": str(e)}
                
        return results

    def segment_and_analyze_with_claude(
        self, client: Any, user_suspicions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Segment the dataset and use Claude to analyze each segment for anomalies.
        
        This is useful for datasets too large to analyze in a single prompt.
        
        Args:
            client: Claude API client
            user_suspicions: Optional dictionary with user-specified suspicions to guide analysis
            
        Returns:
            List of analysis results for each segment, with potential anomalies identified
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")
            
        # Limit to a maximum of 50 rows per segment
        MAX_ROWS_PER_SEGMENT = 50
        
        # If dataset is small enough, analyze it all at once
        if len(self.df) <= MAX_ROWS_PER_SEGMENT:
            return [self._analyze_segment_with_claude(client, self.df, 0, len(self.df), user_suspicions)]
            
        # Divide into segments
        segments = []
        row_count = len(self.df)
        
        # Calculate number of segments (ceiling division)
        num_segments = (row_count + MAX_ROWS_PER_SEGMENT - 1) // MAX_ROWS_PER_SEGMENT
        
        logger.info(f"Segmenting dataset with {row_count} rows into {num_segments} chunks for Claude analysis")
        
        # Process each segment
        for i in range(num_segments):
            start_idx = i * MAX_ROWS_PER_SEGMENT
            end_idx = min((i + 1) * MAX_ROWS_PER_SEGMENT, row_count)
            
            # Extract segment
            segment = self.df.iloc[start_idx:end_idx].copy()
            
            # Analyze with Claude
            try:
                result = self._analyze_segment_with_claude(client, segment, start_idx, end_idx, user_suspicions)
                segments.append(result)
                logger.info(f"Processed segment {i+1}/{num_segments} ({start_idx}-{end_idx})")
            except Exception as e:
                logger.error(f"Error processing segment {i+1}/{num_segments}: {str(e)}")
                segments.append({
                    "chunk": i + 1,
                    "rows": f"{start_idx}-{end_idx}",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
        
        return segments
            
    def _analyze_segment_with_claude(
        self, 
        client: Any, 
        segment: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        user_suspicions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use Claude to analyze a segment of data for anomalies.
        
        Args:
            client: Claude API client
            segment: DataFrame segment to analyze
            start_idx: Starting index of this segment in the original dataframe
            end_idx: Ending index of this segment in the original dataframe
            user_suspicions: Optional dictionary with user-specified suspicions to guide analysis
            
        Returns:
            Dict with Claude's analysis of the segment
        """
        # Convert segment to string representation
        segment_str = segment.to_string()
        
        # Get column information
        column_info = "\n".join([
            f"- {col}: {segment[col].dtype} - {segment[col].nunique()} unique values, {segment[col].isna().sum()} missing"
            for col in segment.columns
        ])
        
        # Add user suspicions if provided
        suspicion_info = ""
        if user_suspicions:
            # Format the user suspicions nicely
            suspicion_parts = []
            
            if "focus_columns" in user_suspicions and user_suspicions["focus_columns"]:
                suspicion_parts.append(f"Focus columns: {', '.join(user_suspicions['focus_columns'])}")
                
            if "potential_issues" in user_suspicions and user_suspicions["potential_issues"]:
                suspicion_parts.append(f"Potential issues: {', '.join(user_suspicions['potential_issues'])}")
                
            if "treatment_columns" in user_suspicions and user_suspicions["treatment_columns"]:
                suspicion_parts.append(f"Treatment columns: {', '.join(user_suspicions['treatment_columns'])}")
                
            if "outcome_columns" in user_suspicions and user_suspicions["outcome_columns"]:
                suspicion_parts.append(f"Outcome columns: {', '.join(user_suspicions['outcome_columns'])}")
                
            if "suspect_grouping" in user_suspicions and user_suspicions["suspect_grouping"]:
                suspicion_parts.append(f"Suspect grouping column: {user_suspicions['suspect_grouping']}")
                
            if "description" in user_suspicions and user_suspicions["description"]:
                suspicion_parts.append(f"Description: {user_suspicions['description']}")
                
            if suspicion_parts:
                suspicion_info = "User-provided suspicions to consider (but not bias your analysis):\n" + "\n".join(suspicion_parts) + "\n\n"
        
        # Create the prompt for Claude
        prompt = f"""Analyze this dataset segment for potential data manipulation or anomalies.

Dataset segment (rows {start_idx}-{end_idx}):
{segment_str}

Column information:
{column_info}

{suspicion_info}I need you to carefully examine this data for potential signs of manipulation or fabrication.

Things to look for:
1. Unusual patterns in numeric data (e.g., too many round numbers, unusual terminal digits)
2. Suspicious sequences or progressions (e.g., too perfect linear relationships)
3. Values that seem to be designed to produce a specific statistical effect
4. Observations that appear to be out of order
5. Unnatural clustering or distribution of values
6. Data that looks manually entered or adjusted

Be sure to:
- Consider what the data is measuring and what patterns would be expected
- Look for individual observations that break clear patterns in the data
- Focus on numeric columns as they often show clearer signs of manipulation
- Look for sorting inconsistencies, especially in IDs or sequential values
- Check for values that seem to perfectly support a research hypothesis

Respond with JSON in this format:
{{
  "anomalies_detected": true/false,
  "confidence": 1-10 (how confident you are in your assessment),
  "explanation": "Detailed explanation of your findings",
  "findings": [
    {{
      "type": "pattern type (e.g. 'sequence anomaly', 'terminal digit anomaly')",
      "description": "Description of the specific anomaly",
      "columns_involved": ["col1", "col2"],
      "row_indices": [list of row indices],
      "severity": 1-10
    }}
  ]
}}"""
        
        # Call Claude
        try:
            logger.info(f"Sending segment {start_idx}-{end_idx} to Claude for analysis")
            response = client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            
            # Extract JSON response
            content = response.content[0].text
            
            # Try to find and parse JSON
            import re
            json_pattern = r"\{.*\}"
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            if matches:
                # Use the longest match as it's likely the full JSON
                json_str = max(matches, key=len)
                analysis = json.loads(json_str)
                
                # Add metadata
                analysis["chunk"] = start_idx // MAX_ROWS_PER_SEGMENT + 1  # 1-indexed chunk number
                analysis["rows"] = f"{start_idx}-{end_idx}"
                
                return analysis
            else:
                logger.warning(f"Could not extract JSON from Claude's response for segment {start_idx}-{end_idx}")
                # Return a structured error
                return {
                    "chunk": start_idx // MAX_ROWS_PER_SEGMENT + 1,
                    "rows": f"{start_idx}-{end_idx}",
                    "error": "Could not extract JSON from Claude's response",
                    "raw_data": content[:500] + ("..." if len(content) > 500 else "")
                }
        except Exception as e:
            logger.error(f"Error analyzing segment {start_idx}-{end_idx}: {str(e)}")
            # Return a structured error
            return {
                "chunk": start_idx // MAX_ROWS_PER_SEGMENT + 1,
                "rows": f"{start_idx}-{end_idx}",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
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
        try:
            with zipfile.ZipFile(self.excel_file, "r") as zip_ref:
                zip_ref.extractall(self.temp_dir)
            self.extracted = True
            logger.info(f"Successfully extracted Excel file to {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error extracting Excel file: {str(e)}")
            self.extracted = False
            return
        
        # Parse calculation chain
        self.parse_calc_chain()

    def parse_calc_chain(self) -> None:
        """Parse Excel calculation chain XML file to extract calculation ordering information."""
        if not self.extracted:
            logger.error("Cannot parse calc chain - Excel file not extracted")
            return

        calc_chain_path = os.path.join(self.temp_dir, "xl", "calcChain.xml")
        logger.info(f"Looking for calcChain.xml at: {calc_chain_path}")
        
        if not os.path.exists(calc_chain_path):
            logger.info("No calculation chain found in this Excel file.")
            return
            
        try:
            # Parse XML
            tree = ET.parse(calc_chain_path)
            root = tree.getroot()
            logger.info(f"Successfully parsed calcChain.xml, root tag: {root.tag}")

            # Get namespace if present
            ns = ''
            if '}' in root.tag:
                ns = root.tag.split('}')[0] + '}'
                logger.info(f"Found namespace: {ns}")
                
            # Process calculation chain entries using namespace-aware approach
            count = 0
            
            # First try with the namespace-specific approach
            if ns:
                c_tag = f"{ns}c"
                for c in root.findall(f".//{c_tag}"):
                    ref = c.get("r")
                    if ref:
                        # Convert Excel cell reference to row and column
                        col_str = "".join(filter(str.isalpha, ref))
                        row_num = int("".join(filter(str.isdigit, ref)))
                        
                        self.calc_chain.append({"ref": ref, "row": row_num, "col": col_str})
                        count += 1
            
            # If no entries found with namespace, or no namespace was detected, try generic way
            if count == 0:
                for c in root.findall(".//*"):
                    # Check if the tag ends with 'c' (for the calculation cell tag)
                    if c.tag.endswith('c'):
                        ref = c.get("r")
                        if ref:
                            # Convert Excel cell reference to row and column
                            col_str = "".join(filter(str.isalpha, ref))
                            row_num = int("".join(filter(str.isdigit, ref)))
                            
                            self.calc_chain.append({"ref": ref, "row": row_num, "col": col_str})
                            count += 1
            
            logger.info(f"Parsed calc chain with {count} entries")
        except Exception as e:
            logger.error(f"Error parsing calc chain: {str(e)}")
            # Return empty list but don't crash
            self.calc_chain = []

    def analyze_row_movement(self, row_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Analyze the calculation chain to detect evidence of row movement.
        
        This analysis is based on the principle that when rows are moved in Excel,
        the calculation order remains intact. For example, if a cell from row 7 was moved
        to row 12, its calculation entry in calcChain.xml would still show it being
        calculated between cells from rows 6 and 8.

        Args:
            row_numbers: List of row numbers to check for movement

        Returns:
            List[Dict[str, Any]]: Findings about row movements, each with details about the evidence
        """
        findings = []
        
        if not row_numbers:
            logger.warning("No row numbers provided to check for movement")
            return findings

        # If there's no calc chain data, return empty findings
        if not self.calc_chain:
            logger.warning("No calculation chain data available to analyze row movement")
            return findings

        try:
            logger.info(f"Analyzing {len(row_numbers)} suspicious rows with {len(self.calc_chain)} calc chain entries")
            
            # Group calc chain entries by row
            row_entries = {}
            for entry in self.calc_chain:
                row = entry["row"]
                if row not in row_entries:
                    row_entries[row] = []
                row_entries[row].append(entry)

            # Convert Excel's 1-based row numbers to match our data indices which are 0-based
            # Try multiple strategies since the exact offset can vary based on headers, etc.
            possible_row_mappings = [
                row_numbers,  # No adjustment
                [r + 1 for r in row_numbers],  # +1 for Excel's 1-based indexing
                [r + 2 for r in row_numbers],  # +2 for header row + 1-based indexing
                [r + 3 for r in row_numbers],  # +3 for multi-row header + 1-based indexing
            ]
            
            # Try each mapping and take the one that finds the most matches
            best_mapping = []
            max_matches = 0
            
            for mapping in possible_row_mappings:
                int_mapping = [int(r) for r in mapping]  # Ensure all are integers
                matching = set(int_mapping).intersection(set(row_entries.keys()))
                if len(matching) > max_matches:
                    max_matches = len(matching)
                    best_mapping = int_mapping
            
            matching_rows = set(best_mapping).intersection(set(row_entries.keys()))
            logger.info(f"Found {len(matching_rows)} suspicious rows in the calculation chain using best mapping")
            
            # Analyze the calculation chain structure
            # We'll build a graph of "expected" row neighbors based on calculation order
            expected_neighbors = {}
            
            # First, convert the calc chain to a sequence of row numbers
            calc_sequence = [entry["row"] for entry in self.calc_chain]
            
            # For each row, find its adjacent rows in the calculation sequence
            for i in range(1, len(calc_sequence) - 1):
                row = calc_sequence[i]
                prev_row = calc_sequence[i - 1]
                next_row = calc_sequence[i + 1]
                
                if row not in expected_neighbors:
                    expected_neighbors[row] = set()
                
                expected_neighbors[row].add(prev_row)
                expected_neighbors[row].add(next_row)
            
            # Also record the calculation order for each row
            row_calc_order = {}
            for i, entry in enumerate(self.calc_chain):
                row = entry["row"]
                if row not in row_calc_order:
                    row_calc_order[row] = []
                row_calc_order[row].append(i)
            
            # Dictionary to track original row positions
            original_positions = {}
            
            # For each suspicious row, check if its calculation order suggests it was moved
            for row in matching_rows:
                # Skip rows without calculation entries
                if row not in row_calc_order:
                    continue
                
                # Get all calculation positions for this row
                calc_positions = row_calc_order[row]
                
                # For each calculation position, check the surrounding entries
                for pos in calc_positions:
                    # Need at least one position before and after
                    if pos > 0 and pos < len(self.calc_chain) - 1:
                        prev_entry = self.calc_chain[pos - 1]
                        next_entry = self.calc_chain[pos + 1]
                        
                        prev_row = prev_entry["row"]
                        next_row = next_entry["row"]
                        
                        # If this calculation is between consecutive row numbers,
                        # but the row itself is not between them, it likely was moved
                        moved_evidence = []
                        
                        # Classic case: calculation between consecutive rows (e.g., rows 7 and 8)
                        if abs(prev_row - next_row) == 1:
                            min_neighbor = min(prev_row, next_row)
                            max_neighbor = max(prev_row, next_row)
                            
                            # If row should be between these neighbors but isn't
                            if (min_neighbor < row < max_neighbor) or (row < min_neighbor) or (row > max_neighbor):
                                likely_original = min_neighbor + 1  # Row was likely between them
                                moved_evidence.append((min_neighbor, max_neighbor, likely_original))
                        
                        # Also check for cases where rows that should be far apart
                        # based on calculation order are adjacent in the current sheet
                        if (abs(row - prev_row) > 3 and abs(prev_row - next_row) <= 2) or \
                           (abs(row - next_row) > 3 and abs(prev_row - next_row) <= 2):
                            # Find where this row likely belongs in the sequence
                            if abs(prev_row - next_row) == 1:
                                # If prev_row and next_row are consecutive, row likely belongs between them
                                original_pos = min(prev_row, next_row) + 1
                                moved_evidence.append((prev_row, next_row, original_pos))
                            elif abs(prev_row - next_row) == 2:
                                # There might be a missing row between them
                                middle = (prev_row + next_row) // 2
                                if middle != row:  # If the middle isn't our current row
                                    moved_evidence.append((prev_row, next_row, middle))
                        
                        # If we found evidence of movement
                        for prev_r, next_r, orig_pos in moved_evidence:
                            finding = {
                                "row": row,
                                "evidence": f"Cell from row {row} is calculated between rows {prev_r} and {next_r} in calcChain",
                                "current_position": row,
                                "likely_original_position": orig_pos,
                                "confidence": "high" if abs(prev_r - next_r) == 1 else "medium"
                            }
                            
                            # Track the original position for consolidation
                            if row not in original_positions:
                                original_positions[row] = []
                            original_positions[row].append(orig_pos)
                            
                            findings.append(finding)
            
            # Consolidate findings to improve confidence
            consolidated_findings = []
            processed_rows = set()
            
            for row in original_positions:
                if row in processed_rows:
                    continue
                    
                # Get all potential original positions for this row
                positions = original_positions[row]
                
                # If we have multiple potential positions, use the most common one
                if len(positions) > 1:
                    from collections import Counter
                    position_counts = Counter(positions)
                    most_common_pos = position_counts.most_common(1)[0][0]
                    most_common_count = position_counts.most_common(1)[0][1]
                    
                    # Calculate confidence based on consistency of evidence
                    confidence = "high" if most_common_count >= 3 else "medium"
                    if most_common_count == 1:
                        confidence = "low"
                        
                    # Create a consolidated finding
                    consolidated_finding = {
                        "row": row,
                        "evidence": f"Row {row} appears to have been moved from position {most_common_pos} (found in {most_common_count} calculation chain entries)",
                        "current_position": row,
                        "likely_original_position": most_common_pos,
                        "confidence": confidence,
                        "evidence_count": most_common_count
                    }
                    consolidated_findings.append(consolidated_finding)
                else:
                    # If we only have one position, use the first finding
                    for finding in findings:
                        if finding["row"] == row:
                            consolidated_findings.append(finding)
                            break
                            
                processed_rows.add(row)
            
            # Sort by confidence and row number
            consolidated_findings.sort(key=lambda x: (0 if x["confidence"] == "high" else 
                                                    1 if x["confidence"] == "medium" else 2, 
                                                    x["row"]))
            
            logger.info(f"Analysis complete - found evidence for {len(consolidated_findings)} instances of row movement")
            return consolidated_findings
            
        except Exception as e:
            logger.error(f"Error in analyze_row_movement: {str(e)}")
            traceback.print_exc()

        return findings