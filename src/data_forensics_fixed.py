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
        self.df: Optional[pd.DataFrame] = None  # Pandas DataFrame
        self.excel_metadata: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self.user_suspicions: Dict[str, Any] = {}

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

        logger.info(f"Analyzing dataset: {filepath}")

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
        logger.info(f"File extension: {ext}")

        # Load data with pandas
        try:
            if ext == ".csv":
                # Try different encodings with pandas
                encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
                for encoding in encodings:
                    try:
                        self.df = pd.read_csv(filepath, encoding=encoding)
                        logger.info(
                            f"Successfully read CSV with pandas using encoding: {encoding}"
                        )
                        break
                    except Exception as enc_err:
                        continue
                else:
                    # No encoding worked, raise error
                    raise ValueError(f"Failed to read CSV file with any encoding")
            elif ext == ".xlsx":
                # For Excel files, read with pandas and extract metadata
                self.df = pd.read_excel(filepath)
                self.excel_metadata = self.extract_excel_metadata(filepath)
                logger.info(
                    f"Successfully read Excel file with pandas, shape: {self.df.shape}"
                )
            elif ext == ".parquet":
                # Use pandas for parquet
                self.df = pd.read_parquet(filepath)
                logger.info(
                    f"Successfully read Parquet file with pandas, shape: {self.df.shape}"
                )
            elif ext == ".dta":
                # Use pandas for Stata files
                self.df = pd.read_stata(filepath)
                logger.info(
                    f"Successfully read Stata file with pandas, shape: {self.df.shape}"
                )
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
                            logger.info(
                                f"Successfully read SPSS file with encoding: {encoding}"
                            )
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

        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            raise ValueError(f"Error reading file {filepath}: {str(e)}")

        # Analyze sorting anomalies if sort columns are provided
        if sort_cols and id_col:
            logger.info(
                f"Checking sorting anomalies for ID column '{id_col}' within groups '{sort_cols}'"
            )

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
                logger.info(f"Checking focus columns: {additional_columns_to_check}")

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
                logger.info(f"Found {len(sorting_issues)} sorting anomalies")

        # Check for duplicate IDs
        if id_col:
            logger.info(f"Checking duplicate IDs for column: {id_col}")
            duplicates = self.check_duplicate_ids(id_col)
            if duplicates:
                self.findings.append({"type": "duplicate_ids", "details": duplicates})
                logger.info(f"Found {len(duplicates)} duplicate IDs")

        return self.findings

    def check_duplicate_ids(self, id_col: str) -> List[Dict[str, Any]]:
        """Check for duplicate ID values that might indicate manipulation.

        Args:
            id_col: Column name for IDs

        Returns:
            List of duplicate ID findings
        """
        # Use pandas for duplicate detection
        if self.df is None:
            raise ValueError("No dataset loaded")

        # Group by ID and count occurrences with pandas
        value_counts = self.df[id_col].value_counts()

        # Create a DataFrame with the ID column and counts
        counts = pd.DataFrame(
            {id_col: value_counts.index, "count": value_counts.values}
        )

        # Filter to only duplicates and sort by count
        counts = counts[counts["count"] > 1].sort_values("count", ascending=False)

        duplicate_details = []

        # For each duplicate ID, get details
        for index, row in counts.iterrows():
            dup_id = row[id_col]
            count = row["count"]

            # Find row indices with this duplicate ID
            # Convert to integers list for serialization
            rows_df = self.df[self.df[id_col] == dup_id]
            # Get row indices using pandas
            row_indices = rows_df.index.tolist()

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

        try:
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as tmpdirname:
                logger.info(f"Extracting Excel file to temp directory: {tmpdirname}")

                # Extract the Excel file (which is a zip file)
                with zipfile.ZipFile(excel_file, "r") as zip_ref:
                    zip_ref.extractall(tmpdirname)

                # Look for calcChain.xml
                calc_chain_path = os.path.join(tmpdirname, "xl", "calcChain.xml")
                logger.info(f"Looking for calcChain.xml at: {calc_chain_path}")

                if os.path.exists(calc_chain_path):
                    logger.info("Found calcChain.xml, parsing...")

                    # Parse the XML
                    tree = ET.parse(calc_chain_path)
                    root = tree.getroot()

                    # Get namespace if present
                    ns = ""
                    if "}" in root.tag:
                        ns = root.tag.split("}")[0] + "}"
                        logger.info(f"Found namespace: {ns}")

                    # Extract calculation order
                    count = 0
                    for c in root.findall(".//*"):
                        if c.tag.endswith("c"):  # Look for calculation cell entries
                            cell_ref = c.get("r")
                            if cell_ref:
                                # Convert Excel cell reference to row and column
                                col_name = "".join(filter(str.isalpha, cell_ref))
                                row_num = int("".join(filter(str.isdigit, cell_ref)))

                                metadata["calc_chain"].append(
                                    {
                                        "cell_ref": cell_ref,
                                        "column": col_name,
                                        "row": row_num,
                                    }
                                )
                                count += 1

                    logger.info(f"Extracted {count} entries from calc chain")
                else:
                    logger.info("No calcChain.xml found in Excel file")
        except Exception as e:
            logger.error(f"Error extracting Excel metadata: {str(e)}")
            traceback.print_exc()

        return metadata

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

        if self.df is None:
            logger.error("DataFrame is None in DataForensics object")
            return [
                {
                    "error": "DataFrame is None in DataForensics object",
                    "hint": "Make sure forensics.df is assigned a valid DataFrame",
                }
            ]

        if len(self.df) == 0:
            logger.error("DataFrame is empty (0 rows)")
            return [
                {
                    "error": "DataFrame is empty (0 rows)",
                    "hint": "The dataset must contain at least one row of data",
                }
            ]

        claude_findings: List[Dict[str, Any]] = []

        # Determine number of chunks needed
        total_rows: int = len(self.df)
        num_chunks: int = max(
            1, (total_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
        )

        logger.info(
            f"Segmenting dataset with {total_rows} rows into {num_chunks} chunks"
        )

        # Process each chunk
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_rows_per_chunk
            end_idx = min(start_idx + max_rows_per_chunk, total_rows)

            logger.info(
                f"Analyzing chunk {chunk_idx + 1}/{num_chunks} with rows {start_idx} to {end_idx - 1}"
            )

            # Get this chunk of data
            chunk_df = self.df.iloc[start_idx:end_idx].copy()

            # Convert to simple representation for Claude analysis
            chunk_data = chunk_df.to_dict(orient="records")

            # Include calc chain entries for this chunk of rows if available
            calc_chain_entries = []
            if (
                hasattr(self, "excel_metadata")
                and self.excel_metadata
                and "calc_chain" in self.excel_metadata
            ):
                calc_chain = self.excel_metadata["calc_chain"]

                # Filter to get only entries for rows in this chunk
                for entry in calc_chain:
                    if (
                        start_idx <= entry["row"] - 1 < end_idx
                    ):  # Convert from 1-indexed to 0-indexed
                        calc_chain_entries.append(entry)

                logger.info(
                    f"Including {len(calc_chain_entries)} calcChain entries for chunk {chunk_idx + 1}"
                )

            # Build a prompt for Claude to analyze the chunk
            user_suspicion_section = ""
            if user_suspicions:
                user_suspicion_section = "\n\nUser has the following suspicions to guide (but not bias) your analysis:\n"
                for key, value in user_suspicions.items():
                    if value and key != "description":
                        if isinstance(value, list):
                            user_suspicion_section += (
                                f"- {key}: {', '.join(str(v) for v in value)}\n"
                            )
                        else:
                            user_suspicion_section += f"- {key}: {value}\n"

                # Add user description if provided
                if "description" in user_suspicions and user_suspicions["description"]:
                    user_suspicion_section += (
                        f"\nUser description: {user_suspicions['description']}\n"
                    )

                # Add important note to avoid bias
                user_suspicion_section += "\nIMPORTANT: While considering these areas, maintain objectivity and report what the data actually shows, not what is expected. Do not let these suggestions narrow your analysis or bias your findings. Thoroughly analyze all patterns in the data.\n"

            prompt = f"""Analyze this dataset chunk for potential data manipulation or anomalies.{
                user_suspicion_section
            }

            Dataset Chunk (Rows {start_idx} to {end_idx - 1}):
            {json.dumps(chunk_data, indent=2)}

            {
                f"Excel calculation chain entries for these rows: {json.dumps(calc_chain_entries, indent=2)}"
                if calc_chain_entries
                else ""
            }

            Analyze this data for potential anomalies or manipulation such as:
            1. Out-of-order observations or suspicious sorting
            2. Unusual patterns in values that suggest data fabrication
            3. Duplicate IDs or other suspicious patterns
            4. Statistical anomalies like outliers with unusual influence
            5. Evidence of values being changed to achieve specific results

            Provide your analysis as a single structured JSON object with the following format:
            {{
              "anomalies_detected": true or false,
              "confidence": 1-10 scale (higher means more confident that manipulation occurred),
              "findings": [
                {{
                  "type": "category of anomaly",
                  "column": "affected column",
                  "rows": [list of affected row indices within the chunk],
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

            Only report anomalies if you have strong evidence they exist in the data. Do not invent problems.
            Focus on data patterns that would be genuinely concerning in a research context.
            """

            try:
                # Call Claude API
                response = client.messages.create(
                    model="claude-3-7-sonnet-latest",
                    max_tokens=4000,
                    system="You are an expert data forensic analyst detecting subtle patterns of potential data manipulation in research datasets.",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )

                # Extract and parse the result
                result_text = response.content[0].text

                # Try to find a JSON object in the result
                json_match = re.search(r"```json\s*(.*?)\s*```", result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If not in a code block, try to find JSON block directly
                    json_match = re.search(r"({[\s\S]*})", result_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = result_text

                try:
                    # Parse the JSON result
                    chunk_findings = json.loads(json_str)
                    claude_findings.append(chunk_findings)

                    # Log the main results
                    if chunk_findings.get("anomalies_detected", False):
                        logger.info(
                            f"Claude found {len(chunk_findings.get('findings', []))} anomalies in chunk {chunk_idx + 1} with confidence {chunk_findings.get('confidence', 'N/A')}"
                        )
                    else:
                        logger.info(
                            f"Claude found no anomalies in chunk {chunk_idx + 1}"
                        )
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, add error information
                    logger.warning(f"Error parsing Claude's response as JSON: {str(e)}")
                    claude_findings.append(
                        {
                            "error": "Failed to parse Claude response",
                            "raw_response": result_text[:500] + "..."
                            if len(result_text) > 500
                            else result_text,
                            "explanation": str(e),
                        }
                    )
            except Exception as e:
                # Handle any other errors
                logger.error(f"Error calling Claude API: {str(e)}")
                claude_findings.append(
                    {
                        "error": f"Error calling Claude API: {str(e)}",
                        "chunk": f"{chunk_idx + 1}/{num_chunks}",
                    }
                )

        return claude_findings

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
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset first.")

        anomalies = []
        logger.info(f"Starting sorting anomaly detection for ID column '{id_col}'")

        # Identify potential dependent variables (numeric columns that aren't used for sorting)
        dependent_vars = []
        prioritized_vars = []

        # Convert sort_cols to list if it's a string
        if isinstance(sort_cols, str):
            sort_cols_list = [sort_cols]
        else:
            sort_cols_list = sort_cols.copy() if sort_cols else []

        if check_dependent_vars:
            # Find numeric columns using pandas
            numeric_cols = list(self.df.select_dtypes(include=["number"]).columns)
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
                logger.info(
                    f"Will prioritize {len(prioritized_vars)} columns: {prioritized_vars}"
                )

        # Process each sort column
        for col in sort_cols_list:
            logger.info(f"Analyzing sort column: {col}")

            # Get unique values
            unique_vals = self.df[col].dropna().unique()
            logger.info(f"Found {len(unique_vals)} unique values in sort column {col}")

            # For each unique value in the sorting column
            for val in unique_vals:
                # Get all rows with this value
                subset = self.df[self.df[col] == val].copy()

                if len(subset) <= 1:
                    continue  # Need at least 2 rows to detect ordering issues

                logger.info(f"Analyzing {len(subset)} rows for {col}={val}")

                # Sort by ID to detect out-of-order entries
                sorted_by_id = subset.sort_values(id_col)

                # Get the sorted IDs and their original indices
                ids = sorted_by_id[id_col].values
                indices = sorted_by_id.index.tolist()

                # Check for out-of-order entries
                for i in range(1, len(ids)):
                    curr_id = ids[i]
                    prev_id = ids[i - 1]

                    # Skip if IDs are equal (can't determine order)
                    if curr_id == prev_id:
                        continue

                    # Check if IDs are out of order
                    if curr_id < prev_id:
                        row_idx = indices[i]  # The index in the original dataframe

                        # Create anomaly object
                        anomaly = {
                            "id": curr_id
                            if not isinstance(curr_id, (int, np.integer))
                            else int(curr_id),
                            "previous_id": prev_id
                            if not isinstance(prev_id, (int, np.integer))
                            else int(prev_id),
                            "row_index": int(row_idx),
                            "sort_column": col,
                            "sort_value": val
                            if not isinstance(val, (int, np.integer))
                            else int(val),
                        }

                        # Check if dependent variables also follow an unusual pattern
                        if dependent_vars:
                            out_of_order_analysis = (
                                self._analyze_out_of_order_dependent_vars(
                                    subset,
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

        logger.info(f"Found {len(anomalies)} sorting anomalies")
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
            logger.error(f"Error calculating statistical impact: {str(e)}")
            return "Error calculating statistical impact"


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
            ns = ""
            if "}" in root.tag:
                ns = root.tag.split("}")[0] + "}"
                logger.info(f"Found namespace: {ns}")

            # Process calculation chain entries
            count = 0
            for c in root.findall(".//*"):
                if c.tag.endswith("c"):  # Look for calculation cell entries
                    ref = c.get("r")
                    if ref:
                        # Convert Excel cell reference to row and column
                        col_str = "".join(filter(str.isalpha, ref))
                        row_num = int("".join(filter(str.isdigit, ref)))

                        self.calc_chain.append(
                            {"ref": ref, "row": row_num, "col": col_str}
                        )
                        count += 1

            logger.info(f"Parsed calc chain with {count} entries")
        except Exception as e:
            logger.error(f"Error parsing calc chain: {str(e)}")
            # Return empty list but don't crash
            self.calc_chain = []

    def analyze_row_movement(self, row_numbers: List[int]) -> List[Dict[str, Any]]:
        """
        Analyze the calculation chain to detect evidence of row movement.

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
            logger.warning(
                "No calculation chain data available to analyze row movement"
            )
            return findings

        try:
            logger.info(
                f"Analyzing {len(row_numbers)} suspicious rows with {len(self.calc_chain)} calc chain entries"
            )

            # Group calc chain entries by row
            row_entries = {}
            for entry in self.calc_chain:
                row = entry["row"]
                if row not in row_entries:
                    row_entries[row] = []
                row_entries[row].append(entry)

            # Find rows in our calc chain that match suspicious rows
            matching_rows = set(row_numbers).intersection(set(row_entries.keys()))
            logger.info(
                f"Found {len(matching_rows)} suspicious rows in the calculation chain"
            )

            # Find the position of each suspicious row in the calculation chain
            for row in matching_rows:
                entries = row_entries[row]
                logger.info(f"Row {row} has {len(entries)} entries in calc chain")

                # For each entry in this row, find adjacent entries in the chain
                for entry in entries:
                    idx = self.calc_chain.index(entry)

                    # Check entries before and after
                    if idx > 0 and idx < len(self.calc_chain) - 1:
                        prev_entry = self.calc_chain[idx - 1]
                        next_entry = self.calc_chain[idx + 1]

                        # Check if adjacent entries are from different rows
                        # This can indicate that the row was moved
                        if prev_entry["row"] != row and next_entry["row"] != row:
                            # More suspicious if the adjacent entries are consecutive rows
                            if abs(prev_entry["row"] - next_entry["row"]) == 1:
                                findings.append(
                                    {
                                        "row": row,
                                        "evidence": f"Cell {entry['ref']} calculation is between rows {prev_entry['row']} and {next_entry['row']}",
                                        "likely_original_position": f"between rows {prev_entry['row']} and {next_entry['row']}",
                                    }
                                )
                                logger.info(f"Found evidence of movement for row {row}")

            logger.info(
                f"Analysis complete - found {len(findings)} instances of row movement"
            )
        except Exception as e:
            logger.error(f"Error in analyze_row_movement: {str(e)}")
            traceback.print_exc()

        return findings
