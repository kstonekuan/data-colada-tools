#!/usr/bin/env python3
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DataForensics:
    """Class for detecting potential data manipulation in research datasets."""

    def __init__(self):
        """Initialize the DataForensics object."""
        self.findings = []
        self.plots = []

    def analyze_dataset(self, filepath, id_col=None, sort_cols=None):
        """Analyze a dataset for potential manipulation."""
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
                encodings = ["latin1", "cp1252", None]  # None lets pyreadstat try to detect encoding
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
                    raise ValueError(f"Could not read SPSS file with any encoding: {last_error}")
            except ImportError:
                raise ValueError("The pyreadstat package is required to read SPSS files")
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

        return self.findings

    def check_sorting_anomalies(self, id_col, sort_cols):
        """Check for anomalies in sorting order that might indicate manipulation."""
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

    def check_duplicate_ids(self, id_col):
        """Check for duplicate ID values that might indicate manipulation."""
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

    def extract_excel_metadata(self, excel_file):
        """Extract metadata from Excel file including calcChain information."""
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

    def analyze_calc_chain(self, suspect_rows):
        """Analyze the calcChain to detect row movement."""
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

    def analyze_statistical_anomalies(self):
        """Look for statistical anomalies in the data."""
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

    def analyze_suspicious_observations(self, suspicious_rows, group_col, outcome_vars):
        """Analyze whether suspicious observations show a strong effect in the expected direction."""
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

    def generate_report(self):
        """Generate an HTML report of the findings."""
        if not self.findings:
            return "No anomalies detected in the data."

        report = "<html><head><title>Data Forensics Report</title>"
        report += "<style>body{font-family:Arial; margin:20px;} "
        report += ".finding{margin:20px 0; padding:15px; border:1px solid #ddd; border-radius:5px;} "
        report += ".important{color:red; font-weight:bold;}</style></head><body>"
        report += "<h1>Data Forensics Report</h1>"

        # Summarize findings
        report += "<h2>Summary of Findings</h2>"
        report += f"<p>Found {len(self.findings)} potential issues in the data.</p>"

        # Detailed findings
        report += "<h2>Detailed Findings</h2>"

        for i, finding in enumerate(self.findings):
            report += f"<div class='finding'>"
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

            report += "</div>"

        report += "</body></html>"
        return report


class ExcelForensics:
    """Specialized class for forensic analysis of Excel files."""

    def __init__(self, excel_file):
        """Initialize with an Excel file path."""
        self.excel_file = excel_file
        self.temp_dir = None
        self.extracted = False
        self.calc_chain = []

    def __enter__(self):
        """Context manager entry - extract the Excel file."""
        self.temp_dir = tempfile.mkdtemp()
        self.extract_excel()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def extract_excel(self):
        """Extract Excel file contents."""
        with zipfile.ZipFile(self.excel_file, "r") as zip_ref:
            zip_ref.extractall(self.temp_dir)
        self.extracted = True

        # Parse calculation chain
        self.parse_calc_chain()

    def parse_calc_chain(self):
        """Parse Excel calculation chain XML file."""
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
            print("No calculation chain found in this Excel file.")

    def analyze_row_movement(self, row_numbers):
        """
        Analyze the calculation chain to detect evidence of row movement.

        Args:
            row_numbers: List of row numbers to check for movement

        Returns:
            List of findings about row movements
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
