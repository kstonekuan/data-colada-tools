#!/usr/bin/env python3
import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional

# Set up module logger
logger = logging.getLogger(__name__)


class ExcelForensics:
    """Class for performing forensic analysis on Excel files to detect manipulation."""

    def __init__(self, filepath: str) -> None:
        """Initialize with path to Excel file.

        Args:
            filepath: Path to Excel file for analysis
        """
        self.filepath = filepath
        self.temp_dir = None
        self.calc_chain = None
        self.namespaces = {}
        self.workbook_part = None
        self.sheet_data = {}

    def __enter__(self) -> "ExcelForensics":
        """Context manager entry method for safely working with temp files.

        Returns:
            ExcelForensics: Instance for use in a context
        """
        self.temp_dir = tempfile.mkdtemp()
        try:
            self.extract_excel()
        except Exception as e:
            if self.temp_dir:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            raise e
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit method for cleaning up temp files.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def extract_excel(self) -> None:
        """Extract Excel file (which is a ZIP archive) to temporary directory."""
        if not self.temp_dir:
            raise ValueError("Temporary directory not initialized")

        try:
            with zipfile.ZipFile(self.filepath, "r") as zip_ref:
                zip_ref.extractall(self.temp_dir)

            # Get namespaces from workbook.xml
            workbook_path = os.path.join(self.temp_dir, "xl", "workbook.xml")
            if os.path.exists(workbook_path):
                self.workbook_part = workbook_path
                tree = ET.parse(workbook_path)
                root = tree.getroot()

                # Extract namespaces
                # XML namespaces are like {http://schemas.openxmlformats.org/spreadsheetml/2006/main}
                # We extract this into a dict like {'ns': 'http://schemas...'}
                matches = re.findall(r"\{([^}]+)\}", str(root))
                if matches:
                    self.namespaces["ns"] = matches[0]

            # Parse the calculation chain if it exists
            calc_chain_path = os.path.join(self.temp_dir, "xl", "calcChain.xml")
            if os.path.exists(calc_chain_path):
                self.calc_chain = self.parse_calc_chain(calc_chain_path)

            # For each sheet, load its data
            sheets_path = os.path.join(self.temp_dir, "xl", "worksheets")
            if os.path.exists(sheets_path):
                for sheet_file in os.listdir(sheets_path):
                    if sheet_file.startswith("sheet") and sheet_file.endswith(".xml"):
                        sheet_path = os.path.join(sheets_path, sheet_file)
                        sheet_index = int(re.search(r"sheet(\d+)", sheet_file).group(1))
                        self.sheet_data[sheet_index] = sheet_path

        except Exception as e:
            logger.error(f"Error extracting Excel file: {e}")
            if self.temp_dir:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            raise e

    def parse_calc_chain(self, calc_chain_path: str) -> List[Dict[str, Any]]:
        """Parse Excel's calcChain.xml to extract calculation order.

        Args:
            calc_chain_path: Path to calcChain.xml file

        Returns:
            List[Dict[str, Any]]: List of calculation entries with sheet, row, column info
        """
        try:
            # Parse the XML file
            tree = ET.parse(calc_chain_path)
            root = tree.getroot()

            # Extract the namespace if present in calcChain.xml
            matches = re.findall(r"\{([^}]+)\}", str(root))
            if matches:
                ns = {"ns": matches[0]}
            else:
                # Fall back to the namespace from workbook.xml
                ns = self.namespaces

            # Build xpath with namespace
            if ns:
                prefix = "{" + ns.get("ns", "") + "}"
                xpath = f".//{prefix}c"
            else:
                # Try without namespace
                xpath = ".//c"

            # Extract calculation entries
            entries = []
            for i, c_elem in enumerate(root.findall(xpath, ns) or root.findall(".//c")):
                entry = {
                    "index": i,  # Position in calc chain
                    "sheet_id": c_elem.get("i"),  # Sheet index
                    "cell_ref": c_elem.get("r"),  # Cell reference (e.g., A1)
                }

                # Parse the cell reference to extract row and column
                if entry["cell_ref"]:
                    match = re.match(r"([A-Z]+)(\d+)", entry["cell_ref"])
                    if match:
                        col_letter, row_num = match.groups()
                        entry["column"] = col_letter
                        entry["row"] = int(row_num)

                entries.append(entry)

            return entries

        except Exception as e:
            logger.error(f"Error parsing calcChain.xml: {e}")
            return []

    def analyze_row_movement(self, row_numbers: List[int]) -> List[Dict[str, Any]]:
        """Analyze the calculation chain to detect evidence of row movement.

        This analysis is based on the principle that when rows are moved in Excel,
        the calculation order remains intact. For example, if a cell from row 7 was moved
        to row 12, its calculation entry in calcChain.xml would still show it being
        calculated between cells from rows 6 and 8.

        Args:
            row_numbers: List of suspicious row numbers to check

        Returns:
            List[Dict[str, Any]]: Evidence of potential row movement
        """
        if not self.calc_chain:
            logger.warning("No calculation chain found in Excel file")
            return []

        # Multiple mapping strategies to handle different Excel formats
        # Strategy 1: Data row index is Excel row number - 1 (Excel is 1-indexed)
        # Strategy 2: Data row index is Excel row number - 2 (headers + 1-indexed)
        mapping_strategies = [
            {
                "offset": 1,
                "description": "Excel row number (1-indexed) = Data row index + 1",
            },
            {
                "offset": 2,
                "description": "Excel row number (1-indexed) = Data row index + 2",
            },
        ]

        all_findings = []

        for strategy in mapping_strategies:
            offset = strategy["offset"]

            # Map suspicious row indices to Excel row numbers
            excel_rows = [r + offset for r in row_numbers]

            # Look for out-of-sequence calculations
            findings = self._analyze_calculation_order(excel_rows, strategy)

            if findings:
                all_findings.extend(findings)

        # Consolidate findings if the same row appears multiple times
        consolidated = {}
        for finding in all_findings:
            data_row = finding["data_row_index"]
            if data_row not in consolidated:
                consolidated[data_row] = {
                    "data_row_index": data_row,
                    "evidence": [],
                    "confidence": "low",
                    "confidence_score": 0,
                    "explanations": [],
                    "mapping_strategies": [],
                    "probable_original_position": None,
                }

            # Add this piece of evidence
            consolidated[data_row]["evidence"].append(finding["evidence"])
            consolidated[data_row]["explanations"].append(finding["explanation"])

            # Track which mapping strategy was used
            if (
                finding["mapping_strategy"]
                not in consolidated[data_row]["mapping_strategies"]
            ):
                consolidated[data_row]["mapping_strategies"].append(
                    finding["mapping_strategy"]
                )

            # Update confidence
            evidence_score = finding.get("confidence_score", 1)
            consolidated[data_row]["confidence_score"] += evidence_score

            # Collect possible original positions
            if finding.get("probable_original_position"):
                if not consolidated[data_row]["probable_original_position"]:
                    consolidated[data_row]["probable_original_position"] = finding[
                        "probable_original_position"
                    ]
                # If multiple positions, choose the one with higher confidence
                elif finding.get("confidence_score", 0) > consolidated[data_row].get(
                    "confidence_score", 0
                ):
                    consolidated[data_row]["probable_original_position"] = finding[
                        "probable_original_position"
                    ]

        # Set the final confidence level
        for data_row, finding in consolidated.items():
            score = finding["confidence_score"]
            if score >= 5:
                finding["confidence"] = "high"
            elif score >= 3:
                finding["confidence"] = "medium"
            else:
                finding["confidence"] = "low"

            # Join explanations into a single string
            finding["explanation"] = " Furthermore, ".join(finding["explanations"])

        return list(consolidated.values())

    def _analyze_calculation_order(
        self, excel_rows: List[int], strategy: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze calculation order to detect anomalies that suggest row movement.

        Args:
            excel_rows: List of suspicious Excel row numbers to check
            strategy: Row mapping strategy information

        Returns:
            List[Dict[str, Any]]: Evidence of potential row movement
        """
        if not self.calc_chain:
            return []

        findings = []

        # Filter calc chain entries with valid row information
        calc_entries = [entry for entry in self.calc_chain if "row" in entry]

        # Group entries by sheet
        sheet_entries = {}
        for entry in calc_entries:
            sheet_id = entry.get(
                "sheet_id", "1"
            )  # Default to first sheet if not specified
            if sheet_id not in sheet_entries:
                sheet_entries[sheet_id] = []
            sheet_entries[sheet_id].append(entry)

        # For each sheet, analyze the calculation order
        for sheet_id, entries in sheet_entries.items():
            # Sort entries by their position in the calc chain
            entries.sort(key=lambda e: e["index"])

            # Find suspicious rows in this sheet's calc chain
            suspicious_entries = []
            for entry in entries:
                if entry["row"] in excel_rows:
                    suspicious_entries.append(entry)

            # For each suspicious entry, check if it's calculated between rows
            # that would indicate it was moved
            for entry in suspicious_entries:
                evidence = self._check_entry_for_movement(
                    entry, entries, excel_rows, strategy
                )
                if evidence:
                    # Calculate data row index from Excel row
                    data_row_index = entry["row"] - strategy["offset"]

                    finding = {
                        "data_row_index": data_row_index,
                        "excel_row": entry["row"],
                        "evidence": evidence["evidence"],
                        "explanation": evidence["explanation"],
                        "confidence_score": evidence["confidence"],
                        "mapping_strategy": strategy["description"],
                    }

                    # If we can determine a probable original position
                    if (
                        "probable_original_position" in evidence
                        and evidence["probable_original_position"] is not None
                    ):
                        # Convert Excel row number to data row index
                        orig_excel_row = evidence["probable_original_position"]
                        orig_data_row = orig_excel_row - strategy["offset"]
                        finding["probable_original_position"] = orig_data_row

                    findings.append(finding)

        return findings

    def _check_entry_for_movement(
        self,
        entry: Dict[str, Any],
        all_entries: List[Dict[str, Any]],
        suspicious_rows: List[int],
        strategy: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check if a calculation entry shows evidence of row movement.

        Args:
            entry: The calculation entry to check
            all_entries: All calculation entries for this sheet
            suspicious_rows: Excel row numbers that are suspicious
            strategy: Row mapping strategy information

        Returns:
            Optional[Dict[str, Any]]: Evidence of row movement, if found
        """
        # Find the index of this entry in the calculation chain
        try:
            entry_idx = all_entries.index(entry)
        except ValueError:
            return None

        # If this is the first or last entry, we can't check neighbors
        if entry_idx == 0 or entry_idx == len(all_entries) - 1:
            return None

        # Get neighboring entries to check if they're calculated in sequence
        prev_entries = all_entries[max(0, entry_idx - 5) : entry_idx]
        next_entries = all_entries[entry_idx + 1 : min(len(all_entries), entry_idx + 6)]

        evidence = None
        confidence = 0
        explanation = ""
        probable_original_position = None

        # Check if calculated between rows that suggest movement
        if prev_entries and next_entries:
            # Get valid rows from neighboring entries
            valid_prev = [e for e in prev_entries if "row" in e]
            valid_next = [e for e in next_entries if "row" in e]

            if valid_prev and valid_next:
                # Last valid previous entry
                prev_entry = valid_prev[-1]
                # First valid next entry
                next_entry = valid_next[0]

                # If the row is calculated between two consecutive rows
                if (
                    next_entry["row"] - prev_entry["row"] > 1
                    and prev_entry["row"] < entry["row"] < next_entry["row"]
                ):
                    # This is a strong indicator - high confidence
                    confidence = 4
                    evidence = {
                        "type": "calculation_order_anomaly",
                        "suspicious_row": entry["row"],
                        "prev_row": prev_entry["row"],
                        "next_row": next_entry["row"],
                        "calc_index": entry["index"],
                        "cell_ref": entry["cell_ref"],
                    }
                    explanation = (
                        f"Cell {entry['cell_ref']} in row {entry['row']} was "
                        + f"calculated between rows {prev_entry['row']} and {next_entry['row']}, "
                        + "strongly suggesting it was moved from its original position."
                    )

                    # The probable original position is between the prev and next rows
                    probable_original_position = prev_entry["row"] + 1

                # Check for more complex patterns
                elif (
                    abs(prev_entry["row"] - entry["row"]) > 3
                    or abs(next_entry["row"] - entry["row"]) > 3
                ):
                    # Significant gap - moderate confidence
                    confidence = 2
                    evidence = {
                        "type": "calculation_gap",
                        "suspicious_row": entry["row"],
                        "prev_row": prev_entry["row"],
                        "next_row": next_entry["row"],
                        "calc_index": entry["index"],
                        "cell_ref": entry["cell_ref"],
                    }
                    explanation = (
                        f"Cell {entry['cell_ref']} in row {entry['row']} is calculated "
                        + f"in sequence with rows {prev_entry['row']} and {next_entry['row']}, "
                        + "which are significantly distant, suggesting row movement."
                    )

                    # Calculate the middle of the range as a possible original position
                    if prev_entry["row"] < entry["row"] < next_entry["row"]:
                        probable_original_position = prev_entry["row"] + 1
                    elif next_entry["row"] < entry["row"] < prev_entry["row"]:
                        probable_original_position = next_entry["row"] + 1

        # Look for patterns in a broader context - see if this row's position is inconsistent
        if not evidence:
            # Get rows that are calculated near this entry
            context_entries = all_entries[
                max(0, entry_idx - 10) : min(len(all_entries), entry_idx + 11)
            ]
            context_rows = [e["row"] for e in context_entries if "row" in e]

            if context_rows:
                # Check if the row is an outlier in terms of position
                if len(context_rows) >= 5:
                    median_row = sorted(context_rows)[len(context_rows) // 2]
                    distance = abs(entry["row"] - median_row)

                    if distance > 5:
                        confidence = 1
                        evidence = {
                            "type": "position_outlier",
                            "suspicious_row": entry["row"],
                            "median_context_row": median_row,
                            "distance": distance,
                            "calc_index": entry["index"],
                            "cell_ref": entry["cell_ref"],
                        }
                        explanation = (
                            f"Cell {entry['cell_ref']} in row {entry['row']} is calculated "
                            + f"together with rows around {median_row}, suggesting it may "
                            + f"have been moved {distance} rows from its original position."
                        )

                        # Estimate the original position as being near the median row
                        probable_original_position = median_row

        if evidence:
            return {
                "evidence": evidence,
                "confidence": confidence,
                "explanation": explanation,
                "probable_original_position": probable_original_position,
            }

        return None
