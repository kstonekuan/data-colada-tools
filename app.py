#!/usr/bin/env python3
import datetime
import json
import logging
import os
import re
import sys
import uuid

# Use Agg backend for matplotlib to avoid GUI dependencies - must be set before any matplotlib or seaborn imports
import matplotlib
matplotlib.use('Agg')

# Core dependencies that don't require visualization
import numpy as np
import pandas as pd
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

# Import required functions from main - lazy load other dependencies
from src.main import (
    analyze_column_unique_values,
    detect_data_manipulation,
    setup_client,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("app.log")],
)


# Log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Handle keyboard interrupt differently
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

# Try to import markdown, fallback to basic HTML conversion if not available
try:
    import markdown

    HAS_MARKDOWN = True
except ImportError:
    logging.warning("Markdown package not found, using basic HTML conversion instead")
    HAS_MARKDOWN = False

# Matplotlib backend is already set to Agg at the top of the file


# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)


# Register error handlers
@app.errorhandler(403)
def forbidden_error(error):
    logging.error(f"403 Forbidden error: {error}")
    flash("Access denied: You don't have permission to access this resource")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found_error(error):
    logging.error(f"404 Not Found error: {error}")
    flash("Resource not found: The requested page or file does not exist")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(error):
    logging.error(f"500 Internal Server error: {error}")
    flash("Server error: An unexpected error occurred. Please try again later.")
    return redirect(url_for("index"))


def basic_markdown_to_html(md_text):
    """Simple markdown to HTML converter for fallback when markdown package isn't available"""
    html = md_text

    # Convert headers
    html = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)

    # Convert code blocks
    html = re.sub(
        r"```json\n(.*?)\n```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL
    )
    html = re.sub(r"```(.*?)```", r"<pre><code>\1</code></pre>", html, flags=re.DOTALL)

    # Convert images
    html = re.sub(r"!\[(.*?)\]\((.*?)\)", r'<img src="\2" alt="\1">', html)

    # Convert bold
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)

    # Convert italics
    html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

    # Convert paragraphs (simplified)
    html = "<p>" + html.replace("\n\n", "</p><p>") + "</p>"

    return f'<div class="markdown-content">{html}</div>'


app.config["UPLOAD_FOLDER"] = "uploads"
app.config["RESULTS_FOLDER"] = "results"
app.config["SAMPLES_FOLDER"] = "samples"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["ALLOWED_EXTENSIONS"] = {"xlsx", "csv", "dta", "sav"}

# Create necessary directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs(app.config["SAMPLES_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(app.config["SAMPLES_FOLDER"], "datasets"), exist_ok=True)
os.makedirs(os.path.join(app.config["SAMPLES_FOLDER"], "papers"), exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    # Get available sample datasets
    samples = []
    samples_dir = os.path.join(app.config["SAMPLES_FOLDER"], "datasets")
    papers_dir = os.path.join(app.config["SAMPLES_FOLDER"], "papers")

    if os.path.exists(samples_dir):
        for filename in os.listdir(samples_dir):
            if allowed_file(filename):
                sample = {
                    "filename": filename,
                    "path": os.path.join(samples_dir, filename),
                    "name": os.path.splitext(filename)[0].replace("_", " ").title(),
                    "type": os.path.splitext(filename)[1][1:].upper(),
                }

                # Check if there's a matching paper
                paper_name = os.path.splitext(filename)[0] + ".pdf"
                if os.path.exists(os.path.join(papers_dir, paper_name)):
                    sample["paper"] = paper_name
                    sample["paper_path"] = os.path.join(papers_dir, paper_name)

                # Check if there's a description file
                desc_name = os.path.splitext(filename)[0] + ".txt"
                if os.path.exists(os.path.join(samples_dir, desc_name)):
                    with open(os.path.join(samples_dir, desc_name), "r") as f:
                        sample["description"] = f.read().strip()

                samples.append(sample)

    return render_template("index.html", samples=samples)


@app.route("/use-sample/<path:filename>")
def use_sample(filename):
    # Validate the filename
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", os.path.basename(filename)):
        flash("Invalid sample dataset filename")
        return redirect(url_for("index"))

    # Construct the sample file path
    sample_path = os.path.join(app.config["SAMPLES_FOLDER"], "datasets", filename)

    if not os.path.exists(sample_path):
        flash("Sample dataset not found")
        return redirect(url_for("index"))

    try:
        # Create a unique folder for this analysis
        analysis_id = str(uuid.uuid4())
        analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
        os.makedirs(analysis_folder, exist_ok=True)

        # Check if there's a matching paper
        paper_path = None
        paper_name = os.path.splitext(filename)[0] + ".pdf"
        potential_paper_path = os.path.join(
            app.config["SAMPLES_FOLDER"], "papers", paper_name
        )

        if os.path.exists(potential_paper_path):
            paper_path = os.path.join(analysis_folder, paper_name)
            # Copy the paper to the analysis folder
            import shutil

            shutil.copy2(potential_paper_path, paper_path)
            has_paper = True
            paper_is_text = False
        else:
            has_paper = False
            paper_is_text = False

        # Initialize Claude client
        client = setup_client()

        # Run analysis - pass the paper path if provided
        if has_paper and paper_path:
            _report = detect_data_manipulation(
                client,
                sample_path,
                analysis_folder,
                paper_path=paper_path,
                use_claude_segmentation=True,
            )
        else:
            _report = detect_data_manipulation(
                client, sample_path, analysis_folder, use_claude_segmentation=True
            )

        # Read the dataset to get columns/data
        try:
            ext = os.path.splitext(sample_path)[1].lower()
            original_columns = []
            original_data = []

            if ext == ".xlsx":
                df = pd.read_excel(sample_path)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()
            elif ext == ".csv":
                df = pd.read_csv(sample_path)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()
            elif ext == ".dta":
                df = pd.read_stata(sample_path)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()
            elif ext == ".sav":
                # Import pyreadstat lazily
                from src.lazy_imports import get_pyreadstat
                pyreadstat = get_pyreadstat()

                df, meta = pyreadstat.read_sav(sample_path, encoding=None)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()

            # Convert non-serializable values to Python native types
            for i, row in enumerate(original_data):
                for j, val in enumerate(row):
                    if isinstance(val, (np.integer, np.floating)):
                        original_data[i][j] = val.item()
                    elif isinstance(val, np.ndarray):
                        original_data[i][j] = val.tolist()
                    elif pd.isna(val):
                        original_data[i][j] = None
        except Exception as e:
            print(f"Error preparing original data for browser: {e}")
            original_columns = []
            original_data = []

        # Store analysis information
        report_filename = f"report_{filename}.md"
        analysis_info = {
            "id": analysis_id,
            "filename": filename,
            "timestamp": datetime.datetime.now().timestamp(),
            "report_path": os.path.join(analysis_folder, report_filename),
            "report_filename": report_filename,
            "has_paper": has_paper,
            "paper_path": paper_path if paper_path else None,
            "paper_is_text": paper_is_text,
            "original_columns": original_columns,
            "original_data": original_data,
            "is_sample": True,
            "sample_name": os.path.splitext(filename)[0].replace("_", " ").title(),
        }

        # Save analysis metadata
        with open(os.path.join(analysis_folder, "analysis_info.json"), "w") as f:
            json.dump(analysis_info, f)

        return redirect(url_for("view_results", analysis_id=analysis_id))

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()

        # Log the error
        error_msg = f"CRITICAL: Sample analysis failed for {filename}: {str(e)}"
        error_box = "#" * len(error_msg)
        print(f"\n{error_box}\n{error_msg}\n{error_box}\n")
        print(f"ERROR DETAILS:\n{error_details}")

        logging.error(f"Sample analysis failed for {filename}: {str(e)}")
        logging.error(f"Traceback: {error_details}")

        error_summary = str(e)
        if len(error_summary) > 100:
            error_summary = error_summary[:100] + "..."

        flash(
            f"Analysis failed: {error_summary} (See server logs for details)", "error"
        )
        return redirect(url_for("index"))


def generate_data_preview(file_path, json_findings):
    """Generate a HTML preview of the dataset with suspicious data highlighted."""
    # Determine file type and read data
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".xlsx":
            df = pd.read_excel(file_path)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".dta":
            df = pd.read_stata(file_path)
        elif ext == ".sav":
            try:
                # Import pyreadstat lazily
                from src.lazy_imports import get_pyreadstat
                pyreadstat = get_pyreadstat()

                df, meta = pyreadstat.read_sav(file_path, encoding="latin1")
            except ImportError:
                return "<div class='alert alert-warning'>The pyreadstat package is required to read SPSS (.sav) files. Please install it with 'pip install pyreadstat'.</div>"
            except Exception as e:
                try:
                    # Try alternative encodings if the first attempt fails
                    df, meta = pyreadstat.read_sav(file_path, encoding="cp1252")
                except Exception:
                    try:
                        # Try with automatic encoding detection
                        df, meta = pyreadstat.read_sav(file_path, encoding=None)
                    except Exception:
                        return f"<div class='alert alert-danger'>Error reading SPSS file: {str(e)}. Try converting the file to CSV format first.</div>"
        else:
            return "<div class='alert alert-warning'>Unsupported file format for preview.</div>"

        # Maps for storing suspicious rows and cells
        suspicious_rows = {}
        suspicious_cells = {}

        # Debug output
        print(f"Processing {len(json_findings)} findings")

        # Extract information about suspicious data
        for finding in json_findings:
            print(f"Processing finding type: {finding['type']}")
            if finding["type"] == "sorting_anomaly":
                for anomaly in finding["details"]:
                    row_idx = int(anomaly["row_index"])  # Ensure it's an integer
                    id_val = anomaly["id"]
                    prev_id = anomaly["previous_id"]
                    sort_col = anomaly["sort_column"]
                    sort_val = anomaly["sort_value"]

                    print(f"Found sorting anomaly at row {row_idx}")

                    # Check if the anomaly has out-of-order analysis
                    out_of_order_info = ""
                    if (
                        "out_of_order_analysis" in anomaly
                        and anomaly["out_of_order_analysis"]
                    ):
                        oo_analysis = anomaly["out_of_order_analysis"]
                        sorted_by = ", ".join(oo_analysis.get("sorted_by", []))
                        breaking_pattern = oo_analysis.get("breaking_pattern", "")

                        if sorted_by and breaking_pattern:
                            out_of_order_info = f" | Values appear sorted by {sorted_by}, but {breaking_pattern}"

                        # If there are imputed original values, include them
                        if (
                            "imputed_original_values" in oo_analysis
                            and oo_analysis["imputed_original_values"]
                        ):
                            imputed_vals = oo_analysis["imputed_original_values"]
                            imputed_info = []

                            for imp in imputed_vals:
                                if all(
                                    k in imp
                                    for k in ["column", "current", "likely_original"]
                                ):
                                    imputed_info.append(
                                        f"{imp['column']}: {imp['current']} → {imp['likely_original']}"
                                    )

                            if imputed_info:
                                out_of_order_info += f" | Likely original values: {', '.join(imputed_info)}"

                        # Include statistical impact if available
                        if (
                            "statistical_impact" in oo_analysis
                            and oo_analysis["statistical_impact"]
                        ):
                            out_of_order_info += f" | Statistical impact: {oo_analysis['statistical_impact']}"

                    # Mark row as suspicious
                    if row_idx not in suspicious_rows:
                        suspicious_rows[row_idx] = []
                    suspicious_rows[row_idx].append(
                        {
                            "type": "sorting_anomaly",
                            "css_class": "sorting-anomaly",
                            "explanation": f"Sorting anomaly: ID {id_val} comes after ID {prev_id} in group {sort_col}={sort_val}{out_of_order_info}",
                            "out_of_order_analysis": anomaly.get(
                                "out_of_order_analysis", {}
                            ),
                        }
                    )

                    # Find the ID column - try common patterns for ID columns
                    id_columns = [
                        col
                        for col in df.columns
                        if any(
                            pattern in col.lower()
                            for pattern in ["id", "participant", "subject", "case"]
                        )
                    ]
                    if id_columns:
                        id_col = id_columns[0]
                        cell_key = f"{row_idx}_{id_col}"
                        # Base explanation
                        explanation = (
                            f"ID {id_val} out of sequence (previous ID: {prev_id})"
                        )

                        # Check if we have out-of-order analysis
                        css_class = "cell-highlight-sorting"
                        if (
                            "out_of_order_analysis" in anomaly
                            and anomaly["out_of_order_analysis"]
                        ):
                            explanation += " | Potential data manipulation detected"
                            # Add the out-of-order class to highlight it more prominently
                            css_class += " cell-highlight-out-of-order"

                            # Add details about specific columns that may have been manipulated
                            oo_analysis = anomaly["out_of_order_analysis"]
                            if (
                                "imputed_original_values" in oo_analysis
                                and oo_analysis["imputed_original_values"]
                            ):
                                for imp in oo_analysis["imputed_original_values"]:
                                    explanation += f" | Column {imp['column']} likely altered from {imp['likely_original']} to {imp['current']}"

                                    # Also mark the specific manipulated column cells
                                    if imp["column"] in df.columns:
                                        affected_col = imp["column"]
                                        affected_cell_key = f"{row_idx}_{affected_col}"
                                        affected_explanation = f"Value {imp['current']} is inconsistent with surrounding values. Likely original value: {imp['likely_original']}"

                                        if "statistical_impact" in oo_analysis:
                                            affected_explanation += f" | {oo_analysis['statistical_impact']}"

                                        suspicious_cells[affected_cell_key] = {
                                            "type": "sorting_value_manipulation",
                                            "css_class": "cell-highlight-out-of-order",
                                            "explanation": affected_explanation,
                                        }

                        suspicious_cells[cell_key] = {
                            "type": "sorting",
                            "css_class": css_class,
                            "explanation": explanation,
                        }

            elif finding["type"] == "duplicate_ids":
                for duplicate in finding["details"]:
                    duplicate_id = duplicate["id"]
                    row_indices = duplicate["row_indices"]

                    print(f"Found duplicate ID: {duplicate_id} in rows {row_indices}")

                    # Mark rows as suspicious
                    for row_idx in row_indices:
                        row_idx = int(row_idx)  # Ensure it's an integer
                        if row_idx not in suspicious_rows:
                            suspicious_rows[row_idx] = []
                        suspicious_rows[row_idx].append(
                            {
                                "type": "duplicate_id",
                                "css_class": "duplicate-id",
                                "explanation": f"Duplicate ID: {duplicate_id} appears {duplicate['count']} times",
                            }
                        )

                        # Find the ID column
                        id_columns = [
                            col
                            for col in df.columns
                            if any(
                                pattern in col.lower()
                                for pattern in ["id", "participant", "subject", "case"]
                            )
                        ]
                        if id_columns:
                            id_col = id_columns[0]
                            cell_key = f"{row_idx}_{id_col}"
                            suspicious_cells[cell_key] = {
                                "type": "duplicate",
                                "css_class": "cell-highlight-duplicate",
                                "explanation": f"Duplicate ID {duplicate_id} appears {duplicate['count']} times",
                            }

            elif finding["type"] == "statistical_anomaly":
                col = finding["column"]
                print(f"Found statistical anomalies in column {col}")

                for row_idx in finding["outlier_rows"]:
                    row_idx = int(row_idx)  # Ensure it's an integer
                    print(f"  - Outlier at row {row_idx}")

                    if row_idx not in suspicious_rows:
                        suspicious_rows[row_idx] = []
                    suspicious_rows[row_idx].append(
                        {
                            "type": "statistical_anomaly",
                            "css_class": "statistical-anomaly",
                            "explanation": f"Statistical outlier in {col} column (z-score > 3)",
                        }
                    )

                    cell_key = f"{row_idx}_{col}"
                    suspicious_cells[cell_key] = {
                        "type": "outlier",
                        "css_class": "cell-highlight-outlier",
                        "explanation": "Statistical outlier (z-score > 3)",
                    }

            elif finding["type"] == "excel_row_movement":
                for movement in finding["details"]:
                    row_idx = int(movement["row"])  # Ensure it's an integer
                    explanation = movement["evidence"]

                    print(f"Found Excel row movement evidence for row {row_idx}")

                    if row_idx not in suspicious_rows:
                        suspicious_rows[row_idx] = []
                    suspicious_rows[row_idx].append(
                        {
                            "type": "excel_movement",
                            "css_class": "excel-movement",
                            "explanation": explanation,
                        }
                    )

            elif finding["type"] == "claude_detected_anomaly":
                # Process anomalies detected by Claude in data segments
                detail = finding["details"]
                print(f"Processing Claude anomaly: {json.dumps(detail, indent=2)}")

                # Handle both string and list row indices formats
                row_indices = []
                if "row_indices" in detail:
                    if isinstance(detail["row_indices"], list):
                        row_indices = detail["row_indices"]
                    elif isinstance(detail["row_indices"], str):
                        # Try to parse string as a list or as a single value
                        try:
                            row_indices = json.loads(detail["row_indices"])
                            if not isinstance(row_indices, list):
                                row_indices = [row_indices]
                        except (json.JSONDecodeError, TypeError, ValueError):
                            # If JSON parsing fails, treat as a single value
                            row_indices = [detail["row_indices"]]

                # If we still don't have row indices, look for "rows" field which might contain range
                if not row_indices and "rows" in finding:
                    rows_str = finding["rows"]
                    # Parse patterns like "100-200"
                    range_match = re.match(r"(\d+)-(\d+)", rows_str)
                    if range_match:
                        start, end = (
                            int(range_match.group(1)),
                            int(range_match.group(2)),
                        )
                        # Add a sample of rows from this range to highlight (limit to 10 for performance)
                        sample_size = min(10, end - start + 1)
                        sample_indices = list(range(start, start + sample_size))
                        row_indices = sample_indices
                        print(
                            f"Using sample of {sample_size} rows from range {rows_str}"
                        )

                if row_indices:
                    for row_idx in row_indices:
                        try:
                            row_idx = int(row_idx)  # Ensure it's an integer
                            explanation = detail.get(
                                "description", "No description provided"
                            )
                            cols_involved = ", ".join(
                                detail.get("columns_involved", ["Unknown"])
                            )
                            severity = detail.get("severity", "N/A")

                            # Add handling for out_of_order_analysis
                            out_of_order_info = ""
                            if (
                                "out_of_order_analysis" in detail
                                and detail["out_of_order_analysis"]
                            ):
                                oo_analysis = detail["out_of_order_analysis"]
                                sorted_by = ", ".join(oo_analysis.get("sorted_by", []))
                                breaking_pattern = oo_analysis.get(
                                    "breaking_pattern", ""
                                )

                                if sorted_by and breaking_pattern:
                                    out_of_order_info = f" | Data appears sorted by {sorted_by}, but {breaking_pattern}"

                                # If there are imputed original values, include them
                                if (
                                    "imputed_original_values" in oo_analysis
                                    and oo_analysis["imputed_original_values"]
                                ):
                                    imputed_vals = oo_analysis[
                                        "imputed_original_values"
                                    ]
                                    imputed_info = []

                                    for imp in imputed_vals:
                                        if all(
                                            k in imp
                                            for k in [
                                                "column",
                                                "current",
                                                "likely_original",
                                            ]
                                        ):
                                            imputed_info.append(
                                                f"{imp['column']}: {imp['current']} → {imp['likely_original']}"
                                            )

                                    if imputed_info:
                                        out_of_order_info += f" | Likely original values: {', '.join(imputed_info)}"

                                # Include statistical impact if available
                                if (
                                    "statistical_impact" in oo_analysis
                                    and oo_analysis["statistical_impact"]
                                ):
                                    out_of_order_info += f" | Statistical impact: {oo_analysis['statistical_impact']}"

                            print(
                                f"Found Claude-detected anomaly at row {row_idx}, severity: {severity}"
                            )

                            if row_idx not in suspicious_rows:
                                suspicious_rows[row_idx] = []
                            suspicious_rows[row_idx].append(
                                {
                                    "type": "claude_detected_anomaly",
                                    "css_class": "claude-anomaly",
                                    "explanation": f"{explanation} (Columns: {cols_involved}, Severity: {severity}/10){out_of_order_info}",
                                    "out_of_order_analysis": detail.get(
                                        "out_of_order_analysis", {}
                                    ),
                                }
                            )

                            # Mark the specific columns as suspicious
                            columns_to_mark = detail.get("columns_involved", [])
                            if not columns_to_mark and len(df.columns) > 0:
                                # If no columns specified, mark the first column
                                columns_to_mark = [df.columns[0]]

                            for col in columns_to_mark:
                                if col in df.columns:
                                    cell_key = f"{row_idx}_{col}"

                                    # Check if this is an out-of-order cell
                                    css_class = "cell-highlight-claude"
                                    if (
                                        "out_of_order_analysis" in detail
                                        and detail["out_of_order_analysis"]
                                    ):
                                        # Check if this specific column is mentioned in imputed values
                                        oo_analysis = detail["out_of_order_analysis"]
                                        if (
                                            "imputed_original_values" in oo_analysis
                                            and oo_analysis["imputed_original_values"]
                                        ):
                                            for imp in oo_analysis[
                                                "imputed_original_values"
                                            ]:
                                                if imp.get("column") == col:
                                                    css_class += (
                                                        " cell-highlight-out-of-order"
                                                    )
                                                    explanation += f" | Original value likely {imp.get('likely_original')} (current: {imp.get('current')})"

                                    suspicious_cells[cell_key] = {
                                        "type": "claude_anomaly",
                                        "css_class": css_class,
                                        "explanation": explanation,
                                    }
                        except Exception as e:
                            import traceback

                            error_details = traceback.format_exc()
                            print(
                                f"ERROR: Processing Claude anomaly row {row_idx} failed: {str(e)}"
                            )
                            print(f"ERROR DETAILS: {error_details}")
                            continue

        # Log the suspicious rows and cells found
        print(
            f"Found {len(suspicious_rows)} suspicious rows and {len(suspicious_cells)} suspicious cells"
        )

        if not suspicious_rows:
            return "<div class='alert alert-info'>No suspicious data points were found in this dataset.</div>"

        # Group suspicious rows by type for organized display
        anomaly_types = {
            "sorting_anomaly": {
                "title": "Sorting Anomalies",
                "rows": [],
                "description": "Rows with IDs out of sequence",
            },
            "duplicate_id": {
                "title": "Duplicate IDs",
                "rows": [],
                "description": "Rows with duplicate ID values",
            },
            "statistical_anomaly": {
                "title": "Statistical Outliers",
                "rows": [],
                "description": "Rows with values that are statistical outliers (z-score > 3)",
            },
            "excel_movement": {
                "title": "Excel Row Movement",
                "rows": [],
                "description": "Rows with evidence of Excel manipulation from metadata",
            },
            "claude_detected_anomaly": {
                "title": "Claude AI Detected Anomalies",
                "rows": [],
                "description": "Anomalies detected by Claude's direct analysis of data segments",
                "css_class": "claude-anomaly",
                "cell_class": "cell-highlight-claude",
                "border_color": "#6f42c1",  # Purple
            },
        }

        # Categorize suspicious rows
        for idx, issues in suspicious_rows.items():
            for issue in issues:
                issue_type = issue["type"]
                if issue_type in anomaly_types:
                    if idx not in [r["idx"] for r in anomaly_types[issue_type]["rows"]]:
                        anomaly_types[issue_type]["rows"].append(
                            {"idx": idx, "explanation": issue["explanation"]}
                        )

        # Start building HTML output
        html_output = '<div class="suspicious-data-summary mb-4">\n'
        html_output += '  <div class="alert alert-warning">\n'
        html_output += f'    <i class="fas fa-exclamation-triangle me-2"></i>Found {len(suspicious_rows)} suspicious rows in the dataset.\n'
        html_output += "  </div>\n"

        # Add a legend to explain the borders
        html_output += '  <div class="card mb-3">\n'
        html_output += '    <div class="card-header bg-light">\n'
        html_output += '      <h6 class="m-0">Legend</h6>\n'
        html_output += "    </div>\n"
        html_output += '    <div class="card-body p-3">\n'
        html_output += '      <div class="row">\n'

        # Sorting anomalies
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #dc3545; margin-right: 8px;"></div>\n'
        html_output += (
            "            <span><strong>Red:</strong> Sorting Anomalies</span>\n"
        )
        html_output += "          </div>\n"
        html_output += "        </div>\n"

        # Duplicate IDs
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #ffc107; margin-right: 8px;"></div>\n'
        html_output += (
            "            <span><strong>Yellow:</strong> Duplicate IDs</span>\n"
        )
        html_output += "          </div>\n"
        html_output += "        </div>\n"

        # Statistical outliers
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #0dcaf0; margin-right: 8px;"></div>\n'
        html_output += (
            "            <span><strong>Blue:</strong> Statistical Outliers</span>\n"
        )
        html_output += "          </div>\n"
        html_output += "        </div>\n"

        # Excel movement
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #6c757d; margin-right: 8px;"></div>\n'
        html_output += (
            "            <span><strong>Gray:</strong> Excel Movement</span>\n"
        )
        html_output += "          </div>\n"
        html_output += "        </div>\n"

        # Claude detected anomalies - add this in the second row
        html_output += "      </div>\n"
        html_output += '      <div class="row mt-2">\n'
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #6f42c1; margin-right: 8px;"></div>\n'
        html_output += "            <span><strong>Purple:</strong> Claude AI Anomalies (strikethrough = out-of-order values)</span>\n"
        html_output += "          </div>\n"
        html_output += "        </div>\n"

        html_output += "      </div>\n"
        html_output += '      <div class="small text-muted mt-2">Hover over highlighted cells for detailed explanations of what makes them suspicious.</div>\n'
        html_output += "    </div>\n"
        html_output += "  </div>\n"
        html_output += "</div>\n"

        # Create tabs for different types of suspicious data
        html_output += (
            '<ul class="nav nav-tabs mb-3" id="suspiciousDataTabs" role="tablist">\n'
        )

        # Add tab headers
        active = True
        for anomaly_id, anomaly_info in anomaly_types.items():
            if anomaly_info["rows"]:
                tab_active = "active" if active else ""
                html_output += '  <li class="nav-item" role="presentation">\n'
                html_output += f'    <button class="nav-link {tab_active}" id="{anomaly_id}-tab" data-bs-toggle="tab" data-bs-target="#{anomaly_id}" type="button" role="tab">\n'
                html_output += (
                    f"      {anomaly_info['title']} ({len(anomaly_info['rows'])})\n"
                )
                html_output += "    </button>\n"
                html_output += "  </li>\n"
                active = False

        html_output += "</ul>\n"

        # Add tab content
        html_output += '<div class="tab-content" id="suspiciousDataTabsContent">\n'

        # Process each anomaly type
        active = True
        for anomaly_id, anomaly_info in anomaly_types.items():
            if not anomaly_info["rows"]:
                continue

            tab_active = "show active" if active else ""
            html_output += f'  <div class="tab-pane fade {tab_active}" id="{anomaly_id}" role="tabpanel">\n'
            html_output += '    <div class="card mb-4">\n'
            html_output += '      <div class="card-header bg-light">\n'
            html_output += f'        <h5 class="mb-0">{anomaly_info["title"]}</h5>\n'
            html_output += f'        <p class="small text-muted mb-0">{anomaly_info["description"]}</p>\n'
            html_output += "      </div>\n"
            html_output += '      <div class="card-body">\n'

            # Create data table for this anomaly type
            html_output += '        <div class="table-responsive">\n'
            html_output += f'          <table id="{anomaly_id}-table" class="table table-hover table-bordered">\n'
            html_output += '            <thead class="table-light">\n'
            html_output += "              <tr>\n"
            html_output += "                <th>Row</th>\n"

            # Only include relevant columns for each anomaly type to avoid information overload
            relevant_columns = []

            if anomaly_id == "sorting_anomaly":
                # For sorting anomalies, show ID columns and sort columns
                id_columns = [
                    col
                    for col in df.columns
                    if any(
                        pattern in col.lower()
                        for pattern in ["id", "participant", "subject", "case"]
                    )
                ]

                # Safely extract sort columns
                sort_columns = []
                for idx, issues in suspicious_rows.items():
                    for anomaly in issues:
                        if (
                            anomaly["type"] == "sorting_anomaly"
                            and "sort_column" in anomaly
                        ):
                            sort_columns.append(anomaly["sort_column"])

                sort_columns = list(set(sort_columns))

                relevant_columns = id_columns + sort_columns
                relevant_columns = list(
                    dict.fromkeys(relevant_columns)
                )  # Remove duplicates while preserving order

            elif anomaly_id == "duplicate_id":
                # For duplicate IDs, show ID columns and a few other key columns
                id_columns = [
                    col
                    for col in df.columns
                    if any(
                        pattern in col.lower()
                        for pattern in ["id", "participant", "subject", "case"]
                    )
                ]
                relevant_columns = id_columns[:1]  # Just the primary ID column

                # Add a few other discriminating columns
                other_columns = [col for col in df.columns if col not in id_columns]
                if other_columns:
                    relevant_columns.extend(
                        other_columns[:3]
                    )  # Add up to 3 more columns

            elif anomaly_id == "statistical_anomaly":
                # For statistical anomalies, show columns with outliers and ID columns
                outlier_columns = []

                # Safely extract outlier columns
                if json_findings:
                    for finding in json_findings:
                        if (
                            finding.get("type") == "statistical_anomaly"
                            and "column" in finding
                        ):
                            outlier_columns.append(finding["column"])

                outlier_columns = list(set(outlier_columns))

                id_columns = [
                    col
                    for col in df.columns
                    if any(
                        pattern in col.lower()
                        for pattern in ["id", "participant", "subject", "case"]
                    )
                ]
                relevant_columns = (
                    id_columns[:1] + outlier_columns
                )  # ID column + columns with outliers

            elif anomaly_id == "excel_movement":
                # For Excel movement, show ID columns and a few key columns
                id_columns = [
                    col
                    for col in df.columns
                    if any(
                        pattern in col.lower()
                        for pattern in ["id", "participant", "subject", "case"]
                    )
                ]
                relevant_columns = id_columns[:1]  # Just the primary ID column

                # Add a few other discriminating columns
                group_columns = [
                    col
                    for col in df.columns
                    if any(
                        pattern in col.lower()
                        for pattern in ["group", "condition", "treatment"]
                    )
                ]
                if group_columns:
                    relevant_columns.extend(group_columns[:2])

                # Add outcome columns if we can detect them
                outcome_columns = [
                    col
                    for col in df.columns
                    if any(
                        pattern in col.lower()
                        for pattern in ["score", "result", "outcome", "dependent"]
                    )
                ]
                if outcome_columns:
                    relevant_columns.extend(outcome_columns[:2])

            # Ensure we have some columns (fallback)
            if not relevant_columns and len(df.columns) > 0:
                relevant_columns = df.columns[: min(5, len(df.columns))]

            # Add column headers
            for col in relevant_columns:
                html_output += f"                <th>{col}</th>\n"

            # Add explanation header
            html_output += "                <th>Issue</th>\n"
            html_output += "              </tr>\n"
            html_output += "            </thead>\n"
            html_output += "            <tbody>\n"

            # Sort rows by index for consistent display
            try:
                sorted_rows = sorted(
                    anomaly_info.get("rows", []), key=lambda x: x.get("idx", 0)
                )
            except Exception as sort_error:
                print(f"Error sorting rows: {sort_error}")
                sorted_rows = anomaly_info.get("rows", [])

            # Add row data
            for row_info in sorted_rows:
                try:
                    idx = row_info.get("idx", -1)
                    if idx >= 0 and idx < len(df):  # Ensure the index is valid
                        row_data = df.iloc[idx]

                        # Get CSS class for this row
                        css_class = ""
                        for issue in suspicious_rows.get(idx, []):
                            if issue.get("type") == anomaly_id and "css_class" in issue:
                                css_class = issue["css_class"]
                                break

                        html_output += f'              <tr class="{css_class}">\n'
                        html_output += f"                <td><strong>{idx + 1}</strong></td>\n"  # Display 1-based row numbers

                        # Add cell data for relevant columns
                        for col in relevant_columns:
                            if col in row_data:
                                cell_value = row_data[col]
                                cell_key = f"{idx}_{col}"

                                # Make sure we have a value to display
                                if pd.isna(cell_value):
                                    cell_display = ""
                                else:
                                    cell_display = str(cell_value)

                                # Check if this cell is specifically highlighted
                                if cell_key in suspicious_cells:
                                    cell_info = suspicious_cells[cell_key]
                                    html_output += f'                <td class="{cell_info.get("css_class", "")}" data-bs-toggle="tooltip" data-bs-placement="top" title="{cell_info.get("explanation", "")}">{cell_display}</td>\n'
                                else:
                                    html_output += (
                                        f"                <td>{cell_display}</td>\n"
                                    )
                            else:
                                html_output += (
                                    "                <td></td>\n"  # Column not found
                                )

                        # Add explanation
                        explanation = row_info.get("explanation", "Unknown issue")
                        html_output += f"                <td>{explanation}</td>\n"
                        html_output += "              </tr>\n"
                except Exception as row_error:
                    print(f"Error processing row {row_info}: {row_error}")
                    continue

            html_output += "            </tbody>\n"
            html_output += "          </table>\n"
            html_output += "        </div>\n"
            html_output += "      </div>\n"
            html_output += "    </div>\n"
            html_output += "  </div>\n"

            active = False

        html_output += "</div>\n"

        # Add a link to view full dataset
        html_output += '<div class="text-center mt-4">\n'
        html_output += (
            '  <button id="show-full-data" class="btn btn-outline-primary">\n'
        )
        html_output += (
            '    <i class="fas fa-table me-2"></i>Show Full Dataset Preview\n'
        )
        html_output += "  </button>\n"
        html_output += "</div>\n"

        # Create a hidden div with the full dataset table
        html_output += (
            '<div id="full-dataset-preview" style="display: none; margin-top: 30px;">\n'
        )
        html_output += (
            '  <div class="d-flex justify-content-between align-items-center mb-3">\n'
        )
        html_output += '    <h5 class="mb-0">Full Dataset Preview</h5>\n'
        html_output += '    <span class="badge bg-light text-dark border">Highlighted rows contain suspicious data</span>\n'
        html_output += "  </div>\n"

        # Create full table
        html_output += (
            '  <table id="data-preview-table" class="table table-hover table-sm">\n'
        )
        html_output += "    <thead>\n"
        html_output += "      <tr>\n"
        html_output += "        <th>#</th>\n"

        for col in df.columns:
            html_output += f"        <th>{col}</th>\n"

        html_output += "      </tr>\n"
        html_output += "    </thead>\n"
        html_output += "    <tbody>\n"

        # Process all rows, but optimize the display to focus on suspicious rows
        suspicious_indices = list(suspicious_rows.keys())

        # Debug the suspicious rows
        print(f"Full dataset preview - Suspicious rows: {suspicious_indices}")

        # Determine which rows to display:
        # 1. Always include suspicious rows
        # 2. Include some rows before and after each suspicious row for context
        # 3. Add ellipses between non-adjacent sections

        context_rows = (
            2  # Number of rows to show before/after each suspicious row for context
        )
        rows_to_display = set()

        # Add suspicious rows and their context rows
        for sus_idx in suspicious_indices:
            # Add the suspicious row itself
            rows_to_display.add(sus_idx)

            # Add context rows before and after
            for i in range(
                max(0, sus_idx - context_rows), min(len(df), sus_idx + context_rows + 1)
            ):
                rows_to_display.add(i)

        # Always include the first few rows for reference
        for i in range(min(5, len(df))):
            rows_to_display.add(i)

        # Get sorted list of rows to display
        display_indices = sorted(list(rows_to_display))

        # Process rows and add ellipses between non-adjacent sections
        last_idx = -1
        row_num = 1

        for i, idx in enumerate(display_indices):
            # Check if we need to add an ellipsis row
            if last_idx != -1 and idx > last_idx + 1:
                # Add ellipsis row
                html_output += '      <tr class="table-secondary">\n'
                html_output += f'        <td colspan="{len(df.columns) + 1}" class="text-center">\n'
                html_output += f"          <em>⋮ ⋮ ⋮ {idx - last_idx - 1} rows without issues omitted ⋮ ⋮ ⋮</em>\n"
                html_output += "        </td>\n"
                html_output += "      </tr>\n"

            # Get the row data
            row = df.iloc[idx]

            # Check if this is a suspicious row
            row_classes = []
            title_text = []

            # Use original DataFrame index to check for suspicious rows
            if idx in suspicious_rows:
                print(
                    f"Found suspicious row at index {idx} with issues: {suspicious_rows[idx]}"
                )
                row_classes.append("suspicious-row")

                # Add all issue classes for this row
                for issue in suspicious_rows[idx]:
                    if "css_class" in issue:
                        row_classes.append(issue["css_class"])
                        print(
                            f"Adding class {issue['css_class']} to full dataset row {idx}"
                        )

                    if "explanation" in issue:
                        title_text.append(issue["explanation"])

            row_class = " ".join(row_classes)
            row_title = " | ".join(title_text)

            if row_class:
                html_output += f'      <tr class="{row_class}" title="{row_title}">\n'
            else:
                html_output += "      <tr>\n"

            # Add row number cell (showing the actual row number from the dataset)
            html_output += f"        <td><strong>{idx + 1}</strong></td>\n"

            # Add data cells
            for col in df.columns:
                try:
                    cell_value = row[col]
                    cell_key = f"{idx}_{col}"

                    # Make sure we have a value to display
                    if pd.isna(cell_value):
                        cell_display = ""
                    else:
                        cell_display = str(cell_value)

                    if cell_key in suspicious_cells:
                        cell_info = suspicious_cells[cell_key]
                        html_output += f'        <td class="{cell_info.get("css_class", "")}" data-bs-toggle="tooltip" data-bs-placement="top" title="{cell_info.get("explanation", "")}">{cell_display}</td>\n'
                    else:
                        html_output += f"        <td>{cell_display}</td>\n"
                except Exception as cell_error:
                    print(f"Error processing cell {col} in row {idx}: {cell_error}")
                    html_output += "        <td></td>\n"

            html_output += "      </tr>\n"

            # Update last_idx
            last_idx = idx
            row_num += 1

        html_output += "    </tbody>\n"
        html_output += "  </table>\n"

        # Add note about the smart preview
        displayed_rows = len(display_indices)
        skipped_rows = len(df) - displayed_rows
        if skipped_rows > 0:
            html_output += '  <div class="alert alert-info mt-3 mb-0">'
            html_output += '    <i class="fas fa-info-circle me-2"></i>'
            html_output += f"    Smart preview: Showing {displayed_rows} rows (including all suspicious rows and their context) "
            html_output += f"    from a total of {len(df)} rows. {skipped_rows} rows without issues are condensed."
            html_output += "  </div>\n"

        html_output += "</div>\n"

        # Add JavaScript to toggle full dataset view
        html_output += "<script>\n"
        html_output += '  document.addEventListener("DOMContentLoaded", function() {\n'
        html_output += (
            '    const toggleButton = document.getElementById("show-full-data");\n'
        )
        html_output += '    const fullDatasetDiv = document.getElementById("full-dataset-preview");\n'
        html_output += "    if (toggleButton && fullDatasetDiv) {\n"
        html_output += '      toggleButton.addEventListener("click", function() {\n'
        html_output += '        if (fullDatasetDiv.style.display === "none") {\n'
        html_output += '          fullDatasetDiv.style.display = "block";\n'
        html_output += "          toggleButton.innerHTML = '<i class=\"fas fa-compress me-2\"></i>Hide Full Dataset Preview';\n"
        html_output += "        } else {\n"
        html_output += '          fullDatasetDiv.style.display = "none";\n'
        html_output += "          toggleButton.innerHTML = '<i class=\"fas fa-table me-2\"></i>Show Full Dataset Preview';\n"
        html_output += "        }\n"
        html_output += "      });\n"
        html_output += "    }\n"
        html_output += "  });\n"
        html_output += "</script>\n"

        return html_output

    except Exception as e:
        import logging
        import traceback

        error_details = traceback.format_exc()

        # Create a more visible error message in the server logs
        error_msg = f"ERROR: Data preview generation failed: {str(e)}"
        error_box = "*" * len(error_msg)
        print(f"\n{error_box}\n{error_msg}\n{error_box}\n")
        print(f"ERROR DETAILS:\n{error_details}")

        # Log to application log if logger is configured
        try:
            logging.error(f"Data preview generation failed: {str(e)}")
            logging.error(f"Traceback: {error_details}")
        except Exception:
            pass  # Silently handle if logging isn't configured

        return f"<div class='alert alert-danger'>Error generating data preview: {str(e)}<br><small>Check server logs for details.</small></div>"


@app.route("/previous_results")
def previous_results():
    results = []

    # Get all analysis directories
    for analysis_id in os.listdir(app.config["RESULTS_FOLDER"]):
        analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)

        # Skip non-directories and special directories like 'visual_test'
        if not os.path.isdir(analysis_folder) or analysis_id == "visual_test":
            continue

        # Try to load the analysis info
        info_path = os.path.join(analysis_folder, "analysis_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r") as f:
                    analysis_info = json.load(f)

                # Convert timestamp to readable date
                if "timestamp" in analysis_info:
                    ts = analysis_info["timestamp"]
                    date_str = datetime.datetime.fromtimestamp(ts).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    analysis_info["date"] = date_str

                # Count visualizations
                vis_count = len(
                    [
                        f
                        for f in os.listdir(analysis_folder)
                        if f.endswith((".png", ".jpg", ".jpeg", ".svg"))
                    ]
                )
                analysis_info["visualization_count"] = vis_count

                # Check for manipulation rating
                manipulation_rating = None
                report_path = os.path.join(
                    analysis_folder, analysis_info.get("report_filename")
                )
                if os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report_content = f.read()
                        rating_match = re.search(
                            r"MANIPULATION_RATING:\s*(\d+)", report_content
                        )
                        if rating_match:
                            manipulation_rating = int(rating_match.group(1))
                analysis_info["manipulation_rating"] = manipulation_rating

                results.append(analysis_info)
            except Exception as e:
                print(f"Error loading analysis {analysis_id}: {str(e)}")

    # Sort results by timestamp (newest first)
    results.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

    return render_template("previous_results.html", results=results)


@app.route("/analyze-columns", methods=["POST"])
def analyze_columns():
    # Check if file was uploaded
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))

    file = request.files["file"]

    # Check if the file is valid
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("index"))

    if not file or not allowed_file(file.filename):
        allowed_ext = ", ".join(app.config["ALLOWED_EXTENSIONS"])
        flash(f"Invalid file type. Allowed types: {allowed_ext}")
        return redirect(url_for("index"))

    # Generate a unique analysis ID
    analysis_id = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    )
    analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
    os.makedirs(analysis_folder, exist_ok=True)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(analysis_folder, filename)
    file.save(file_path)

    # Get columns to analyze (if specified)
    columns_to_analyze = request.form.get("columns", "").strip()
    columns = columns_to_analyze.split(",") if columns_to_analyze else None

    # Set up Claude client
    try:
        client = setup_client()
    except Exception as e:
        flash(f"Error setting up Claude API client: {str(e)}")
        return redirect(url_for("index"))

    # Run the column analysis
    try:
        column_analysis = analyze_column_unique_values(
            client, file_path, columns, analysis_folder
        )

        # Save the analysis results
        analysis_path = os.path.join(analysis_folder, "column_analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(column_analysis, f, indent=2)

        # Create a human-readable report
        report_content = "# Column Analysis Report\n\n"
        report_content += f"Dataset: {filename}\n\n"
        report_content += f"Total columns analyzed: {column_analysis['summary']['total_columns_analyzed']}\n"
        report_content += f"Average suspicion rating: {column_analysis['summary']['average_suspicion']:.2f}/10\n\n"

        # Add highly suspicious columns
        if column_analysis["summary"]["suspicious_columns"]:
            report_content += "## Highly Suspicious Columns\n\n"
            for col in column_analysis["summary"]["suspicious_columns"]:
                report_content += (
                    f"### {col['column']} (Rating: {col['rating']}/10)\n\n"
                )
                if col["column"] in column_analysis:
                    report_content += (
                        column_analysis[col["column"]]["analysis"] + "\n\n"
                    )

        # Add all other columns
        report_content += "## All Columns\n\n"
        for col_name, col_data in column_analysis.items():
            if col_name != "summary" and col_name not in [
                c["column"] for c in column_analysis["summary"]["suspicious_columns"]
            ]:
                suspicion = col_data.get("suspicion_rating", "N/A")
                report_content += f"### {col_name} (Rating: {suspicion}/10)\n\n"
                report_content += (
                    col_data.get("analysis", "No analysis available") + "\n\n"
                )

        # Save the report
        report_path = os.path.join(analysis_folder, "column_analysis_report.md")
        with open(report_path, "w") as f:
            f.write(report_content)

        # Redirect to results page
        return redirect(url_for("view_column_analysis", analysis_id=analysis_id))

    except Exception as e:
        logging.error(f"Error analyzing columns: {str(e)}")
        flash(f"Error analyzing columns: {str(e)}")
        return redirect(url_for("index"))


@app.route("/column-analysis/<analysis_id>")
def view_column_analysis(analysis_id):
    # Validate analysis ID format to prevent security issues
    import logging

    if not analysis_id or not re.match(r"^[a-zA-Z0-9_\-]+$", analysis_id):
        logging.warning(f"Invalid analysis ID format: {analysis_id}")
        flash("Invalid analysis ID format")
        return redirect(url_for("index"))

    analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)

    # Check if the analysis folder exists
    if not os.path.exists(analysis_folder):
        flash("Analysis not found")
        return redirect(url_for("index"))

    # Load the column analysis results
    try:
        analysis_path = os.path.join(analysis_folder, "column_analysis.json")
        with open(analysis_path, "r") as f:
            column_analysis = json.load(f)

        # Load the report for display
        report_path = os.path.join(analysis_folder, "column_analysis_report.md")
        with open(report_path, "r") as f:
            report_content = f.read()

        # Convert markdown to HTML
        if HAS_MARKDOWN:
            report_html = markdown.markdown(report_content)
        else:
            report_html = basic_markdown_to_html(report_content)

        return render_template(
            "results.html",
            analysis_id=analysis_id,
            report=report_html,
            title="Column Analysis Results",
            has_suspicious_columns=len(column_analysis["summary"]["suspicious_columns"])
            > 0,
        )
    except Exception as e:
        logging.error(f"Error loading column analysis: {str(e)}")
        flash(f"Error loading column analysis: {str(e)}")
        return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload_file():
    # Check if file was uploaded
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    # Check if file was selected
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    # Validate file type
    if not allowed_file(file.filename):
        flash(
            f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
        )
        return redirect(request.url)

    # Check if a research paper was uploaded
    paper_file = None
    paper_path = None
    has_paper = False
    paper_is_text = False

    # Check for text-based article content
    article_text = request.form.get("article-text", "")
    if (
        article_text and len(article_text.strip()) > 100
    ):  # Ensure we have substantial content
        paper_is_text = True
        print(f"Article text provided ({len(article_text)} characters)")

    # We always have include-paper field set to 'on' now
    if "paper-file" in request.files:
        paper_file = request.files["paper-file"]

        # Check if a PDF was actually uploaded
        if paper_file.filename != "" and paper_file.filename.lower().endswith(".pdf"):
            has_paper = True
            print(f"Research paper uploaded: {paper_file.filename}")
        else:
            # No PDF uploaded or wrong file type, continue without paper
            paper_file = None
            if not paper_is_text:
                print("No valid research paper uploaded or paper was not provided")

    try:
        # Create a unique folder for this analysis
        analysis_id = str(uuid.uuid4())
        analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
        os.makedirs(analysis_folder, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Handle paper content (either file or text)
        if has_paper and paper_file:
            paper_filename = secure_filename(paper_file.filename)
            paper_path = os.path.join(analysis_folder, paper_filename)
            paper_file.save(paper_path)
            print(f"Saved research paper to: {paper_path}")
        elif paper_is_text:
            # Save article text to a text file
            paper_path = os.path.join(analysis_folder, "article_text.txt")
            with open(paper_path, "w") as text_file:
                text_file.write(article_text)
            has_paper = True
            print(f"Saved article text to: {paper_path}")

        # Initialize Claude client
        client = setup_client()

        # Check for encoding issues with .sav files
        if filename.lower().endswith(".sav"):
            try:
                # Test read the file to verify it can be processed
                # Import pyreadstat lazily
                from src.lazy_imports import get_pyreadstat
                pyreadstat = get_pyreadstat()

                test_df = None
                encodings = [
                    "latin1",
                    "cp1252",
                    "utf-8",
                    "iso-8859-1",
                    None,
                ]  # Try more encodings, including None for auto-detection

                for encoding in encodings:
                    try:
                        print(f"Trying to read SPSS file with encoding: {encoding}")
                        test_df, meta = pyreadstat.read_sav(
                            file_path, encoding=encoding
                        )
                        print(f"Successfully read SPSS file with encoding: {encoding}")
                        # If successful, break out of the loop
                        break
                    except Exception as e:
                        last_error = str(e)
                        print(f"Failed with encoding {encoding}: {last_error}")
                        continue

                if test_df is None:
                    raise Exception(
                        f"Could not read SPSS file with any encoding: {last_error}"
                    )

                # Clear memory
                del test_df
            except ImportError:
                flash(
                    "The pyreadstat package is required to read SPSS files. Please install it with 'pip install pyreadstat'."
                )
                return redirect(url_for("index"))
            except Exception as e:
                flash(
                    f"Error reading SPSS file: {str(e)}. Try converting to CSV format first."
                )
                return redirect(url_for("index"))

        # Process any user-provided suspicions
        user_suspicions = None
        if any(
            param in request.form
            for param in [
                "description",
                "focus_columns",
                "treatment_columns",
                "outcome_columns",
                "suspicious_rows",
                "suspect_grouping",
            ]
        ):
            user_suspicions = {}

            # Process text fields
            for field in ["description", "suspect_grouping"]:
                if field in request.form and request.form[field].strip():
                    user_suspicions[field] = request.form[field].strip()

            # Process comma-separated list fields
            for field in ["focus_columns", "treatment_columns", "outcome_columns"]:
                if field in request.form and request.form[field].strip():
                    user_suspicions[field] = [
                        col.strip()
                        for col in request.form[field].split(",")
                        if col.strip()
                    ]

            # Process multi-select fields - use getlist to handle multiple selections
            if "potential_issues" in request.form:
                user_suspicions["potential_issues"] = request.form.getlist(
                    "potential_issues"
                )

            # Process suspicious rows (handle ranges like "10-20")
            if (
                "suspicious_rows" in request.form
                and request.form["suspicious_rows"].strip()
            ):
                suspicious_rows = []
                for part in request.form["suspicious_rows"].split(","):
                    part = part.strip()
                    if "-" in part:
                        # Handle range
                        try:
                            start, end = part.split("-")
                            suspicious_rows.extend(range(int(start), int(end) + 1))
                        except Exception as e:
                            print(f"Error parsing row range {part}: {e}")
                    else:
                        # Handle single row
                        try:
                            suspicious_rows.append(int(part))
                        except Exception as e:
                            print(f"Error parsing row index {part}: {e}")

                if suspicious_rows:
                    user_suspicions["suspicious_rows"] = suspicious_rows

            # Log what we're passing to the analysis
            if user_suspicions:
                print(
                    f"Using user-provided suspicions to guide analysis: {json.dumps(user_suspicions, default=str)}"
                )

        # Run analysis - pass the paper path and suspicions if provided
        if has_paper and paper_path:
            _report = detect_data_manipulation(
                client,
                file_path,
                analysis_folder,
                paper_path=paper_path,
                use_claude_segmentation=True,
                user_suspicions=user_suspicions,
            )
        else:
            _report = detect_data_manipulation(
                client,
                file_path,
                analysis_folder,
                use_claude_segmentation=True,
                user_suspicions=user_suspicions,
            )

        # Store original dataset for browsing
        try:
            ext = os.path.splitext(file_path)[1].lower()
            original_columns = []
            original_data = []

            if ext == ".xlsx":
                df = pd.read_excel(file_path)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()
            elif ext == ".csv":
                df = pd.read_csv(file_path)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()
            elif ext == ".dta":
                df = pd.read_stata(file_path)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()
            elif ext == ".sav":
                # Import pyreadstat lazily to improve startup time
                from src.lazy_imports import get_pyreadstat
                pyreadstat = get_pyreadstat()

                df, meta = pyreadstat.read_sav(file_path, encoding=None)
                original_columns = df.columns.tolist()
                original_data = df.values.tolist()

            # Convert non-serializable values (like numpy types) to Python native types
            for i, row in enumerate(original_data):
                for j, val in enumerate(row):
                    if isinstance(val, (np.integer, np.floating)):
                        original_data[i][j] = val.item()
                    elif isinstance(val, np.ndarray):
                        original_data[i][j] = val.tolist()
                    elif pd.isna(val):
                        original_data[i][j] = None

        except Exception as e:
            print(f"Error preparing original data for browser: {e}")
            original_columns = []
            original_data = []

        # Store analysis information
        report_filename = f"report_{filename}.md"
        analysis_info = {
            "id": analysis_id,
            "filename": filename,
            "timestamp": os.path.getmtime(file_path),
            "report_path": os.path.join(analysis_folder, report_filename),
            "report_filename": report_filename,
            "has_paper": has_paper,
            "paper_path": paper_path if paper_path else None,
            "paper_is_text": paper_is_text,
            "original_columns": original_columns,
            "original_data": original_data,
        }

        # Save analysis metadata
        with open(os.path.join(analysis_folder, "analysis_info.json"), "w") as f:
            json.dump(analysis_info, f)

        return redirect(url_for("view_results", analysis_id=analysis_id))

    except Exception as e:
        import logging
        import traceback

        error_details = traceback.format_exc()

        # Create a highly visible error message in the server logs
        error_msg = f"CRITICAL: Analysis failed for file {filename}: {str(e)}"
        error_box = "#" * len(error_msg)
        print(f"\n{error_box}\n{error_msg}\n{error_box}\n")
        print(f"ERROR DETAILS:\n{error_details}")

        # Log to application log if logger is configured
        try:
            logging.error(f"Analysis failed for file {filename}: {str(e)}")
            logging.error(f"Traceback: {error_details}")
        except Exception:
            pass  # Silently handle if logging isn't configured

        # Show a more detailed error message to the user
        error_summary = str(e)
        if len(error_summary) > 100:  # Truncate if too long
            error_summary = error_summary[:100] + "..."

        flash(
            f"Analysis failed: {error_summary} (See server logs for details)", "error"
        )
        return redirect(url_for("index"))


@app.route("/results/<analysis_id>")
def view_results(analysis_id):
    # Validate analysis ID format to prevent security issues
    import logging

    if not analysis_id or not re.match(r"^[a-zA-Z0-9_\-]+$", analysis_id):
        logging.warning(f"Invalid analysis ID format: {analysis_id}")
        flash("Invalid analysis ID format")
        return redirect(url_for("index"))

    # Validate analysis ID exists
    analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
    if not os.path.exists(analysis_folder):
        logging.warning(f"Analysis not found: {analysis_id}")
        flash("Analysis not found")
        return redirect(url_for("index"))

    # Verify analysis folder is a directory and not a file
    if not os.path.isdir(analysis_folder):
        logging.error(f"Analysis ID points to a file, not a directory: {analysis_id}")
        flash("Invalid analysis ID")
        return redirect(url_for("index"))

    # Verify analysis folder is a directory and not a file
    if not os.path.isdir(analysis_folder):
        logging.error(f"Analysis ID points to a file, not a directory: {analysis_id}")
        flash("Invalid analysis ID")
        return redirect(url_for("index"))

    # Load analysis info
    try:
        with open(os.path.join(analysis_folder, "analysis_info.json"), "r") as f:
            analysis_info = json.load(f)

        # Read report content
        report_file_path = os.path.join(
            analysis_folder,
            analysis_info.get(
                "report_filename", f"report_{analysis_info['filename']}.md"
            ),
        )
        if not os.path.exists(report_file_path):
            report_file_path = analysis_info["report_path"]

        with open(report_file_path, "r") as f:
            report_content = f.read()

        # Extract manipulation rating if present
        manipulation_rating = None
        rating_match = re.search(r"MANIPULATION_RATING:\s*(\d+)", report_content)
        if rating_match:
            manipulation_rating = int(rating_match.group(1))
            # Remove the rating line from the report content for cleaner display
            report_content = re.sub(
                r"MANIPULATION_RATING:\s*\d+\s*\n", "", report_content
            )

        # Extract JSON content for better formatting
        json_findings = []
        json_columns = []

        # Find all JSON code blocks
        json_blocks = re.findall(r"```json\n(.*?)\n```", report_content, re.DOTALL)

        if len(json_blocks) >= 2:
            # First block is typically column categories
            try:
                json_columns = json.loads(json_blocks[0])
            except (json.JSONDecodeError, IndexError, ValueError):
                json_columns = {}

            # Second block is typically findings
            try:
                json_findings = json.loads(json_blocks[1])
            except (json.JSONDecodeError, IndexError, ValueError):
                json_findings = []

        # Clean up the report content to remove raw JSON blocks
        clean_report = report_content

        # Replace JSON blocks with more readable messages
        if len(json_blocks) >= 2:
            # Replace first JSON block (column categories)
            column_count = sum(
                len(cols) for cols in json_columns.values() if isinstance(cols, list)
            )
            column_replacement = f"""<div class="alert alert-info">
<strong>Column Analysis:</strong> Identified {column_count} columns categorized by their function.
<em>See the "Column Analysis" tab for details.</em>
</div>"""
            clean_report = re.sub(
                r"```json\n" + re.escape(json_blocks[0]) + r"\n```",
                column_replacement,
                clean_report,
            )

            # Replace second JSON block (findings) with more detail
            finding_types = [f["type"] for f in json_findings] if json_findings else []
            finding_count = len(json_findings)

            if finding_count > 0:
                finding_summary = ", ".join(
                    [t.replace("_", " ").title() for t in set(finding_types)]
                )

                # Create a more detailed explanation based on finding types
                finding_explanations = []
                if "sorting_anomaly" in finding_types:
                    sorting_anomalies = [
                        f for f in json_findings if f["type"] == "sorting_anomaly"
                    ]
                    if sorting_anomalies and "details" in sorting_anomalies[0]:
                        anomaly_count = len(sorting_anomalies[0]["details"])

                        # Check if we have any out-of-order analysis
                        out_of_order_findings = False
                        for anomaly in sorting_anomalies:
                            if "details" in anomaly:
                                for detail in anomaly["details"]:
                                    if "out_of_order_analysis" in detail:
                                        out_of_order_findings = True
                                        break
                                if out_of_order_findings:
                                    break

                        if out_of_order_findings:
                            finding_explanations.append(
                                f"<li><strong>Sorting Anomalies:</strong> Found {anomaly_count} rows out of sequence, with <span style='color: #dc3545;'>out-of-order observations</span> that strongly suggest manual data manipulation.</li>"
                            )
                        else:
                            finding_explanations.append(
                                f"<li><strong>Sorting Anomalies:</strong> Found {anomaly_count} rows out of sequence, suggesting manual row manipulation.</li>"
                            )

                if "excel_row_movement" in finding_types:
                    excel_findings = [
                        f for f in json_findings if f["type"] == "excel_row_movement"
                    ]
                    if excel_findings and "details" in excel_findings[0]:
                        movement_count = len(excel_findings[0]["details"])
                        finding_explanations.append(
                            f"<li><strong>Excel Manipulation:</strong> Direct evidence of {movement_count} rows being moved in Excel.</li>"
                        )

                if "duplicate_ids" in finding_types:
                    duplicate_findings = [
                        f for f in json_findings if f["type"] == "duplicate_ids"
                    ]
                    if duplicate_findings and "details" in duplicate_findings[0]:
                        duplicate_count = len(duplicate_findings[0]["details"])
                        finding_explanations.append(
                            f"<li><strong>Duplicate IDs:</strong> Found {duplicate_count} duplicate IDs that may indicate copying or duplication.</li>"
                        )

                if "effect_size_analysis" in finding_types:
                    finding_explanations.append(
                        "<li><strong>Effect Size Analysis:</strong> Statistical analysis of treatment effects shows unusual patterns.</li>"
                    )

                if "with_without_comparison" in finding_types:
                    comparison_findings = [
                        f
                        for f in json_findings
                        if f["type"] == "with_without_comparison"
                    ]
                    if comparison_findings and "details" in comparison_findings[0]:
                        details = comparison_findings[0]["details"]
                        suspicious_count = details.get("suspicious_rows_count", 0)

                        # Check if any results show significant changes
                        significant_changes = False
                        for result in details.get("comparison_results", []):
                            if result.get("significance_changed", False):
                                significant_changes = True
                                break

                        if significant_changes:
                            finding_explanations.append(
                                f"<li><strong>Row Comparison Analysis:</strong> <span style='color: #dc3545;'>Removing {suspicious_count} suspicious rows changes statistical significance of results.</span></li>"
                            )
                        else:
                            finding_explanations.append(
                                f"<li><strong>Row Comparison Analysis:</strong> Comparison of results with and without {suspicious_count} suspicious rows.</li>"
                            )

                if "claude_detected_anomaly" in finding_types:
                    claude_findings = [
                        f
                        for f in json_findings
                        if f["type"] == "claude_detected_anomaly"
                    ]
                    if claude_findings:
                        anomaly_count = len(claude_findings)

                        # Check if we have any out-of-order analysis
                        out_of_order_findings = False
                        for finding in claude_findings:
                            if finding.get("details") and finding["details"].get(
                                "out_of_order_analysis"
                            ):
                                out_of_order_findings = True
                                break

                        if out_of_order_findings:
                            finding_explanations.append(
                                f"<li><strong>Claude AI Analysis:</strong> Detected {anomaly_count} anomalies in specific data segments, including <span style='color: #6f42c1;'>out-of-order observations</span> that may indicate manual data manipulation.</li>"
                            )
                        else:
                            finding_explanations.append(
                                f"<li><strong>Claude AI Analysis:</strong> Detected {anomaly_count} anomalies in specific data segments that may indicate manipulation.</li>"
                            )

                if "claude_chunk_analysis" in finding_types:
                    chunk_findings = [
                        f for f in json_findings if f["type"] == "claude_chunk_analysis"
                    ]
                    if chunk_findings and "details" in chunk_findings[0]:
                        detail = chunk_findings[0]["details"]
                        anomaly_count = detail.get("total_anomalies", 0)
                        chunk_count = detail.get("anomaly_chunks", 0)
                        anomaly_types_list = detail.get("anomaly_types", [])
                        anomaly_types_text = (
                            ", ".join(anomaly_types_list)
                            if anomaly_types_list
                            else "various anomalies"
                        )
                        finding_explanations.append(
                            f"<li><strong>Claude Segment Analysis:</strong> Found {anomaly_count} high-confidence anomalies ({anomaly_types_text}) across {chunk_count} data segments.</li>"
                        )

                finding_html = (
                    "<ul>" + "".join(finding_explanations) + "</ul>"
                    if finding_explanations
                    else ""
                )

                finding_replacement = f"""<div class="alert alert-warning">
<strong>Technical Findings:</strong> Detected {finding_count} potential issues: {finding_summary}.
{finding_html}
<em>See the "Technical Findings" tab for detailed analysis and the "Visualizations" tab for graphical evidence.</em>
</div>"""
                clean_report = re.sub(
                    r"```json\n" + re.escape(json_blocks[1]) + r"\n```",
                    finding_replacement,
                    clean_report,
                )

        # Convert markdown to HTML
        if HAS_MARKDOWN:
            try:
                report_html = markdown.markdown(
                    clean_report, extensions=["fenced_code", "tables"]
                )
            except Exception as e:
                print(f"Error rendering markdown: {e}")
                report_html = basic_markdown_to_html(clean_report)
        else:
            report_html = basic_markdown_to_html(clean_report)

        # Fix image paths in HTML - handle all possible formats
        base_url = url_for("get_file", analysis_id=analysis_id, filename="").rstrip("/")

        # Fix normal image tag patterns
        report_html = report_html.replace('src="', f'src="{base_url}/')

        # Fix any other URL patterns
        report_html = report_html.replace('href="', f'href="{base_url}/')

        # Special case for image URLs that already include a full path
        # Don't double up the base_url if the URL already has it
        report_html = report_html.replace(
            f'src="{base_url}/{base_url}/', f'src="{base_url}/'
        )

        # Get image files
        image_files = [
            f
            for f in os.listdir(analysis_folder)
            if f.endswith((".png", ".jpg", ".jpeg", ".svg"))
        ]

        # Generate data preview with highlighted suspicious data
        data_preview = None
        try:
            # Get original file path
            original_file_path = os.path.join(
                app.config["UPLOAD_FOLDER"], analysis_info["filename"]
            )
            if os.path.exists(original_file_path):
                data_preview = generate_data_preview(original_file_path, json_findings)
            else:
                data_preview = "<div class='alert alert-warning'>Original dataset file not found for preview.</div>"
        except Exception as e:
            data_preview = f"<div class='alert alert-danger'>Error generating data preview: {str(e)}</div>"

        # Check if this is a sample dataset
        is_sample = analysis_info.get("is_sample", False)
        sample_name = analysis_info.get("sample_name", "")

        return render_template(
            "results.html",
            analysis=analysis_info,
            report=report_content,
            report_html=report_html,
            images=image_files,
            analysis_id=analysis_id,
            manipulation_rating=manipulation_rating,
            json_findings=json_findings,
            json_columns=json_columns,
            data_preview=data_preview,
            is_sample=is_sample,
            sample_name=sample_name,
        )

    except Exception as e:
        import logging
        import traceback

        error_details = traceback.format_exc()

        # Create a highly visible error message in the server logs
        error_msg = (
            f"CRITICAL: Failed to load results for analysis {analysis_id}: {str(e)}"
        )
        error_box = "#" * len(error_msg)
        print(f"\n{error_box}\n{error_msg}\n{error_box}\n")
        print(f"ERROR DETAILS:\n{error_details}")

        # Log to application log if logger is configured
        try:
            logging.error(f"Results page failed for analysis {analysis_id}: {str(e)}")
            logging.error(f"Traceback: {error_details}")
        except Exception:
            pass  # Silently handle if logging isn't configured

        # Show a more detailed error message to the user
        error_summary = str(e)
        if len(error_summary) > 100:  # Truncate if too long
            error_summary = error_summary[:100] + "..."

        flash(
            f"Error loading results: {error_summary} (See server logs for details)",
            "error",
        )
        return redirect(url_for("index"))


@app.route("/file/<analysis_id>/<path:filename>")
def get_file(analysis_id, filename):
    """Serve files from the results directory"""
    try:
        # Validate that the analysis ID exists
        analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
        if not os.path.exists(analysis_folder):
            logging.warning(
                f"Attempt to access file in non-existent analysis folder: {analysis_id}"
            )
            flash("Analysis not found")
            return redirect(url_for("index"))

        # Validate that the file exists
        file_path = os.path.join(analysis_folder, filename)
        if not os.path.exists(file_path):
            logging.warning(f"Attempt to access non-existent file: {file_path}")
            flash(f"File not found: {filename}")
            return redirect(url_for("view_results", analysis_id=analysis_id))

        # Validate that the file is within the analysis folder (prevent directory traversal)
        if not os.path.realpath(file_path).startswith(
            os.path.realpath(analysis_folder)
        ):
            logging.error(
                f"Security issue: Attempt to access file outside analysis folder: {file_path}"
            )
            flash("Access denied for security reasons")
            return redirect(url_for("index"))

        # Make sure the file is readable
        if not os.access(file_path, os.R_OK):
            logging.error(
                f"Permission error: File exists but is not readable: {file_path}"
            )
            flash("File access error: The file exists but cannot be read")
            return redirect(url_for("view_results", analysis_id=analysis_id))

        # If it's an image, ensure it's a valid image type to prevent security issues
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            # Check for valid image file
            try:
                from PIL import Image

                Image.open(file_path).verify()  # Just verify it's a valid image
            except Exception:
                logging.warning(f"Invalid image file: {file_path}")
                # Continue anyway, as it might be a corrupted image we still want to serve

        # Use Flask's secure way to serve files
        try:
            # Use the correct form of send_from_directory based on the Flask version
            try:
                # Try the newer Flask 2.0+ form first
                return send_from_directory(
                    analysis_folder, filename, as_attachment=False
                )
            except TypeError:
                # Fall back to older Flask versions which didn't have as_attachment
                return send_from_directory(analysis_folder, filename)
        except Exception:
            # Fall back to basic file serving if Flask's method fails
            from flask import Response

            with open(file_path, "rb") as f:
                content = f.read()

            # Try to determine MIME type
            import mimetypes

            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

            return Response(content, mimetype=mime_type)
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()

        # Log the error
        logging.error(
            f"Error accessing file {filename} in analysis {analysis_id}: {str(e)}"
        )
        logging.error(f"Traceback: {error_details}")

        flash(f"Error accessing file: {str(e)}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    # Log application startup
    logging.info("=" * 50)
    logging.info("Data Forensics Tool starting up")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logging.info(f"Results folder: {app.config['RESULTS_FOLDER']}")
    logging.info("=" * 50)

    # Fix the duplicate app.run call issue
    app.run(debug=True)
