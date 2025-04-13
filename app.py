#!/usr/bin/env python3
import datetime
import json
import os
import re
import uuid

import matplotlib
import pandas as pd

# Try to import markdown, fallback to basic HTML conversion if not available
try:
    import markdown

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# Use non-interactive backend to avoid GUI issues
matplotlib.use("Agg")

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

from src.data_forensics import DataForensics
from src.main import detect_data_manipulation, setup_client

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)


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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["ALLOWED_EXTENSIONS"] = {"xlsx", "csv", "dta"}

# Create necessary directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


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

                    # Mark row as suspicious
                    if row_idx not in suspicious_rows:
                        suspicious_rows[row_idx] = []
                    suspicious_rows[row_idx].append(
                        {
                            "type": "sorting_anomaly",
                            "css_class": "sorting-anomaly",
                            "explanation": f"Sorting anomaly: ID {id_val} comes after ID {prev_id} in group {sort_col}={sort_val}",
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
                        suspicious_cells[cell_key] = {
                            "type": "sorting",
                            "css_class": "cell-highlight-sorting",
                            "explanation": f"ID {id_val} out of sequence (previous ID: {prev_id})",
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
                        "explanation": f"Statistical outlier (z-score > 3)",
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

        # Log the suspicious rows and cells found
        print(
            f"Found {len(suspicious_rows)} suspicious rows and {len(suspicious_cells)} suspicious cells"
        )
        
        if not suspicious_rows:
            return "<div class='alert alert-info'>No suspicious data points were found in this dataset.</div>"
            
        # Group suspicious rows by type for organized display
        anomaly_types = {
            "sorting_anomaly": {"title": "Sorting Anomalies", "rows": [], "description": "Rows with IDs out of sequence"},
            "duplicate_id": {"title": "Duplicate IDs", "rows": [], "description": "Rows with duplicate ID values"},
            "statistical_anomaly": {"title": "Statistical Outliers", "rows": [], "description": "Rows with values that are statistical outliers (z-score > 3)"},
            "excel_movement": {"title": "Excel Row Movement", "rows": [], "description": "Rows with evidence of Excel manipulation from metadata"}
        }
        
        # Categorize suspicious rows
        for idx, issues in suspicious_rows.items():
            for issue in issues:
                issue_type = issue["type"]
                if issue_type in anomaly_types:
                    if idx not in [r["idx"] for r in anomaly_types[issue_type]["rows"]]:
                        anomaly_types[issue_type]["rows"].append({"idx": idx, "explanation": issue["explanation"]})
        
        # Start building HTML output
        html_output = '<div class="suspicious-data-summary mb-4">\n'
        html_output += '  <div class="alert alert-warning">\n'
        html_output += f'    <i class="fas fa-exclamation-triangle me-2"></i>Found {len(suspicious_rows)} suspicious rows in the dataset.\n'
        html_output += '  </div>\n'
        
        # Add a legend to explain the borders
        html_output += '  <div class="card mb-3">\n'
        html_output += '    <div class="card-header bg-light">\n'
        html_output += '      <h6 class="m-0">Legend</h6>\n'
        html_output += '    </div>\n'
        html_output += '    <div class="card-body p-3">\n'
        html_output += '      <div class="row">\n'
        
        # Sorting anomalies
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #dc3545; margin-right: 8px;"></div>\n'
        html_output += '            <span><strong>Red:</strong> Sorting Anomalies</span>\n'
        html_output += '          </div>\n'
        html_output += '        </div>\n'
        
        # Duplicate IDs
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #ffc107; margin-right: 8px;"></div>\n'
        html_output += '            <span><strong>Yellow:</strong> Duplicate IDs</span>\n'
        html_output += '          </div>\n'
        html_output += '        </div>\n'
        
        # Statistical outliers
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #0dcaf0; margin-right: 8px;"></div>\n'
        html_output += '            <span><strong>Blue:</strong> Statistical Outliers</span>\n'
        html_output += '          </div>\n'
        html_output += '        </div>\n'
        
        # Excel movement
        html_output += '        <div class="col-md-3 mb-2">\n'
        html_output += '          <div class="d-flex align-items-center">\n'
        html_output += '            <div style="width: 24px; height: 24px; border: 2px solid #6c757d; margin-right: 8px;"></div>\n'
        html_output += '            <span><strong>Gray:</strong> Excel Movement</span>\n'
        html_output += '          </div>\n'
        html_output += '        </div>\n'
        
        html_output += '      </div>\n'
        html_output += '      <div class="small text-muted mt-2">Hover over highlighted cells for detailed explanations of what makes them suspicious.</div>\n'
        html_output += '    </div>\n'
        html_output += '  </div>\n'
        html_output += '</div>\n'
        
        # Create tabs for different types of suspicious data
        html_output += '<ul class="nav nav-tabs mb-3" id="suspiciousDataTabs" role="tablist">\n'
        
        # Add tab headers
        active = True
        for anomaly_id, anomaly_info in anomaly_types.items():
            if anomaly_info["rows"]:
                tab_active = 'active' if active else ''
                html_output += f'  <li class="nav-item" role="presentation">\n'
                html_output += f'    <button class="nav-link {tab_active}" id="{anomaly_id}-tab" data-bs-toggle="tab" data-bs-target="#{anomaly_id}" type="button" role="tab">\n'
                html_output += f'      {anomaly_info["title"]} ({len(anomaly_info["rows"])})\n'
                html_output += f'    </button>\n'
                html_output += f'  </li>\n'
                active = False
        
        html_output += '</ul>\n'
        
        # Add tab content
        html_output += '<div class="tab-content" id="suspiciousDataTabsContent">\n'
        
        # Process each anomaly type
        active = True
        for anomaly_id, anomaly_info in anomaly_types.items():
            if not anomaly_info["rows"]:
                continue
                
            tab_active = 'show active' if active else ''
            html_output += f'  <div class="tab-pane fade {tab_active}" id="{anomaly_id}" role="tabpanel">\n'
            html_output += f'    <div class="card mb-4">\n'
            html_output += f'      <div class="card-header bg-light">\n'
            html_output += f'        <h5 class="mb-0">{anomaly_info["title"]}</h5>\n'
            html_output += f'        <p class="small text-muted mb-0">{anomaly_info["description"]}</p>\n'
            html_output += f'      </div>\n'
            html_output += f'      <div class="card-body">\n'
            
            # Create data table for this anomaly type
            html_output += f'        <div class="table-responsive">\n'
            html_output += f'          <table id="{anomaly_id}-table" class="table table-hover table-bordered">\n'
            html_output += f'            <thead class="table-light">\n'
            html_output += f'              <tr>\n'
            html_output += f'                <th>Row</th>\n'
            
            # Only include relevant columns for each anomaly type to avoid information overload
            relevant_columns = []
            
            if anomaly_id == "sorting_anomaly":
                # For sorting anomalies, show ID columns and sort columns
                id_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in ["id", "participant", "subject", "case"])]
                
                # Safely extract sort columns
                sort_columns = []
                for idx, issues in suspicious_rows.items():
                    for anomaly in issues:
                        if anomaly["type"] == "sorting_anomaly" and "sort_column" in anomaly:
                            sort_columns.append(anomaly["sort_column"])
                
                sort_columns = list(set(sort_columns))
                
                relevant_columns = id_columns + sort_columns
                relevant_columns = list(dict.fromkeys(relevant_columns))  # Remove duplicates while preserving order
                
            elif anomaly_id == "duplicate_id":
                # For duplicate IDs, show ID columns and a few other key columns
                id_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in ["id", "participant", "subject", "case"])]
                relevant_columns = id_columns[:1]  # Just the primary ID column
                
                # Add a few other discriminating columns
                other_columns = [col for col in df.columns if col not in id_columns]
                if other_columns:
                    relevant_columns.extend(other_columns[:3])  # Add up to 3 more columns
                
            elif anomaly_id == "statistical_anomaly":
                # For statistical anomalies, show columns with outliers and ID columns
                outlier_columns = []
                
                # Safely extract outlier columns
                if json_findings:
                    for finding in json_findings:
                        if finding.get("type") == "statistical_anomaly" and "column" in finding:
                            outlier_columns.append(finding["column"])
                
                outlier_columns = list(set(outlier_columns))
                
                id_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in ["id", "participant", "subject", "case"])]
                relevant_columns = id_columns[:1] + outlier_columns  # ID column + columns with outliers
                
            elif anomaly_id == "excel_movement":
                # For Excel movement, show ID columns and a few key columns
                id_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in ["id", "participant", "subject", "case"])]
                relevant_columns = id_columns[:1]  # Just the primary ID column
                
                # Add a few other discriminating columns
                group_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in ["group", "condition", "treatment"])]
                if group_columns:
                    relevant_columns.extend(group_columns[:2])
                    
                # Add outcome columns if we can detect them
                outcome_columns = [col for col in df.columns if any(pattern in col.lower() for pattern in ["score", "result", "outcome", "dependent"])]
                if outcome_columns:
                    relevant_columns.extend(outcome_columns[:2])
            
            # Ensure we have some columns (fallback)
            if not relevant_columns and len(df.columns) > 0:
                relevant_columns = df.columns[:min(5, len(df.columns))]
            
            # Add column headers
            for col in relevant_columns:
                html_output += f'                <th>{col}</th>\n'
                
            # Add explanation header
            html_output += f'                <th>Issue</th>\n'
            html_output += f'              </tr>\n'
            html_output += f'            </thead>\n'
            html_output += f'            <tbody>\n'
            
            # Sort rows by index for consistent display
            try:
                sorted_rows = sorted(anomaly_info.get("rows", []), key=lambda x: x.get("idx", 0))
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
                        html_output += f'                <td><strong>{idx + 1}</strong></td>\n'  # Display 1-based row numbers
                        
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
                                    html_output += f'                <td>{cell_display}</td>\n'
                            else:
                                html_output += f'                <td></td>\n'  # Column not found
                        
                        # Add explanation
                        explanation = row_info.get("explanation", "Unknown issue")
                        html_output += f'                <td>{explanation}</td>\n'
                        html_output += f'              </tr>\n'
                except Exception as row_error:
                    print(f"Error processing row {row_info}: {row_error}")
                    continue
            
            html_output += f'            </tbody>\n'
            html_output += f'          </table>\n'
            html_output += f'        </div>\n'
            html_output += f'      </div>\n'
            html_output += f'    </div>\n'
            html_output += f'  </div>\n'
            
            active = False
        
        html_output += '</div>\n'
        
        # Add a link to view full dataset
        html_output += '<div class="text-center mt-4">\n'
        html_output += '  <button id="show-full-data" class="btn btn-outline-primary">\n'
        html_output += '    <i class="fas fa-table me-2"></i>Show Full Dataset Preview\n'
        html_output += '  </button>\n'
        html_output += '</div>\n'
        
        # Create a hidden div with the full dataset table
        html_output += '<div id="full-dataset-preview" style="display: none; margin-top: 30px;">\n'
        html_output += '  <div class="d-flex justify-content-between align-items-center mb-3">\n'
        html_output += '    <h5 class="mb-0">Full Dataset Preview</h5>\n'
        html_output += '    <span class="badge bg-light text-dark border">Highlighted rows contain suspicious data</span>\n'
        html_output += '  </div>\n'
        
        # Create full table
        html_output += '  <table id="data-preview-table" class="table table-hover table-sm">\n'
        html_output += '    <thead>\n'
        html_output += '      <tr>\n'
        html_output += '        <th>#</th>\n'
        
        for col in df.columns:
            html_output += f'        <th>{col}</th>\n'
        
        html_output += '      </tr>\n'
        html_output += '    </thead>\n'
        html_output += '    <tbody>\n'
        
        # Process all rows, but optimize the display to focus on suspicious rows
        suspicious_indices = list(suspicious_rows.keys())
        
        # Debug the suspicious rows
        print(f"Full dataset preview - Suspicious rows: {suspicious_indices}")
        
        # Determine which rows to display:
        # 1. Always include suspicious rows
        # 2. Include some rows before and after each suspicious row for context
        # 3. Add ellipses between non-adjacent sections
        
        context_rows = 2  # Number of rows to show before/after each suspicious row for context
        rows_to_display = set()
        
        # Add suspicious rows and their context rows
        for sus_idx in suspicious_indices:
            # Add the suspicious row itself
            rows_to_display.add(sus_idx)
            
            # Add context rows before and after
            for i in range(max(0, sus_idx - context_rows), min(len(df), sus_idx + context_rows + 1)):
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
                html_output += f'          <em>⋮ ⋮ ⋮ {idx - last_idx - 1} rows without issues omitted ⋮ ⋮ ⋮</em>\n'
                html_output += '        </td>\n'
                html_output += '      </tr>\n'
            
            # Get the row data
            row = df.iloc[idx]
            
            # Check if this is a suspicious row
            row_classes = []
            title_text = []
            
            # Use original DataFrame index to check for suspicious rows
            if idx in suspicious_rows:
                print(f"Found suspicious row at index {idx} with issues: {suspicious_rows[idx]}")
                row_classes.append("suspicious-row")
                
                # Add all issue classes for this row
                for issue in suspicious_rows[idx]:
                    if "css_class" in issue:
                        row_classes.append(issue["css_class"])
                        print(f"Adding class {issue['css_class']} to full dataset row {idx}")
                    
                    if "explanation" in issue:
                        title_text.append(issue["explanation"])
            
            row_class = " ".join(row_classes)
            row_title = " | ".join(title_text)
            
            if row_class:
                html_output += f'      <tr class="{row_class}" title="{row_title}">\n'
            else:
                html_output += '      <tr>\n'
            
            # Add row number cell (showing the actual row number from the dataset)
            html_output += f'        <td><strong>{idx + 1}</strong></td>\n'
            
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
                        html_output += f'        <td>{cell_display}</td>\n'
                except Exception as cell_error:
                    print(f"Error processing cell {col} in row {idx}: {cell_error}")
                    html_output += f'        <td></td>\n'
            
            html_output += '      </tr>\n'
            
            # Update last_idx
            last_idx = idx
            row_num += 1
        
        html_output += '    </tbody>\n'
        html_output += '  </table>\n'
        
        # Add note about the smart preview
        displayed_rows = len(display_indices)
        skipped_rows = len(df) - displayed_rows
        if skipped_rows > 0:
            html_output += f'  <div class="alert alert-info mt-3 mb-0">'
            html_output += f'    <i class="fas fa-info-circle me-2"></i>'
            html_output += f'    Smart preview: Showing {displayed_rows} rows (including all suspicious rows and their context) '
            html_output += f'    from a total of {len(df)} rows. {skipped_rows} rows without issues are condensed.'
            html_output += f'  </div>\n'
        
        html_output += '</div>\n'
        
        # Add JavaScript to toggle full dataset view
        html_output += '<script>\n'
        html_output += '  document.addEventListener("DOMContentLoaded", function() {\n'
        html_output += '    const toggleButton = document.getElementById("show-full-data");\n'
        html_output += '    const fullDatasetDiv = document.getElementById("full-dataset-preview");\n'
        html_output += '    if (toggleButton && fullDatasetDiv) {\n'
        html_output += '      toggleButton.addEventListener("click", function() {\n'
        html_output += '        if (fullDatasetDiv.style.display === "none") {\n'
        html_output += '          fullDatasetDiv.style.display = "block";\n'
        html_output += '          toggleButton.innerHTML = \'<i class="fas fa-compress me-2"></i>Hide Full Dataset Preview\';\n'
        html_output += '        } else {\n'
        html_output += '          fullDatasetDiv.style.display = "none";\n'
        html_output += '          toggleButton.innerHTML = \'<i class="fas fa-table me-2"></i>Show Full Dataset Preview\';\n'
        html_output += '        }\n'
        html_output += '      });\n'
        html_output += '    }\n'
        html_output += '  });\n'
        html_output += '</script>\n'
        
        return html_output

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in generate_data_preview: {str(e)}\n{error_details}")
        return f"<div class='alert alert-danger'>Error generating data preview: {str(e)}</div>"


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

    try:
        # Create a unique folder for this analysis
        analysis_id = str(uuid.uuid4())
        analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
        os.makedirs(analysis_folder, exist_ok=True)

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Initialize Claude client
        client = setup_client()

        # Run analysis
        report = detect_data_manipulation(client, file_path, analysis_folder)

        # Store analysis information
        report_filename = f"report_{filename}.md"
        analysis_info = {
            "id": analysis_id,
            "filename": filename,
            "timestamp": os.path.getmtime(file_path),
            "report_path": os.path.join(analysis_folder, report_filename),
            "report_filename": report_filename,
        }

        # Save analysis metadata
        with open(os.path.join(analysis_folder, "analysis_info.json"), "w") as f:
            json.dump(analysis_info, f)

        return redirect(url_for("view_results", analysis_id=analysis_id))

    except Exception as e:
        flash(f"Error during analysis: {str(e)}")
        return redirect(url_for("index"))


@app.route("/results/<analysis_id>")
def view_results(analysis_id):
    # Validate analysis ID
    analysis_folder = os.path.join(app.config["RESULTS_FOLDER"], analysis_id)
    if not os.path.exists(analysis_folder):
        flash("Analysis not found")
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
            except:
                json_columns = {}

            # Second block is typically findings
            try:
                json_findings = json.loads(json_blocks[1])
            except:
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

            # Replace second JSON block (findings)
            finding_types = [f["type"] for f in json_findings] if json_findings else []
            finding_count = len(json_findings)

            if finding_count > 0:
                finding_summary = ", ".join(
                    [t.replace("_", " ").title() for t in set(finding_types)]
                )
                finding_replacement = f"""<div class="alert alert-warning">
<strong>Technical Findings:</strong> Detected {finding_count} potential issues: {finding_summary}.
<em>See the "Technical Findings" tab for detailed analysis.</em>
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

        # Fix image paths in HTML
        base_url = url_for("get_file", analysis_id=analysis_id, filename="")
        report_html = report_html.replace('src="', f'src="{base_url}')

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
        )

    except Exception as e:
        flash(f"Error loading results: {str(e)}")
        return redirect(url_for("index"))


@app.route("/file/<analysis_id>/<path:filename>")
def get_file(analysis_id, filename):
    """Serve files from the results directory"""
    return send_from_directory(
        os.path.join(app.config["RESULTS_FOLDER"], analysis_id), filename
    )


if __name__ == "__main__":
    app.run(debug=True)
    app.run(debug=True)
