#!/usr/bin/env python3
import argparse
import json
import os

import pandas as pd
from anthropic import Anthropic

from src.data_forensics import DataForensics, ExcelForensics
from src.visualize import ForensicVisualizer


def load_config():
    """Load configuration from config.json or environment variables."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config.json"
    )

    # Default config
    config = {"api_key": os.environ.get("CLAUDE_API_KEY", "")}

    # Load from file if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")

    return config


def setup_client(api_key=None):
    """Set up the Claude API client."""
    config = load_config()

    # Command line API key takes precedence
    if api_key:
        config["api_key"] = api_key

    if not config.get("api_key"):
        raise ValueError(
            "Claude API key is required. Please provide it via config.json, CLAUDE_API_KEY environment variable, or --api-key flag."
        )

    return Anthropic(api_key=config["api_key"])


def identify_columns(client, df):
    """Use Claude to identify and categorize columns in the dataset."""
    columns = df.columns.tolist()
    column_types = [str(df[col].dtype) for col in columns]
    sample_data = df.head(5).to_string()

    prompt = f"""Analyze this dataset and categorize the columns:
- ID columns (unique identifiers for observations)
- Grouping/condition columns (variables that indicate experimental conditions, treatments, or groups)
- Outcome/dependent variables (variables that measure the results)
- Demographic columns (age, gender, etc.)
- Other columns (any that don't fit above categories)

Column names and data types:
{list(zip(columns, column_types))}

Sample data (first 5 rows):
{sample_data}

Respond with JSON in this format:
{{
  "id_columns": ["col1", "col2"],
  "group_columns": ["col3"],
  "outcome_columns": ["col4", "col5"],
  "demographic_columns": ["col6", "col7"],
  "other_columns": ["col8"]
}}

Be concise and only include the JSON in your response."""

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract JSON from response
    try:
        # Extract JSON portion from the response
        content = response.content[0].text
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        json_str = content[json_start:json_end]
        column_categories = json.loads(json_str)
        return column_categories
    except Exception as e:
        print(f"Error parsing column categories: {e}")
        # Fallback
        return {
            "id_columns": [],
            "group_columns": [],
            "outcome_columns": [],
            "demographic_columns": [],
            "other_columns": columns,
        }


def detect_data_manipulation(client, data_path, output_dir=None):
    """Detect potential data manipulation in research data."""
    print(f"Analyzing data file: {data_path}")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine file type and read data
    file_ext = os.path.splitext(data_path)[1].lower()

    # Initialize data forensics
    forensics = DataForensics()

    # Read data based on file type
    if file_ext == ".xlsx":
        df = pd.read_excel(data_path)
        # Create Excel forensics object for deeper analysis
        excel_forensics = ExcelForensics(data_path)
    elif file_ext == ".csv":
        df = pd.read_csv(data_path)
    elif file_ext == ".dta":
        df = pd.read_stata(data_path)
    else:
        print(f"Unsupported file format: {file_ext}")
        return

    # Use Claude to identify column categories
    column_categories = identify_columns(client, df)

    # Basic analysis with DataForensics
    findings = []

    # If we have ID and group columns, check for sorting anomalies
    if column_categories["id_columns"] and column_categories["group_columns"]:
        id_col = column_categories["id_columns"][0]  # Use first ID column
        group_col = column_categories["group_columns"][0]  # Use first group column

        print(
            f"Checking sorting anomalies for ID column '{id_col}' within groups '{group_col}'"
        )
        forensics.df = df
        sorting_issues = forensics.check_sorting_anomalies(id_col, group_col)

        if sorting_issues:
            findings.append({"type": "sorting_anomaly", "details": sorting_issues})
            print(f"Found {len(sorting_issues)} sorting anomalies")

            # If Excel file, check calc chain for evidence of row movement
            if file_ext == ".xlsx":
                with excel_forensics as ef:
                    suspicious_rows = [int(issue["row_index"]) for issue in sorting_issues]
                    movement_evidence = ef.analyze_row_movement(suspicious_rows)

                    if movement_evidence:
                        findings.append(
                            {"type": "excel_row_movement", "details": movement_evidence}
                        )
                        print(f"Found evidence of row movement in Excel file")

            # Check if suspicious observations show strong effect in the expected direction
            if column_categories["outcome_columns"]:
                try:
                    suspicious_rows = [int(issue["row_index"]) for issue in sorting_issues]
                    effects = forensics.analyze_suspicious_observations(
                        suspicious_rows, group_col, column_categories["outcome_columns"]
                    )
                    
                    if effects and not (isinstance(effects, dict) and "error" in effects):
                        findings.append({"type": "effect_size_analysis", "details": effects})
                except Exception as e:
                    print(f"Error analyzing effect sizes: {e}")
                    # Don't add to findings if there was an error

    # Check for duplicate IDs
    if column_categories["id_columns"]:
        id_col = column_categories["id_columns"][0]
        duplicate_ids = forensics.check_duplicate_ids(id_col)

        if duplicate_ids:
            findings.append({"type": "duplicate_ids", "details": duplicate_ids})
            print(f"Found {len(duplicate_ids)} duplicate IDs")

    # Generate Claude prompt with findings
    findings_json = json.dumps(findings, indent=2)

    prompt = f"""I need you to analyze potential data manipulation in a research dataset.

Here are the column categories I've identified:
{json.dumps(column_categories, indent=2)}

My analysis has found the following potential issues:
{findings_json}

Based on these findings:
1. Assess the likelihood that this data has been manipulated
2. Explain what manipulation techniques may have been used
3. Discuss the patterns in the suspicious observations - do they show a particularly strong effect?
4. Provide a detailed explanation of why these patterns are unlikely to occur naturally

Reference: The article "Data Colada" describes a case where researchers found evidence of manipulation in a dataset about dishonesty research. They found:
- Rows that were out of order when sorted by ID within experimental conditions
- Evidence in Excel's calcChain.xml showing rows had been moved between conditions
- The suspicious observations showed extremely strong effects in the predicted direction

Your assessment:"""

    print("Generating forensic analysis with Claude...")
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )

    analysis = response.content[0].text

    # Generate report
    report = f"""# Data Forensics Report: {os.path.basename(data_path)}

## Column Categories
```json
{json.dumps(column_categories, indent=2)}
```

## Technical Findings
```json
{json.dumps(findings, indent=2)}
```

## Claude's Analysis
{analysis}
"""

    # Create visualizations if we have suspicious rows
    plots = []
    if findings and output_dir:
        try:
            visualizer = ForensicVisualizer(output_dir)

            # Extract suspicious rows from findings
            suspicious_rows = []
            for finding in findings:
                if finding["type"] == "sorting_anomaly":
                    suspicious_rows.extend(
                        [issue["row_index"] for issue in finding["details"]]
                    )

            if (
                suspicious_rows
                and column_categories["id_columns"]
                and column_categories["group_columns"]
            ):
                id_col = column_categories["id_columns"][0]
                group_col = column_categories["group_columns"][0]

                try:
                    # Plot ID sequence
                    id_plot = visualizer.plot_id_sequence(
                        df, id_col, group_col, suspicious_rows
                    )
                    if id_plot:
                        plots.append({"type": "id_sequence", "path": id_plot})
                        print(f"Created ID sequence plot: {id_plot}")
                except Exception as e:
                    print(f"Error creating ID sequence plot: {e}")

                # Plot suspicious vs normal observations for outcome variables
                if column_categories["outcome_columns"]:
                    try:
                        outcome_plot = visualizer.plot_suspicious_vs_normal(
                            df, group_col, column_categories["outcome_columns"], suspicious_rows
                        )
                        if outcome_plot:
                            plots.append({"type": "suspicious_vs_normal", "path": outcome_plot})
                            print(f"Created suspicious vs normal plot: {outcome_plot}")
                    except Exception as e:
                        print(f"Error creating suspicious vs normal plot: {e}")

                    try:
                        # Plot effect sizes
                        effect_plot = visualizer.plot_effect_sizes(
                            df, group_col, column_categories["outcome_columns"], suspicious_rows
                        )
                        if effect_plot:
                            plots.append({"type": "effect_sizes", "path": effect_plot})
                            print(f"Created effect sizes plot: {effect_plot}")
                    except Exception as e:
                        print(f"Error creating effect sizes plot: {e}")
        except Exception as e:
            print(f"Error during visualization: {e}")

    # Update the report with plot information
    if plots:
        plots_section = "\n## Visualizations\n\n"
        for plot in plots:
            rel_path = (
                os.path.relpath(plot["path"], output_dir)
                if output_dir
                else plot["path"]
            )
            plots_section += f"### {plot['type'].replace('_', ' ').title()}\n"
            plots_section += f"![{plot['type']}]({rel_path})\n\n"

        report += plots_section

    # Save report
    if output_dir:
        report_path = os.path.join(
            output_dir, f"report_{os.path.basename(data_path)}.md"
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Data Manipulation Detection with Claude"
    )
    parser.add_argument("--api-key", help="Claude API key")
    parser.add_argument("--data", required=True, help="Path to input data file")
    parser.add_argument("--output", help="Directory to save output reports")

    args = parser.parse_args()

    try:
        client = setup_client(args.api_key)
        report = detect_data_manipulation(client, args.data, args.output)
        print("\nAnalysis complete!")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
