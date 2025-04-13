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
        model="claude-3-7-sonnet-latest",
        max_tokens=4096,
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


def detect_data_manipulation(client, data_path, output_dir=None, paper_path=None):
    """Detect potential data manipulation in research data.

    Args:
        client: Claude API client
        data_path: Path to the dataset file
        output_dir: Directory to save output files
        paper_path: Optional path to research paper PDF for context
    """
    print(f"Analyzing data file: {data_path}")
    if paper_path:
        print(f"Using research paper for context: {paper_path}")

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
    elif file_ext == ".sav":
        try:
            # Try multiple encodings for SPSS files
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
                    df, meta = pyreadstat.read_sav(data_path, encoding=encoding)
                    read_success = True
                    print(
                        f"Successfully read SPSS file with encoding: {encoding if encoding else 'auto-detected'}"
                    )
                    break
                except Exception as e:
                    last_error = str(e)
                    continue

            if not read_success:
                error_msg = f"Could not read SPSS file with any encoding: {last_error}"
                print(error_msg)
                return error_msg
        except ImportError:
            print(
                "The pyreadstat package is required to read SPSS files. Please install it with 'pip install pyreadstat'"
            )
            return "The pyreadstat package is required to read SPSS files. Please install it with 'pip install pyreadstat'"
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
                    suspicious_rows = [
                        int(issue["row_index"]) for issue in sorting_issues
                    ]
                    movement_evidence = ef.analyze_row_movement(suspicious_rows)

                    if movement_evidence:
                        findings.append(
                            {"type": "excel_row_movement", "details": movement_evidence}
                        )
                        print(f"Found evidence of row movement in Excel file")

            # Check if suspicious observations show strong effect in the expected direction
            if column_categories["outcome_columns"]:
                try:
                    suspicious_rows = [
                        int(issue["row_index"]) for issue in sorting_issues
                    ]
                    effects = forensics.analyze_suspicious_observations(
                        suspicious_rows, group_col, column_categories["outcome_columns"]
                    )

                    if effects and not (
                        isinstance(effects, dict) and "error" in effects
                    ):
                        findings.append(
                            {"type": "effect_size_analysis", "details": effects}
                        )
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

    # Extract text from research paper if provided
    paper_context = ""
    if paper_path and os.path.exists(paper_path):
        try:
            import PyPDF2

            # Extract text from PDF
            with open(paper_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                paper_text = ""
                for page_num in range(
                    min(len(pdf_reader.pages), 20)
                ):  # Extract up to 20 pages
                    page = pdf_reader.pages[page_num]
                    paper_text += page.extract_text()

                # If we have text, use another Claude call to extract the most important parts
                if paper_text:
                    # Create a prompt to extract the important parts of the research paper without summarizing
                    extraction_prompt = f"""
Please extract the most important segments verbatim from this research paper. Do NOT summarize or paraphrase - I need the exact original text.

Focus on identifying and extracting relevant sections that address:

1. The research question or hypothesis (e.g., abstract, introduction statements)
2. The methodology used (e.g., method sections, experimental design)
3. The expected findings or hypothesized effects
4. The research domain and its key concepts

Remove references, citations, acknowledgments, and formatting elements like headers/page numbers, but preserve the exact original text of important content. 

DO NOT summarize or paraphrase - only extract complete sentences and paragraphs directly from the original. The extraction must use only words that appear in the exact same order as in the original text.

Research Paper:
{paper_text}

Your extraction should ONLY contain original text directly quoted from the paper.
"""
                    try:
                        # Call Claude to extract the key information
                        print(
                            "Calling Claude to extract key sections of the research paper..."
                        )
                        extraction_response = client.messages.create(
                            model="claude-3-7-sonnet-latest",
                            max_tokens=4096,
                            messages=[{"role": "user", "content": extraction_prompt}],
                        )

                        paper_extracts = extraction_response.content[0].text.strip()

                        # Add paper context to prompt
                        if paper_extracts:
                            paper_context = f"""
I've also extracted the key sections from the research paper that provide context about the data:

Research Paper Excerpts (Original Text):
{paper_extracts}
"""
                            print(
                                "Added extracted paper sections to prompt (original text preserved)"
                            )
                    except Exception as e:
                        print(f"Error in Claude summary extraction: {e}")
                        # Fallback to original method if Claude call fails
                        paper_excerpt = f"Research Paper (original text excerpt):\n{paper_text[:5000]}"
                        paper_context = f"I've also extracted text from the research paper that might provide context about the data:\n\n{paper_excerpt}"
                        print(
                            "Added raw research paper context to prompt (fallback method)"
                        )
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Continue without paper context if there's an error

    prompt = f"""I need you to analyze potential data manipulation in a research dataset.

Here are the column categories I've identified:
{json.dumps(column_categories, indent=2)}

My analysis has found the following potential issues:
{findings_json}

{paper_context}

Based on these findings:
1. Start your response with a manipulation rating on a scale of 1-10, with 10 being absolutely certain the data was manipulated and 1 being no evidence of manipulation. Format this exactly as: "MANIPULATION_RATING: [1-10]"
2. Assess the likelihood that this data has been manipulated
3. Explain what manipulation techniques may have been used
4. Discuss the patterns in the suspicious observations - do they show a particularly strong effect?
5. Provide a detailed explanation of why these patterns are unlikely to occur naturally

Reference: The article "Data Colada" describes a case where researchers found evidence of manipulation in a dataset about dishonesty research. They found:
- Rows that were out of order when sorted by ID within experimental conditions
- Evidence in Excel's calcChain.xml showing rows had been moved between conditions
- The suspicious observations showed extremely strong effects in the predicted direction

If I provided a research paper, use its context to guide your analysis about what kinds of effects or patterns would be considered suspicious for this specific research domain.

Your assessment:"""

    print("Generating forensic analysis with Claude...")
    response = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=4096,
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
                            df,
                            group_col,
                            column_categories["outcome_columns"],
                            suspicious_rows,
                        )
                        if outcome_plot:
                            plots.append(
                                {"type": "suspicious_vs_normal", "path": outcome_plot}
                            )
                            print(f"Created suspicious vs normal plot: {outcome_plot}")
                    except Exception as e:
                        print(f"Error creating suspicious vs normal plot: {e}")

                    try:
                        # Plot effect sizes
                        effect_plot = visualizer.plot_effect_sizes(
                            df,
                            group_col,
                            column_categories["outcome_columns"],
                            suspicious_rows,
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
