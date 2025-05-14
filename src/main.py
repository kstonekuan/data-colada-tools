#!/usr/bin/env python3
import argparse
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from anthropic import Anthropic

# Import forensics modules
from src.data_forensics import DataForensics
from src.excel_forensics import ExcelForensics
from src.sorting_forensics import SortingForensics
from src.statistical_forensics import StatisticalForensics
from src.visualize import ForensicVisualizer

# Set up module logger
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, str]:
    """Load configuration from config.json or environment variables.

    Returns:
        Dict[str, str]: Configuration dictionary with API keys and other settings
    """
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
            logger.error(f"Error loading config file: {e}")

    return config


def setup_client(api_key: Optional[str] = None) -> Anthropic:
    """Set up the Claude API client.

    Args:
        api_key: Optional API key to override the one in config/environment

    Returns:
        Anthropic: Configured Claude API client

    Raises:
        ValueError: If no API key is provided in any form
    """
    config = load_config()

    # Command line API key takes precedence
    if api_key:
        config["api_key"] = api_key

    if not config.get("api_key"):
        raise ValueError(
            "Claude API key is required. Please provide it via config.json, CLAUDE_API_KEY environment variable, or --api-key flag."
        )

    return Anthropic(api_key=config["api_key"])


def identify_columns(client: Anthropic, df: pd.DataFrame) -> Dict[str, List[str]]:
    """Use Claude to identify and categorize columns in the dataset.

    Args:
        client: Claude API client
        df: Pandas DataFrame to analyze

    Returns:
        Dict[str, List[str]]: Dictionary mapping column categories to lists of column names
    """
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
        logger.error(f"Error parsing column categories: {e}")
        # Fallback
        return {
            "id_columns": [],
            "group_columns": [],
            "outcome_columns": [],
            "demographic_columns": [],
            "other_columns": columns,
        }


def analyze_column_unique_values(
    client: Anthropic,
    data_path: str,
    columns: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze unique values in each column using Claude.

    Args:
        client: Claude API client
        data_path: Path to the dataset file
        columns: Optional list of columns to analyze. If None, analyze all columns.
        output_dir: Directory to save output file

    Returns:
        Dictionary mapping column names to analysis results
    """
    logger.info(f"Analyzing unique column values in: {data_path}")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize data forensics
    forensics = DataForensics()

    # Read the data file
    try:
        forensics.analyze_dataset(data_path)
    except Exception as e:
        error_msg = f"Error loading dataset: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Analyze unique values in each column
    try:
        results = forensics.analyze_column_unique_values(client, columns)
    except Exception as e:
        error_msg = f"Error analyzing column values: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Generate a summary report of highly suspicious columns
    suspicious_columns = []
    for column, result in results.items():
        if (
            "suspicion_rating" in result
            and result["suspicion_rating"]
            and result["suspicion_rating"] >= 7
        ):
            suspicious_columns.append(
                {
                    "column": column,
                    "rating": result["suspicion_rating"],
                    "unique_count": result["unique_count"],
                }
            )

    # Sort by suspicion rating (descending)
    suspicious_columns.sort(key=lambda x: x["rating"], reverse=True)

    # Include the summary in the results
    results["summary"] = {
        "total_columns_analyzed": len(results) - 1,  # Exclude the summary key
        "suspicious_columns": suspicious_columns,
        "average_suspicion": np.mean(
            [
                r.get("suspicion_rating", 0) or 0
                for c, r in results.items()
                if c != "summary"
            ]
        ),
    }

    # Save results to file if output directory provided
    if output_dir:
        report_path = os.path.join(
            output_dir, f"column_analysis_{os.path.basename(data_path)}.json"
        )
        with open(report_path, "w") as f:
            # Use the enhanced JSON serializer to handle all types
            def json_serializer(obj):
                if isinstance(obj, (np.integer, int, float)):
                    return int(obj) if isinstance(obj, np.integer) else obj
                if isinstance(obj, (np.floating, float)):
                    return float(obj)
                if isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                if isinstance(obj, (np.bool_, bool)):
                    return str(
                        obj
                    ).lower()  # Convert boolean to string "true" or "false"
                return str(obj)  # Convert any other types to strings

            json.dump(results, f, indent=2, default=json_serializer)
        logger.info(f"Column analysis saved to {report_path}")

    return results


def detect_data_manipulation(
    client: Anthropic,
    data_path: str,
    output_dir: Optional[str] = None,
    paper_path: Optional[str] = None,
    use_claude_segmentation: bool = False,
    user_suspicions: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Detect potential data manipulation in research data.

    Args:
        client: Claude API client
        data_path: Path to the dataset file
        output_dir: Directory to save output files
        paper_path: Optional path to research paper PDF for context
        use_claude_segmentation: Whether to use Claude to analyze data in segments
        user_suspicions: Optional dictionary with user-specified suspicions to guide analysis.
            Format: {
                "focus_columns": List[str],  # Columns to prioritize checking
                "potential_issues": List[str],  # e.g., "sorting", "out_of_order", "duplicates"
                "treatment_columns": List[str],  # Potential treatment indicator columns
                "outcome_columns": List[str],  # Outcome variables to analyze
                "suspicious_rows": List[int],  # Specific rows to check more carefully
                "suspect_grouping": str,  # Potential column to check for group-based manipulation
                "description": str  # Free-text description of suspicions
            }

    Returns:
        Optional[str]: Report content as a string, or None/error message on failure
    """
    logger.info(f"Analyzing data file: {data_path}")
    if paper_path:
        logger.info(f"Using research paper for context: {paper_path}")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine file type and read data
    file_ext = os.path.splitext(data_path)[1].lower()

    # Initialize data forensics
    forensics = DataForensics()

    # Load dataset by calling analyze_dataset
    try:
        forensics.analyze_dataset(data_path)
    except Exception as e:
        error_msg = f"Error loading dataset: {str(e)}"
        logger.error(error_msg)
        return error_msg

    # Get dataframe from forensics object
    df = forensics.df

    # Create sorting forensics object
    sorting_forensics = SortingForensics(df)

    # Create statistical forensics object
    statistical_forensics = StatisticalForensics(df)

    # Create Excel forensics object for deeper analysis if needed
    excel_forensics = None
    if file_ext == ".xlsx":
        excel_forensics = ExcelForensics(data_path)

    # Use Claude to identify column categories
    column_categories = identify_columns(client, df)

    # Basic analysis with DataForensics
    findings = []

    # If we have ID and group columns, check for sorting anomalies
    if column_categories["id_columns"] and column_categories["group_columns"]:
        id_col = column_categories["id_columns"][0]  # Use first ID column
        group_col = column_categories["group_columns"][0]  # Use first group column

        logger.info(
            f"Checking sorting anomalies for ID column '{id_col}' within groups '{group_col}'"
        )

        # Pass user suspicions if provided
        if user_suspicions:
            logger.info(
                "Using user-provided suspicions to guide (but not bias) analysis"
            )
            sorting_issues = sorting_forensics.check_sorting_anomalies(
                id_col,
                group_col,
                check_dependent_vars=True,
                prioritize_columns=user_suspicions.get("focus_columns", [])
                + user_suspicions.get("outcome_columns", []),
                prioritize_out_of_order=any(
                    issue.lower()
                    in ["out of order", "out-of-order", "out_of_order", "sorting"]
                    for issue in user_suspicions.get("potential_issues", [])
                ),
            )
        else:
            sorting_issues = sorting_forensics.check_sorting_anomalies(
                id_col, group_col
            )

        if sorting_issues:
            findings.append({"type": "sorting_anomaly", "details": sorting_issues})
            logger.info(f"Found {len(sorting_issues)} sorting anomalies")

            # If Excel file, check calc chain for evidence of row movement
            if file_ext == ".xlsx" and excel_forensics:
                with excel_forensics as ef:
                    suspicious_rows = [
                        int(issue["row_index"]) for issue in sorting_issues
                    ]
                    # For Excel analysis, we need to adjust row numbers to account for Excel's 1-based indexing
                    logger.info(
                        f"Checking for Excel row movements in rows: {suspicious_rows}"
                    )
                    movement_evidence = ef.analyze_row_movement(suspicious_rows)

                    if movement_evidence:
                        findings.append(
                            {"type": "excel_row_movement", "details": movement_evidence}
                        )
                        logger.info("Found evidence of row movement in Excel file")

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

                    # Compare results with and without suspicious rows
                    comparison_results = forensics.compare_with_without_suspicious_rows(
                        suspicious_rows, group_col, column_categories["outcome_columns"]
                    )

                    if comparison_results and not (
                        isinstance(comparison_results, dict)
                        and "error" in comparison_results
                    ):
                        findings.append(
                            {
                                "type": "with_without_comparison",
                                "details": comparison_results,
                            }
                        )
                        logger.info(
                            "Completed comparison of results with and without suspicious rows"
                        )
                except Exception as e:
                    logger.error(
                        f"Error analyzing effect sizes or comparing results: {e}"
                    )
                    # Don't add to findings if there was an error

    # Check for duplicate IDs
    if column_categories["id_columns"]:
        id_col = column_categories["id_columns"][0]
        duplicate_ids = forensics.check_duplicate_ids(id_col)

        if duplicate_ids:
            findings.append({"type": "duplicate_ids", "details": duplicate_ids})
            logger.info(f"Found {len(duplicate_ids)} duplicate IDs")

    # Check for terminal digit anomalies if there are outcome columns
    if column_categories["outcome_columns"]:
        terminal_digit_results = statistical_forensics.analyze_terminal_digits(
            column_categories["outcome_columns"]
        )

        # Filter to just suspicious columns
        suspicious_terminal_digits = {
            col: result
            for col, result in terminal_digit_results.items()
            if "suspicion" in result and result["suspicion"] in ["medium", "high"]
        }

        if suspicious_terminal_digits:
            findings.append(
                {
                    "type": "terminal_digit_anomaly",
                    "details": suspicious_terminal_digits,
                }
            )
            logger.info(
                f"Found terminal digit anomalies in {len(suspicious_terminal_digits)} columns"
            )

    # Check for distribution anomalies (multimodality)
    if column_categories["outcome_columns"] and column_categories["group_columns"]:
        multimodal_results = statistical_forensics.detect_multimodality(
            column_categories["outcome_columns"],
            by_group=column_categories["group_columns"][0],
        )

        # Filter to just suspicious columns
        suspicious_distributions = {}
        for col, result in multimodal_results.items():
            if isinstance(result, dict) and "suspicion" in result:
                if result["suspicion"] in ["medium", "high"]:
                    suspicious_distributions[col] = result
            elif isinstance(result, dict):  # Group results
                suspicious_groups = {
                    group: group_result
                    for group, group_result in result.items()
                    if "suspicion" in group_result
                    and group_result["suspicion"] in ["medium", "high"]
                }
                if suspicious_groups:
                    suspicious_distributions[col] = suspicious_groups

        if suspicious_distributions:
            findings.append(
                {"type": "distribution_anomaly", "details": suspicious_distributions}
            )
            logger.info(
                f"Found distribution anomalies in {len(suspicious_distributions)} columns"
            )

    # Check for inlier anomalies
    if column_categories["outcome_columns"] and column_categories["group_columns"]:
        inlier_results = statistical_forensics.check_inlier_anomalies(
            column_categories["outcome_columns"],
            by_group=column_categories["group_columns"][0],
        )

        # Filter to just suspicious columns
        suspicious_inliers = {}
        for col, result in inlier_results.items():
            if isinstance(result, dict) and "suspicion" in result:
                if result["suspicion"] in ["medium", "high"]:
                    suspicious_inliers[col] = result
            elif isinstance(result, dict):  # Group results
                suspicious_groups = {
                    group: group_result
                    for group, group_result in result.items()
                    if "suspicion" in group_result
                    and group_result["suspicion"] in ["medium", "high"]
                }
                if suspicious_groups:
                    suspicious_inliers[col] = suspicious_groups

        if suspicious_inliers:
            findings.append({"type": "inlier_anomaly", "details": suspicious_inliers})
            logger.info(f"Found inlier anomalies in {len(suspicious_inliers)} columns")

    # Use Claude to analyze data in segments if requested
    claude_segment_findings = []
    if use_claude_segmentation:
        dataset_info = f"{len(df)} rows, {len(df.columns)} columns"
        logger.info(
            f"Starting dataset segmentation and Claude-based anomaly detection on dataset with {dataset_info}..."
        )
        # Pass user suspicions to Claude analysis if provided
        if user_suspicions:
            claude_segment_findings = forensics.segment_and_analyze_with_claude(
                client, user_suspicions=user_suspicions
            )
        else:
            claude_segment_findings = forensics.segment_and_analyze_with_claude(client)

        # Add a summary of Claude's segment analysis to findings
        if claude_segment_findings:
            # Filter out non-dictionary items and count high-confidence anomalies
            valid_findings: List[Dict[str, Any]] = [
                f for f in claude_segment_findings if isinstance(f, dict)
            ]
            anomaly_chunks: List[Dict[str, Any]] = [
                f
                for f in valid_findings
                if f.get("anomalies_detected", False) and f.get("confidence", 0) >= 7
            ]

            if anomaly_chunks:
                total_anomalies: int = 0
                anomaly_types: Set[str] = set()
                for chunk in anomaly_chunks:
                    if "findings" in chunk:
                        total_anomalies += len(chunk["findings"])
                        for finding in chunk["findings"]:
                            if "type" in finding:
                                anomaly_types.add(finding["type"])

                logger.info(
                    f"Claude detected {total_anomalies} high-confidence anomalies across {len(anomaly_chunks)} chunks"
                )
                findings.append(
                    {
                        "type": "claude_chunk_analysis",
                        "details": {
                            "anomaly_chunks": len(anomaly_chunks),
                            "total_anomalies": total_anomalies,
                            "anomaly_types": list(anomaly_types),
                            "total_chunks": len(claude_segment_findings),
                        },
                    }
                )

    # Generate Claude prompt with findings
    # Define a custom JSON serializer to handle non-serializable types
    def custom_json_serializer(obj):
        if isinstance(obj, (np.integer, int, float)):
            return int(obj) if isinstance(obj, np.integer) else obj
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return str(obj).lower()  # Convert boolean to string "true" or "false"
        elif isinstance(obj, dict):
            return {k: custom_json_serializer(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [custom_json_serializer(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(custom_json_serializer(item) for item in obj)
        elif obj is None:
            return None
        else:
            return str(obj)  # Convert any other types to strings

    # Apply the custom serializer to the findings data
    serialized_findings = custom_json_serializer(findings)
    findings_json = json.dumps(serialized_findings, indent=2)

    # Extract text from research paper if provided
    paper_context = ""
    if paper_path and os.path.exists(paper_path):
        try:
            # Import PyPDF2 lazily to improve startup time
            from src.lazy_imports import get_pypdf2

            PyPDF2 = get_pypdf2()

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
Extract the most important segments verbatim from this research paper. Do NOT summarize or paraphrase - keep the exact original text.

Focus on identifying and extracting relevant sections that address:

1. The research question or hypothesis (e.g., abstract, introduction statements)
2. The methodology used (e.g., method sections, experimental design)
3. The expected findings or hypothesized effects
4. The research domain and its key concepts

Remove references, citations, acknowledgments, and formatting elements like headers/page numbers, but preserve the exact original text of important content. 

DO NOT summarize or paraphrase - only extract complete sentences and paragraphs directly from the original. The extraction must use only words that appear in the exact same order as in the original text.

Research Paper:
{paper_text}
"""
                    try:
                        # Call Claude to extract the key information
                        logger.info(
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
                            logger.info(
                                "Added extracted paper sections to prompt (original text preserved)"
                            )
                    except Exception as e:
                        logger.error(f"Error in Claude summary extraction: {e}")
                        # Fallback to original method if Claude call fails
                        paper_excerpt = f"Research Paper (original text excerpt):\n{paper_text[:5000]}"
                        paper_context = f"I've also extracted text from the research paper that might provide context about the data:\n\n{paper_excerpt}"
                        logger.info(
                            "Added raw research paper context to prompt (fallback method)"
                        )
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
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
5. If there are comparison results showing what happens when suspicious rows are removed, explain the significance of these changes
6. Provide a detailed explanation of why these patterns are unlikely to occur naturally

Reference: Research on data manipulation detection has identified several patterns that may indicate fabrication:

1. Structural evidence:
   - Rows that are out of order when sorted by ID within experimental conditions
   - Evidence in Excel's calcChain.xml showing rows had been moved between conditions
   - Duplicate IDs where identical observations appear in multiple conditions

2. Statistical anomalies:
   - Terminal digit anomalies (non-uniform distribution of last digits)
   - Variance anomalies (suspiciously low variance in certain groups)
   - Excessive repetition of digit patterns
   - Too many "inliers" (values clustered unnaturally close to means)
   - Perfect linear sequences or progressions
   - Suspicious observations showing extremely strong effects in the predicted direction

If I provided a research paper, use its context to guide your analysis about what kinds of effects or patterns would be considered suspicious for this specific research domain.

Your assessment:"""

    logger.info("Generating forensic analysis with Claude...")
    try:
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        analysis = response.content[0].text
    except Exception as e:
        error_details = traceback.format_exc()

        # Create a highly visible error message in the server logs
        error_msg = f"CRITICAL: Claude API call failed: {str(e)}"
        logger.critical(error_msg)
        logger.error(f"Error details: {error_details}")

        # Provide a fallback analysis
        analysis = f"""
MANIPULATION_RATING: 5

⚠️ **API Error: Could not get final analysis from Claude**

There was an error communicating with the Claude API: {str(e)}

However, the preliminary analysis has been completed and can be reviewed in the Technical Findings section. I've assigned a neutral manipulation rating (5/10) since the final analysis could not be completed.

Please review the technical findings and visualizations to make your own assessment of the data. If the error persists, please contact support.
"""

    # Generate report
    # Use custom serializer for column categories too
    serialized_column_categories = custom_json_serializer(column_categories)
    column_categories_json = json.dumps(serialized_column_categories, indent=2)

    report = f"""# Data Forensics Report: {os.path.basename(data_path)}

## Column Categories
```json
{column_categories_json}
```

## Technical Findings
```json
{findings_json}
```

## Claude's Analysis
{analysis}
"""

    # Add Claude's segment analysis if available
    if claude_segment_findings:
        segment_report = "\n## Claude Segment Analysis\n\n"
        segment_report += "Claude analyzed the dataset in segments to identify potential anomalies.\n\n"

        # Sanitize the findings to ensure we only have dictionaries
        sanitized_findings: List[Dict[str, Any]] = []
        for f in claude_segment_findings:
            if not isinstance(f, dict):
                logger.warning(
                    f"Found non-dictionary item in claude_segment_findings: {type(f)}: {f}"
                )
                # Convert to error dictionary
                sanitized_findings.append(
                    {
                        "error": f"Invalid finding type: {type(f)}",
                        "raw_data": str(f)[:200]
                        if isinstance(f, str)
                        else "Non-string type",
                    }
                )
            else:
                sanitized_findings.append(f)

        # Check for any error results
        error_chunks: List[Dict[str, Any]] = [
            f for f in sanitized_findings if isinstance(f, dict) and "error" in f
        ]
        if error_chunks:
            segment_report += f"⚠️ **Note:** {len(error_chunks)} of {len(sanitized_findings)} chunks had errors during processing.\n\n"

            # Add error details to help with debugging
            segment_report += "**Error Details:**\n\n"
            for i, error_chunk in enumerate(error_chunks):
                error_message = error_chunk.get("error", "Unknown error")
                chunk_info = f"chunk {error_chunk.get('chunk', i + 1)}"
                rows_info = error_chunk.get("rows", "unknown rows")

                segment_report += (
                    f"- Error in {chunk_info} ({rows_info}): {error_message}\n"
                )

                # Include raw data snippet if available, for better debugging
                if "raw_data" in error_chunk and error_chunk["raw_data"]:
                    raw_data = error_chunk["raw_data"]
                    if len(raw_data) > 100:
                        raw_data = raw_data[:100] + "..."
                    segment_report += f"  Raw data: `{raw_data}`\n"

                # Include traceback if available
                if "traceback" in error_chunk and error_chunk["traceback"]:
                    segment_report += (
                        f"  Details: ```\n{error_chunk['traceback']}\n```\n"
                    )

            segment_report += "\n"

        # Add summary (filter out error chunks for anomaly detection)
        valid_chunks: List[Dict[str, Any]] = [
            f for f in sanitized_findings if isinstance(f, dict) and "error" not in f
        ]
        anomaly_chunks: List[Dict[str, Any]] = [
            f
            for f in valid_chunks
            if isinstance(f, dict) and f.get("anomalies_detected", False)
        ]
        high_confidence_chunks: List[Dict[str, Any]] = [
            f
            for f in anomaly_chunks
            if isinstance(f, dict) and f.get("confidence", 0) >= 7
        ]

        segment_report += f"- Total segments analyzed: {len(sanitized_findings)}\n"
        segment_report += f"- Segments with anomalies: {len(anomaly_chunks)}\n"
        segment_report += f"- Segments with high-confidence anomalies: {len(high_confidence_chunks)}\n\n"

        # Add details for high-confidence findings
        if high_confidence_chunks:
            segment_report += "### High Confidence Anomalies\n\n"

            for i, chunk in enumerate(high_confidence_chunks):
                chunk_idx = chunk.get("chunk", i + 1)
                confidence = chunk.get("confidence", "N/A")
                explanation = chunk.get("explanation", "No explanation provided")

                segment_report += (
                    f"#### Segment {chunk_idx} (Confidence: {confidence}/10)\n\n"
                )
                segment_report += f"{explanation}\n\n"

                if "findings" in chunk and chunk["findings"]:
                    segment_report += "Specific findings:\n\n"
                    for j, finding in enumerate(chunk["findings"]):
                        f_type = finding.get("type", "Unknown")
                        f_desc = finding.get("description", "No description")
                        f_sev = finding.get("severity", "N/A")
                        f_cols = ", ".join(finding.get("columns_involved", ["Unknown"]))

                        segment_report += (
                            f"**Finding {j + 1}**: {f_type} (Severity: {f_sev}/10)\n"
                        )
                        segment_report += f"- Description: {f_desc}\n"
                        segment_report += f"- Columns involved: {f_cols}\n"

                        if "row_indices" in finding and finding["row_indices"]:
                            if len(finding["row_indices"]) <= 10:
                                row_str = ", ".join(
                                    str(r) for r in finding["row_indices"]
                                )
                            else:
                                first_5 = ", ".join(
                                    str(r) for r in finding["row_indices"][:5]
                                )
                                last_5 = ", ".join(
                                    str(r) for r in finding["row_indices"][-5:]
                                )
                                row_str = f"{first_5}, ... ({len(finding['row_indices']) - 10} more rows) ..., {last_5}"

                            segment_report += f"- Rows affected: {row_str}\n"

                        segment_report += "\n"

                segment_report += "---\n\n"

        report += segment_report

    # Create visualizations if we have suspicious rows
    plots: List[Dict[str, str]] = []
    if findings and output_dir:
        try:
            visualizer: ForensicVisualizer = ForensicVisualizer(output_dir)

            # Extract suspicious rows from findings
            suspicious_rows: List[int] = []
            for finding in findings:
                if finding["type"] == "sorting_anomaly":
                    suspicious_rows.extend(
                        [int(issue["row_index"]) for issue in finding["details"]]
                    )

            if (
                suspicious_rows
                and column_categories["id_columns"]
                and column_categories["group_columns"]
            ):
                id_col: str = column_categories["id_columns"][0]
                group_col: str = column_categories["group_columns"][0]

                try:
                    # Plot ID sequence
                    id_plot = visualizer.plot_id_sequence(
                        df, id_col, group_col, suspicious_rows
                    )
                    if id_plot:
                        plots.append({"type": "id_sequence", "path": id_plot})
                        logger.info(f"Created ID sequence plot: {id_plot}")
                except Exception as e:
                    logger.error(f"Error creating ID sequence plot: {e}")

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
                            logger.info(
                                f"Created suspicious vs normal plot: {outcome_plot}"
                            )
                    except Exception as e:
                        logger.error(f"Error creating suspicious vs normal plot: {e}")

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
                            logger.info(f"Created effect sizes plot: {effect_plot}")
                    except Exception as e:
                        logger.error(f"Error creating effect sizes plot: {e}")

                    # Try to plot with/without comparison if we have that data
                    try:
                        comparison_findings = [
                            f
                            for f in findings
                            if f.get("type") == "with_without_comparison"
                        ]
                        if comparison_findings and "details" in comparison_findings[0]:
                            comparison_results = comparison_findings[0]["details"]
                            comparison_plot = visualizer.plot_with_without_comparison(
                                comparison_results
                            )
                            if comparison_plot:
                                plots.append(
                                    {
                                        "type": "with_without_comparison",
                                        "path": comparison_plot,
                                    }
                                )
                                logger.info(
                                    f"Created with/without comparison plot: {comparison_plot}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error creating with/without comparison plot: {e}"
                        )
        except Exception as e:
            logger.error(f"Error during visualization: {e}")

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
        logger.info(f"Report saved to {report_path}")

    return report


def main() -> int:
    """Main entry point for the application.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Data Manipulation Detection with Claude"
    )
    parser.add_argument("--api-key", help="Claude API key")
    parser.add_argument("--data", required=True, help="Path to input data file")
    parser.add_argument("--output", help="Directory to save output reports")
    parser.add_argument(
        "--analyze-columns",
        action="store_true",
        help="Analyze unique values in each column using Claude",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Specific columns to analyze (for use with --analyze-columns)",
    )

    args: argparse.Namespace = parser.parse_args()

    try:
        client: Anthropic = setup_client(args.api_key)

        if args.analyze_columns:
            # Run column unique value analysis
            logger.info("Running column unique value analysis...")
            results = analyze_column_unique_values(
                client, args.data, args.columns, args.output
            )

            # Print a summary to the console
            if "error" in results:
                logger.error(f"Error: {results['error']}")
                return 1

            if "summary" in results:
                summary = results["summary"]
                print("\nColumn Analysis Summary:")
                print(f"- Analyzed {summary['total_columns_analyzed']} columns")
                print(
                    f"- Average suspicion rating: {summary['average_suspicion']:.2f}/10"
                )

                if summary["suspicious_columns"]:
                    print("\nHighly suspicious columns:")
                    for col in summary["suspicious_columns"]:
                        print(
                            f"- {col['column']}: {col['rating']}/10 ({col['unique_count']} unique values)"
                        )
                else:
                    print("\nNo highly suspicious columns found.")

                if args.output:
                    print(
                        f"\nDetailed results saved to: {os.path.join(args.output, f'column_analysis_{os.path.basename(args.data)}.json')}"
                    )

            logger.info("Column analysis complete!")
        else:
            # Run full data manipulation detection
            detect_data_manipulation(client, args.data, args.output)
            logger.info("Analysis complete!")

        return 0
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
