import io
import json
import os

import pandas as pd
from Bio import SeqIO


def detect_file_type(file_path):
    """Detect the type of data file based on extension and content."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".csv", ".tsv", ".txt"]:
        return "tabular"
    elif ext in [".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn"]:
        return "sequence"
    elif ext in [".json"]:
        return "json"
    elif ext in [".vcf"]:
        return "variant"
    else:
        # Try to infer from content
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith(">"):
                return "sequence"
            elif first_line.startswith("##fileformat=VCF"):
                return "variant"
            elif "," in first_line or "\t" in first_line:
                return "tabular"

    return "unknown"


def load_data(file_path):
    """Load data from various file formats."""
    file_type = detect_file_type(file_path)

    if file_type == "tabular":
        # Detect delimiter
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
            if "\t" in first_line:
                delimiter = "\t"
            else:
                delimiter = ","

        return pd.read_csv(file_path, delimiter=delimiter)

    elif file_type == "sequence":
        records = list(SeqIO.parse(file_path, "fasta"))
        return records

    elif file_type == "json":
        return pd.read_json(file_path)

    elif file_type == "variant":
        # Very basic VCF handling - in a real app, use specialized libraries
        with open(file_path, "r") as f:
            lines = [line for line in f if not line.startswith("##")]

        return pd.read_csv(io.StringIO("".join(lines)), sep="\t", comment="#")

    else:
        # Just return the raw text
        with open(file_path, "r") as f:
            return f.read()


def format_data_for_claude(data, max_chars=100000):
    """Format data for Claude API in a way that's suitable for analysis."""
    if isinstance(data, pd.DataFrame):
        # Get basic stats
        stats = {
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "null_counts": data.isnull().sum().to_dict(),
            "sample": data.head(5).to_dict(orient="records"),
        }

        formatted = f"DataFrame Statistics:\n{json.dumps(stats, indent=2)}\n\n"

        # Add more detailed sample if space allows
        if len(formatted) < max_chars / 2:
            formatted += (
                f"DataFrame Sample (first 10 rows):\n{data.head(10).to_markdown()}\n\n"
            )

        return formatted

    elif isinstance(data, list) and hasattr(data[0], "seq"):
        # Sequence data
        formatted = f"Sequence Data ({len(data)} sequences):\n\n"

        for i, record in enumerate(data[:5]):
            formatted += f"Sequence {i + 1}: {record.id}\n"
            formatted += f"Description: {record.description}\n"
            formatted += f"Length: {len(record.seq)} bp\n"
            formatted += f"First 50 bases: {record.seq[:50]}...\n\n"

        return formatted

    elif isinstance(data, str):
        # Raw text, possibly truncate
        if len(data) > max_chars:
            return data[:max_chars] + "... [truncated]"
        return data

    else:
        # Fallback
        return str(data)
