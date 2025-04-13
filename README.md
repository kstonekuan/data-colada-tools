# Data Falsification Detector

A tool for detecting data manipulation in academic research using Anthropic's Claude AI.

This system is inspired by the investigative techniques described in the "Data Colada" blog series, which uncovered evidence of fraud in multiple academic papers.

## Features

- **Excel Metadata Analysis**: Examines metadata in Excel files, including calcChain.xml, to detect manual row manipulation
- **Sorting Anomaly Detection**: Identifies observations that are out of sequence in datasets
- **Statistical Pattern Recognition**: Checks if suspicious rows show unusually strong effects
- **Duplicate ID Detection**: Identifies duplicate participant IDs
- **Claude-Powered Analysis**: Uses Claude AI to categorize columns and provide expert analysis

## Setup

1. Clone this repository

2. Set up a virtual environment and install dependencies:
   ```
   ./setup_venv.sh
   ```
   
   This will:
   - Create a Python virtual environment in the `venv` directory
   - Install required dependencies
   - Install the package in development mode
   
   Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

3. Set up your Claude API key by running:
   ```
   python setup_api_key.py
   ```
   
   Alternatively, you can:
   - Copy `config.json.example` to `config.json` and add your API key
   - Set the `CLAUDE_API_KEY` environment variable

## Usage

Make sure your virtual environment is activated:
```
source venv/bin/activate
```

Analyze a dataset for potential manipulation:

```
python src/main.py --data /path/to/your/data.xlsx --output ./reports
```

Arguments:
- `--data`: Path to the data file (supports Excel, CSV, Stata formats)
- `--output`: Directory to save analysis reports
- `--api-key`: Provide Claude API key directly (overrides config file and environment variable)

## How It Works

The system implements the data forensic techniques described in the Data Colada blog:

1. **Sorting Anomalies**: Detects rows that are out of sequence when sorted by ID within experimental conditions
2. **Excel Forensics**: For Excel files, examines calcChain.xml to find evidence that rows were moved between conditions
3. **Statistical Analysis**: Checks if suspicious observations show unusually strong effects in the predicted direction
4. **Claude Analysis**: Uses Claude to provide expert assessment of detected patterns

## Example Output

The system generates a comprehensive report with:
- Column categorization (ID columns, group columns, outcome columns)
- Technical findings (sorting anomalies, Excel metadata evidence)
- Claude's expert analysis of the likelihood of data manipulation

## Contributing

Contributions to improve the detection methods are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## Limitations

This tool provides evidence that may suggest data manipulation but cannot definitively prove fraud. Always conduct a thorough investigation before making accusations.

## License

MIT

## Acknowledgements

Inspired by the forensic methods developed by Uri Simonsohn, Joe Simmons, and Leif Nelson as described in their Data Colada blog.