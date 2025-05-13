# Data Colada Tools

A tool for detecting data manipulation in academic research using Anthropic's Claude AI.

This system is inspired by the investigative techniques described in the ["Data Colada"](https://datacolada.org/) blog series, which uncovered evidence of fraud in multiple academic papers.

## Features

- **Excel Metadata Analysis**: Examines metadata in Excel files, including calcChain.xml, to detect manual row manipulation
- **Sorting Anomaly Detection**: Identifies observations that are out of sequence in datasets
- **Statistical Pattern Recognition**: Checks if suspicious rows show unusually strong effects
- **Duplicate ID Detection**: Identifies duplicate participant IDs
- **Claude-Powered Analysis**: Uses Claude AI to categorize columns and provide expert analysis
- **Web Interface**: User-friendly browser interface for analyzing datasets without command line experience

## Setup

1. Clone this repository

2. Set up a virtual environment and install dependencies:
   ```
   ./setup_venv.sh
   ```
   
   This will:
   - Check for and install [uv](https://github.com/astral-sh/uv) if needed
   - Create a Python virtual environment using uv in the `venv` directory
   - Install required dependencies with uv
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

### Command Line Interface

Analyze a dataset for potential manipulation:

```
python src/main.py --data /path/to/your/data.xlsx --output ./reports
```

Arguments:
- `--data`: Path to the data file (supports Excel, CSV, Stata formats)
- `--output`: Directory to save analysis reports
- `--api-key`: Provide Claude API key directly (overrides config file and environment variable)

### Web Application

Run the Flask web application:
```
python app.py
```

Access the web interface at http://127.0.0.1:5000/

#### Using the Web Interface

1. **Upload your data file**:
   - Drag and drop your Excel file into the upload area, or click "Browse Files"
   - Supported formats: Excel (.xlsx), CSV (.csv), Stata (.dta), SPSS (.sav)

2. **Analyze the data**:
   - Click the "Analyze Data" button
   - The analysis will take a few minutes to process

3. **View the results**:
   - **Report Tab**: Shows a detailed analysis of potential data manipulation
   - **Technical Findings Tab**: Shows detailed evidence of data manipulation
   - **Visualizations Tab**: Shows graphs and charts of the analysis results
   - **Data Preview Tab**: Shows the dataset with suspicious data highlighted

4. **Using Sample Datasets**:
   - The web interface includes sample datasets that demonstrate the tool's capabilities
   - These samples include datasets with known manipulations for educational purposes

## Directory Structure

- `src/`: Source code for the data analysis library
- `static/`: CSS and JavaScript files for the web interface
- `templates/`: HTML templates for the web interface
- `uploads/`: Stores uploaded data files (created automatically)
- `results/`: Contains analysis results, reports, and visualizations (created automatically)
- `samples/`: Sample datasets for demonstration purposes

## Development

### Linting and Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. To check your code:

```
./lint.sh
```

This will:
1. Check for linting issues
2. Offer to fix them automatically
3. Format the code according to the project's style rules

You can also run specific Ruff commands:
```
# Just check for issues
ruff check src/ app.py

# Fix issues automatically
ruff check --fix src/ app.py

# Format code
ruff format src/ app.py
```

The linting configuration is defined in `pyproject.toml`.

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
- Visualizations of suspicious data points
- Interactive data preview with highlighted suspicious observations

## Troubleshooting

- If you encounter an error, check the error message on the screen or in the terminal
- Make sure your Claude API key is properly configured
- Verify your data file is in one of the supported formats
- For issues with the web interface, check the console logs in your browser's developer tools
- If running into memory issues with large datasets, try running the analysis in smaller chunks

## Contributing

Contributions to improve the detection methods are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run the linting script to ensure code quality
5. Submit a pull request

## Limitations

This tool provides evidence that may suggest data manipulation but cannot definitively prove fraud. Always conduct a thorough investigation before making accusations.

## License

MIT

## Acknowledgements

Inspired by the forensic methods developed by Uri Simonsohn, Joe Simmons, and Leif Nelson as described in their Data Colada blog.