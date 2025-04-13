# Data Forensics Web Interface

This web interface allows you to easily upload and analyze Excel files for potential data manipulation without using the command line.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your Claude API key:

```bash
python setup_api_key.py
```

## Running the Web Interface

Start the web server:

```bash
python app.py
```

Then open your browser and navigate to: http://127.0.0.1:5000/

## Using the Web Interface

1. **Upload your data file**:
   - Drag and drop your Excel file into the upload area, or click "Browse Files"
   - Supported formats: Excel (.xlsx), CSV (.csv), Stata (.dta)

2. **Analyze the data**:
   - Click the "Analyze Data" button
   - The analysis will take a few minutes to process

3. **View the results**:
   - **Report Tab**: Shows a detailed analysis of potential data manipulation
   - **Visualizations Tab**: Shows graphs and charts of the analysis results

## Analysis Features

The web interface performs the same comprehensive analysis as the command-line tool:

- Detection of sorting anomalies in ID sequences
- Analysis of Excel file metadata for evidence of row manipulation
- Statistical analysis of suspicious observations
- Effect size comparison between suspicious and normal data points
- Visual representation of findings
- AI-powered analysis of potential manipulation patterns

## Directory Structure

When you upload and analyze a file, the following directories are used:

- `uploads/`: Stores uploaded data files
- `results/`: Contains analysis results, reports, and visualizations

## Troubleshooting

- If you encounter an error, check the error message on the screen
- Make sure your Claude API key is properly configured
- Verify your data file is in one of the supported formats