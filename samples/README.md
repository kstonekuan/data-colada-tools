# Sample Datasets for Data Colada Tools

This folder contains sample datasets and their corresponding research papers that can be used to test the Data Colada data forensics tool.

## Structure

- `/datasets`: Contains the sample datasets in supported formats (.xlsx, .csv, .dta, .sav)
- `/papers`: Contains research papers in PDF format that correspond to the datasets

## Adding Your Own Samples

To add a new sample:

1. Place your dataset file in the `/datasets` folder
2. If you have a corresponding research paper, place it in the `/papers` folder with the same base name as your dataset
3. To add a description for the dataset, create a text file with the same base name in the `/datasets` folder

### Example:

For a dataset named `example_study.csv`:
- Dataset: `/datasets/example_study.csv`
- Description: `/datasets/example_study.txt`
- Research paper: `/papers/example_study.pdf`

## Naming Convention

- Use descriptive names for your datasets (e.g., `sequential_data_manipulation.csv`, `fabricated_data_example.xlsx`)
- Use lowercase with underscores for spaces
- Match the base filename across all related files

## Included Samples

- `example_dataset.csv`: A simple clean dataset for demonstration purposes
- `manipulated_data.csv`: A dataset with deliberate sorting anomalies to demonstrate detection capabilities

## Using Samples

Samples can be selected from the "Sample Datasets" section on the main page. The analysis will be performed with the same capabilities as user-uploaded datasets.