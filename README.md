# PropTax Analyzer: AI-Driven Government Property Tax Prediction

This project focuses on analyzing property tax data using machine learning techniques to predict government property taxes.

## Project Overview

The PropTax Analyzer uses artificial intelligence to process and analyze property tax data, identify patterns, and make predictions about property tax assessments.

## Data Processing

The project includes data preprocessing and exploratory data analysis (EDA) on the `2024_Property_Tax_Roll.csv` dataset.

### Data Preprocessing

The data preprocessing steps include:
- Handling missing values
- Data type conversion
- Feature normalization
- Outlier detection and handling

### Feature Engineering

The project includes a dedicated feature engineering module that:
- Extracts geographical coordinates from location data
- Creates derived features (tax rate, exemption percentage)
- Generates property type indicators
- Calculates ZIP code aggregate statistics
- Computes percentile ranks for property values and taxes

### Exploratory Data Analysis (EDA)

The EDA process generates insights through:
- Distribution analysis of key features
- Correlation analysis between variables
- Categorical data visualization
- Outlier identification through box plots
- Geographic distribution of properties
- Tax rate distributions
- Property type analysis
- Tax vs assessment value relationships
- Percentile comparisons

A comprehensive summary of EDA findings is available in the [EDA Summary](eda_summary.md) document.

## Setup and Usage

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the data preprocessing and EDA script:
   ```
   python preprocess_and_eda.py
   ```

3. Check the generated visualizations in the `figures/` directory and the preprocessed dataset in the `data/` directory.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Project Structure

```
.
├── 2024_Property_Tax_Roll.csv       # Raw property tax dataset
├── preprocess_and_eda.py            # Data preprocessing and EDA script
├── feature_engineering.py           # Feature engineering module
├── eda_summary.md                   # Summary of EDA findings and insights
├── data/                            # Directory for processed data
│   └── preprocessed_property_tax_data.csv  # Preprocessed dataset
├── figures/                         # Directory for EDA visualizations
│   ├── numerical_distributions.png
│   ├── correlation_matrix.png
│   ├── categorical_distributions.png
│   ├── boxplots_outliers.png
│   ├── geographic_distribution.png
│   ├── tax_rate_distribution.png
│   ├── property_type_distribution.png
│   ├── tax_vs_assessment.png
│   └── percentile_comparison.png
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
``` 