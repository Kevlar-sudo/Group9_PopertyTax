import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import feature_engineering as fe

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create output directories if they don't exist
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the dataset
print("Loading the dataset...")
df = pd.read_csv('2024_Property_Tax_Roll.csv', low_memory=False)

# Display basic information
print("\n==== Dataset Overview ====")
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Display the first few rows
print("\n==== First 5 rows ====")
print(df.head())

# Data types and missing values
print("\n==== Data Types and Missing Values ====")
missing_data = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_data)

# Summary statistics for numerical columns
print("\n==== Summary Statistics for Numerical Columns ====")
print(df.describe().T)

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

# Data preprocessing functions
def preprocess_data(df):
    """Preprocess the dataset for modeling"""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    for col in numerical_cols:
        if processed_df[col].isnull().sum() > 0:
            # Fill missing numerical values with median
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    for col in categorical_cols:
        if processed_df[col].isnull().sum() > 0:
            # Fill missing categorical values with mode
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    # Apply feature engineering
    print("\n==== Feature Engineering ====")
    processed_df = fe.engineer_features(processed_df)
    
    # Create advanced features
    processed_df = fe.create_advanced_features(processed_df)
    
    # Save the preprocessed data
    processed_df.to_csv('data/preprocessed_property_tax_data.csv', index=False)
    
    return processed_df

# EDA function
def perform_eda(df):
    """Perform exploratory data analysis and save visualizations"""
    
    # Distribution of key numerical features
    plt.figure(figsize=(16, 12))
    plt.suptitle('Distribution of Key Numerical Features', fontsize=20)
    
    # Plot histograms for the first 6 numerical columns (or all if less than 6)
    num_plots = min(6, len(numerical_cols))
    for i, col in enumerate(numerical_cols[:num_plots]):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'{col} Distribution')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/numerical_distributions.png')
    plt.close()
    
    # Correlation matrix for numerical features
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png')
    plt.close()
    
    # Bar plots for categorical features (top 5 categories)
    plt.figure(figsize=(16, 12))
    plt.suptitle('Distribution of Top Categories in Categorical Features', fontsize=20)
    
    # Plot bar charts for the first 6 categorical columns (or all if less than 6)
    cat_plots = min(6, len(categorical_cols))
    for i, col in enumerate(categorical_cols[:cat_plots]):
        plt.subplot(2, 3, i+1)
        df[col].value_counts().nlargest(5).plot(kind='bar')
        plt.title(f'Top 5 {col} Categories')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/categorical_distributions.png')
    plt.close()
    
    # Boxplots to detect outliers in numerical features
    plt.figure(figsize=(16, 12))
    plt.suptitle('Boxplots for Numerical Features (Outlier Detection)', fontsize=20)
    
    for i, col in enumerate(numerical_cols[:num_plots]):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=df[col])
        plt.title(f'{col} Boxplot')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('figures/boxplots_outliers.png')
    plt.close()
    
    # Additional visualizations for engineered features
    print("\n==== Additional EDA for Engineered Features ====")
    perform_additional_eda(df)

def perform_additional_eda(df):
    """Perform additional EDA for engineered features"""
    
    # Check if the engineered features exist
    engineered_features = ['longitude', 'latitude', 'tax_rate', 'exemption_pct', 
                           'is_residential', 'is_commercial', 'is_industrial', 'zip_numeric',
                           'value_percentile_in_zip', 'tax_percentile_in_zip']
    
    existing_features = [col for col in engineered_features if col in df.columns]
    
    if len(existing_features) > 0:
        # Geographical distribution of properties
        if 'longitude' in df.columns and 'latitude' in df.columns:
            plt.figure(figsize=(12, 10))
            plt.scatter(df['longitude'], df['latitude'], alpha=0.5, s=1)
            plt.title('Geographic Distribution of Properties', fontsize=16)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.tight_layout()
            plt.savefig('figures/geographic_distribution.png')
            plt.close()
        
        # Tax rate distribution
        if 'tax_rate' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['tax_rate'].clip(0, df['tax_rate'].quantile(0.95)), bins=50, kde=True)
            plt.title('Tax Rate Distribution (Clipped at 95th percentile)', fontsize=16)
            plt.xlabel('Tax Rate')
            plt.tight_layout()
            plt.savefig('figures/tax_rate_distribution.png')
            plt.close()
        
        # Property type distribution
        property_types = ['is_residential', 'is_commercial', 'is_industrial']
        existing_types = [col for col in property_types if col in df.columns]
        
        if len(existing_types) > 0:
            plt.figure(figsize=(10, 6))
            property_counts = [df[col].sum() for col in existing_types]
            plt.bar(existing_types, property_counts)
            plt.title('Property Type Distribution', fontsize=16)
            plt.xlabel('Property Type')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('figures/property_type_distribution.png')
            plt.close()
        
        # Tax vs Assessment scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(df['TOTAL_ASSMT'], df['TOTAL_TAXES'], alpha=0.5, s=3)
        plt.title('Property Tax vs Assessment Value', fontsize=16)
        plt.xlabel('Total Assessment ($)')
        plt.ylabel('Total Taxes ($)')
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('figures/tax_vs_assessment.png')
        plt.close()
        
        # Tax percentile vs Value percentile
        if 'tax_percentile_in_zip' in df.columns and 'value_percentile_in_zip' in df.columns:
            plt.figure(figsize=(10, 8))
            plt.scatter(df['value_percentile_in_zip'], df['tax_percentile_in_zip'], alpha=0.5, s=3)
            plt.title('Tax Percentile vs Value Percentile (within ZIP code)', fontsize=16)
            plt.xlabel('Value Percentile')
            plt.ylabel('Tax Percentile')
            plt.plot([0, 1], [0, 1], 'r--')  # Perfect correlation line
            plt.tight_layout()
            plt.savefig('figures/percentile_comparison.png')
            plt.close()

# Main execution
if __name__ == "__main__":
    try:
        # Process the data
        print("\n==== Processing Data ====")
        processed_df = preprocess_data(df)
        print("Data preprocessing completed and saved to data/preprocessed_property_tax_data.csv")
        
        # Perform EDA
        print("\n==== Performing Exploratory Data Analysis ====")
        perform_eda(processed_df)
        print("EDA completed. Visualizations saved to the 'figures' directory.")
        
    except Exception as e:
        print(f"An error occurred: {e}") 