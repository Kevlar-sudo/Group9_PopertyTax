import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import re

def extract_coordinates(geom_str):
    """Extract coordinates from geometry string"""
    if pd.isna(geom_str):
        return np.nan, np.nan
    
    # Extract coordinates from POINT format
    match = re.search(r'POINT \(([^ ]+) ([^)]+)\)', geom_str)
    if match:
        try:
            lon = float(match.group(1))
            lat = float(match.group(2))
            return lon, lat
        except (ValueError, IndexError):
            return np.nan, np.nan
    return np.nan, np.nan

def engineer_features(df):
    """Create new features from existing data"""
    # Make a copy to avoid modifying the original
    df_new = df.copy()
    
    # Extract coordinates from Property_Location
    print("Extracting coordinates from Property_Location...")
    df_new[['longitude', 'latitude']] = pd.DataFrame(
        df_new['Property_Location'].apply(extract_coordinates).tolist(),
        index=df_new.index
    )
    
    # Create tax rate feature
    print("Creating tax rate feature...")
    # Avoid division by zero by adding a small constant
    df_new['tax_rate'] = df_new['TOTAL_TAXES'] / (df_new['TOTAL_ASSMT'] + 1e-10)
    
    # Create exemption percentage feature
    print("Creating exemption percentage feature...")
    df_new['exemption_pct'] = df_new['TOTAL_EXEMPT'] / (df_new['TOTAL_ASSMT'] + 1e-10)
    
    # Create property type indicators based on CLASS and descriptions
    print("Creating property type indicators...")
    
    # Create binary indicators for residential properties
    df_new['is_residential'] = df_new['CLASS'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 0)
    
    # Create binary indicators for commercial properties
    df_new['is_commercial'] = df_new['CLASS'].apply(lambda x: 1 if x in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] else 0)
    
    # Create binary indicators for industrial properties
    df_new['is_industrial'] = df_new['CLASS'].apply(lambda x: 1 if x in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] else 0)
    
    # Create ZIP code as a numeric feature if possible
    print("Processing ZIP codes...")
    
    def extract_zip_numeric(zip_str):
        if pd.isna(zip_str):
            return np.nan
        # Extract first 5 digits if it's a ZIP+4 code
        match = re.search(r'^(\d{5})', str(zip_str))
        if match:
            return int(match.group(1))
        return np.nan
    
    df_new['zip_numeric'] = df_new['ZIP_POSTAL'].apply(extract_zip_numeric)
    
    return df_new

def create_advanced_features(df):
    """Create more advanced features for modeling"""
    df_advanced = df.copy()
    
    # Group by ZIP and calculate statistics
    print("Creating ZIP code aggregate features...")
    zip_stats = df.groupby('zip_numeric').agg({
        'TOTAL_ASSMT': ['mean', 'median', 'std'],
        'TOTAL_TAXES': ['mean', 'median', 'std'],
        'tax_rate': ['mean', 'median', 'std']
    }).reset_index()
    
    # Flatten the multi-index columns
    zip_stats.columns = ['zip_numeric'] + [
        f'{col[0]}_{col[1]}_by_zip' for col in zip_stats.columns[1:]
    ]
    
    # Merge the statistics back to the main dataframe
    df_advanced = df_advanced.merge(zip_stats, on='zip_numeric', how='left')
    
    # Calculate property value percentile within each zip code
    print("Calculating property value percentiles...")
    
    # Group by zip_numeric and calculate percentile rank
    df_advanced['value_percentile_in_zip'] = df_advanced.groupby('zip_numeric')['TOTAL_ASSMT'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Calculate tax percentile within each zip code
    df_advanced['tax_percentile_in_zip'] = df_advanced.groupby('zip_numeric')['TOTAL_TAXES'].transform(
        lambda x: x.rank(pct=True)
    )
    
    return df_advanced

def create_preprocessing_pipeline(df, numerical_cols, categorical_cols):
    """Create a preprocessing pipeline for the data"""
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Only include a subset of categorical columns to avoid dimensionality explosion
    # Choose a reasonable number of categorical columns with low cardinality
    cat_cols_filtered = []
    for col in categorical_cols:
        if col in df.columns and df[col].nunique() < 20:  # Only include columns with fewer than 20 unique values
            cat_cols_filtered.append(col)
    
    # Combine the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, cat_cols_filtered)
        ])
    
    return preprocessor

if __name__ == "__main__":
    # For testing
    print("This is a feature engineering module designed to be imported, not run directly.") 