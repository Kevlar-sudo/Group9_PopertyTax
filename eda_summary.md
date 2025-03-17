# Exploratory Data Analysis Summary: 2024 Property Tax Roll

## Dataset Overview

- **Records**: 44,034 properties
- **Features**: 30 original columns + engineered features
- **Memory Usage**: 10.08 MB (original dataset)

## Key Observations

### Data Quality
- Most numerical columns have minimal missing values (< 1%)
- Notable missing data in categorical fields:
  - COMPANY: 79.26% missing
  - UNIT: 89.65% missing
  - FIRST_NAME/LAST_NAME: ~22% missing
  - ZIP_POSTAL: 3.26% missing
  - SUFFIX: 3.62% missing

### Property Assessment and Taxes
- Significant variation in assessment values:
  - Median: $311,800
  - Mean: $633,305 (highly skewed due to outliers)
  - Max: $708,726,600
- Tax distribution:
  - Median: $3,890
  - Mean: $7,544 (skewed due to outliers)
  - Max: $2,918,528

### Property Types
- Class distribution shows majority is likely residential properties
- Some properties have unique classifications (up to Class 84)

### Geographic Patterns
- Properties show clustering patterns based on longitude/latitude
- Data likely represents a specific geographic region/municipality

### Tax Correlations
- Strong correlation between property assessment and property taxes
- Tax rates (tax/assessment) show variation, which may indicate:
  - Different tax policies for different property types
  - Special tax districts or exemptions
  - Assessment variations

### Exemptions
- Many properties (majority) have no tax exemption
- Some properties have significant exemptions, even 100% exemption

## Insights for Modeling

1. **Key Predictive Features**:
   - Property assessment value
   - Property class/type
   - Geographic location (latitude/longitude)
   - ZIP code (neighborhood effects)

2. **Data Preparation Requirements**:
   - Handle skewed distributions through transformations
   - Consider log transformations for monetary values
   - Normalize geographical coordinates
   - Address missing categorical data
   - Create meaningful aggregations by location

3. **Modeling Considerations**:
   - Tax prediction likely has a strong linear component with the assessment value
   - Neighborhood effects suggest hierarchical/mixed models might be appropriate
   - Outliers require special handling or robust modeling techniques
   - Different property types may require separate models or categorical embeddings

4. **Evaluation Metrics**:
   - Consider using both absolute (RMSE) and percentage (MAPE) error metrics due to wide range of values
   - Stratified evaluation across different property types and value ranges

5. **Feature Engineering Value**:
   - Derived features like tax_rate and zip-based aggregations show potential predictive value
   - Percentile ranks within neighborhoods help normalize property values
   - Property type indicators separate different tax treatment categories

## Next Steps

1. Develop predictive models for property tax estimation
2. Apply feature selection to identify most impactful variables
3. Consider time-series analysis if historical data becomes available
4. Evaluate model performance across different property segments
5. Develop visualizations and tools for tax prediction and analysis 