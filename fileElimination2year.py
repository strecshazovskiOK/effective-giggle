import pandas as pd

# Step 1: Load the dataset
input_file = '/Users/omercankahveci/Desktop/dataset/household_data_1min_singleindex.csv'  # Replace with your file path
output_file = 'output_cleaned_usage_generation_aggregation.csv'  # Output file

# Load the dataset and suppress warnings
data = pd.read_csv(input_file, low_memory=False)

# Step 2: Standardize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
print("Standardized Columns:", data.columns)

# Step 3: Dynamically identify and map columns
column_mapping = {
    'usage': ['usage', 'grid_import', 'consumption'],
    'generation': ['generation', 'pv', 'solar', 'solar_generation'],
    'timestamp': ['timestamp', 'date', 'utc_timestamp', 'date_time']
}

mapped_columns = {}

for standard_name, keywords in column_mapping.items():
    for col in data.columns:
        if any(keyword in col for keyword in keywords):
            mapped_columns[standard_name] = col
            break

if 'usage' not in mapped_columns or 'generation' not in mapped_columns or 'timestamp' not in mapped_columns:
    raise ValueError("Could not map required columns: Usage, Generation, or Timestamp. Please verify dataset structure.")

usage_column = mapped_columns['usage']
generation_column = mapped_columns['generation']
datetime_column = mapped_columns['timestamp']

print(f"Mapped Usage column: {usage_column}")
print(f"Mapped Generation column: {generation_column}")
print(f"Mapped Timestamp column: {datetime_column}")

# Step 4: Convert the Timestamp column to a proper datetime format
data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
data = data.dropna(subset=[datetime_column])  # Drop rows where datetime parsing failed
data = data.sort_values(by=datetime_column).reset_index(drop=True)

# Step 5: Eliminate rows where Usage or Generation are missing (NaN)
data = data.dropna(subset=[usage_column, generation_column])
print(f"Remaining rows after eliminating missing Usage/Generation: {len(data)}")

# Step 6: Eliminate rows where both Usage and Generation are zero
data = data[~((data[usage_column] == 0) & (data[generation_column] == 0))]
print(f"Remaining rows after eliminating rows with both zeros: {len(data)}")

# Step 7: Compute actual usage and generation differences
usage_actual = data[usage_column].diff().fillna(0)
generation_actual = data[generation_column].diff().fillna(0)

# Correct negative differences by carrying forward values
usage_actual = usage_actual.where(usage_actual >= 0, data[usage_column])
generation_actual = generation_actual.where(generation_actual >= 0, data[generation_column])

# Step 8: Create the output DataFrame
output_data = pd.DataFrame({
    'Date & Time': data[datetime_column].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'Usage [kW]': usage_actual,
    'Generation [kW]': generation_actual
})

# Step 9: Save the output to a new CSV file
output_data.to_csv(output_file, index=False)
print(f"Output file saved to: {output_file}")
