import pandas as pd
import numpy as np

# Step 1: Define Input and Output Files
input_file = '/Users/omercankahveci/Desktop/dataset/HomeF-meter3_2015.csv'  # Replace with your file
output_file = 'output_cleaned_usage_generation.csv'

# Step 2: Load Dataset and Standardize Column Names
data = pd.read_csv(input_file, low_memory=False)

data.columns = (data.columns.str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('[^a-z0-9_]', '', regex=True))
print("Standardized Columns:", data.columns)

# Step 3: Dynamically Map Required Columns
column_mapping = {
    'usage': ['usage', 'grid_import', 'consumption'],
    'generation': ['generation', 'pv', 'solar', 'solar_generation'],
    'timestamp': ['timestamp', 'date', 'utc_timestamp', 'date_time']
}

mapped_columns = {}
for key, keywords in column_mapping.items():
    for col in data.columns:
        if any(keyword in col for keyword in keywords):
            mapped_columns[key] = col
            break

if 'usage' not in mapped_columns or 'generation' not in mapped_columns or 'timestamp' not in mapped_columns:
    raise ValueError("Could not map required columns: Usage, Generation, or Timestamp.")

usage_column = mapped_columns['usage']
generation_column = mapped_columns['generation']
datetime_column = mapped_columns['timestamp']

print(f"Mapped Usage column: {usage_column}")
print(f"Mapped Generation column: {generation_column}")
print(f"Mapped Timestamp column: {datetime_column}")

# Step 4: Convert Timestamp to Datetime Format
data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
data = data.dropna(subset=[datetime_column])
data = data.sort_values(by=datetime_column).reset_index(drop=True)

# Step 5: Calculate Incremental Usage and Generation
incremental_usage = data[usage_column].diff().fillna(data[usage_column])
incremental_generation = data[generation_column].diff().fillna(data[generation_column])

# Replace negative increments with NaN and forward-fill gaps
incremental_usage[incremental_usage < 0] = np.nan
incremental_generation[incremental_generation < 0] = np.nan

incremental_usage = incremental_usage.fillna(method='bfill').fillna(0)
incremental_generation = incremental_generation.fillna(method='bfill').fillna(0)

# Step 6: Create Cleaned Output DataFrame
output_data = pd.DataFrame({
    'Date & Time': data[datetime_column].dt.strftime('%Y-%m-%d %H:%M:%S'),
    'Usage [kW]': incremental_usage,
    'Generation [kW]': incremental_generation
})

# Step 7: Eliminate Rows Where Both Usage and Generation are Zero
output_data = output_data[~((output_data['Usage [kW]'] == 0) & (output_data['Generation [kW]'] == 0))]

# Step 8: Save the Cleaned Data
output_data.to_csv(output_file, index=False)
print(f"Output file saved to: {output_file}")
