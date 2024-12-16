import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Step 1: Generate Random Historical Energy Consumption Data
np.random.seed(42)  # For reproducibility

# Settings for the data generation
start_date = datetime(2023, 1, 1, 0, 0)  # Starting timestamp
num_hours = 100  # Total number of hours to simulate
timestamps = pd.date_range(start=start_date, periods=num_hours, freq='H')  # Hourly data

# Generate random values for energy consumption and generation in kWh
energy_generation = np.random.uniform(450, 600, size=num_hours)  # Between 450 and 600 kWh
energy_usage = np.random.uniform(400, 550, size=num_hours)       # Between 400 and 550 kWh

# Create a DataFrame
data = pd.DataFrame({
    "Date & Time": timestamps,
    "Generation [kW]": energy_generation,
    "Usage [kW]": energy_usage
})

# Step 2: Save Raw Data to CSV
file_path = "sample_historical_energy_data.csv"  # Adjust the path as needed
data.to_csv(file_path, index=False)
print(f"Sample historical energy data saved to: {file_path}")

# Step 3: Standardize Column Names
data.columns = (data.columns.str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('[^a-z0-9_]', '', regex=True))
print("Standardized Columns:", data.columns)

# Step 4: Dynamically Identify and Map Columns
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
    raise ValueError("Required columns (Usage, Generation, Timestamp) could not be mapped!")

usage_column = mapped_columns['usage']
generation_column = mapped_columns['generation']
datetime_column = mapped_columns['timestamp']

print(f"Mapped Usage column: {usage_column}")
print(f"Mapped Generation column: {generation_column}")
print(f"Mapped Timestamp column: {datetime_column}")

# Step 5: Convert the Timestamp Column to Proper Datetime Format
data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
data = data.dropna(subset=[datetime_column])

# Step 6: Filter Data to Include Only the First 2 Years
start_date = data[datetime_column].min()
end_date = start_date + pd.DateOffset(years=2)

filtered_data = data[(data[datetime_column] >= start_date) & (data[datetime_column] < end_date)]
print(f"Filtered data from {start_date} to {end_date}. Remaining rows: {len(filtered_data)}")

# Step 7: Eliminate Rows Where Usage or Generation are Missing (NaN)
filtered_data = filtered_data.dropna(subset=[usage_column, generation_column])
print(f"Remaining rows after eliminating missing Usage/Generation: {len(filtered_data)}")

# Step 8: Eliminate Rows Where Both Usage and Generation are Zero
filtered_data = filtered_data[~((filtered_data[usage_column] == 0) & (filtered_data[generation_column] == 0))]
print(f"Remaining rows after eliminating rows with both zeros: {len(filtered_data)}")

# Step 9: Save Cleaned Data to a New CSV File
output_file = "output_cleaned_usage_generation_with_mapping.csv"
filtered_data.reset_index(drop=True).to_csv(output_file, index=False)

print(f"Cleaned and filtered data saved to: {output_file}")
