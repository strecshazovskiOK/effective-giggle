import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time

# Set up a local cache directory for Hugging Face models
cache_dir = "./huggingface_cache/"

# Load the Hugging Face GPT-2 model and tokenizer
print("Loading Hugging Face model...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained("gpt2", truncation=True, cache_dir=cache_dir)
model_hf = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=cache_dir)
generator = pipeline(
    "text-generation",
    model=model_hf,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)
print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

# Load the trained LSTM energy forecasting model
print("Loading LSTM forecasting model...")
lstm_model = load_model('lstm_multitarget_model.keras')
print("LSTM model loaded successfully.")

# Step 1: Load and Preprocess Data
data_file = "output_cleaned_usage_generation.without.aggregation.csv"
data = pd.read_csv(data_file)

# Step 2: Standardize column names
data.columns = (data.columns.str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('[^a-z0-9_]', '', regex=True))
print("Standardized Columns:", data.columns)

# Step 3: Dynamically map columns
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

# Step 4: Convert timestamp and clean data
data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')
data = data.dropna(subset=[datetime_column])  # Drop invalid timestamps
data = data.dropna(subset=[usage_column, generation_column])  # Drop rows with NaN in key columns

# Remove rows where both usage and generation are zero
data = data[~((data[usage_column] == 0) & (data[generation_column] == 0))]
print(f"Remaining rows after cleaning: {len(data)}")

# Step 5: Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[[usage_column, generation_column]])

# Define time steps for LSTM input
time_steps = 60

# Step 6: Create sequences
def create_sequences(data, time_steps=60):
    sequences, targets = [], []
    for i in range(len(data) - time_steps):
        sequences.append(data[i: i + time_steps])
        targets.append(data[i + time_steps])
    return np.array(sequences), np.array(targets)

# Create input sequences
X_input, _ = create_sequences(data_scaled, time_steps)

# Check if the last input is valid (not all zeros)
if np.all(X_input[-1] == 0):
    raise ValueError("Error: Historical input data contains only zeros. Please check the input dataset.")

# Step 7: Make prediction
print("Making predictions...")
scaled_prediction = lstm_model.predict(X_input[-1].reshape(1, time_steps, 2))  # Last valid sequence
prediction = scaler.inverse_transform(scaled_prediction)[0]

# Step 8: Generate insights with Hugging Face GPT-2
latest_usage = data[usage_column].iloc[-1]
latest_generation = data[generation_column].iloc[-1]

prompt = f"""
The user's energy usage and PV generation for the last {time_steps} hours are analyzed.
Latest observations:
- Energy Usage: {latest_usage:.2f} kW
- PV Generation: {latest_generation:.2f} kW
- Predicted Energy Usage for the next hour: {prediction[0]:.2f} kW
- Predicted PV Generation for the next hour: {prediction[1]:.2f} kW.

Provide a concise and insightful summary for the user.
"""

print("Generating insights...")
response = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

# Step 9: Output the generated response
print("\nEnergy Assistant's Response:")
print(response)
