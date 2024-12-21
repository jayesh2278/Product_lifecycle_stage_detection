# Imports
import pandas as pd
import numpy as np
import ruptures as rpt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load Sales Data
df = pd.read_csv('/content/sample_sales.csv')

# Select a Product by Grouping
product_name = input("Enter the product name you want to analyze: ")

df_product = df[df['product_name'] == product_name].copy()

# Ensure timestamp is properly converted
df_product['timestamp'] = pd.to_datetime(df_product['date'])
df_product = df_product[['timestamp', 'quantity']].rename(columns={'quantity': 'sales'})

# Add Smoothed Sales Column (Rolling Average)
df_product['smoothed_sales'] = df_product['sales'].rolling(window=5, min_periods=1).mean()

# Sort Data by Timestamp and Compute Cumulative Sales
df_product = df_product.sort_values(by='timestamp')
df_product['cumulative_sales'] = df_product['sales'].cumsum()

# Logistic Growth Function Definition
def logistic_growth(t, a, b, c):
    """Logistic growth function: c / (1 + a * exp(-b * t))"""
    return c / (1 + a * np.exp(-b * t))

# Normalize Time Variable (Days since Launch)
df_product['days'] = (df_product['timestamp'] - df_product['timestamp'].min()).dt.days

# Fit Logistic Curve to the Cumulative Sales Data
initial_guess = [1, 1, df_product['cumulative_sales'].max()]  # Initial guess for fitting
try:
    params, _ = curve_fit(
        logistic_growth,
        df_product['days'],
        df_product['cumulative_sales'],
        p0=initial_guess,
        maxfev=10000,
        bounds=(0, np.inf)
    )
except RuntimeError as e:
    print(f"Error fitting curve: {e}")
    params = [1, 1, df_product['cumulative_sales'].max()]  # Fallback values

# Extract Parameters
a, b, c = params

# Generate Predictions for Logistic Growth
df_product['logistic_prediction'] = logistic_growth(df_product['days'], a, b, c)

# Detect Change Points Using Ruptures Library
growth_rate = np.diff(df_product['logistic_prediction'])
algo = rpt.Pelt(model="rbf").fit(growth_rate)
change_points = algo.predict(pen=15)  # Detect change points

# Add Final Day as a Change Point if Missing
if change_points[-1] != df_product['days'].iloc[-1]:
    change_points.append(df_product['days'].iloc[-1])

# Ensure Change Points are Valid
if len(change_points) < 2:
    print("Not enough change points detected; skipping segmentation.")
    change_points = [0, df_product['days'].iloc[-1]]  # Default to full range

# Function to Determine Lifecycle Stages
def determine_stage(segment_data, is_first_segment=False):
    """Determine the stage of the product lifecycle based on sales trends."""
    if segment_data.empty:
        return 'Unknown'  # Fallback for empty segments

    sales_change = segment_data['sales'].diff().mean()  # Average change in sales
    recent_sales = segment_data['sales'].sum()

    if is_first_segment and sales_change > 0:
        return 'Introduction'
    elif sales_change > 0:
        return 'Growth'
    elif sales_change < 0 and recent_sales > 0.1 * df_product['cumulative_sales'].max():
        return 'Maturity'
    else:
        return 'Decline'

# Segment Data by Detected Change Points
segments = [(0, change_points[0])] + [
    (change_points[i], change_points[i + 1]) for i in range(len(change_points) - 1)
]
stages = []

# Assign Stages to Segments
for i, (start, end) in enumerate(segments):
    segment_data = df_product[(df_product['days'] >= start) & (df_product['days'] < end)]
    is_first_segment = (i == 0)  # First segment can be 'Introduction'
    stage = determine_stage(segment_data, is_first_segment)
    stages.append(stage)

# Colors for Stages
colors = {
    'Introduction': 'lightblue',
    'Growth': 'lightgreen',
    'Maturity': 'lightyellow',
    'Decline': 'lightcoral',
    'Unknown': 'grey'
}

# -------------------
# Plot 1: Smoothed Sales Trend with Lifecycle Stages
# -------------------
plt.figure(figsize=(10, 5))  # Adjust size for better visibility

# Plot original and smoothed sales
plt.plot(df_product['timestamp'], df_product['sales'], label='Original Sales', alpha=0.5)
plt.plot(df_product['timestamp'], df_product['smoothed_sales'], label='Smoothed Sales', linewidth=2)

# Overlay lifecycle stages using axvspan
for i, (start, end) in enumerate(segments):
    stage_label = stages[i]
    stage_color = colors.get(stage_label, 'grey')

    # Ensure segment dates exist
    start_date = df_product[df_product['days'] == start]['timestamp']
    end_date = df_product[df_product['days'] == end]['timestamp']

    if not start_date.empty and not end_date.empty:
        plt.axvspan(start_date.values[0], end_date.values[0], color=stage_color, alpha=0.3, label=f'{stage_label} Stage')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Original Sales with Lifecycle Stages')

# Handle legend to avoid duplicate entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.show()

# -------------------
# Plot 2: Cumulative Sales with Lifecycle Detection
# -------------------
plt.figure(figsize=(10, 5))

# Plot cumulative sales and logistic prediction
plt.plot(df_product['days'], df_product['cumulative_sales'], label='Actual Sales', linewidth=2)
plt.plot(df_product['days'], df_product['logistic_prediction'], label='Logistic Prediction', linestyle='--')

# Highlight lifecycle stages on the x-axis
for i, (start, end) in enumerate(segments):
    stage_label = stages[i]
    stage_color = colors.get(stage_label, 'grey')
    plt.axvspan(start, end, color=stage_color, alpha=0.3, label=f'{stage_label} Stage')

# Add vertical lines for change points
for cp in change_points[:-1]:
    plt.axvline(x=cp, color='red', linestyle='--', alpha=0.7)

# Display the current stage with a text box
current_stage = stages[-1]
plt.text(df_product['days'].iloc[-1] / 2, df_product['cumulative_sales'].max() * 0.8,
        f'Current Stage: {current_stage}', fontsize=12, ha='center',
        bbox=dict(facecolor='white', alpha=0.6))

# Add labels and title
plt.xlabel('Days since Launch')
plt.ylabel('Cumulative Sales')
plt.title(f'Lifecycle Detection for {product_name}')
plt.legend(loc='upper left')

plt.show()