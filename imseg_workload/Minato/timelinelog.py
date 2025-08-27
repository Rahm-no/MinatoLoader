import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

# Open the log file
with open('/dl-bench/rnouaj/mlcomns_imseg_1/output/unet3d.log', 'r') as file:
    log_data = file.readlines()

# Define the specific events you want to plot
specific_events = ["init_start", "init_stop", "Hey it is Rahma tracing", "run_start"]

# Initialize lists to store event types, start timestamps, and durations
event_types = []
start_timestamps = []
durations = []

# Iterate over each log entry
for line in log_data:
    # Convert log entry string to dictionary
    log_entry = json.loads(line.split(':::MLLOG ')[1])
    
    # Access the value of the "key" field
    key_value = log_entry['key']
    timestamp = log_entry['time_ms']

    # Check if the event type is one of the specific events
    if key_value in specific_events:
        event_types.append(key_value)
        if key_value == "init_start":
            start_timestamps.append(timestamp)
        elif key_value == "init_stop":
            # Calculate the duration between "init_start" and "init_stop"
            if len(start_timestamps) > 0:
                start_timestamp = start_timestamps.pop()  # Get the last recorded "init_start" timestamp
                duration = timestamp - start_timestamp
                durations.append(duration)

# Convert start timestamps to datetime objects
start_dates = [datetime.fromtimestamp(ts / 1000.0).strftime("%H:%M") for ts in start_timestamps]  # Convert to normal time format
print(start_dates)
# Convert durations to timedelta objects
durations = [datetime.fromtimestamp(duration / 1000.0).strftime("%H:%M") for duration in durations]  # Convert to normal time format

# Make sure the number of start dates and durations match
num_events = min(len(start_dates), len(durations))
start_dates = start_dates[:num_events]
durations = durations[:num_events]

# Plotting the timeline
plt.figure(figsize=(12, 6))
plt.barh(range(len(start_dates)), [datetime.strptime(duration, "%H:%M") - datetime.strptime("00:00", "%H:%M") for duration in durations], left=[datetime.strptime(start_date, "%H:%M") - datetime.strptime("00:00", "%H:%M") for start_date in start_dates], height=0.5)
plt.yticks(range(len(start_dates)), event_types[:num_events])
plt.xlabel('Time')

# Create the output directory if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save the plot as an image
output_path = os.path.join(output_dir, 'timeline.png')
plt.savefig(output_path)
