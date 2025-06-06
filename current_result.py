import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

# Load the data
df = pd.read_csv(
    'detection_results/detections.csv',
    header=None,
    names=['timestamp', 'frame', 'class_and_name', 'confidence', 'resolution']
)

# Split 'class_and_name' into 'detection_id' and 'detection_name'
df[['detection_id', 'detection_name']] = df['class_and_name'].str.split(pat=' ', n=1, expand=True)
df.drop(columns=['class_and_name'], inplace=True)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['time_minute'] = df['timestamp'].dt.floor('min')

# Extract numeric confidence
df['confidence'] = df['confidence'].str.replace('%', '').astype(float)

# Plot 1: Detection Frequency Over Time
plt.figure(figsize=(12, 6))
top_signs = df['detection_name'].value_counts().nlargest(5).index
df_top = df[df['detection_name'].isin(top_signs)]

for sign in top_signs:
    sign_data = df_top[df_top['detection_name'] == sign]
    sign_counts = sign_data.resample('min', on='timestamp').size()
    plt.plot(sign_counts.index, sign_counts.values, label=sign)

plt.title('Detection Frequency Over Time (Top 5 Signs)')
plt.xlabel('Time')
plt.ylabel('Detections per minute')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Plot 2: Confidence Distribution
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_top, x='detection_name', y='confidence')
plt.title('Confidence Distribution by Sign Type')
plt.xlabel('Sign Type')
plt.ylabel('Confidence (%)')
plt.xticks(rotation=45)
plt.grid(True)

# Plot 3: Top Detected Signs
plt.figure(figsize=(12, 6))
sign_counts = df['detection_name'].value_counts().nlargest(10)
sign_counts.plot(kind='bar')
plt.title('Top 10 Most Detected Signs')
plt.xlabel('Sign Type')
plt.ylabel('Detection Count')
plt.xticks(rotation=45)
plt.grid(True)

# Plot 4: Confidence Over Time for Key Signs
plt.figure(figsize=(12, 6))
for sign in top_signs:
    sign_data = df_top[df_top['detection_name'] == sign]
    plt.scatter(sign_data['timestamp'], sign_data['confidence'], label=sign, alpha=0.5)

plt.title('Confidence Levels Over Time (Top 5 Signs)')
plt.xlabel('Time')
plt.ylabel('Confidence (%)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()