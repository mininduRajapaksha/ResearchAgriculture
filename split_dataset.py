import matplotlib.pyplot as plt

# Define the labels and their corresponding percentages
labels = ['Train', 'Validation', 'Test']
percentages = [75, 15, 15]  # Expected distribution in percentage

# Create the bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, percentages, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Add text labels above each bar indicating the percentage value
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold')

plt.xlabel('Dataset Splits')
plt.ylabel('Percentage (%)')
plt.title('Dataset Image Distribution (Total Images: 9750)')
plt.ylim(0, 100)  # Set y-axis limit to 100%
plt.show()
