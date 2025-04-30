import os
import matplotlib.pyplot as plt

def count_images_in_directory(root_dir, extensions=('.jpg', '.jpeg', '.png')):
    """
    Counts the number of image files (with given extensions) in a directory (recursively).
    """
    count = 0
    for root, dirs, files in os.walk(root_dir):
        count += len([f for f in files if f.lower().endswith(extensions)])
    return count

# Define dataset paths
train_dir = 'D:/Research project/Datasets/Split70_15_15/Train'
valid_dir = 'D:/Research project/Datasets/Split70_15_15/Valid'
test_dir  = 'D:/Research project/Datasets/Split70_15_15/Test'

# Count the images in each directory
num_train = count_images_in_directory(train_dir)
num_valid = count_images_in_directory(valid_dir)
num_test  = count_images_in_directory(test_dir)

# Calculate total images
total_images = num_train + num_valid + num_test

print("Total images:", total_images)
print("Train images:", num_train)
print("Validation images:", num_valid)
print("Test images:", num_test)

# Calculate the percentage distribution
if total_images > 0:
    percent_train = (num_train / total_images) * 100
    percent_valid = (num_valid / total_images) * 100
    percent_test  = (num_test / total_images) * 100
else:
    percent_train = percent_valid = percent_test = 0

# Prepare data for the bar chart
labels = ['Train', 'Validation', 'Test']
percentages = [percent_train, percent_valid, percent_test]

# Create a bar chart to display the distribution
plt.figure(figsize=(8, 6))
bars = plt.bar(labels, percentages, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

# Annotate each bar with its percentage value
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.xlabel("Dataset Split")
plt.ylabel("Percentage (%)")
plt.title(f"Image Distribution (Total Images: {total_images})")
plt.ylim(0, 100)  # Ensure the y-axis goes from 0 to 100%
plt.show()
