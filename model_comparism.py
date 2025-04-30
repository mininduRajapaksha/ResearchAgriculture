import numpy as np
import matplotlib.pyplot as plt

#test accuracy
test_accuracy = [73.52, 82.10, 99.48]

#model names
model_names = ['ResNet50', 'EfficientNetB0', 'Custom CNN']

# Create a bar plot
plt.figure(figsize=(6, 5))
plt.bar(model_names, test_accuracy, color=['lightgreen'])

#labeling the bars
plt.ylim(0, 100)
plt.xlabel('Model')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy of Different Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()