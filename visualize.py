# import json
# import matplotlib.pyplot as plt

# def plot_training_history(history_path='history.json'):
#     # Load the training history
#     with open(history_path, 'r') as f:
#         history = json.load(f)

# #     # Plot Training and Validation Accuracy
#     plt.figure(figsize=(8, 6))
#     plt.plot(history['accuracy'], label='Training Accuracy')
#     plt.plot(history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig('accuracy.png')
#     plt.show()

# #     # Plot Training and Validation Loss
#     plt.figure(figsize=(8, 6))
#     plt.plot(history['loss'], label='Training Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('loss.png')
#     plt.show()

# if __name__ == "__main__":
#     plot_training_history()