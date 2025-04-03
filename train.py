# import os
# import json
# from data_loader import load_data
# from model import create_model

# def train_model(epochs=10):
#     # Set paths
#     train_dir = 'D:/Research project/Datasets/Banana Dataset/Train'
#     test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
#     valid_dir = 'D:/Research project/Datasets/Banana Dataset/Valid'

#     # Load data
#     train_set, val_set, test_set = load_data(train_dir, valid_dir, test_dir)

#     # Create and train model
#     model = create_model()
#     model.summary()

#     # Train the model
#     history = model.fit(
#         train_set,
#         validation_data=val_set,
#         epochs=epochs
#     )

#     # Save training history
#     with open('history.json', 'w') as f:
#         json.dump(history.history, f)

#     # Evaluate model
#     test_loss, test_accuracy = model.evaluate(test_set)
#     print("Test Accuracy:", test_accuracy)

#     return history, model

# if __name__ == "__main__":
#     os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#     history, model = train_model()