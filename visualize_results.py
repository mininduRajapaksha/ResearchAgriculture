import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

def visualize_model_performance():
    # Load the model
    model = load_model('banana_quality_model.h5')
    
    # Load test dataset
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_dir = 'D:/Research project/Datasets/Banana Dataset/Test'
    
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get predictions
    predictions = model.predict(test_set)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_set.classes
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = list(test_set.class_indices.keys())
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    visualize_model_performance()