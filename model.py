# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# def create_model(input_shape=(416, 416, 3), num_classes=4):
#     model = Sequential([
#         # First convolutional block
#         Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         MaxPooling2D((2, 2)),
#         BatchNormalization(),

#         # Second convolutional block
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         BatchNormalization(),

#         # Third convolutional block
#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         BatchNormalization(),

#         # Flatten and dense layers
#         Flatten(),
#         Dropout(0.5),
#         Dense(128, activation='relu'),
#         Dense(num_classes, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     return model