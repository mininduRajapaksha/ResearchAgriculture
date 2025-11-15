# import os
# from keras.preprocessing.image import ImageDataGenerator

# def load_data(train_dir, valid_dir, test_dir, img_size=(416, 416), batch_size=32):
#     # Data augmentation for training
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True
#     )

#     # Only rescaling for validation and test
#     val_datagen = ImageDataGenerator(rescale=1./255)
#     test_datagen = ImageDataGenerator(rescale=1./255)

#     # Load datasets
#     train_set = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     val_set = val_datagen.flow_from_directory(
#         valid_dir,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     test_set = test_datagen.flow_from_directory(
#         test_dir,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     return train_set, val_set, test_set