import json
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam


# Load the saved datasets
def load_from_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

X = load_from_pickle('pickles/X.pkl')
y = load_from_pickle('pickles/y.pkl')
df_skin = load_from_pickle('pickles/df_skin.pickle')

# Convert labels to one-hot encoding
y_one_hot = to_categorical(y, num_classes=7)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_one_hot, 
    test_size=0.3,
    random_state=1,
    stratify=y
)

# Output the shapes of training and test data
print('Train dataset shape:', X_train.shape)
print('Test dataset shape:', X_test.shape)

# Compute class weights for imbalanced classes
# compute weights for the loss function, because the problem is unbalanced
class_weights = np.around(compute_class_weight(class_weight='balanced',classes=np.unique(np.array(df_skin['lesion_id'])),y=y),2)
class_weights = dict(zip(np.unique(np.array(df_skin['lesion_id'])),class_weights))
print('class_weights ')
print(class_weights)


# Define the Neural Network architecture
model = Sequential([
    # 1st convolutional layer
    Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(100,100,3)),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    # 2nd convolutional layer
    Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPool2D(pool_size=(3,3), strides=(2,2)),
    
    Flatten(),
    
    # 6th, Dense layer
    Dense(4096, activation='relu'),
    Dropout(0.5),
    
    # 8th output layer
    Dense(7, activation='softmax')
])


checkpoint_path = "model_checkpoint/cp.weights.h5" 
checkpoint_dir = os.path.dirname(checkpoint_path)

# Number of epochs to save checkpoints at
x = 50
batch_size = 32
# Calculate the number of batches in one epoch
number_of_batches_per_epoch = len(X_train) 
if len(X_train) % batch_size > 0:
    number_of_batches_per_epoch += 1  # Accounting for the last batch which may be smaller than batch_size

# Calculate total batches for 'x' epochs. 
save_freq_in_batches = number_of_batches_per_epoch * x
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=False, verbose=1, save_freq=save_freq_in_batches)

from tensorflow.keras.callbacks import EarlyStopping
# Define early stopping callback
early_stop = EarlyStopping(monitor='val_accuracy', patience=100)
# #We compile our model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(X_train)

history = model.fit(datagen.flow(X_train,y_train), epochs=200, validation_data=(X_test, y_test), batch_size=batch_size, class_weight=class_weights, callbacks=[cp_callback,early_stop])

from keras.models import load_model,save_model

# Save the model using Keras's built-in save method
model.save('model.h5')

save_model(model, "model_keras.keras", overwrite=True)

# Load the model from file
loaded_model = load_model('model.h5')
# Save the training history to a file
with open('history.json', 'w') as f:
    json.dump(history.history, f)
# Print loaded model summary
loaded_model.summary()

scores = loaded_model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


import os
import matplotlib.pyplot as plt

# Ensure the directory exists
figs_dir = 'figs'
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

# We evaluate our model
fig, axis = plt.subplots(1, 2, figsize=(20, 7))

axis[0].plot(history.history['accuracy'])
axis[0].plot(history.history['val_accuracy'])
axis[0].set_title('model accuracy')
axis[0].set_ylabel('accuracy')
axis[0].set_xlabel('epoch')
axis[0].legend(['train', 'val'], loc='upper left')

axis[1].plot(history.history['loss'])
axis[1].plot(history.history['val_loss'])
axis[1].set_title('model loss')
axis[1].set_ylabel('loss')
axis[1].set_xlabel('epoch')
axis[1].legend(['train', 'val'], loc='upper left')

# Save the figure
fig_path = os.path.join(figs_dir, 'training_metrics.png')
plt.savefig(fig_path)

# Close the figure to release memory
plt.close(fig)