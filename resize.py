import pickle
import cv2
import os
import numpy as np
import pandas as pd

# Load the dataset
df_skin = pd.read_csv('HAM10000_metadata.csv')

# Define lesion types and IDs
lesion_type_map = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
lesion_id_map = {k: v for v, k in enumerate(lesion_type_map.keys())}

# Add lesion type and id columns to the DataFrame
df_skin['lesion_type'] = df_skin['dx'].map(lesion_type_map)
df_skin['lesion_id'] = df_skin['dx'].map(lesion_id_map)

# Initialize image (X) and label (y) lists
X = []
y = []

# Function to augment images to avoid overfitting
def augment_images(image):
    return [cv2.rotate(image, rot) for rot in (cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180)] + [cv2.flip(image, flip) for flip in (0, 1)]

# Function to preprocess images from a given directory
def preprocess_images(directory, images_list):
    for idx, image_name in enumerate(images_list):
        # Extract image ID and read the image
        image_id = image_name.rstrip('.jpg')
        image_path = os.path.join(directory, image_name)
        image = cv2.resize(cv2.imread(image_path), (100, 100))
        
        # Append the original image
        X.append(image)
        
        # Retrieve the corresponding lesion id
        lesion_id = df_skin.loc[df_skin['image_id'] == image_id, 'lesion_id'].values[0]
        y.append(lesion_id)
        
        # Augment images for all classes except for 'nv' (0)
        if lesion_id != 0:
            for augmented_image in augment_images(image):
                X.append(augmented_image)
                y.append(lesion_id)
        
        # Optional progress printout
        if idx % 1000 == 0:
            print(f'Processed {idx}/{len(images_list)} images from {directory}')

# Process images from both directories
preprocess_images('HAM10000_images_part_1', os.listdir('HAM10000_images_part_1'))
preprocess_images('HAM10000_images_part_2', os.listdir('HAM10000_images_part_2'))

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Save numpy arrays and dataframe to disk in 'pickles' directory
os.makedirs('pickles', exist_ok=True)
pickle.dump(X, open('pickles/X.pkl', 'wb'))
pickle.dump(y, open('pickles/y.pkl', 'wb'))
pickle.dump(df_skin, open('pickles/df_skin.pickle', 'wb'))
