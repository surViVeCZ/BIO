from PIL import Image
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import exposure
from skimage.color import rgb2gray
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
from pathlib import Path

# Function to load data
def load_data(metadata_path, data_directory, num_folders=5):
    metadata = pd.read_excel(metadata_path)
    
    images = []
    labels = []
    
    # Count total images in the dataset directory
    total_images = sum(1 for _ in Path(data_directory).rglob('*.png'))
    
    finger_folders = ["l_index", "l_little", "l_middle", "l_ring", "l_thumb", 
                      "r_index", "r_little", "r_middle", "r_ring", "r_thumb"]
    
    progress_bar = tqdm(total=total_images, desc='Loading images', dynamic_ncols=True)
    
    for index, row in metadata.iterrows():
        if index >= num_folders:
            break
            
        for finger_folder in finger_folders:
            folder_path = Path(data_directory) / str(row['id']) / finger_folder
            
            if folder_path.exists():
                image_files = list(folder_path.glob("*.png"))
                
                for img_path in image_files:
                    img = Image.open(img_path)
                    img = np.array(img)
                    
                    images.append(img)
                    
                    label = {
                        'id': row['id'],
                        'gender': row['gender'],
                        'age': row['age'],
                        'melanin': row['melanin'],
                        'cardiovascular_disease': row['cardiovascular disease'],
                        'smoker': row['smoker'],
                        'sport_hobby_with_fingers': row['sport/hobby with fingers'],
                        'alcohol_before_scan': row['alcohol before scan'],
                        'skin_disease': row['skin disease'],
                        'finger': finger_folder
                    }
                    labels.append(label)
                    progress_bar.update(1)
                
    progress_bar.close()
    return np.array(images), labels

# Function to preprocess a single image
def preprocess_image(image, target_size=(1024, 1024)):
    # Convert image to numpy array if not already
    img_array = np.array(image)

    # Check if the image has an alpha channel (transparency)
    if img_array.shape[-1] == 4:
        # Convert to RGB first, then grayscale
        grayscale_img = color.rgb2gray(img_array[..., :3])
    elif img_array.shape[-1] == 3:
        # Convert to grayscale
        grayscale_img = color.rgb2gray(img_array)
    else:
        # Image is already grayscale
        grayscale_img = img_array

    # Apply histogram equalization
    equalized_img = exposure.equalize_adapthist(grayscale_img, clip_limit=0.03)

    # Scale the image to range [-1, 1]
    scaled_img = (equalized_img * 2.0) - 1.0  # This scales the 0-1 range to -1 to 1

    # Convert scaled image back to a PIL image for further processing
    processed_image = Image.fromarray(np.uint8((scaled_img + 1) * 0.5 * 255), mode='L')

    original_size = processed_image.size
    ratio = min(target_size[0]/original_size[0], target_size[1]/original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

    # Resize the image using the calculated size
    processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)

    # Create a new image with the target size and a black background
    new_image = Image.new("L", target_size)

    # Paste the resized image onto the center of the new image
    new_image.paste(processed_image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))

    return new_image

# Function to preprocess labels
def preprocess_labels(labels):
    # Assuming all dicts have the same structure
    keys = labels[0].keys()
    label_data = {key: np.array([dic[key] for dic in labels]) for key in keys}

    # Initialize encoders
    one_hot_encoder = OneHotEncoder()
    label_encoder = LabelEncoder()

    # Process categorical data
    categorical_keys = ['gender', 'smoker', 'sport_hobby_with_fingers', 'alcohol_before_scan', 'skin_disease', 'finger']
    categorical_data = np.stack([label_data[key] for key in categorical_keys], axis=1)
    categorical_data_encoded = one_hot_encoder.fit_transform(categorical_data).toarray()

    # Process numerical/ordinal data
    numerical_keys = ['age']
    numerical_data = np.stack([label_data[key] for key in numerical_keys], axis=1)

    # Process data that may require custom encoding like 'melanin', 'cardiovascular_disease'
    # For example, melanin might be encoded based on the count of 'L' and 'R' occurrences, or any domain-specific method
    # cardiovascular_disease might be one-hot encoded if it has standard categories, or label-encoded if it's more like a ranking

    # For now, we'll just use the age directly and ignore other complex features for simplicity
    condition_input = np.concatenate([numerical_data, categorical_data_encoded], axis=1)

    return condition_input

# Main function to orchestrate the preprocessing
def main():
    metadata_path = "data_description.xlsx"
    data_directory = "dataset"
    num_folders = 5  # Adjust as needed

    images, labels = load_data(metadata_path, data_directory, num_folders)

    # Preprocess images
    preprocessed_images = np.stack([preprocess_image(image) for image in images[0:100]], axis=0)

    # Preprocess labels
    preprocessed_labels = preprocess_labels(labels[0:100])

    # Save preprocessed data to pickle files
    with open('preprocessed_images.pickle', 'wb') as f:
        pickle.dump(preprocessed_images, f)
        
    with open('preprocessed_labels.pickle', 'wb') as f:
        pickle.dump(preprocessed_labels, f)

    print("Preprocessing complete. Files saved.")

if __name__ == "__main__":
    main()
