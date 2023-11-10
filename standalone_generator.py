import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


# set tensorflow keras to silence all 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Generator
class Generator():
    
    def __init__(self, z_dim, condition_shape, img_shape):
        # z_dim: Dimension of the latent space (noise vector)
        # condition_shape: Shape of the condition input (e.g., parameters like age, blood pressure)
        # img_shape: Shape of the generated image output

        self.z_dim = z_dim
        self.condition_shape = condition_shape
        self.img_shape = img_shape

        self.model = self.build_generator()

    def build_generator(self):
        # Noise input (z_dim-dimensional latent vector)
        z_input = layers.Input(shape=(self.z_dim,))
        # Conditional input (additional information you want to condition the generation on)
        condition_input = layers.Input(shape=(self.condition_shape,))
        
        # Combine noise and condition via concatenation
        x = layers.Concatenate()([z_input, condition_input])
        
        # Fully connected layer that takes the combined input
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)  # Helps to stabilize training
        
        # Up-sampling: Increasing the dimensionality to get to the correct image size
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)  # Further stabilization of training
        # The final layer has a size of the product of the image dimensions (width * height * channels)
        x = layers.Dense(np.prod(self.img_shape), activation='tanh')(x)  # 'tanh' activation is common for GANs
        # Reshape the output to the size of the image
        img = layers.Reshape(self.img_shape)(x)
        
        # The generator model takes noise and condition as input and outputs an image
        generator = models.Model([z_input, condition_input], img, name='generator')

        return generator



# limit number of images for testing 
SAMPLE_LIMIT = 1000


with open('preprocessed_images.pickle', 'rb') as f:
    preprocessed_images = pickle.load(f)

with open('preprocessed_labels.pickle', 'rb') as f:
    preprocessed_labels = pickle.load(f)
    
    

preprocessed_images = preprocessed_images[0:SAMPLE_LIMIT]
preprocessed_labels = preprocessed_labels[0:SAMPLE_LIMIT]


# shuffle images and labels in the same order
import sklearn
from  sklearn.utils import shuffle
preprocessed_images, preprocessed_labels = shuffle(preprocessed_images, preprocessed_labels, random_state=0)

PRINT = False

if PRINT:
    # print 25 random images in tight 5x5 grid
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(preprocessed_images[i].reshape(1024, 1024), cmap='gray')
        plt.axis('off')
        # tight layout
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    input("Continue?")




z_dim = 10  # This should match the z_dim you used to initialize your Generator



generator = Generator(z_dim, preprocessed_labels.shape[1], (1024 ,1024, 1))  # Adjust the image shape accordingly



iterations = 5

# Define the optimizer and loss function for the generator
generator_optimizer = Adam(learning_rate=0.02)
mse_loss = MeanSquaredError()

# Compile the generator model
generator.model.compile(optimizer=generator_optimizer, loss=mse_loss)


### TO BE PARALELIZED ###

from concurrent.futures import ThreadPoolExecutor, as_completed

output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)



def loss_func(real_image, synthetic_image):
    return mse_loss(real_image, synthetic_image)

counter = 0
### TO BE PARALELIZED ###
# Training loop
for epoch in range(iterations):
    for i in range(len(preprocessed_images)):
        real_image = preprocessed_images[i:i+1]  # Select a single image slice
        label = preprocessed_labels[i:i+1]       # Select the corresponding label slice

        # Generate noise
        noise = np.random.normal(-1, 1, (1, z_dim))
        
        # Generate a synthetic image
        synthetic_image = generator.model.predict([noise, label])
        
        # Scale the synthetic image to match the real image range
        # if counter mod 10 == 0:
        synthetic_image = (synthetic_image + 1) / 2
        
        # initialize synthetic image to real image
        synthetic_image = real_image

        
        # Compute the loss between the synthetic and real image
        loss = generator.model.train_on_batch([noise, label], real_image)
        
        if counter % 1 == 0:
        # save synthetic image and real image to file synth.jpg and real.jpg
            plt.imsave(os.path.join(output_dir, 'synth.jpg'), synthetic_image.reshape(1024, 1024), cmap='gray')
            plt.imsave(os.path.join(output_dir, 'real.jpg'), real_image.reshape(1024, 1024), cmap='gray')
            print(loss_func(real_image, synthetic_image))
            print(f'Epoch: {epoch}, Loss: {loss}')
            input('Press Enter to continue...')
        
        # Print the loss
        
        counter += 1


### END OF PARALELIZED ###

# Display a synthetic image
plt.imshow(synthetic_image[0].reshape(1024, 1024), cmap='gray')
plt.axis('off')
plt.show()


# save 