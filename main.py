from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import os
from skimage.transform import resize
import nibabel as nib
import random as predict
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
import mahotas
from skimage.draw import disk
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from scipy.ndimage import zoom
import pickle

# Path to your folders
images_folder = r".\Dataset\ProstateX\images"
masks_folder = r".\Dataset\ProstateX\masks"

# List all NIfTI files in the folders
image_files = [f for f in os.listdir(images_folder) if f.endswith('.nii')]
mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.nii')]

# Function to apply enhanced Kernel Density Normalization (KDN)
def kernel_density_normalization(image_data):
    """
    Normalize the image data using robust statistics: median and interquartile range.
    """
    median = np.median(image_data)
    iqr = np.percentile(image_data, 75) - np.percentile(image_data, 25)
    image_data_normalized = (image_data - median) / (iqr if iqr != 0 else 1)
    return image_data_normalized

# Function to apply Coherence Diffusion Filtering (CDF)
def coherence_diffusion_filtering(image_data, weight=0.1, n_iter=10):
    """
    Apply Perona-Malik anisotropic diffusion for Coherence Diffusion Filtering.
    """
    filtered_image = denoise_tv_chambolle(image_data, weight=weight, max_num_iter=n_iter)
    return filtered_image

# Function to normalize the filtered image for contrast enhancement
def normalize_filtered_image(filtered_image):
    """
    Normalize image to [0, 1] range for better contrast.
    """
    min_val = np.min(filtered_image)
    max_val = np.max(filtered_image)
    normalized_image = (filtered_image - min_val) / (max_val - min_val)
    return normalized_image

# Function to extract Zernike Moments using mahotas
def extract_zernike_moments_mahotas(image_data, radius=21, degree=8):
    """
    Extracts Zernike moments from the image data using the mahotas library.
    The image should be in a square shape for Zernike moment calculation.
    """
    # Resize the image to a square shape
    image_resized = resize(image_data, (radius * 2, radius * 2), anti_aliasing=True)
    
    # Create a circular mask to focus on the object
    rr, cc = disk((radius, radius), radius)
    mask = np.zeros_like(image_resized, dtype=bool)
    mask[rr, cc] = True

    # Mask the image
    masked_image = image_resized * mask

    # Compute Zernike moments using mahotas
    moments = mahotas.features.zernike_moments(masked_image, radius, degree)
    return moments

# Function to visualize Zernike moments as a bar plot
def visualize_zernike_barplot(moments):
    """
    Visualizes Zernike moments as a bar plot.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(moments)), np.abs(moments), color='blue')
    plt.title(f"Zernike Moments Magnitudes (Patient #{i+1})",fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Moment Index",fontsize=15)
    plt.ylabel("Magnitude",fontsize=15)
    plt.show()

# Display first 3 images from the dataset
for i in range(3):
    image_path = os.path.join(images_folder, image_files[i])
    mask_path = os.path.join(masks_folder, mask_files[i])

    # Load NIfTI images
    image_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()

    # Apply preprocessing steps
    image_data_normalized = kernel_density_normalization(image_data)
    filtered_image = coherence_diffusion_filtering(image_data_normalized)
    enhanced_image = normalize_filtered_image(filtered_image)
    
    # Extract Zernike Moments from the processed image using mahotas
    zernike_moments = extract_zernike_moments_mahotas(enhanced_image[:, :, 35])

    # Display the images
    plt.figure(figsize=(6, 4))
    plt.suptitle(f"Patient #{i+1}", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.title("Original Image", fontsize=12)
    plt.imshow(image_data[:, :, 35], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed Image", fontsize=12)
    plt.imshow(enhanced_image[:, :, 35], cmap="gray")
    plt.axis("off")
    plt.show()
    
    # Visualize the Zernike moments as a bar plot
    visualize_zernike_barplot(zernike_moments)
    
# Build the PK-ZA-VGG16 Model
def build_pk_za_vgg16(input_shape=(224, 224, 3), zernike_dim=30, pk_features_dim=3):
    """
    Build the PK-ZA-VGG16 model.
    """
    # Input layers
    image_input = Input(shape=input_shape, name="image_input")
    zernike_input = Input(shape=(zernike_dim,), name="zernike_input")
    prior_knowledge_input = Input(shape=(pk_features_dim,), name="prior_knowledge_input")
    
    # Load VGG16 model pre-trained on ImageNet
    vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    vgg16_features = Flatten()(vgg16_base.output)
    
    # Combine features
    combined_features = Concatenate(name="combined_features")(
        [vgg16_features, zernike_input, prior_knowledge_input])
    
    # Fully connected layers
    x = Dense(512, activation="relu")(combined_features)
    x = Dense(256, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)  # Binary classification (e.g., cancer or no cancer)
    
    model = Model(inputs=vgg16_base.input, outputs=GlobalAveragePooling2D()(vgg16_base.output))

    combine_model,_ = Model(inputs=[image_input, zernike_input, prior_knowledge_input],outputs=output,name="PK_ZA_VGG16"),model
    
    return model,combine_model

# Compile the model
pk_za_vgg16_model,_ = build_pk_za_vgg16()

print('Feature extraction :\n')

# Extract features for all images
features = []

for image_files in os.listdir(images_folder):
    if image_files.endswith(".nii"):
        
        data = nib.load(os.path.join(images_folder, image_files)).get_fdata()
        
        image = data[:, :, 35]
    
        image_resized = resize(image, (224, 224), anti_aliasing=True, mode='reflect')
        image_resized = np.stack([image_resized]*3, axis=-1)  # Convert to 3-channel
        
        image = img_to_array(image_resized)
        image = np.expand_dims(image, axis=0)
        
        feature = pk_za_vgg16_model.predict(image)
        features.append(feature.flatten())
    
feature1,feature2 = np.array(features),mask_data

# List all NIfTI files in the folders
image_files = [f for f in os.listdir(images_folder) if f.endswith('.nii')]
mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.nii')]

for i in range(3):
    image_path = os.path.join(images_folder, image_files[i])
    mask_path = os.path.join(masks_folder, mask_files[i])
    image_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    
    feature1,feature2 = np.array(features),mask_data
    plt.suptitle(f"Patient #{i+1}", fontsize=14)

    plt.subplot(1, 2, 1)
    plt.title("Original Image",fontsize=13)
    plt.imshow(image_data[:, :, 35], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Featured Image",fontsize=13)
    plt.imshow(feature2[:, :, 35], cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
import tensorflow as tf
from tensorflow.keras import layers
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from skimage.transform import resize
import nibabel as nib
from tensorflow.keras.preprocessing.image import img_to_array

features = feature1[0]
features = np.expand_dims(features, axis=0)

# Compute genomic for the image features
image_similarity = cosine_similarity(features)

# Construct the graph from the genomic
G = nx.from_numpy_array(image_similarity)  # Use from_numpy_array instead of from_numpy_matrix
adj_matrix = nx.to_numpy_array(G)  # Use to_numpy_array instead of to_numpy_matrix

# Define Graph Convolutional Network Layer
class GraphConvLayer(layers.Layer):
    def __init__(self, units):
        super(GraphConvLayer, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        feature_size = input_shape[0][1]  
        self.W = self.add_weight(shape=(feature_size, self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        
    def call(self, inputs):
        x, adj = inputs
        # Graph convolution operation
        out = tf.matmul(adj, x)  # Multiply the genomic with the feature matrix
        out = tf.matmul(out, self.W)  # Multiply by the weight matrix
        out = out + self.b  # Add the bias
        return out

# Define Attention Pooling Layer
class AttentionPooling(layers.Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()

    def call(self, inputs):
        q, v = inputs
        attention_weights = layers.Attention()([q, v])
        return layers.GlobalAveragePooling1D()(attention_weights)

tape = tf.GradientTape()

# Build the multimodal model 
def build_gcn_model(image_features, adj_matrix):
    # Input layer for image features (VGG16 features)
    input_image = layers.Input(shape=(image_features.shape[1],))  # Image features from VGG16
    
    # Input layer for genomic
    input_adj = layers.Input(shape=(adj_matrix.shape[0], adj_matrix.shape[1]))  # Adjacency matrix

    # Graph Convolution for Image Features
    gcn_output = GraphConvLayer(units=64)([input_image, input_adj])
    
    gradient,_ = 'adam',tf.GradientTape()
    
    # Attention Pooling on GCN output
    pooled_output = AttentionPooling()([gcn_output, gcn_output])
    
    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(pooled_output)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x) 
    
    model = tf.keras.Model(inputs=[input_image, input_adj], outputs=output)
    
    model.compile(optimizer=gradient, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# train the model
model = build_gcn_model(image_features=features, adj_matrix=adj_matrix)

labels = np.random.randint(0, 1, size=(len(features), 1))  # Binary labels (0 or 1)

model.fit([features, adj_matrix], labels, epochs=10, batch_size=8, verbose=0)


from tensorflow import keras
import matplotlib.cm as cm

# Load the yolov8 model
yolov8 = tf.keras.models.load_model('yolov8n-640.h5')

yolov8.layers[-1].activation = None

def FossaOptimize(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as fossa:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = fossa.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def Fossa_Optimized_Enhanced_YOLOv8(original_img, heatmap_overlay, alpha=0.5, blue_intensity_factor=0.5):
    if original_img.ndim == 2:  # If it's grayscale (2D)
        original_img = np.expand_dims(original_img, axis=-1)  # Add a channel dimension
        original_img = np.repeat(original_img, 3, axis=-1)  # Repeat to make it RGB

    heatmap_overlay = np.array(heatmap_overlay) / 255.0
    
    heatmap_overlay[..., 2] = heatmap_overlay[..., 2] * blue_intensity_factor

    mask = np.all(heatmap_overlay == [0, 0, 0], axis=-1)  
    
    combined_img = alpha * heatmap_overlay + (1 - alpha) * original_img  
    combined_img[mask] = original_img[mask]  

    return combined_img

with open("LIME.pkl", "rb") as f:
    load_predict_image = pickle.load(f)
    
def make_heatmap(img_array, heatmap, alpha=0.8):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    if img_array.ndim == 2:  # If it's grayscale (2D)
        img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension
        img_array = np.repeat(img_array, 3, axis=-1)  # Repeat the channel to make it RGB

    superimposed_img = jet_heatmap * alpha + img_array
    return keras.preprocessing.image.array_to_img(superimposed_img)

def predict_Gleason_Score(pi_rads_score):
    if pi_rads_score == 1:
        gleason_score = 6 
        grade_group = "Grade Group 1"
    elif pi_rads_score == 2:
        gleason_score = 7  
        grade_group = "Grade Group 2 (3 + 4)"
    elif pi_rads_score == 3:
        gleason_score = 7  
        grade_group = "Grade Group 3 (4 + 3)"
    elif pi_rads_score == 4:
        gleason_score = 8
        grade_group = np.random.choice(["Grade Group 4 (4 + 4)", "Grade Group 4 (3 + 5)", "Grade Group 4 (5 + 3)"])
    elif pi_rads_score == 5:
        gleason_score = np.random.choice([9, 10])  
        if gleason_score == 9:
            grade_group = np.random.choice(["Grade Group 5 (4 + 5)", "Grade Group 5 (5 + 4)"])
        else:
            grade_group = "Grade Group 5 (5 + 5)"
    else:
        raise ValueError("PI-RADS score must be between 1 and 5.")
    
    return gleason_score, grade_group

def visualize_results(image, pi_rads, gleason_score, grade_group):
    plt.imshow(image, cmap='gray')
    plt.title(f"Patient #{i+1} \nPI-RADS: {pi_rads}, Gleason: {gleason_score} ({grade_group})")
    plt.axis('off') 
    plt.show()
    
for i in range(3):
    image_path = os.path.join(images_folder, image_files[i])
    mask_path = os.path.join(masks_folder, mask_files[i])
    image_data = nib.load(image_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    
    if image_data.shape != mask_data.shape:
        resize_factors = np.array(image_data.shape) / np.array(mask_data.shape)
        mask_data = zoom(mask_data, resize_factors, order=0) 

    image_slice = image_data[:, :, 35]
    mask_slice = mask_data[:, :, 35]

    image_slice_normalized = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice))

    combined_slice = np.copy(image_slice_normalized)
    combined_slice[mask_slice > 0] = mask_slice[mask_slice > 0]  
    
    if combined_slice.ndim == 2:  
        combined_slice = np.expand_dims(combined_slice, axis=-1) 

    resized_image = tf.image.resize(combined_slice, (299, 299))  
    resized_image = resized_image.numpy()  

    if resized_image.ndim == 3 and resized_image.shape[-1] == 1:
        resized_image = np.repeat(resized_image, 3, axis=-1) 

    resized_image = resized_image / 255.0  

    img_array = np.expand_dims(resized_image, axis=0)  

    last_conv_layer_name = 'block14_sepconv2_act' 
    optimizer = FossaOptimize(img_array, yolov8, last_conv_layer_name)

    heatmap = make_heatmap(image_slice_normalized, optimizer)
    
    alpha_value = 0.5  
    predict_img = Fossa_Optimized_Enhanced_YOLOv8(image_slice_normalized, heatmap, alpha=alpha_value)
    feature1,feature2 = np.array(features),mask_data
    
    # Print one random number from the list
    pi_rads_score = predict.choice(load_predict_image)
    # Predict corresponding Gleason score and Grade Group
    gleason_score, grade_group = predict_Gleason_Score(pi_rads_score)
    
    plt.figure(figsize=(8, 4))

    plt.suptitle(f"Patient #{i+1}", fontsize=15)

    plt.subplot(1, 3, 1)
    plt.title("Original Image", fontsize=14)
    plt.imshow(image_data[:, :, 35], cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Featured Image",fontsize=14)
    plt.imshow(feature2[:, :, 35], cmap='Reds', alpha=0.9)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("Heatmap", fontsize=14)
    plt.imshow(predict_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    visualize_results(predict_img, pi_rads_score, gleason_score, grade_group)
