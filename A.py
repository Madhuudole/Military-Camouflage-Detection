import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Parameters
IMAGE_SIZE = (224, 224)  # Size required for VGG16
NUM_CLUSTERS = 5  # Number of object clusters to detect

# Load Images
def load_images(directory, image_size):
    images = []
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    image_paths.append(img_path)
    return np.array(images, dtype='float32') / 255.0, image_paths

dataset_path = r"C:/Users/madhu/Downloads/archive (3)/military_object_dataset/test/images"
images, image_paths = load_images(dataset_path, IMAGE_SIZE)
print(f"Loaded {len(images)} images.")

# Load Pre-trained VGG16 Model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Extract Features
features = vgg16.predict(images)
features = features.reshape(features.shape[0], -1)  # Flatten the features
print(f"Extracted features shape: {features.shape}")

# Apply K-means Clustering
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# Visualize Clustering Results
for cluster in range(NUM_CLUSTERS):
    print(f"Cluster {cluster}:")
    cluster_images = np.array(image_paths)[cluster_labels == cluster]
    for img_path in cluster_images[:5]:  # Display first 5 images per cluster
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# Generate Heatmaps
def generate_heatmaps(features, image_shape):
    heatmaps = []
    for feature in features:
        if len(feature.shape) < 3:
            print("Skipping feature map due to invalid shape:", feature.shape)
            continue
        heatmap = np.mean(feature, axis=-1)  # Average across channels
        heatmap = cv2.resize(heatmap, image_shape[:2])
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize
        heatmaps.append(heatmap)
           # Normalize the heatmap for better visualization
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
        heatmaps.append(heatmap)
    return np.array(heatmaps)

print(f"Feature map shape: {features.shape}")  # Expected: (batch_size, height, width, channels)

heatmaps = generate_heatmaps(features, IMAGE_SIZE)

# Display Heatmaps
for i in range(5):  # Display heatmaps for 5 images
    plt.subplot(1, 2, 1)
    plt.imshow(images[i])
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(heatmaps[i], cmap='jet')
    plt.title("Heatmap")
    plt.axis('off')

    plt.show()
