import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- KMeans Functions (from previous cell) ---
def split_and_stack_image(img):
    """Split image into RGB and stack into (pixels, 3) array."""
    r, g, b = cv2.split(img)
    pixels = np.stack([r.flatten(), g.flatten(), b.flatten()], axis=1)
    return pixels

def initialize_centroids(k):
    """Initialize k random RGB centroids."""
    return np.random.randint(0, 256, (k, 3)).astype(np.float32)

def perform_kmeans_clustering(pixels, k, max_iter=100):
    """Perform KMeans clustering using fast NumPy operations."""
    centroids = initialize_centroids(k)
    for _ in range(max_iter):
        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        assignment = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = (assignment == i)
            if np.any(mask):
                new_centroids[i] = pixels[mask].mean(axis=0)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return assignment, centroids

def reconstruct_clustered_image(pixels, assignment, centroids, img_shape):
    """Reconstruct clustered image."""
    clustered_pixels = centroids[assignment].astype(np.uint8)
    clustered_image = clustered_pixels.reshape(img_shape)
    return clustered_image
def cluster_image_kmeans(image, k=5, max_iter=100):
    """
    Perform K-means clustering on an image and return the clustered image.

    Parameters:
    - image: input RGB image (np.ndarray)
    - k: number of clusters
    - max_iter: maximum number of KMeans iterations

    Returns:
    - clustered_image: image recolored by cluster centroids
    """
    pixels = split_and_stack_image(image)
    assignment, centroids = perform_kmeans_clustering(pixels, k, max_iter)
    clustered_image = reconstruct_clustered_image(pixels, assignment, centroids, image.shape)
    return clustered_image