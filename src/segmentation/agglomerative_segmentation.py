import numpy as np
import cv2


def rgb2lab(image):
    """Convert RGB image to Lab color space using OpenCV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

def initialize_centers(image, S):
    """Initialize centers on a grid with spacing S."""
    h, w, _ = image.shape
    centers = []
    for y in range(S//2, h, S):
        for x in range(S//2, w, S):
            centers.append([y, x, *image[y, x]])  # [row, col, L, a, b]
    print(f"new shape of the image matrix: {image.shape}")
    return np.array(centers, dtype=np.float32)

def create_label_distance_maps(h, w):
    """Create label and distance maps."""
    labels = -1 * np.ones((h, w), dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)
    return labels, distances

def compute_distance(center, lab_image, y, x, m, S):
    """Compute combined color+spatial distance."""
    color_diff = lab_image[y, x] - center[2:]
    dc = np.linalg.norm(color_diff)
    ds = np.linalg.norm(np.array([y, x]) - center[:2])
    return np.sqrt((dc)**2 + (m/S * ds)**2)

def slic_superpixels(image, num_superpixels=100, m=10, num_iterations=5):
    """
    Perform simple SLIC superpixel clustering from scratch.
    
    image: RGB image (np.array)
    num_superpixels: target number of superpixels
    m: compactness factor
    num_iterations: how many iterations to run
    """
    h, w, _ = image.shape
    N = h * w
    S = int(np.sqrt(N / num_superpixels))  # grid interval

    lab_image = rgb2lab(image)
    centers = initialize_centers(lab_image, S)
    labels, distances = create_label_distance_maps(h, w)

    for it in range(num_iterations):
        print(f"Iteration {it+1}/{num_iterations}")

        for idx, center in enumerate(centers):
            y0, x0 = int(center[0]), int(center[1])
            y_min, y_max = max(0, y0-S), min(h, y0+S)
            x_min, x_max = max(0, x0-S), min(w, x0+S)

            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    d = compute_distance(center, lab_image, y, x, m, S)
                    if d < distances[y, x]:
                        distances[y, x] = d
                        labels[y, x] = idx

        # Update centers
        new_centers = np.zeros_like(centers)
        count = np.zeros((centers.shape[0], 1), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                label = labels[y, x]
                if label >= 0:
                    new_centers[label, 0] += y
                    new_centers[label, 1] += x
                    new_centers[label, 2:] += lab_image[y, x]
                    count[label] += 1

        for i in range(centers.shape[0]):
            if count[i] > 0:
                new_centers[i] /= count[i]

        centers = new_centers

    return labels , centers 

def draw_contours(image, labels):
    """Draw superpixel contours on the image."""
    contour_img = image.copy()
    mask = np.zeros(labels.shape, dtype=np.uint8)

    h, w = labels.shape
    for y in range(1, h-1):
        for x in range(1, w-1):
            if (labels[y, x] != labels[y+1, x]) or (labels[y, x] != labels[y, x+1]):
                mask[y, x] = 255

    contour_img[mask == 255] = [0, 255, 0]  # Green contours
    return contour_img


def extract_features(image, labels, num_superpixels):
    features = []
    centers = []
    
    for i in range(num_superpixels):
        # Find the mask of the current superpixel
        mask = (labels == i)
        
        # Get the coordinates of the pixels in the current superpixel
        coords = np.column_stack(np.where(mask))  # (y, x) coordinates
        
        # Get the mean color of the superpixel
        mean_color = np.mean(image[mask], axis=0)
        
        # Get the center of the superpixel (average coordinates)
        center = np.mean(coords, axis=0)
        
        # Combine color and position into a feature vector
        feature_vector = np.hstack([mean_color, center])
        
        features.append(feature_vector)
        centers.append(center)
    
    return np.array(features), np.array(centers)

import matplotlib.pyplot as plt

class AgglomerativeClusteringFromScratch:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, features):
        self.features = features
        self.clusters = {i: [i] for i in range(len(features))}
        self.labels = np.arange(len(features))

        while len(self.clusters) > self.n_clusters:
            cluster_pair = self.find_closest_clusters()

            if cluster_pair is None:
                print("No more clusters can be merged.")
                break

            cluster_1, cluster_2 = cluster_pair
            print(f"Merging clusters {cluster_1} and {cluster_2}")
            self.merge_clusters(cluster_1, cluster_2)

        # Remap cluster IDs to 0...n_clusters-1
        unique_cluster_ids = list(self.clusters.keys())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_cluster_ids)}

        self.final_labels = np.zeros(len(features), dtype=np.int32)
        for cluster_id, points in self.clusters.items():
            new_cluster_id = id_mapping[cluster_id]
            for point in points:
                self.final_labels[point] = new_cluster_id

    def compute_distance_matrix(self, features):
        n = len(features)
        distances = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(features[i] - features[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def find_closest_clusters(self):
        min_dist = np.inf
        cluster_pair = None
        cluster_ids = list(self.clusters.keys())

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster_1 = cluster_ids[i]
                cluster_2 = cluster_ids[j]
                dist = self.compute_linkage_distance(self.clusters[cluster_1], self.clusters[cluster_2])
                if dist < min_dist:
                    min_dist = dist
                    cluster_pair = (cluster_1, cluster_2)

        return cluster_pair

    def compute_linkage_distance(self, cluster1, cluster2):
        if self.linkage == 'single':
            return min(
                np.linalg.norm(self.features[i] - self.features[j])
                for i in cluster1 for j in cluster2
            )
        elif self.linkage == 'complete':
            return max(
                np.linalg.norm(self.features[i] - self.features[j])
                for i in cluster1 for j in cluster2
            )
        else:
            raise ValueError("Unsupported linkage method")

    def merge_clusters(self, cluster_1, cluster_2):
        new_cluster_points = self.clusters[cluster_1] + self.clusters[cluster_2]
        del self.clusters[cluster_1]
        del self.clusters[cluster_2]
        new_cluster_id = max(self.clusters.keys(), default=-1) + 1
        self.clusters[new_cluster_id] = new_cluster_points

def visualize_clusters(image, labels, superpixel_map, num_clusters):
    output_image = image.copy()
    colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)

    h, w = superpixel_map.shape
    clustered_image = np.zeros((h, w, 3), dtype=np.uint8)

    for superpixel_id in range(len(labels)):
        mask = (superpixel_map == superpixel_id)
        clustered_image[mask] = colors[labels[superpixel_id]]

    plt.imshow(clustered_image)
    plt.title(f"Clusters: {num_clusters}")
    plt.axis('off')
    plt.show()


def cluster_image(image, num_superpixels=100, compactness=10, num_iterations=5, num_clusters=5, linkage='single'):
    """
    Perform SLIC superpixel segmentation + Agglomerative clustering and return the clustered image.

    Parameters:
    - image: input RGB image
    - num_superpixels: desired number of superpixels
    - compactness: compactness factor for SLIC
    - num_iterations: number of SLIC iterations
    - num_clusters: number of final clusters after Agglomerative clustering
    - linkage: 'single' or 'complete' linkage method

    Returns:
    - clustered_image: image colored by clusters
    """

    labels, _ = slic_superpixels(image, num_superpixels=num_superpixels, m=compactness, num_iterations=num_iterations)

    features, _ = extract_features(image, labels, np.max(labels)+1)

    clustering = AgglomerativeClusteringFromScratch(n_clusters=num_clusters, linkage=linkage)
    clustering.fit(features)

    h, w = labels.shape
    output_image = np.zeros((h, w, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)

    for superpixel_id in range(len(clustering.final_labels)):
        mask = (labels == superpixel_id)
        output_image[mask] = colors[clustering.final_labels[superpixel_id]]

    return output_image







if __name__ == "__main__":
    # main(image_path, num_clusters=2)

    # Load an image
    image = cv2.imread('../../data/bird.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clustered = cluster_image(image, num_superpixels=200, compactness=20, num_clusters=8, linkage='complete')

    plt.imshow(clustered)
    plt.axis('off')
    plt.show()
