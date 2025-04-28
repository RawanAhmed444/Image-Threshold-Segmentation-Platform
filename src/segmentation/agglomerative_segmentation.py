import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


# Step 1: Canny Edge Detection
def apply_canny_edge_detection(image):
    print(image.shape)
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    return edges




# Step 2: Construct Graph based on pixel similarity (color similarity)
def construct_graph(image, edges):
    height, width = image.shape[:2]
    num_pixels = height * width
    print(num_pixels)
    graph = lil_matrix((num_pixels, num_pixels), dtype=np.float64)

    # Convert image into 2D array of pixel intensities (flattened)
    image_reshaped = image.reshape((-1, 3))  # Shape: (num_pixels, 3)
    print(image_reshaped.shape)
    # Construct a graph based on color similarity and edge connectivity
    for i in range(height):
        for j in range(width):
            idx1 = i * width + j  # Flattened index for pixel (i, j)
            
            for di in range(-1, 2):  # Neighbors' relative positions
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        idx2 = ni * width + nj
                        
                        # Only consider edges detected by Canny
                        if edges[i, j] == 255 and edges[ni, nj] == 255:
                            dist = np.linalg.norm(image_reshaped[idx1] - image_reshaped[idx2])
                            graph[idx1, idx2] = np.exp(-dist**2 / (2.0 * (10**2)))  # Gaussian kernel

    return graph
# Rest of the implementation stays the same, including Laplacian, Spectral Clustering, etc.


# Step 3: Calculate the Laplacian Matrix of the Graph
def compute_laplacian(graph):
    # Create a sparse matrix for the degree matrix
    degree_matrix = lil_matrix((graph.shape[0], graph.shape[0]), dtype=np.float32)
    
    # Compute degree matrix (diagonal with sum of edges for each node)
    row_sums = np.array(graph.sum(axis=1)).flatten()
    degree_matrix.setdiag(row_sums)  # Set diagonal to sum of each row (degree)

    # The Laplacian is L = D - A, where A is the graph adjacency matrix (sparse)
    laplacian_matrix = degree_matrix - graph

    return laplacian_matrix


# Step 4: Eigenvalue Decomposition of the Laplacian Matrix
def eigen_decomposition(laplacian_matrix, num_clusters):
    # Compute the eigenvalues and eigenvectors

    # Convert the Laplacian matrix to sparse format (if it's sparse)
    laplacian_sparse = csr_matrix(laplacian_matrix)

    # Compute the smallest eigenvalues and eigenvectors using sparse eigensolver
    eigenvalues, eigenvectors = eigsh(laplacian_sparse, k=num_clusters + 1, which='SM')

    # Now you can use the eigenvectors (ignoring the first eigenvector)
    embedding = eigenvectors[:, 1:num_clusters + 1]
    # Take the first 'num_clusters' eigenvectors (ignore the first eigenvalue)
    return embedding

# Step 5: Spectral Clustering
def spectral_clustering(laplacian_matrix, num_clusters):
    # Step 4: Eigen decomposition
    print(laplacian_matrix.shape)
    eigenvectors = eigen_decomposition(laplacian_matrix, num_clusters)
    
    # Normalize the eigenvectors row-wise
    rows_norm = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    eigenvectors_normalized = eigenvectors / rows_norm
    
    # Perform K-means clustering on the eigenvectors to get the clusters
    # For simplicity, we use basic K-means (random initialization and iterative refinement)
    
    # Randomly initialize centroids
    centroids = eigenvectors_normalized[np.random.choice(eigenvectors_normalized.shape[0], num_clusters, replace=False)]
    prev_centroids = centroids.copy()
    
    while True:
        # Assign each pixel to the closest centroid
        distances = np.linalg.norm(eigenvectors_normalized[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Recompute centroids
        for k in range(num_clusters):
            centroids[k] = eigenvectors_normalized[labels == k].mean(axis=0)
        
        # Convergence check
        if np.all(centroids == prev_centroids):
            break
        prev_centroids = centroids.copy()
    
    return labels

# Step 6: Main Spectral Agglomerative Segmentation Function
def spectral_agglomerative_segmentation(image, num_clusters=5):
    # Step 1: Apply edge detection
    edges = apply_canny_edge_detection(image)
    
    # Step 2: Construct similarity graph
    graph = construct_graph(image, edges)
    
    # Step 3: Compute the Laplacian matrix
    laplacian_matrix = compute_laplacian(graph)
    
    # Step 4: Perform Spectral Clustering
    labels = spectral_clustering(laplacian_matrix, num_clusters)
    
    # Reshape the labels back into the image shape
    height, width = image.shape[:2]
    segmented_image = labels.reshape((height, width))
    
    return segmented_image

# Step 7: Visualization
def display_results(image, segmented_image):
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    # Display the segmented image
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='nipy_spectral')
    plt.title("Segmented Image")
    
    plt.show()




if __name__ == "__main__":
    # Load an example image
    image = cv2.imread('bird.jpg')  # Replace with your image file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # Resize for faster processing
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Display the results
    edges  =apply_canny_edge_detection(image_gray)
    graph = construct_graph(image, edges)
    laplacian = compute_laplacian(graph)
    labels = spectral_clustering(laplacian, num_clusters=5)
    segmented_image = labels.reshape((image.shape[0], image.shape[1]))
    print(graph.shape)
    display_results(image, segmented_image)
