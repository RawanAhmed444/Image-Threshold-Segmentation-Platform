import cv2
import numpy as np
import torch
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

def compute_superpixel_centers(image_lab, labels, K):
    H, W, _ = image_lab.shape
    device = image_lab.device
    centers = torch.zeros((K, 5), device=device)  # L, A, B, x, y
    count = torch.zeros(K, device=device)

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    flat_lab = image_lab.view(-1, 3)
    flat_labels = labels.view(-1)
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()

    for k in range(K):
        mask = flat_labels == k
        count[k] = mask.sum()
        if count[k] > 0:
            centers[k, :3] = flat_lab[mask].mean(dim=0)
            centers[k, 3] = flat_x[mask].float().mean()
            centers[k, 4] = flat_y[mask].float().mean()

    return centers, count

def compute_distance(image_lab, centers, S, m):
    H, W, _ = image_lab.shape
    device = image_lab.device

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    coords = torch.stack((grid_y, grid_x), dim=-1).float()

    distances = torch.full((H, W), float('inf'), device=device)
    labels = torch.full((H, W), -1, dtype=torch.long, device=device)

    for k, center in enumerate(centers):
        y, x = int(center[4].item()), int(center[3].item())
        y1, y2 = max(y - S, 0), min(y + S, H)
        x1, x2 = max(x - S, 0), min(x + S, W)

        region = image_lab[y1:y2, x1:x2]
        region_coords = coords[y1:y2, x1:x2]

        dc = torch.norm(region - center[:3], dim=2)
        ds = torch.norm(region_coords - center[3:], dim=2)
        D = torch.sqrt(dc**2 + (ds / S)**2 * m**2)

        region_dist = distances[y1:y2, x1:x2]
        region_label = labels[y1:y2, x1:x2]

        mask = D < region_dist
        region_dist[mask] = D[mask]
        region_label[mask] = k

        distances[y1:y2, x1:x2] = region_dist
        labels[y1:y2, x1:x2] = region_label

    return labels

def slic_gpu(image_lab, K, m=10, max_iter=10):
    H, W, _ = image_lab.shape
    N = H * W
    S = int((N / K)**0.5)
    device = image_lab.device

    grid_y = torch.arange(S//2, H, S, device=device)
    grid_x = torch.arange(S//2, W, S, device=device)
    centers = torch.zeros((len(grid_y) * len(grid_x), 5), device=device)

    idx = 0
    for y in grid_y:
        for x in grid_x:
            L, A, B = image_lab[y, x]
            centers[idx] = torch.tensor([L, A, B, x.float(), y.float()], device=device)
            idx += 1

    for _ in range(max_iter):
        labels = compute_distance(image_lab, centers, S, m)
        centers, _ = compute_superpixel_centers(image_lab, labels, centers.shape[0])

    return labels, centers

def extract_superpixel_features(image_lab, labels, centers):
    K = centers.shape[0]
    device = image_lab.device
    H, W, _ = image_lab.shape

    features = torch.zeros((K, 5), device=device)
    count = torch.zeros(K, device=device)

    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    flat_lab = image_lab.view(-1, 3)
    flat_labels = labels.view(-1)
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()

    for k in range(K):
        mask = flat_labels == k
        if mask.sum() > 0:
            features[k, :3] = flat_lab[mask].mean(dim=0)
            features[k, 3] = flat_x[mask].float().mean()
            features[k, 4] = flat_y[mask].float().mean()

    return features

def run_agglomerative_clustering(features, n_clusters):
    features_cpu = features.detach().cpu().numpy()
    clustering = AgglomerativeClusteringFromScratch(n_clusters=n_clusters)
    clustering.fit(features_cpu)
    cluster_labels = clustering.final_labels
    return torch.tensor(cluster_labels, device=features.device)

def visualize_clusters(image_rgb, labels, cluster_labels):
    H, W, _ = image_rgb.shape
    output = image_rgb.copy()
    colors = np.random.randint(0, 255, (cluster_labels.max().item() + 1, 3))

    for y in range(H):
        for x in range(W):
            superpixel_label = labels[y, x].item()
            cluster_id = cluster_labels[superpixel_label]
            output[y, x] = colors[cluster_id]

    return output

def main(image_path, num_superpixels=200, m=10, num_iterations=5, n_clusters=2):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab_tensor = torch.from_numpy(lab).float().to('cuda')

    labels, centers = slic_gpu(lab_tensor, num_superpixels, m, num_iterations)
    features = extract_superpixel_features(lab_tensor, labels, centers)
    cluster_labels = run_agglomerative_clustering(features, n_clusters)
    output_image = visualize_clusters(image_rgb, labels.cpu(), cluster_labels.cpu())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Clustered Superpixels')
    plt.axis('off')
    plt.show()

image_path = '../../data/bird.jpg'
main(image_path)

