import numpy as np

def image_data_norm(img_data):
    mean, std = np.mean(img_data), np.std(img_data)
    return (img_data - mean) / std, mean, std

def covariance_matrix(img_data):
    return np.cov(img_data, rowvar=False)

def compute_sorted_eigen(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

def compute_projected_data(img_data, eigenvectors):
    return np.dot(img_data, eigenvectors)

def compute_pca(image_data):
    img_data_normed, mean, std = image_data_norm(image_data)
    eigen_values, eigen_vectors = compute_sorted_eigen(covariance_matrix(img_data_normed))
    projected_data = compute_projected_data(img_data_normed, eigen_vectors)
    explained_variance_ratio = eigen_values / np.sum(eigen_values)
    return projected_data, eigen_vectors, mean, std, explained_variance_ratio

def pca_compose(img_data):
    img_data = img_data[:, :, :3] if img_data.shape[-1] > 3 else img_data
    return {i: compute_pca(img_data[:, :, i]) for i in range(img_data.shape[-1])}

def pca_find_valuable_comp(pca_channel, threshold=99.995):
    return max(np.argmax(np.cumsum(pca[4]) >= threshold / 100) + 1 for pca in pca_channel.values())

def pca_transform(pca_channel, n_components):
    def reconstruct_channel(pca):
        projected_data, eigen_vectors, mean, std, _ = pca
        return np.dot(np.dot(projected_data[:, :n_components], eigen_vectors.T[:n_components, :]), std) + mean
    
    compressed_image = np.transpose([reconstruct_channel(pca) for pca in pca_channel.values()], (1, 2, 0))
    return np.array(compressed_image, dtype=np.uint8)
