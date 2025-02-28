import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.linalg import sqrtm
from scipy.special import digamma, loggamma
from math import pi

# computes the Frechet Distance between two datasets
def compute_fd(reps1, reps2, eps=1e-6):
    mu1, sigma1 = np.mean(reps1, axis=0), np.cov(reps1, rowvar=False)
    mu2, sigma2 = np.mean(reps2, axis=0), np.cov(reps2, rowvar=False)

    diff = mu1 - mu2
    try:
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except ValueError:
        covmean = sqrtm(sigma1 + eps * np.eye(sigma1.shape[0])).dot(sqrtm(sigma2 + eps * np.eye(sigma2.shape[0])))
        covmean = covmean.real

    return np.dot(diff, diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

# Information-theoretic metrics
def volume_of_unit_ball_log(d):
    return (d / 2) * np.log(pi) - loggamma((d / 2) + 1)

def cross_entropy(N, M, k, nu_k, d):
    return (1 / N) * np.sum(np.log(M) - digamma(k) + volume_of_unit_ball_log(d) + d * np.log(nu_k))

def entropy(N, k, rho_k, d):
    return (1 / N) * np.sum(np.log(N-1) - digamma(k) + volume_of_unit_ball_log(d) + d * np.log(rho_k))

# kNN Metrics
def calc_precision(dist_R, dist_RG_pairs, M):
    radii_R = dist_R[:, -1]
    G_in_radius = (dist_RG_pairs <= radii_R[:, np.newaxis])
    return np.sum(np.any(G_in_radius, axis=0)) / M

def calc_density(dist_R, dist_RG_pairs, k, M):
    radii_R = dist_R[:, -1]
    G_in_radius = (dist_RG_pairs <= radii_R[:, np.newaxis])
    return np.sum(G_in_radius) / (k * M)

def calc_coverage(dist_R, dist_RG_pairs):
    radii_R = dist_R[:, -1]
    G_in_radius = (dist_RG_pairs <= radii_R[:, np.newaxis])
    return np.mean(np.any(G_in_radius, axis=1))

def calc_PC(G, nbrs_G, nbrs_R, M, k, C):
    k_prime = C * k
    dist_G, _ = nbrs_G.kneighbors(G, k_prime+1)
    radii_G = dist_G[:, -1]

    dist_RG, _ = nbrs_R.kneighbors(G, k)
    return np.sum(dist_RG[:, -1] <= radii_G) / M

# Compute all metrics in one function
def compute_metrics(real_features, fake_features, nearest_k=5):
    """
    Computes metrics for generative models.

    Args:
        real_features (np.ndarray): (N, D) array of real data features.
        fake_features (np.ndarray): (M, D) array of fake data features.
        nearest_k (int): Number of nearest neighbors.

    Returns:
        dict: dictionary containing precision, recall, density, coverage, FID, PCE, RCE, and RE.
    """
    N, D = real_features.shape
    M, _ = fake_features.shape

    # Set up nearest neighbors
    nbrs_R = NearestNeighbors(n_neighbors=nearest_k+1, algorithm='auto', n_jobs=-1).fit(real_features)
    dist_R, _ = nbrs_R.kneighbors(real_features, nearest_k+1)

    nbrs_G = NearestNeighbors(n_neighbors=nearest_k+1, algorithm='auto', n_jobs=-1).fit(fake_features)
    dist_G, _ = nbrs_G.kneighbors(fake_features, nearest_k+1)

    dist_RG_pairs = pairwise_distances(real_features, fake_features, n_jobs=-1)
    dist_GR_pairs = pairwise_distances(fake_features, real_features, n_jobs=-1)

    dist_RG, _ = nbrs_G.kneighbors(real_features, nearest_k+1)
    dist_GR, _ = nbrs_R.kneighbors(fake_features, nearest_k+1)


    precision = calc_precision(dist_R, dist_RG_pairs, M)
    recall = calc_precision(dist_G, dist_GR_pairs, N)  # Symmetric to precision
    density = calc_density(dist_R, dist_RG_pairs, nearest_k, M)
    coverage = calc_coverage(dist_R, dist_RG_pairs)


    PCE = cross_entropy(M, N, nearest_k, dist_GR[:, nearest_k-1], D)
    RCE = cross_entropy(N, M, nearest_k, dist_RG[:, nearest_k-1], D)
    RE = entropy(N, nearest_k, dist_R[:, nearest_k], D)


    fd = compute_fd(real_features, fake_features)

    return {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
        "frechet_distance": fd,
        "precision_cross_entropy": PCE,
        "recall_cross_entropy": RCE,
        "recall_entropy": RE
    }
