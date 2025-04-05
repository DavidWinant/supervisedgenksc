import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils import assign_soft_clusters


# Create synthetic data for testing
def generate_synthetic_data(n_samples=100, k=3):
    # Create clusters in 2D space (k-1 dimensions)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])

    # Generate points around these centers
    data = []
    for center in centers:
        cluster = center + np.random.normal(0, 0.2, (n_samples // k, 2))
        data.append(cluster)

    data = np.vstack(data)
    return torch.tensor(data, dtype=torch.float32), centers


def main():
    # Generate synthetic data
    n_samples = 300
    k = 3
    data, centers = generate_synthetic_data(n_samples, k)

    # Create prototype vectors (in this case, we'll use the true centers)
    prototypes = torch.tensor(centers, dtype=torch.float32)

    # Apply soft clustering
    memberships = assign_soft_clusters(data, prototypes)

    # Visualize results
    plt.figure(figsize=(10, 10))

    # Plot original points colored by their highest membership
    colors = ["r", "g", "b"]
    labels = ["Cluster 1", "Cluster 2", "Cluster 3"]

    # Convert memberships to colors
    dominant_clusters = torch.argmax(memberships, dim=1)

    # Plot points
    for i in range(k):
        mask = dominant_clusters == i
        plt.scatter(
            data[mask, 0], data[mask, 1], c=colors[i], alpha=0.6, label=labels[i]
        )

    # Plot prototypes
    plt.scatter(
        prototypes[:, 0],
        prototypes[:, 1],
        c="black",
        marker="x",
        s=200,
        label="Prototypes",
    )

    plt.legend()
    plt.title("Soft Clustering Results")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

    # Print some statistics
    print("\nMembership matrix statistics:")
    print(f"Shape: {memberships.shape}")
    print(f"Min membership value: {memberships.min().item():.3f}")
    print(f"Max membership value: {memberships.max().item():.3f}")
    print(f"Mean membership value: {memberships.mean().item():.3f}")

    # Print example memberships for first few points
    print("\nExample membership values for first 3 points:")
    for i in range(3):
        print(f"Point {i+1}: {memberships[i].tolist()}")


if __name__ == "__main__":
    main()
