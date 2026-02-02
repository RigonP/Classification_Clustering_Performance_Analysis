"""
Advanced Machine Learning - Clustering Exercise
DBScan Algorithm on IRIS Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                            silhouette_score, davies_bouldin_score,
                            homogeneity_score, completeness_score, v_measure_score)
from sklearn.decomposition import PCA
from itertools import product
import warnings
import os
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

print("="*80)
print("CLUSTERING WITH DBSCAN ON IRIS DATASET")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("1. LOADING AND PREPARING DATA")
print("="*80)

# Load IRIS dataset from file
column_names = ['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)', 'class']

# Read the clean CSV file (no missing values - matches sklearn's iris dataset)
df = pd.read_csv('../data/iris_clean.data', header=None, names=column_names)

# Map class names to numeric values
class_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
df['target'] = df['class'].map(class_mapping)

# Extract features and target
X = df[column_names[:-1]].values  # Features only (class ID removed)
y_true = df['target'].values  # True labels (for comparison only)
feature_names = column_names[:-1]
target_names = ['setosa', 'versicolor', 'virginica']

print(f"\nDataset loaded from: ../data/iris_clean.data")
print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"True number of classes: {len(np.unique(y_true))}")
print(f"Class names: {target_names}")

# Standardize features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData has been standardized (mean=0, std=1)")
print("This is crucial for DBSCAN as it uses distance-based clustering")

# ============================================================================
# 2. VISUALIZE DATA BEFORE CLUSTERING
# ============================================================================
print("\n" + "="*80)
print("2. DATA VISUALIZATION (Before Clustering)")
print("="*80)

fig = plt.figure(figsize=(16, 5))

# Plot 1: Original data (Petal features)
plt.subplot(1, 3, 1)
for class_id in np.unique(y_true):
    plt.scatter(X[y_true == class_id, 2],
                X[y_true == class_id, 3],
                label=target_names[class_id],
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Original Data with True Labels\n(Petal Features)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.subplot(1, 3, 2)
for class_id in np.unique(y_true):
    plt.scatter(X_pca[y_true == class_id, 0],
                X_pca[y_true == class_id, 1],
                label=target_names[class_id],
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Projection with True Labels')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Sepal features
plt.subplot(1, 3, 3)
for class_id in np.unique(y_true):
    plt.scatter(X[y_true == class_id, 0],
                X[y_true == class_id, 1],
                label=target_names[class_id],
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Original Data with True Labels\n(Sepal Features)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/iris_before_clustering.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'iris_before_clustering.png'")
plt.show()

# ============================================================================
# 3. DBSCAN WITH DIFFERENT PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("3. DBSCAN CLUSTERING WITH DIFFERENT PARAMETERS")
print("="*80)

# Define parameter ranges
eps_values = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
min_samples_values = [3, 5, 7, 10]

print("\nTesting combinations of:")
print(f"  ε (epsilon): {eps_values}")
print(f"  minPts: {min_samples_values}")
print(f"\nTotal combinations: {len(eps_values) * len(min_samples_values)}")

# Store results
results = []

print("\n" + "-"*80)
print("Running DBSCAN with different parameter combinations...")
print("-"*80)

for eps, min_pts in product(eps_values, min_samples_values):
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_pts)
    labels = dbscan.fit_predict(X_scaled)

    # Calculate metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Only calculate these metrics if we have at least 2 clusters (excluding noise)
    if n_clusters >= 2 and n_noise < len(labels):
        # Silhouette score (only for non-noise points)
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1:
            try:
                silhouette = silhouette_score(X_scaled[non_noise_mask],
                                             labels[non_noise_mask])
            except:
                silhouette = -1
        else:
            silhouette = -1

        # Comparison with true labels
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        homogeneity = homogeneity_score(y_true, labels)
        completeness = completeness_score(y_true, labels)
        v_measure = v_measure_score(y_true, labels)
    else:
        silhouette = -1
        ari = nmi = homogeneity = completeness = v_measure = 0

    # Store results
    results.append({
        'eps': eps,
        'min_samples': min_pts,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'ari': ari,
        'nmi': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'labels': labels.copy()
    })

    print(f"ε={eps:.1f}, minPts={min_pts:2d} → "
          f"Clusters={n_clusters}, Noise={n_noise:3d}, "
          f"ARI={ari:.3f}, Silhouette={silhouette:.3f}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY OF ALL PARAMETER COMBINATIONS")
print("="*80)
print(results_df[['eps', 'min_samples', 'n_clusters', 'n_noise',
                   'ari', 'silhouette']].to_string(index=False))

# ============================================================================
# 4. FIND BEST PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("4. IDENTIFYING BEST PARAMETER COMBINATIONS")
print("="*80)

# Filter results with 3 clusters (we know IRIS has 3 classes)
results_3clusters = results_df[results_df['n_clusters'] == 3].copy()

if len(results_3clusters) > 0:
    print("\nParameter combinations that found exactly 3 clusters:")
    print(results_3clusters[['eps', 'min_samples', 'n_clusters', 'n_noise',
                              'ari', 'nmi', 'silhouette']].to_string(index=False))

    # Best by ARI
    best_ari_idx = results_3clusters['ari'].idxmax()
    best_ari = results_3clusters.loc[best_ari_idx]

    # Best by Silhouette
    best_sil_idx = results_3clusters['silhouette'].idxmax()
    best_sil = results_3clusters.loc[best_sil_idx]

    print(f"\nBest parameters (by ARI):")
    print(f"  ε = {best_ari['eps']}, minPts = {best_ari['min_samples']}")
    print(f"  ARI = {best_ari['ari']:.4f}, Silhouette = {best_ari['silhouette']:.4f}")

    print(f"\nBest parameters (by Silhouette Score):")
    print(f"  ε = {best_sil['eps']}, minPts = {best_sil['min_samples']}")
    print(f"  ARI = {best_sil['ari']:.4f}, Silhouette = {best_sil['silhouette']:.4f}")
else:
    print("\nNo parameter combination found exactly 3 clusters.")
    print("Finding best overall parameters...")

    # Find combination with highest ARI
    best_ari_idx = results_df['ari'].idxmax()
    best_ari = results_df.loc[best_ari_idx]

    print(f"\nBest parameters (by ARI):")
    print(f"  ε = {best_ari['eps']}, minPts = {best_ari['min_samples']}")
    print(f"  Clusters = {best_ari['n_clusters']}, ARI = {best_ari['ari']:.4f}")

# ============================================================================
# 5. DETAILED VISUALIZATION OF SELECTED CONFIGURATIONS
# ============================================================================
print("\n" + "="*80)
print("5. VISUALIZING SELECTED PARAMETER CONFIGURATIONS")
print("="*80)

# Select interesting configurations to visualize
configs_to_plot = [
    {'eps': 0.5, 'min_samples': 5},
    {'eps': 0.7, 'min_samples': 5},
    {'eps': 0.9, 'min_samples': 5},
    {'eps': 0.5, 'min_samples': 3},
    {'eps': 0.5, 'min_samples': 7},
    {'eps': 0.5, 'min_samples': 10},
]

fig = plt.figure(figsize=(18, 12))

for idx, config in enumerate(configs_to_plot, 1):
    # Find result for this configuration
    result = results_df[(results_df['eps'] == config['eps']) &
                        (results_df['min_samples'] == config['min_samples'])]

    if len(result) == 0:
        continue

    result = result.iloc[0]
    labels = result['labels']

    # Plot in PCA space
    plt.subplot(3, 3, idx)

    # Plot clusters
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            color = 'black'
            marker = 'x'
            s = 50
            alpha = 0.5
        else:
            marker = 'o'
            s = 100
            alpha = 0.7

        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], marker=marker, s=s, alpha=alpha,
                   edgecolors='black', linewidth=1,
                   label=f'Cluster {label}' if label != -1 else 'Noise')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'ε={config["eps"]}, minPts={config["min_samples"]}\n'
              f'Clusters={result["n_clusters"]}, Noise={result["n_noise"]}, '
              f'ARI={result["ari"]:.3f}')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)

# Add comparison with true labels
plt.subplot(3, 3, 7)
for class_id in np.unique(y_true):
    plt.scatter(X_pca[y_true == class_id, 0],
                X_pca[y_true == class_id, 1],
                label=target_names[class_id],
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('True Labels (for comparison)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/dbscan_parameter_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Parameter comparison saved as 'dbscan_parameter_comparison.png'")
plt.show()

# ============================================================================
# 6. HEATMAP VISUALIZATION OF METRICS
# ============================================================================
print("\n" + "="*80)
print("6. HEATMAP VISUALIZATION OF CLUSTERING METRICS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

metrics = ['n_clusters', 'n_noise', 'ari', 'nmi', 'silhouette', 'v_measure']
titles = ['Number of Clusters', 'Number of Noise Points',
          'Adjusted Rand Index', 'Normalized Mutual Information',
          'Silhouette Score', 'V-Measure']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 3, idx % 3]

    # Create pivot table
    pivot = results_df.pivot(index='min_samples', columns='eps', values=metric)

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis',
                ax=ax, cbar_kws={'label': metric})
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('ε (epsilon)')
    ax.set_ylabel('minPts')

plt.tight_layout()
plt.savefig('../results/dbscan_metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ Metrics heatmap saved as 'dbscan_metrics_heatmap.png'")
plt.show()

# ============================================================================
# 7. DETAILED COMPARISON WITH TRUE LABELS (BEST RESULT)
# ============================================================================
print("\n" + "="*80)
print("7. DETAILED COMPARISON WITH TRUE LABELS")
print("="*80)

# Use the best configuration (by ARI)
best_idx = results_df['ari'].idxmax()
best_result = results_df.loc[best_idx]
best_labels = best_result['labels']

print(f"\nBest Configuration:")
print(f"  ε = {best_result['eps']}")
print(f"  minPts = {best_result['min_samples']}")
print(f"  Number of clusters found: {best_result['n_clusters']}")
print(f"  Number of noise points: {best_result['n_noise']}")

print("\n" + "-"*80)
print("CLUSTERING QUALITY METRICS")
print("-"*80)
print(f"  Adjusted Rand Index (ARI):        {best_result['ari']:.4f}")
print(f"  Normalized Mutual Information:    {best_result['nmi']:.4f}")
print(f"  Homogeneity:                      {best_result['homogeneity']:.4f}")
print(f"  Completeness:                     {best_result['completeness']:.4f}")
print(f"  V-Measure:                        {best_result['v_measure']:.4f}")
print(f"  Silhouette Score:                 {best_result['silhouette']:.4f}")

# Create confusion-like matrix
print("\n" + "-"*80)
print("CLUSTER vs TRUE LABEL DISTRIBUTION")
print("-"*80)

# Create a mapping table
comparison_df = pd.DataFrame({
    'True Label': [target_names[i] for i in y_true],
    'Cluster': best_labels
})

# Count occurrences
cross_tab = pd.crosstab(comparison_df['True Label'],
                        comparison_df['Cluster'],
                        margins=True)
print("\n", cross_tab)

# Visualize the comparison
fig = plt.figure(figsize=(16, 6))

# Plot 1: True labels
plt.subplot(1, 3, 1)
for class_id in np.unique(y_true):
    plt.scatter(X_pca[y_true == class_id, 0],
                X_pca[y_true == class_id, 1],
                label=target_names[class_id],
                s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('True Labels')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: DBSCAN clustering
plt.subplot(1, 3, 2)
unique_labels = set(best_labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'black'
        marker = 'x'
        s = 50
    else:
        marker = 'o'
        s = 100

    mask = best_labels == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[color], marker=marker, s=s, alpha=0.7,
               edgecolors='black', linewidth=1,
               label=f'Cluster {label}' if label != -1 else 'Noise')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'DBSCAN Clusters (ε={best_result["eps"]}, minPts={best_result["min_samples"]})')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Confusion matrix style heatmap
plt.subplot(1, 3, 3)
# Remove the 'All' row and column for visualization
cross_tab_viz = cross_tab.iloc[:-1, :-1]
sns.heatmap(cross_tab_viz, annot=True, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Count'})
plt.title('Cluster Assignment vs True Labels')
plt.ylabel('True Label')
plt.xlabel('Cluster ID')

plt.tight_layout()
plt.savefig('../results/dbscan_best_result_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Best result comparison saved as 'dbscan_best_result_comparison.png'")
plt.show()

# ============================================================================
# 8. ANALYSIS OF NOISE POINTS
# ============================================================================
print("\n" + "="*80)
print("8. ANALYSIS OF NOISE POINTS")
print("="*80)

if best_result['n_noise'] > 0:
    noise_mask = best_labels == -1
    noise_true_labels = y_true[noise_mask]

    print(f"\nTotal noise points: {best_result['n_noise']}")
    print("\nTrue label distribution of noise points:")
    for class_id in np.unique(noise_true_labels):
        count = np.sum(noise_true_labels == class_id)
        percentage = (count / best_result['n_noise']) * 100
        print(f"  {target_names[class_id]:15}: {count:3d} points ({percentage:.1f}%)")

    # Visualize noise points
    fig = plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    # Plot all points
    plt.scatter(X_pca[~noise_mask, 0], X_pca[~noise_mask, 1],
               c='lightgray', s=50, alpha=0.3, label='Clustered points')
    # Highlight noise points
    for class_id in np.unique(y_true):
        mask = noise_mask & (y_true == class_id)
        if np.sum(mask) > 0:
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=100, alpha=0.8, edgecolors='black', linewidth=2,
                       label=f'Noise: {target_names[class_id]}')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Noise Points (colored by true label)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Show feature characteristics of noise points
    noise_features = X[noise_mask]
    regular_features = X[~noise_mask]

    feature_idx = 2  # Petal length
    plt.hist(regular_features[:, feature_idx], bins=20, alpha=0.5,
             label='Clustered points', color='blue')
    plt.hist(noise_features[:, feature_idx], bins=20, alpha=0.7,
             label='Noise points', color='red')
    plt.xlabel(feature_names[feature_idx])
    plt.ylabel('Frequency')
    plt.title('Feature Distribution: Clustered vs Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/dbscan_noise_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Noise analysis saved as 'dbscan_noise_analysis.png'")
    plt.show()
else:
    print("\nNo noise points detected with the best parameters.")

# ============================================================================
# 9. FINAL REPORT AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("9. FINAL REPORT AND RECOMMENDATIONS")
print("="*80)

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print(f"""
1. OPTIMAL PARAMETERS:
   • Best ε (epsilon): {best_result['eps']}
   • Best minPts: {best_result['min_samples']}
   • These parameters achieved ARI = {best_result['ari']:.4f}

2. CLUSTERING PERFORMANCE:
   • Number of clusters found: {best_result['n_clusters']} (true = 3)
   • Number of noise points: {best_result['n_noise']}
   • Silhouette score: {best_result['silhouette']:.4f}

3. COMPARISON WITH TRUE LABELS:
   • Adjusted Rand Index: {best_result['ari']:.4f}
   • Normalized Mutual Information: {best_result['nmi']:.4f}
   • V-Measure: {best_result['v_measure']:.4f}

4. PARAMETER SENSITIVITY:
   • Smaller ε → More noise points, fewer clusters
   • Larger ε → Fewer noise points, risk of merging clusters
   • Smaller minPts → More clusters, less noise
   • Larger minPts → Fewer clusters, more noise

5. DBSCAN ADVANTAGES FOR IRIS:
   • Can identify outliers (noise points)
   • Does not require specifying number of clusters
   • Can find clusters of arbitrary shape

6. DBSCAN LIMITATIONS FOR IRIS:
   • Sensitive to parameter selection
   • Struggles with clusters of varying density
   • IRIS classes have different densities (Setosa vs others)
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
