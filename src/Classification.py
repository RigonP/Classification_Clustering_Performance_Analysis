import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (confusion_matrix, classification_report,
                            accuracy_score, precision_score, recall_score,
                            f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from itertools import combinations
import warnings
import os
warnings.filterwarnings('ignore')

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

print("="*80)
print("Leave-One-Out Cross-Validation on IRIS Dataset")
print("="*80)

# LOADING AND EXPLORING THE DATA
print("\n" + "="*80)
print("1. LOADING AND EXPLORING THE IRIS DATASET")
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
X = df[column_names[:-1]].values
y = df['target'].values
feature_names = column_names[:-1]
target_names = ['setosa', 'versicolor', 'virginica']

print(f"\nDataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {feature_names}")
print(f"Target classes: {target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Target_name column for display
df['target_name'] = df['target'].map({0: target_names[0],
                                       1: target_names[1],
                                       2: target_names[2]})

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())


# SIMPLE SCATTER PLOT (PETAL LENGTH vs PETAL WIDTH)
print("\n" + "="*80)
print("1.5. PETAL SCATTER PLOT VISUALIZATION")
print("="*80)

plt.figure(figsize=(10, 8))
for class_id in np.unique(y):
    plt.scatter(
        X[y == class_id, 2],  # Petal Length
        X[y == class_id, 3],  # Petal Width
        label=target_names[class_id],
        alpha=0.7,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
plt.xlabel("Petal Length (cm)", fontsize=12)
plt.ylabel("Petal Width (cm)", fontsize=12)
plt.legend(fontsize=11)
plt.title("Scatter Plot of IRIS Dataset - Petal Features", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/iris_petal_scatter.png', dpi=300, bbox_inches='tight')
print("\n✓ Petal scatter plot saved as '../results/iris_petal_scatter.png'")
plt.close()


# DATA VISUALIZATION
print("\n" + "="*80)
print("2. DATA VISUALIZATION")
print("="*80)

# Changed from 3x3 to 3x4 grid to accommodate all plots
fig = plt.figure(figsize=(20, 12))

# Scatter plot matrix (6 plots)
feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for idx, (i, j) in enumerate(feature_pairs, 1):
    plt.subplot(3, 4, idx)
    for target_idx, target_name in enumerate(target_names):
        mask = y == target_idx
        plt.scatter(X[mask, i], X[mask, j], label=target_name, alpha=0.6, s=50)
    plt.xlabel(feature_names[i])
    plt.ylabel(feature_names[j])
    plt.legend()
    plt.grid(True, alpha=0.3)

# Feature distributions 4 plots one for each feature
for idx, feature_idx in enumerate(range(4)):
    plt.subplot(3, 4, 9 + idx)
    for target_idx, target_name in enumerate(target_names):
        mask = y == target_idx
        plt.hist(X[mask, feature_idx], alpha=0.5, label=target_name, bins=15)
    plt.xlabel(feature_names[feature_idx])
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/iris_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as '../results/iris_visualization.png'")
plt.close()


# FEATURE SELECTION ANALYSIS
print("\n" + "="*80)
print("3. FEATURE SELECTION ANALYSIS")
print("="*80)

# Method 1: ANOVA F-statistic
selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X, y)
f_scores = selector_f.scores_

# Method 2: Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
selector_mi.fit(X, y)
mi_scores = selector_mi.scores_

print("\nFeature Importance Scores:")
print("\nFeature Name                    | ANOVA F-score | Mutual Info")
print("-" * 65)
for i, name in enumerate(feature_names):
    print(f"{name:30} | {f_scores[i]:12.4f} | {mi_scores[i]:11.4f}")

# Rank features
feature_ranking_f = np.argsort(f_scores)[::-1]
feature_ranking_mi = np.argsort(mi_scores)[::-1]

print("\nFeature Ranking (by ANOVA F-score):")
for rank, idx in enumerate(feature_ranking_f, 1):
    print(f"  {rank}. {feature_names[idx]} (score: {f_scores[idx]:.4f})")

print("\nFeature Ranking (by Mutual Information):")
for rank, idx in enumerate(feature_ranking_mi, 1):
    print(f"  {rank}. {feature_names[idx]} (score: {mi_scores[idx]:.4f})")


# IMPLEMENTING LEAVE-ONE-OUT CROSS-VALIDATION
print("\n" + "="*80)
print("4. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)")
print("="*80)

def perform_loocv(X, y, classifier, classifier_name):
    """
    Perform Leave-One-Out Cross-Validation
    """
    loo = LeaveOneOut()
    n_samples = X.shape[0]

    y_true = []
    y_pred = []

    print(f"\nPerforming LOOCV for {classifier_name}...")
    print(f"Training {n_samples} times (leaving one sample out each time)...")

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train and predict
        classifier.fit(X_train_scaled, y_train)
        pred = classifier.predict(X_test_scaled)

        y_true.append(y_test[0])
        y_pred.append(pred[0])

    return np.array(y_true), np.array(y_pred)


# DEFINE CLASSIFIERS
print("\n" + "="*80)
print("5. DEFINING CLASSIFIERS")
print("="*80)

classifiers = {
    'SVM (Linear)': SVC(kernel='linear', C=1.0, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'SVM (Poly)': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

print("\nClassifiers to be evaluated:")
for name, clf in classifiers.items():
    print(f"  • {name}: {clf}")


# EVALUATE WITH ALL FEATURES
print("\n" + "="*80)
print("6. EVALUATION WITH ALL FEATURES (LOOCV)")
print("="*80)

results_all_features = {}

for clf_name, clf in classifiers.items():
    print(f"\n{'='*70}")
    print(f"Evaluating: {clf_name}")
    print(f"{'='*70}")

    y_true, y_pred = perform_loocv(X, y, clf, clf_name)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    results_all_features[clf_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred
    }

    print(f"\nResults:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))


# FEATURE SUBSET EVALUATION
print("\n" + "="*80)
print("7. EVALUATION WITH DIFFERENT FEATURE SUBSETS")
print("="*80)

# Test different feature combinations
feature_subsets = {
    'Top 2 Features': feature_ranking_f[:2],
    'Top 3 Features': feature_ranking_f[:3],
    'All Features': [0, 1, 2, 3],
    'Petal Features': [2, 3],  # petal length and width
    'Sepal Features': [0, 1],  # sepal length and width
}

results_by_features = {}

print("\nTesting different feature subsets with best classifier...")

# Best classifier
best_clf_name = max(results_all_features,
                    key=lambda x: results_all_features[x]['accuracy'])
best_clf = classifiers[best_clf_name]

print(f"Best classifier: {best_clf_name}")
print(f"Accuracy with all features: {results_all_features[best_clf_name]['accuracy']:.4f}")

for subset_name, feature_indices in feature_subsets.items():
    print(f"\n{'-'*70}")
    print(f"Testing: {subset_name}")
    print(f"Features: {[feature_names[i] for i in feature_indices]}")

    X_subset = X[:, feature_indices]

    y_true, y_pred = perform_loocv(X_subset, y,
                                    type(best_clf)(**best_clf.get_params()),
                                    subset_name)

    accuracy = accuracy_score(y_true, y_pred)
    results_by_features[subset_name] = accuracy

    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")


# PARAMETER TUNING FOR BEST CLASSIFIER
print("\n" + "="*80)
print("8. PARAMETER TUNING")
print("="*80)

if 'SVM' in best_clf_name:
    print("\nTuning SVM parameters...")
    param_results = {}

    # Test different C values
    C_values = [0.1, 1.0, 10.0, 100.0]
    for C in C_values:
        clf = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
        y_true, y_pred = perform_loocv(X, y, clf, f"SVM (C={C})")
        accuracy = accuracy_score(y_true, y_pred)
        param_results[f'C={C}'] = accuracy
        print(f"  C={C:6.1f}: Accuracy = {accuracy:.4f}")

    # Test different gamma values
    gamma_values = ['scale', 'auto', 0.1, 1.0]
    for gamma in gamma_values:
        clf = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
        y_true, y_pred = perform_loocv(X, y, clf, f"SVM (gamma={gamma})")
        accuracy = accuracy_score(y_true, y_pred)
        param_results[f'gamma={gamma}'] = accuracy
        print(f"  gamma={gamma:6}: Accuracy = {accuracy:.4f}")

elif 'Random Forest' in best_clf_name:
    print("\nTuning Random Forest parameters...")
    param_results = {}

    # Test different n_estimators
    n_est_values = [10, 50, 100, 200]
    for n_est in n_est_values:
        clf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        y_true, y_pred = perform_loocv(X, y, clf, f"RF (n_est={n_est})")
        accuracy = accuracy_score(y_true, y_pred)
        param_results[f'n_estimators={n_est}'] = accuracy
        print(f"  n_estimators={n_est:3d}: Accuracy = {accuracy:.4f}")


# VISUALIZATION OF RESULTS
print("\n" + "="*80)
print("9. CREATING VISUALIZATIONS")
print("="*80)

# Comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# Plot 1: Comparison of classifiers (all features)
plt.subplot(2, 3, 1)
clf_names = list(results_all_features.keys())
accuracies = [results_all_features[name]['accuracy'] for name in clf_names]
colors = plt.cm.viridis(np.linspace(0, 1, len(clf_names)))
bars = plt.bar(range(len(clf_names)), accuracies, color=colors)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Comparison (All Features, LOOCV)')
plt.xticks(range(len(clf_names)), clf_names, rotation=45, ha='right')
plt.ylim([0.9, 1.0])
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Feature subset comparison
plt.subplot(2, 3, 2)
subset_names = list(results_by_features.keys())
subset_accs = list(results_by_features.values())
colors = plt.cm.plasma(np.linspace(0, 1, len(subset_names)))
bars = plt.bar(range(len(subset_names)), subset_accs, color=colors)
plt.xlabel('Feature Subset')
plt.ylabel('Accuracy')
plt.title(f'Feature Subset Comparison ({best_clf_name})')
plt.xticks(range(len(subset_names)), subset_names, rotation=45, ha='right')
plt.ylim([0.8, 1.0])
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(subset_accs):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Feature importance
plt.subplot(2, 3, 3)
x_pos = np.arange(len(feature_names))
plt.barh(x_pos, f_scores, color='steelblue', alpha=0.7, label='ANOVA F-score')
plt.yticks(x_pos, feature_names)
plt.xlabel('Score')
plt.title('Feature Importance (ANOVA F-score)')
plt.grid(True, alpha=0.3, axis='x')
plt.legend()

# Plot 4-6: Confusion matrices for top 3 classifiers
top_3_classifiers = sorted(results_all_features.items(),
                           key=lambda x: x[1]['accuracy'],
                           reverse=True)[:3]

for idx, (clf_name, results) in enumerate(top_3_classifiers, 4):
    plt.subplot(2, 3, idx)
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix: {clf_name}\nAcc: {results["accuracy"]:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('../results/classification_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Results visualization saved as '../results/classification_results.png'")
plt.close()


# SUMMARY REPORT
print("\n" + "="*80)
print("10. SUMMARY REPORT")
print("="*80)

print("\n" + "="*80)
print("CLASSIFICATION PERFORMANCE SUMMARY (LOOCV)")
print("="*80)

# Create summary table
print(f"\n{'Classifier':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
print("-" * 75)
for clf_name in sorted(results_all_features.keys(),
                       key=lambda x: results_all_features[x]['accuracy'],
                       reverse=True):
    res = results_all_features[clf_name]
    print(f"{clf_name:<25} {res['accuracy']:<12.4f} {res['precision']:<12.4f} "
          f"{res['recall']:<12.4f} {res['f1']:<12.4f}")

print("\n" + "="*80)
print("FEATURE SUBSET PERFORMANCE")
print("="*80)
print(f"\n{'Feature Subset':<25} {'Accuracy':<12} {'Difference from All':<20}")
print("-" * 60)
all_features_acc = results_by_features['All Features']
for subset_name in sorted(results_by_features.keys(),
                         key=lambda x: results_by_features[x],
                         reverse=True):
    acc = results_by_features[subset_name]
    diff = acc - all_features_acc
    sign = '+' if diff >= 0 else ''
    print(f"{subset_name:<25} {acc:<12.4f} {sign}{diff:.4f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

best_acc = max(results_all_features.values(), key=lambda x: x['accuracy'])
best_classifier = max(results_all_features.items(),
                     key=lambda x: x[1]['accuracy'])[0]

print(f"""
1. BEST CLASSIFIER: {best_classifier}
   - Accuracy: {best_acc['accuracy']:.4f} ({best_acc['accuracy']*100:.2f}%)
   - This classifier achieved the highest accuracy using LOOCV

2. FEATURE IMPORTANCE:
   - Most important: {feature_names[feature_ranking_f[0]]}
   - Second: {feature_names[feature_ranking_f[1]]}
   - Petal features (length and width) are more discriminative than sepal features

3. FEATURE SUBSET ANALYSIS:
   - Best subset: {max(results_by_features, key=results_by_features.get)}
   - Accuracy: {max(results_by_features.values()):.4f}
   - Using fewer features can sometimes improve performance by reducing overfitting

4. COMPUTATIONAL EFFICIENCY:
   - LOOCV performed {X.shape[0]} iterations (one for each sample)
   - This provides an unbiased estimate of generalization performance
   - For large datasets, k-fold CV (e.g., 10-fold) is recommended instead

5. CLASS SEPARABILITY:
   - Setosa is perfectly separable from other classes
   - Versicolor and Virginica have some overlap, causing misclassifications
   - This is visible in the confusion matrices
""")
