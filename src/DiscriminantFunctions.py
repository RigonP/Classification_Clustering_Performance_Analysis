import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import inv
import os

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# Data
A = np.array([[2, 2, 4, 4],
              [6, 0, 6, 0]])

B = np.array([[8, 8, 10, 10],
              [2, 4, 2, 4]])

# Format (n_samples, n_features)
A_T = A.T  # 4x2
B_T = B.T  # 4x2

print("=" * 60)
print("MANUAL CALCULATIONS")
print("=" * 60)

# Mean vector
mean_A = np.mean(A_T, axis=0)
mean_B = np.mean(B_T, axis=0)

print("\n1. Mean vectors:")
print(f"   μ_A = {mean_A}")
print(f"   μ_B = {mean_B}")

# Covariance Matrices
cov_A = np.cov(A_T.T)
cov_B = np.cov(B_T.T)

print("\n2. Covariance Matrices:")
print(f"   Σ_A = \n{cov_A}")
print(f"\n   Σ_B = \n{cov_B}")

# Pooled covariances
n_A = A_T.shape[0]
n_B = B_T.shape[0]
cov_pooled = ((n_A - 1) * cov_A + (n_B - 1) * cov_B) / (n_A + n_B - 2)

print(f"\n3. Pooled Covariance Matrix:")
print(f"   Σ_pooled = \n{cov_pooled}")

# Discriminant function coefficients
# g_A(x) - g_B(x) = w^T*x + w_0
# ku w = Σ^(-1) * (μ_A - μ_B)

cov_inv = inv(cov_pooled)
w = cov_inv @ (mean_A - mean_B)
w_0 = -0.5 * (mean_A.T @ cov_inv @ mean_A - mean_B.T @ cov_inv @ mean_B)

print(f"\n4. Discriminant Function Coefficients:")
print(f"   w = {w}")
print(f"   w_0 = {w_0}")
print(f"\n   Function: g(x) = {w[0]:.4f}*x₁ + {w[1]:.4f}*x₂ + {w_0:.4f}")
print(f"   Decision Boundary: g(x) = 0")

# Perdorimi i sklearn LDA
print("\n" + "=" * 60)
print("RESULTS WITH SKLEARN (Python Library)")
print("=" * 60)

# Prepare data for LDA
X = np.vstack([A_T, B_T])
y = np.array([0] * n_A + [1] * n_B)  # 0 for class A, 1 for class B

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

print(f"\n5. LinearDiscriminantAnalysis Results:")
print(f"   Means: \n{lda.means_}")

try:
    print(f"   Covariance: \n{lda.covariance_}")
except AttributeError:
    print(f"   Covariance: \n{cov_pooled} (manual calculations)")

print(f"   Coefficients: {lda.coef_[0]}")
print(f"   Intercept: {lda.intercept_[0]}")

# 6. VISUALIZATION
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(12, 10))

# Plot 1: Points and means
plt.subplot(2, 2, 1)
plt.scatter(A[0], A[1], c='blue', s=100, marker='o',
            label='Class A', edgecolors='darkblue', linewidths=2)
plt.scatter(B[0], B[1], c='red', s=100, marker='s',
            label='Class B', edgecolors='darkred', linewidths=2)

# Means
plt.scatter(mean_A[0], mean_A[1], c='blue', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_A', zorder=5)
plt.scatter(mean_B[0], mean_B[1], c='red', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_B', zorder=5)

# Line connecting means
plt.plot([mean_A[0], mean_B[0]], [mean_A[1], mean_B[1]],
         'k--', linewidth=2, label='Connection μ_A - μ_B')

plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Points and Means', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: With decision boundary (manual)
plt.subplot(2, 2, 2)
plt.scatter(A[0], A[1], c='blue', s=100, marker='o',
            label='Class A', edgecolors='darkblue', linewidths=2)
plt.scatter(B[0], B[1], c='red', s=100, marker='s',
            label='Class B', edgecolors='darkred', linewidths=2)

# Means
plt.scatter(mean_A[0], mean_A[1], c='blue', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_A', zorder=5)
plt.scatter(mean_B[0], mean_B[1], c='red', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_B', zorder=5)

# Decision boundary (manual calculation)
# g(x) = w[0]*x₁ + w[1]*x₂ + w_0 = 0
# If w[1] is close to zero, boundary is vertical
if abs(w[1]) < 0.01:
    # Vertical boundary: x₁ = -w_0/w[0]
    x1_boundary = -w_0 / w[0]
    plt.axvline(x=x1_boundary, color='purple', linewidth=3,
                label=f'Boundary: x₁ = {x1_boundary:.2f}', linestyle='--')
else:
    # General boundary
    x1_range = np.linspace(0, 12, 100)
    x2_boundary = -(w[0] * x1_range + w_0) / w[1]
    plt.plot(x1_range, x2_boundary, 'purple', linewidth=3,
             label='Decision boundary (manual)', linestyle='--')

plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Decision Boundary (Manual)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(-1, 7)

# Plot 3: With decision boundary (sklearn)
plt.subplot(2, 2, 3)
plt.scatter(A[0], A[1], c='blue', s=100, marker='o',
            label='Class A', edgecolors='darkblue', linewidths=2)
plt.scatter(B[0], B[1], c='red', s=100, marker='s',
            label='Class B', edgecolors='darkred', linewidths=2)

# Means
plt.scatter(mean_A[0], mean_A[1], c='blue', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_A', zorder=5)
plt.scatter(mean_B[0], mean_B[1], c='red', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_B', zorder=5)

# Decision Boundary (sklearn)
if abs(lda.coef_[0][1]) < 0.01:
    # Vertical boundary
    x1_boundary_sklearn = -lda.intercept_[0] / lda.coef_[0][0]
    plt.axvline(x=x1_boundary_sklearn, color='green', linewidth=3,
                label=f'Boundary: x₁ = {x1_boundary_sklearn:.2f}', linestyle='-.')
else:
    x1_range = np.linspace(0, 12, 100)
    x2_boundary_sklearn = -(lda.coef_[0][0] * x1_range + lda.intercept_[0]) / lda.coef_[0][1]
    plt.plot(x1_range, x2_boundary_sklearn, 'green', linewidth=3,
             label='Decision Boundary (sklearn)', linestyle='-.')

plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Decision Boundary (sklearn LDA)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(-1, 7)

# Plot 4: Both boundaries together for comparison
plt.subplot(2, 2, 4)
plt.scatter(A[0], A[1], c='blue', s=100, marker='o',
            label='Class A', edgecolors='darkblue', linewidths=2)
plt.scatter(B[0], B[1], c='red', s=100, marker='s',
            label='Class B', edgecolors='darkred', linewidths=2)

# Means
plt.scatter(mean_A[0], mean_A[1], c='blue', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_A', zorder=5)
plt.scatter(mean_B[0], mean_B[1], c='red', s=300, marker='*',
            edgecolors='black', linewidths=2, label='μ_B', zorder=5)

# Both boundaries
if abs(w[1]) < 0.01:
    x1_boundary = -w_0 / w[0]
    plt.axvline(x=x1_boundary, color='purple', linewidth=2,
                label='Manual', linestyle='--', alpha=0.7)
    x1_boundary_sklearn = -lda.intercept_[0] / lda.coef_[0][0]
    plt.axvline(x=x1_boundary_sklearn, color='green', linewidth=2,
                label='sklearn', linestyle='-.', alpha=0.7)
else:
    x1_range = np.linspace(0, 12, 100)
    x2_boundary = -(w[0] * x1_range + w_0) / w[1]
    x2_boundary_sklearn = -(lda.coef_[0][0] * x1_range + lda.intercept_[0]) / lda.coef_[0][1]
    plt.plot(x1_range, x2_boundary, 'purple', linewidth=2,
             label='Manual', linestyle='--', alpha=0.7)
    plt.plot(x1_range, x2_boundary_sklearn, 'green', linewidth=2,
             label='sklearn', linestyle='-.', alpha=0.7)

plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Comparison: Manual vs sklearn', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(-1, 7)


plt.tight_layout()
plt.savefig('../results/discriminant_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Plot saved as 'discriminant_analysis.png'")
plt.show()

# 7. COMPARISON AND DISCUSSION
print("\n" + "=" * 60)
print("COMPARISON AND DISCUSSION OF RESULTS")
print("=" * 60)

print("\n1. Coefficient comparison:")
print(f"   Manual: w = {w}, w_0 = {w_0:.4f}")
print(f"   sklearn: w = {lda.coef_[0]}, w_0 = {lda.intercept_[0]:.4f}")
print(f"   Difference: Δw = {w - lda.coef_[0]}")

print("\n2. Mean comparison:")
print(f"   Manual μ_A = {mean_A}")
print(f"   sklearn μ_A = {lda.means_[0]}")
print(f"   Manual μ_B = {mean_B}")
print(f"   sklearn μ_B = {lda.means_[1]}")

# Calculate decision boundary
if abs(w[1]) < 0.01:
    boundary_x1 = -w_0 / w[0]
    print(f"\n3. The decision boundary is a vertical line at x₁ = {boundary_x1:.4f}")
else:
    print(f"\n3. Decision boundary: {w[0]:.4f}*x₁ + {w[1]:.4f}*x₂ + {w_0:.4f} = 0")

print("\n4. Conclusions:")
print("   • Manual and sklearn results are identical (or very close)")
print("   • The decision boundary optimally separates the two classes")
print("   • The line passes through the middle of the connection between μ_A and μ_B")
print("   • The line is perpendicular to the vector connecting μ_A and μ_B")
print(f"   • Distance between means: {np.linalg.norm(mean_A - mean_B):.4f}")
print(f"   • Class A has greater variance in the x₂ direction (σ²={cov_A[1, 1]:.2f})")
print(f"   • Class B has equal variance in both directions (σ²={cov_B[1, 1]:.2f})")

# 8. Classification testing
print("\n5. Classification testing:")
test_points = np.array([[3, 3], [9, 3], [6, 3], [5, 2], [7, 4]])
predictions_manual = []
predictions_sklearn = lda.predict(test_points)

for point in test_points:
    g_val = w[0] * point[0] + w[1] * point[1] + w_0
    pred = 'A' if g_val > 0 else 'B'
    predictions_manual.append(pred)

print("\n   Point     | Manual | sklearn | Value g(x)")
print("   " + "-" * 52)
for i, point in enumerate(test_points):
    g_val = w[0] * point[0] + w[1] * point[1] + w_0
    sklearn_class = 'A' if predictions_sklearn[i] == 0 else 'B'
    match = "✓" if predictions_manual[i] == sklearn_class else "✗"
    print(f"   {point}  |   {predictions_manual[i]}    |    {sklearn_class}    | {g_val:8.4f}  {match}")

print("\n" + "=" * 60)
print("CONCLUSION")

