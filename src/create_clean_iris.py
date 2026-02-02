import pandas as pd
from sklearn import datasets

# Load the original sklearn iris dataset
iris = datasets.load_iris()

# Create DataFrame with sklearn data
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['class'] = iris.target

# Map target numbers to class names
class_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
df['class'] = df['class'].map(class_names)

# Save to CSV in the correct format
df.to_csv('../data/iris_clean.data',
          header=False,
          index=False,
          columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

print("✓ Created clean iris_clean.data file with sklearn's exact data")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())