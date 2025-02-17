import numpy as np
import pandas as pd
from matplotlib import pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load dataset from CSV file
dataset_path = r'C:\Users\rrpra\OneDrive\Documents\GitHub\FDA-practical-\bengaluru_house_prices.csv'
dataset = pd.read_csv(dataset_path)

# Display basic information
print(dataset.info())
print(dataset.head())
print(dataset.describe())

# Define feature (X) and target (y) columns
x = dataset.select_dtypes(include=[np.number]).iloc[:, :-1]  # Only numeric columns
y = dataset.iloc[:, -1]

# Boxplot for each feature
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
axes = axes.flatten()

for i, name in enumerate(x.columns):
    if i < len(axes):  # Avoid index errors if columns < 8
        axes[i].boxplot(x[name].dropna())  # Drop NaNs to avoid errors
        axes[i].set_xlabel(name)

plt.tight_layout()
plt.show()

# Violin plot for each feature
plt.figure(figsize=(10, 6))
numeric_x = x.select_dtypes(include=np.number)  # Filter only numeric columns
sns.violinplot(data=numeric_x)
plt.xticks(rotation=90)
plt.title("Violin Plot of Features")
plt.show()

# Crosstab analysis with pie chart
categorical_columns = dataset.select_dtypes(exclude=np.number).columns  # Identify categorical columns
if len(categorical_columns) > 0:
    categorical_column = categorical_columns[0]  # Use the first categorical column
    tab = pd.crosstab(index=dataset[categorical_column], columns='count')
    print(tab)
    
    tab.plot.pie(subplots=True, autopct='%1.1f%%', figsize=(8, 8), legend=False)
    plt.title(f'Pie Chart of {categorical_column}')
    plt.ylabel('')  # Hide y-label
    plt.show()
else:
    print("No categorical columns found for crosstab pie chart.")

# Cylindrical Bar Graph Example
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 8, 20]

fig, ax = plt.subplots()
bars = ax.bar(range(len(categories)), values, width=0.5, align='edge')  

# Add shadow effect
for bar in bars:
    x = bar.get_x()
    y = bar.get_height()
    ax.fill([x, x + bar.get_width(), x + bar.get_width(), x], 
            [y, y, y - 0.5, y - 0.5], color='gray', alpha=0.5)

# Set labels
plt.xticks(range(len(categories)), categories)
plt.ylabel('Values')
plt.title('Cylindrical Bar Chart')
plt.show()