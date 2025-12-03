import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plotting_features_relation(df):
    """
    This function visualizes the correlation between numerical features in the dataset using a heatmap.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    plt.title('Feature Correlation Heatmap')
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.show()

def features_distribution(df):
    """
    Plots histogram + KDE distribution for all numerical features in the DataFrame.
    """
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    n = len(numerical_features)
    
    if n == 0:
        print("No numerical features to plot.")
        return

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(feature, fontsize=9)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout(pad=1)
    plt.show()


def features_boxplots(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    n = len(numeric_cols)
    cols = 5
    rows = (n + cols - 1) // cols 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_cols):
        sns.boxplot(x=df[feature], ax=axes[i])
        axes[i].set_title(f'Boxplot of {feature}', fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.show()

