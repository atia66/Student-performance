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

