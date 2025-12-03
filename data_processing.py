import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from torch import cat
import seaborn as sns
import seaborn as sns



def features_valuecounts(df):
    """
    Draws bar plots for value counts of all categorical features in the DataFrame.
    Continuous/numerical columns are skipped automatically.
    """
    # تحديد الأعمدة التصنيفية فقط
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    n = len(categorical_features)
    

    cols = 4
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()
    
    for i, feature in enumerate(categorical_features):
        counts = df[feature].value_counts()
        axes[i].bar(counts.index.astype(str), counts.values)
        axes[i].set_title(feature, fontsize=9)
        axes[i].tick_params(axis='x', rotation=45)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout(pad=0.8)
    plt.show()




def features_distribution(df):
    """
    Plots histogram + KDE distribution for all numerical features in the DataFrame.
    """
    # اختيار الأعمدة العددية فقط
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    n = len(numerical_features)
    
    if n == 0:
        print("No numerical features to plot.")
        return

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(feature, fontsize=9)
    
    # إخفاء أي Subplots زيادة
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout(pad=1)
    plt.show()


def features_correlation(df, target):
    """
    Plots scatter plots of all numerical features against the target variable.
    """
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    # استبعاد العمود target نفسه إذا موجود بين numerical_features
    numerical_features = [f for f in numerical_features if f != target]
    
    n = len(numerical_features)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_features):
        sns.scatterplot(x=df[feature], y=df[target], ax=axes[i])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f"{feature} vs {target}", fontsize=9)
    
    # إخفاء أي Subplots زيادة
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout(pad=1)
    plt.show()




def encode_categorical_features(df):
    """
    Encode categorical features as integers.
    Real NaN values are preserved.
    Text 'None' is treated as a regular class.
    """
    
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(f"Encoding column: {col}")

        categories = df[col].unique()
        
        for i, category in enumerate(categories):
            df.loc[df[col] == category, col] = i
        
        
def imbalance_checking(df):
    """
    that function to check the class distribution of the target variable 'GradeClass' and visualize it using a bar plot.
    """
    target=df['GradeClass']
    print("Class Distribution:\n", target.value_counts())
    target.value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Grade Classes')
    plt.show()

def handle_imbalance(df):
    """
    that functions to downsample the majority classes to handle class imbalance be reducing the number of samples in the majority class (GradeClass 2) to match the median count of the other classes.
    """
    target=df['GradeClass']
    class_4=df[target==2]
    other_classes=df[target!=2]
    
    target_counts=other_classes['GradeClass'].value_counts().median()
    
    drop_count = len(class_4) - target_counts
    rand_index=(np.random.permutation(class_4.index))

    drop_indices = rand_index[:int(drop_count)]
    df_balanced = df.drop(drop_indices)
    df_balanced['GradeClass'] = df_balanced['GradeClass'].astype(int)
    return df_balanced

def balance_all_classes_resample(df):
    """
    Fully balance all classes by resampling each class to the median class size.
    - Upsamples minority classes if below median.
    - Downsamples majority classes if above median.
    """
    target_counts = df['GradeClass'].value_counts()
    median_count = int(target_counts.max())

    balanced_classes = []
    for cls, count in target_counts.items():
        df_cls = df[df['GradeClass'] == cls]
        df_resampled = resample(
            df_cls,
            replace=True if count < median_count else False,  
            n_samples=median_count,
            random_state=42
        )
        balanced_classes.append(df_resampled)

    df_balanced = pd.concat(balanced_classes).sample(frac=1, random_state=42).reset_index(drop=True)
    df_balanced['GradeClass'] = df_balanced['GradeClass'].astype(int)
        
    return df_balanced

