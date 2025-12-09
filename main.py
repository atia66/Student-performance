import test
from modeling import Modeling, show_reports_plot, accuracy_comparison_plot

from EDA import plotting_features_relation, features_boxplots

from data_processing import features_valuecounts, features_distribution, encode_categorical_features, imbalance_checking, handle_imbalance ,balance_all_classes_resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
df = pd.read_csv('./Balanced_Student_performance.csv')


## first data processing

def Data_Processing(df):
    
    features_valuecounts(df)

    features_distribution(df)

    encode_categorical_features(df)

    imbalance_checking(df)
    df_balanced=handle_imbalance(df)
    imbalance_checking(df_balanced)
    # df_balanced2=balance_all_classes_resample(df)
    # imbalance_checking(df_balanced2)
    return df_balanced
def EDA(df):
    plotting_features_relation(df)
    features_boxplots(df)

def testing(df):
    

    report1,randforest_acc=Modeling(RandomForestClassifier, df, random_state=42,n_estimators=10)
    report2,decTree_acc=Modeling(DecisionTreeClassifier, df)
    report3,Gr_acc=Modeling(GradientBoostingClassifier, df)
    report4,SVM_acc=Modeling(KNeighborsClassifier, df, n_neighbors=5)
    report5,LG_acc=Modeling(SVC, df, kernel='poly', C=8.0)

    accuracy_comparison_plot(randforest_acc, decTree_acc , Gr_acc, SVM_acc,LG_acc)
    show_reports_plot(report1, report2 , report3, report4,report5)
df_balanced2=Data_Processing(df)
EDA(df_balanced2)
testing(df_balanced2)