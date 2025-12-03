from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def split_data(df):
    X = df.drop(['GradeClass', 'StudentID'], axis=1)
    y = df['GradeClass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def Evaluate_Model(model, X_train, X_test, y_train, y_test):
    y_tr_predict = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_tr_predict)
    print(f'Train Accuracy: {train_acc:.2f}')
    
    y_ts_predict = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_ts_predict)
    print(f'Test Accuracy: {test_acc:.2f}')
    
    cm = confusion_matrix(y_test, y_ts_predict)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(model.__class__.__name__ + ' Confusion Matrix')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    report_dict = classification_report(y_test, y_ts_predict, output_dict=True)
    acc_feature=pd.DataFrame({model.__class__.__name__:[train_acc,test_acc]},index=['Train Accuracy','Test Accuracy'])
    report_df = pd.DataFrame(report_dict).transpose()
    return report_df ,acc_feature

def Modeling(Model_class, df, **params):
    X_train, X_test, y_train, y_test = split_data(df)    
    model = Model_class(**params)  
    model.fit(X_train, y_train)
    report,acc_feature=Evaluate_Model(model, X_train, X_test, y_train, y_test)
    report=report[['precision', 'recall', 'f1-score']]
    report=report.iloc[:-3 , :]
    return report ,acc_feature


def show_reports_plot(*reports):
    metrics = ['precision', 'recall', 'f1-score']
    n_reports = len(reports)
    bar_width = 0.15
    x = np.arange(len(reports[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(metrics):
        for j, report in enumerate(reports):
            axes[i].bar(
                x + j * bar_width,
                report[metric],
                width=bar_width,
                label=f'Model {j+1}'
            )

        axes[i].set_xticks(x + bar_width * (n_reports - 1) / 2)
        axes[i].set_xticklabels(reports[0].index)
        axes[i].set_title(f"{metric} Comparison")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend(loc='lower right')
        axes[i].set_ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

def accuracy_comparison_plot(*accuracy_tables):
    merged = pd.concat(accuracy_tables, axis=1)
    merged.plot(kind='bar', figsize=(10, 6))
    
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy Comparison")
    plt.ylim(0, 1.1)
    plt.legend( loc='lower right')
    plt.tight_layout()
    plt.show()
    
# # Example usage
# report1,randforest_acc=Modeling(RandomForestClassifier, df, random_state=42,n_estimators=10)
# report2,decTree_acc=Modeling(DecisionTreeClassifier, df)
# report3,Gr_acc=Modeling(GradientBoostingClassifier, df)
# report4,SVM_acc=Modeling(KNeighborsClassifier, df, n_neighbors=5)
# report5,LG_acc=Modeling(LogisticRegression, df, max_iter=1500)

# accuracy_comparison_plot(randforest_acc, decTree_acc , Gr_acc, SVM_acc,LG_acc)
# show_reports_plot(report1, report2 , report3, report4,report5)
