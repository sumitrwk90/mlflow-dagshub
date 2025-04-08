import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import pickle


# # remote tracking info
# import dagshub
# dagshub.init(repo_owner='sumitrwk90', repo_name='mlflow-dagshub', mlflow=True)

# # Autologging
# mlflow.set_experiment("gb_exp_3_autologging")

# Mannual logging
mlflow.set_experiment("hyper_rscv_exp_08_04_2025_1")

# # remote tracking link
# mlflow.set_tracking_uri("https://dagshub.com/sumitrwk90/mlflow-dagshub.mlflow")

# solo tracking url
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = pd.read_csv(r"C:\Users\Lenovo\mlflow\data\water_potability.csv")
data.isnull().sum()

from sklearn.model_selection import RandomizedSearchCV, train_test_split
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
import pickle

# # Numpy array formate
# X_train = train_processed_data.iloc[:,0:-1].values
# y_train = train_processed_data.iloc[:,-1].values

# Pandas DataFrame formate
X_train = train_processed_data.drop(columns=['Potability'], axis=1)
y_train = train_processed_data['Potability']

# n_estimators: int=1000
# learning_rate: float=0.001
# max_depth: int=3

# mlflow.autolog()



# Defines the model
rf = RandomForestClassifier(random_state=42)
param_dist ={
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40]
}

random_search = RandomizedSearchCV(estimator = rf, param_distributions= param_dist, cv=5, verbose=2)

with mlflow.start_run(ru_name = "Random forest Tunning") as parent_run:

    random_search.fit(X_train,y_train)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
            mlflow.log_params(random_search.cv_results_['params']['i'])
            mlflow.log_metrics("mean_test_score", random_search.cv_results_['mean_test_score']['i'])

    # print best parameters
    print("Best parameters: ", random_search.best_params_)

    #log best param
    mlflow.log_params(random_search.best_params_)

    # train best model with best params
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # save 
    pickle.dump(random_search, open("model.pkl","wb"))

    # X_test = test_processed_data.iloc[:,0:-1].values
    # y_test = test_processed_data.iloc[:,-1].values

    X_test = test_processed_data.drop(columns=['Potability'], axis=1)
    y_test = test_processed_data['Potability']

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    model = pickle.load(open('model.pkl',"rb"))

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_score = f1_score(y_test,y_pred)

    mlflow.log_metric("acc",acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score",f1_score)

    # mlflow.log_param("n_estimators", n_estimators)
    # mlflow.log_param("learning_rate", learning_rate)
    # mlflow.log_param("max_depth", max_depth)

    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    # data logging
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")


    # log model
    mlflow.sklearn.log_model(random_search.best_estimator_, "RandomSearchCV")

    # log file
    mlflow.log_artifact(__file__)

    # set tags
    mlflow.set_tag("auther", "Sumit Kumar")
    mlflow.set_tag("model", "RSCV")

    print("acc",acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score",f1_score)