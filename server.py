from flask import Flask, render_template, request, jsonify, Response
from collections import Counter
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import plotly
import plotly.express as px
import json

def obtain_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df.dropna()
    df = df.drop(["id"], axis=1)
    return df

def categorical2dummy(df):
    df_encoded = df.copy()
    labelencoder = dict()
    categorical_col = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    # Convert categorical variables to numerical variables
    for feature in categorical_col:
        labelencoder[feature] = preprocessing.LabelEncoder()
        df_encoded[feature+'_n'] = labelencoder[feature].fit_transform(df_encoded[feature])
    df_encoded = df_encoded.drop(categorical_col, axis=1)
    return df_encoded

def train_XGB(df_encoded, validation_num, test_size):
    X_train_v, X_test, y_train_v, y_test = train_test_split(df_encoded.drop(['stroke'], axis=1), 
                                                        df_encoded['stroke'], 
                                                        train_size=1-test_size)
    bst_dict = dict()
    bst_AUC = []
    bst_graph =[]
    for i in range(validation_num):
        bst_dict[i] = XGBClassifier()
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_v, 
                                                            y_train_v, 
                                                            train_size=.8)
        bst_dict[i].fit(X_train, y_train)
        y_pred_proba = bst_dict[i].predict_proba(X_validation)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_validation, y_pred_proba)
        auc_ = metrics.roc_auc_score(y_validation, y_pred_proba)
        bst_AUC.append(auc_)
        bst_graph.append([fpr, tpr])
    bst = bst_dict[np.argmax(bst_AUC)]
    y_pred_proba = bst.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc_ = metrics.roc_auc_score(y_test, y_pred_proba)
    y_pred = bst.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    return confusion_matrix, auc_, fpr, tpr

def train_RForest(df_encoded, tree_num, validation_num, test_size):
    X_train_v, X_test, y_train_v, y_test = train_test_split(df_encoded.drop(['stroke'], axis=1), 
                                                        df_encoded['stroke'], 
                                                        train_size=1-test_size)
    rfst_dict = dict()
    rfst_AUC = []
    rfst_graph =[]
    for i in range(validation_num):
        rfst_dict[i] = RandomForestClassifier(n_estimators=tree_num)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_v, 
                                                            y_train_v, 
                                                            train_size=.8)
        rfst_dict[i].fit(X_train, y_train)
        y_pred_proba = rfst_dict[i].predict_proba(X_validation)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_validation, y_pred_proba)
        auc_ = metrics.roc_auc_score(y_validation, y_pred_proba)
        rfst_AUC.append(auc_)
        rfst_graph.append([fpr, tpr])
    rfst = rfst_dict[np.argmax(rfst_AUC)]
    y_pred_proba = rfst.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc_ = metrics.roc_auc_score(y_test, y_pred_proba)
    y_pred = rfst.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    return confusion_matrix, auc_, fpr, tpr

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/dataset")
def dataset():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df.head(20)
    return render_template('dataset.html', column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip)

@app.route("/analysis_stat")
def analysis1():
    return render_template('analysis_stat.html')

@app.route("/analysis_uni")
def analysis2():
    return render_template('analysis_uni.html')

@app.route("/analysis_biv")
def analysis3():
    return render_template('analysis_biv.html')

@app.route("/analysis_corr")
def analysis4():
    df = obtain_data()
    df_encode = categorical2dummy(df)
    cormat = df_encode.drop(['stroke'], axis=1).corr()
    fig = px.imshow(cormat)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('analysis_corr.html', graphJSON=graphJSON)

@app.route("/prediction")
def prediction():
    return render_template('prediction.html')

@app.route("/prediction/xgboost")
def xgboost():
    return render_template('xgboost.html')


@app.route("/prediction/xgboost/result", methods=["GET"])
def xgboost_rst():
    num_validation = request.args.get("num_validation", None)
    test_size = request.args.get("test_size", None)
    df = obtain_data()
    df_encode = categorical2dummy(df)
    confusion_matrix, auc, fpr, tpr = train_XGB(df_encode, int(num_validation), float(test_size))
    fig1 = px.line(x=fpr, y=tpr, labels={'x':'False Positive Fraction', 'y':'True Positive Fraction'})
    fig2 = px.imshow(confusion_matrix, x=['False', 'True'], y=['False', 'True'], labels={'x':'Predicted Label', 'y':'True Label'})
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('xgboost_rst.html', num_validation=num_validation, test_size=test_size, auc=auc, graphJSON1=graphJSON1, graphJSON2=graphJSON2)

@app.route("/prediction/rforest")
def rforest():
    return render_template('rforest.html')


@app.route("/prediction/rforest/result", methods=["GET"])
def rforest_rst():
    num_tree = request.args.get("num_tree", None)
    num_validation = request.args.get("num_validation", None)
    test_size = request.args.get("test_size", None)
    df = obtain_data()
    df_encode = categorical2dummy(df)
    confusion_matrix, auc, fpr, tpr = train_RForest(df_encode, int(num_tree), int(num_validation), float(test_size))
    fig1 = px.line(x=fpr, y=tpr, labels={'x':'False Positive Fraction', 'y':'True Positive Fraction'})
    fig2 = px.imshow(confusion_matrix, x=['False', 'True'], y=['False', 'True'], labels={'x':'Predicted Label', 'y':'True Label'})
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('rforest_rst.html', num_validation=num_validation, test_size=test_size, num_tree=num_tree, auc=auc, graphJSON1=graphJSON1, graphJSON2=graphJSON2)



if __name__ == "__main__":
    app.run()
