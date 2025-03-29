#!/usr/bin/env python3
"""Main file to trigger the project."""
###############################################################################
# Project - Customer Churn Prediction Model
# Filename - Customer_Churn_Prediction.py
# Arguments - Datafile, BinaryClassificationColumnArray, CategoricalColumnArray
#             DropColumnArray
# Created By - Vivek Salgia
# Creation Date - 03/28/2025
# Reviewed By -
# Reviewed Date -
# Change logs -
# Version   Date         Type   Changed By                  Comments
# =======   ============ ====   ==============  ===============================
# 1.0       03/28/2025   I     Vivek Salgia    Initial Creation
###############################################################################

import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Train_Model import learnandpredict
import numpy as np


class DataCleansing:
    # This is a constructor method
    def __init__(self):
        self.df = pd.DataFrame()  # This will store the original dataframe
        self.df1 = pd.DataFrame()  # This will store the binary classified dataframe
        self.df2 = pd.DataFrame()  # This will store the final dataframe
        self.binClassCols = (
            []
        )  # This will store the columns which can be binary classified
        self.categCols = []  # This will store the categorical columns
        self.dropCols = (
            []
        )  # This will store the list of columns to be dropped from initial dataframe

    def readData(self, file):
        self.df = pd.read_csv(file)
        for dropColName in self.dropCols:
            self.df.drop(dropColName, axis="columns", inplace=True)

    def binClassUpd(self, replaceDict):
        self.df1 = self.df
        for col in self.binClassCols:
            self.df1[col].replace(replaceDict, inplace=True)

    def getDumCatCols(self):
        self.df2 = pd.get_dummies(data=self.df1, columns=self.categCols)
        for cols in self.df2.columns:
            if self.df2[cols].dtypes == "bool":
                self.df2[cols].replace({True: 1, False: 0}, inplace=True)

    def getMinMaxScaled(self, minMaxScaleCols):
        scaler = MinMaxScaler()
        self.df2[minMaxScaleCols] = scaler.fit_transform(self.df2[minMaxScaleCols])


if __name__ == "__main__":
    obj = DataCleansing()
    tm = learnandpredict()
    obj.binClassCols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn",
    ]
    obj.dropCols = ["customerID"]
    obj.categCols = ["InternetService", "Contract", "PaymentMethod"]
    obj.readData("customer_churn_semicleaned.csv")
    obj.binClassUpd({"Yes": 1, "No": 0})

    obj.binClassCols = ["gender"]
    obj.binClassUpd({"Female": 1, "Male": 0})

    obj.getDumCatCols()
    obj.getMinMaxScaled(["tenure", "MonthlyCharges", "TotalCharges"])

    obj.df2.to_csv("customer_churn_cleaned.csv")

    tm.df = obj.df2

    # tm.train()

    y_pred = tm.predict(tm.df)

    obj.df = pd.read_csv("customer_churn_semicleaned.csv")

    obj.df["Predictions"] = y_pred["Predictions"]

    # finalDataframe = pd.concat([obj.df, y_pred], axis=1)
    # finalDataframe.to_csv("predicted_churn.csv")
    obj.df.to_csv("predicted_churn_semicleaned.csv")
