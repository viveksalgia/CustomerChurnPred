#!/usr/bin/env python3
"""File to train the model"""
###############################################################################
# Project - Customer Churn Prediction Model
# Filename - Train_Model.py
# Arguments -
# Created By - Vivek Salgia
# Creation Date - 03/28/2025
# Reviewed By -
# Reviewed Date -
# Change logs -
# Version   Date         Type   Changed By                  Comments
# =======   ============ ====   ==============  ===============================
# 1.0       03/28/2025   I     Vivek Salgia    Initial Creation
###############################################################################

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import load_model


class learnandpredict:

    # This is a constructor method
    def __init__(self):
        self.df = pd.DataFrame()  # This is the dataframe
        self.X = pd.DataFrame()  # This is the X frame
        self.y = pd.DataFrame()  # This is y frame
        self.X_train = pd.DataFrame()  # This is training dataframe
        self.X_test = pd.DataFrame()  # This is testing dataframe
        self.y_train = pd.DataFrame()  # This is training dataframe for predictions
        self.y_test = pd.DataFrame()  # This is the prediction dataframe
        self.model = keras.Model()  # Initialize the Model

    # This is the training model
    def train(self):
        self.X = self.df.drop(columns="Churn", axis="columns")
        self.y = self.df["Churn"]

        # Split the dataframe into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=5,  # Random State could be any number. This is to make sure all the runs get similar results
        )

        input_shape_rows, input_shape_cols = self.X_train.shape

        print(f"X_train Input Shape - {self.X_train.shape}")
        print(f"X_test Input Shape - {self.X_test.shape}")

        self.model = keras.Sequential(
            [
                keras.layers.Dense(
                    input_shape_cols, input_shape=(input_shape_cols,), activation="relu"
                ),
                keras.layers.Dense(
                    round(input_shape_cols - (input_shape_cols / 3)), activation="relu"
                ),
                keras.layers.Dense(round(input_shape_cols / 2), activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        self.model.fit(self.X_train, self.y_train, epochs=50)

        self.model.save("cust_churn_model.keras")

    # This is a predict method
    def predict(self, df):
        self.model = load_model(
            "cust_churn_model.keras", custom_objects=None, compile=True, safe_mode=True
        )
        if df.empty:
            self.model.evaluate(self.X_test, self.y_test)
            yp = self.model.predict(self.X_test)
        else:
            # print(df.dtypes)
            df.drop(columns="Churn", axis="columns", inplace=True)
            yp = self.model.predict(df)

        y_pred = []
        for prd in yp:
            if prd > 0.5:
                y_pred.append("Yes")
            else:
                y_pred.append("No")

        yp_df = pd.DataFrame(y_pred, columns=["Predictions"])

        if df.empty:
            finalDataframe = pd.concat([self.X_test, yp_df], axis=1)
        else:
            finalDataframe = yp_df

        return finalDataframe
        # finalDataframe.to_csv("predicted_churn.csv")
