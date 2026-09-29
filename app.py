import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Sonar Rock vs Mine Prediction")

sonar_data = pd.read_csv("sonar_data.csv.csv", header=None)

X = sonar_data.drop(columns=60)
Y = sonar_data[60]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.write("Accuracy on training data:", training_data_accuracy)
st.write("Accuracy on test data:", test_data_accuracy)

input_text = st.text_area(
    "Enter 60 sonar values separated by commas:",
    "0.0587,0.1210,0.1268,0.1498,0.1436,0.0561,0.0832,0.0672,0.1372,0.2352,0.3208,0.4257,0.5201,0.4914,0.5950,0.7221,0.9039,0.9111,0.8723,0.7686,0.7326,0.5222,0.3097,0.3172,0.2270,0.1640,0.1746,0.1835,0.2048,0.1674,0.2767,0.3104,0.3399,0.4441,0.5046,0.2814,0.1681,0.2633,0.3198,0.1933,0.0934,0.0443,0.0780,0.0722,0.0405,0.0553,0.1081,0.1139,0.0767,0.0265,0.0215,0.0331,0.0111,0.0088,0.0158,0.0122,0.0038,0.0101,0.0228,0.0124"
)

if st.button("Predict"):

    input_data = tuple(
        float(value.strip())
        for value in input_text.split(",")
        if value.strip()
    )

    if len(input_data) != 60:
        st.error("Please enter exactly 60 values.")
    else:
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)

        if prediction[0] == "R":
            st.success("The object is a rock")
        else:
            st.error("The object is a mine")