import pandas as pd
import streamlit as st
import pickle
import numpy as np

data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('LinearModel.pkl', 'rb'))
def predict_price(sqft, bath, balcony, location, bhk):
    input_data = pd.DataFrame([[sqft, bath, balcony, location, bhk]], columns=[
                              'total_sqft', 'bath', 'balcony', 'site_location', 'bhk'])
    prediction = pipe.predict(input_data)[0] * 1e5
    return np.round(prediction, 2)


def main():
    st.title("Pune House Price Predictor")

    # Get unique locations from the data
    locations = sorted(data['site_location'].unique())

    # Input components
    location = st.selectbox("Select Location", locations)
    bhk = st.number_input("Enter BHK",min_value=0)
    sqft = st.number_input("Enter total house area in sqft",min_value=0)
    bath = st.number_input("Enter number of bathroom(s)",min_value=0)
    balcony = st.number_input("Enter number of balcony(ies)",min_value=0)

    # Predict button
    if st.button("Predict Price"):
        if location and bhk and sqft and bath and balcony:
            prediction = predict_price(float(sqft), int(
                bath), int(balcony), location, int(bhk))
            st.write(f"Prediction: ₹{prediction}")
        else:
            st.warning("Please fill in all the input fields.")


if __name__ == "__main__":

    main()
