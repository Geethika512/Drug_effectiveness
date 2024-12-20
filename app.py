import streamlit as st
import pandas as pd
import pickle

# Header
st.header("Predicting Drug Effectiveness")

# Image in the center column
cola, colb, colc = st.columns(3)
with colb:
    st.image("image.jpg")

# Display the dataframe
st.write("Predictive Model Built on Below Sample Data")
df = pd.read_csv("na_drug.csv")
def lowercase(df):
    for col in df.select_dtypes("O"):
        df[col] = df[col].str.lower()
lowercase(df)
df['Drug'] = df['Drug'].str.replace(",", " ").str.split().str[0]
df = df.drop(columns=['Price', 'Reviews', 'Type'])
st.dataframe(df.head())

# Taking user input
col1, col2 = st.columns(2)
with col1:
    Condition = st.selectbox('Select the condition:', df['Condition'].unique())
with col2:
    Drug = st.selectbox('Select the drug:', df['Drug'].unique())

col3, col4 = st.columns(2)
with col3:
    EaseOfUse = st.number_input(f"Enter EaseOfUse Value (Min {df['EaseOfUse'].min()} to Max {df['EaseOfUse'].max()}):")
with col4:
    Form = st.selectbox('Select the Form:', df['Form'].unique())

col5, col6 = st.columns(2)
with col5:
    Indication = st.selectbox('Select the Indication:', df['Indication'].unique())
with col6:
    Satisfaction = st.number_input(f"Enter Satisfaction Value (Min {df['Satisfaction'].min()} to Max {df['Satisfaction'].max()}):")

xdata = [Condition,Drug,EaseOfUse,Form,Indication,Satisfaction]

# Load the model and encoder
with open('Healthcare_randomforest.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoder.pkl', 'rb') as p:
    encoder = pickle.load(p)

# Prepare the input data for prediction
x = pd.DataFrame([xdata], columns=['Condition', 'Drug', 'EaseOfUse', 'Form', 'Indication', 'Satisfaction'])
st.write("Given Input:")
st.dataframe(x)

cat_cols = ['Condition', 'Drug', 'Form']  # Ensure these match the encoder's training columns
x_cat = x[cat_cols]  # Extract categorical columns
encoded_data = encoder.transform(x_cat).toarray()  # Transform using the encoder
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out()) 

# Drop original categorical columns and add encoded features
x = x.drop(columns=cat_cols)
x = pd.concat([x, encoded_df], axis=1)

# Reorder columns to match the model's training feature order
model_feature_names = model.feature_names_in_  # Ensure this is available
x = x.reindex(columns=model_feature_names, fill_value=0)


# Replace nominal values like 'Indication' if necessary
replacement_mappings = {'Indication': {'on label': 1, 'off label': 0}}
for column, mapping in replacement_mappings.items():
    if column in x.columns:
        x[column] = x[column].replace(mapping)

# Prediction and output
if st.button("Predict"):
    prediction = round(model.predict(x)[0],2)
    st.write(f"Prediction: {prediction}")
  



