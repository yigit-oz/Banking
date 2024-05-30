import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the best model
best_model = joblib.load('best_model.pkl')

# Define preprocessing steps (same as during training)
ordinal_features = ["education", "month", "day_of_week"]
ordinal_transformer = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder())
])

categorical_features = ["job", "marital", "default", "housing", "loan", "contact", "poutcome"]
onehot_transformer = Pipeline(steps=[
    ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", 
                    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_features),
        ('onehot', onehot_transformer, categorical_features),
        ('numeric', numeric_transformer, numeric_features)
    ]
)

# Streamlit app
st.title('Bank Marketing Campaign Predictor')

# Collect user input
def user_input_features():
    age = st.number_input('Age', min_value=0)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                               'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 
                                           'professional.course', 'university.degree', 'unknown'])
    default = st.selectbox('Default', ['no', 'yes', 'unknown'])
    housing = st.selectbox('Housing Loan', ['no', 'yes', 'unknown'])
    loan = st.selectbox('Personal Loan', ['no', 'yes', 'unknown'])
    contact = st.selectbox('Contact', ['cellular', 'telephone'])
    month = st.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.number_input('Duration', min_value=0)
    campaign = st.number_input('Campaign', min_value=0)
    pdays = st.number_input('Pdays', min_value=0)
    previous = st.number_input('Previous', min_value=0)
    emp_var_rate = st.number_input('Employment Variation Rate')
    cons_price_idx = st.number_input('Consumer Price Index')
    cons_conf_idx = st.number_input('Consumer Confidence Index')
    euribor3m = st.number_input('Euribor 3 Month Rate')
    nr_employed = st.number_input('Number of Employees')
    
    data = {'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'month': month,
            'day_of_week': day_of_week,
            'duration': duration,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'emp.var.rate': emp_var_rate,
            'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx,
            'euribor3m': euribor3m,
            'nr.employed': nr_employed}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the input data
preprocessed_input = preprocessor.transform(input_df)

# Make predictions
prediction = best_model.predict(preprocessed_input)

# Display the results
st.subheader('Prediction')
st.write('The model predicts:', 'Subscribed' if prediction[0] == 1 else 'Not Subscribed')