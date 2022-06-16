import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import  StandardScaler
import streamlit as st
import mitosheet
import plotly.express as px
import streamlit.components.v1 as components

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df= pd.read_csv(uploaded_file)
    st.write(df.head(20))
    params =  st.slider('Masukkan angka max 1000',min_value=1,max_value=1000)
    p = st.number_input('Masukkan angka max 100',min_value=1,max_value=100)
    ts =  st.number_input('Masukkan angka test size max 1,00',min_value= 0.01,max_value=1.0,value=0.01)
    rs = st.number_input('Masukkan angka random state max 10',min_value=0,max_value=10)
    # Menggunakan MITO
    mitosheet.sheet()
    st.subheader('Hasil Proses Dataset')
    
    heart_failure_clinical_records_dataset = pd.read_csv(r'heart_failure_clinical_records_dataset.csv')

    # Changed age to dtype int
    heart_failure_clinical_records_dataset['age'] = heart_failure_clinical_records_dataset['age'].fillna(0).astype('int')

    # Filtered age
    heart_failure_clinical_records_dataset = heart_failure_clinical_records_dataset[heart_failure_clinical_records_dataset['age'].notnull()]

    # Filtered diabetes
    heart_failure_clinical_records_dataset = heart_failure_clinical_records_dataset[heart_failure_clinical_records_dataset['diabetes'].notnull()]

    # Filtered sex
    heart_failure_clinical_records_dataset = heart_failure_clinical_records_dataset[heart_failure_clinical_records_dataset['sex'].notnull()]
    a = heart_failure_clinical_records_dataset.iloc[:,[0,3,9]]
    b = heart_failure_clinical_records_dataset.iloc[:,-1]
    st.subheader('Dataset yang digunakan')
    st.write(a)

    x_train, x_test, y_train, y_test = train_test_split(a, b, test_size =ts, random_state = rs)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
                    
                    #Klasifikasi KNN
    classifier = KNeighborsClassifier(n_neighbors = params, metric = 'minkowski', p = p)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
    st.write(f'Akurasi = ',acc)
    
    import plotly.express as px
    # Construct the graph and style it. Further customize your graph by editing this code.
    # See Plotly Documentation for help: https://plotly.com/python/plotly-express/
    fig = px.ecdf(heart_failure_clinical_records_dataset, x=['age', 'diabetes', 'sex'], y='DEATH_EVENT')
    fig.update_layout(
        title='age, diabetes, sex, DEATH_EVENT ecdf', 
        xaxis = dict(
            rangeslider = dict(
                visible=True, 
                thickness=0.05
            )
        ), 
        yaxis = dict(

        ), 
        paper_bgcolor='#FFFFFF', 
        showlegend=True
    )
    fig.show(renderer="iframe")
    st.write(fig)