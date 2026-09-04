import streamlit as st  
import pickle  
import numpy as np  
import pandas as pd  
  
pipe = pickle.load(open('pipe.pkl', 'rb'))  
df = pickle.load(open('df.pkl', 'rb'))  
    
df.columns = df.columns.str.strip()  
  
st.title("Laptop Predictor")  
  
 
company = st.selectbox('Brand', df['Company'].unique())  
  
 
type = st.selectbox('Type', df['TypeName'].unique())  
  

ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])  
  
 
weight = st.number_input('Weight Of the laptop')  
  
  
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])  
  
  
ips = st.selectbox('IPS', ['No', 'Yes'])  
  
screen_size = st.slider(  
    'Screen size in inches',  
    10.0,  
    18.0,  
    13.0  
)  
  
 
resolution = st.selectbox(  
    'Screen Resolution',  
    [  
        '1920x1080',  
        '1366x768',  
        '1600x900',  
        '3840x2160',  
        '3200x1800',  
        '2880x1800',  
        '2560x1600',  
        '2560x1440',  
        '2304x1440'  
    ]  
)  
  
# CPU  
cpu = st.selectbox(  
    'CPU',  
    df['Cpu brand'].unique()  
) 
 

generation = st.selectbox( 
    'Generation', 
    [ 
        'NAN',
        '4th Generation', 
        '5th Generation', 
        '6th Generation', 
        '7th Generation', 
        '8th Generation', 
        '9th Generation', 
        '10th Generation', 
        '11th Generation', 
        '12th Generation', 
        '13th Generation' 
    ] 
) 

 
hdd = st.selectbox(  
    'HDD(in GB)',  
    [0, 128, 256, 512, 1024, 2048]  
)  
  
 
ssd = st.selectbox(  
    'SSD(in GB)',  
    [0, 8, 128, 256, 512, 1024]  
)  
  
 
gpu = st.selectbox(  
    'GPU',  
    df['Gpu brand'].unique()  
)  
  
  
os = st.selectbox(  
    'OS',  
    df['os'].unique()  
)  
  
  

if st.button('Predict Price'):  
  
    touchscreen_val = 1 if touchscreen == 'Yes' else 0  
    ips_val = 1 if ips == 'Yes' else 0  
  
 
    X_res = int(resolution.split('x')[0])  
    Y_res = int(resolution.split('x')[1])  
  
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size  
   
    input_dict = {  
        'Company': company,  
        'TypeName': type,  
        'Ram': ram,  
        'Weight': weight,  
        'Touchscreen': touchscreen_val,  
        'IPS': ips_val,  
        'ppi': ppi,  
        'Cpu brand': cpu,  
        'HDD': hdd,  
        'SSD': ssd,  
        'Gpu brand': gpu,  
        'os': os  
    }  
  
    try:   
        expected_cols = pipe.named_steps['step1'].feature_names_in_  
  
        input_df = pd.DataFrame([input_dict])  
  
         
        input_df = input_df[expected_cols]  
  
        
        prediction = pipe.predict(input_df)[0]  
  
      
        predicted_price = np.exp(prediction)  
  
        st.title(  
            f"The predicted price of this configuration is PKR {int(predicted_price):,}"  
        )  
  
    except KeyError as e:  
  
        st.error(  
            f"Column mismatch between app inputs and trained model: {e}"  
        )  
  
        st.write(  
            "Model expects these columns:",  
            list(pipe.named_steps['step1'].feature_names_in_)  
        )  
  
        st.write(  
            "App provided these columns:",  
            list(input_dict.keys())  
        )  
  
    except Exception as e:  
  
        st.error(  
            f"Something went wrong during prediction: {e}"  
        )