import streamlit as st
import pandas as pd
import joblib
import urllib.request

def rf_predict(data):
  if data == 0:
    return "not_cyberbullying"
  elif data == 1:
    return "ethnicity"
  elif data == 2:
    return "gender"
  elif data == 3:
    return "age"
  else:
    return "religion"

data = pd.read_csv('cyberbullying_tweets.csv')
df = pd.DataFrame(data)



st.title("PREDIKSI KLASIFIKASI CYBERBULLYING")
st.text("Web ini digunakan untuk prediksi jenis cyberbullying")
st.write("Cyberbulyying Dataset")
st.dataframe(data)

#model = joblib.load(urlopen("https://drive.google.com/file/d/12OSbIa8Bv3DEgO2oMUeq9KWgMb7PKt9S/view?usp=drive_link"))
model = joblib.load("model_nb.pkl")
vector = joblib.load("vectorizer.pkl")

with st.form("nlpForm"):
    text = st.text_area("Masukkan Text Disini")
    submit_button = st.form_submit_button(label='Proses')
    
    col1,col2 = st.columns(2)
    if submit_button:
        
        with col1:

            st.info("Hasil")
        
            vector_text = vector.transform([text])
            hasil = model.predict(vector_text)
            kategori = rf_predict(hasil)
        with col2:
            st.success(kategori)
