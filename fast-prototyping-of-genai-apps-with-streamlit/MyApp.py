import streamlit as st
import model as m

st.write("Hello World")

window = st.slider("Forecasted Sales", 0, 100, 50)
st.write(m.run(window))