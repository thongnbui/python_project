from dotenv import load_dotenv
import openai
import streamlit as st

@st.cache_data
def get_response(user_prompt, temperature):
    response = client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": user_prompt}],
        temperature=temperature, 
        max_output_tokens=100 # limit the response length
    )
    return response

load_dotenv()

client = openai.OpenAI()

st.title("Generate AI")
st.write("This is a simple app that uses the OpenAI API to generate text.")

user_prompt = st.text_input("Enter a prompt:")

temperature = st.slider("Model temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
    help="Controls the randomness: 0.0 is the most deterministic, 1.0 is the most random.")

with st.spinner("Generating..."):
    response = get_response(user_prompt, temperature)
    # Print the response from OpenAI
    st.write(response.output[0].content[0].text)
