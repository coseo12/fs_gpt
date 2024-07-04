import streamlit as st
from langchain.prompts import PromptTemplate

# 문자열
st.write("hello")

# Data Structure
st.write([1,2,3,4])

# Data dict
st.write({"x": 1})

# Class
st.write(PromptTemplate)

p = PromptTemplate.from_template("xxxx")

p

st.selectbox("Choose your model", {"GPT-4": "gpt-4", "GPT-3": "gpt3", "GPT-2": "gpt2"})