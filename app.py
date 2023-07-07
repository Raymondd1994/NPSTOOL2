#import dependencies
import os 
from apikey import apikey
import langchain
import streamlit as st 
import time
import pandas as pd
import pydeck as pdk

from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain
os.environ['OPENAI_API_KEY'] = apikey

#App framework

st.title ('The NPS label tool 😄🥲😊')
st.markdown("""
Introducing our AI-powered feedback analysis tool, designed to unlock customer insights at scale. Utilizing the power of OpenAI's GPT-3, our tool automates the process of deciphering customer sentiment and identifying key themes from your feedback data. It's built to save time, enhance understanding, and empower businesses to act on customer insights faster than ever before. With our tool, make every piece of feedback count!
""")

#NPSanswer = st.text_input(' *Enter your NPS open answer* ')
company_name = st.text_input("Enter your company name")
language = st.selectbox("Select language", ["English", "Nederlands","Deutsch","French"])


#Chatmodel

chat_model= ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")

#Prompt template

system_message_prompt = SystemMessagePromptTemplate.from_template("Pretend you're a customer journey specialist who is an expert in labeling customer feedback.")
human_message_prompt = HumanMessagePromptTemplate.from_template(" ##Context For the company {company_name}, we receive feedback of customers. ##Assignment In the following assignment, you need to derive from the given text the reasons why they are not satisfied and process this into labels. Use a maximum of 5 labels for all combined answers. Focus the labeling on the cause of the dissatisfaction. Also provide an explanation of the label, keep general and do not use specific examples. Also provide a weight to each label according to how much it is mentioned in the answers. Use 100 points to divide over the 5 labels.   ##FormatMake sure the labels are as short and concrete as possible.  Use only {language} language to create labels and all other output. ##Input Label the following NPS input {NPSanswer}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

#LLM CHain

nps_chain = LLMChain(llm=chat_model, prompt=chat_prompt, verbose=True)

# This line will allow the user to upload an Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# This line will convert the uploaded file into a pandas DataFrame
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write(df)  # Optional: This will display the DataFrame in the app

    # Combine all feedback stories into a single string
    feedback_text = '\n'.join(df['Feedback'].astype(str))
    NPSanswer = feedback_text
    #NPSanswer = st.text_input(' *Enter your NPS open answer* ', value=feedback_text)

# Show stuff on the screen when there is a prompt 
if st.button('Generate'):
    try:
        if NPSanswer:
            response = nps_chain.run({"NPSanswer": NPSanswer, "company_name": company_name, "language": language})
            st.write(response)   

    except Exception as e:
        st.error(f"An error occurred: {e}")
