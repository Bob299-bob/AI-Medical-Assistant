from fastapi import FastAPI
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# Load env
load_dotenv()

# Gemini Client
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
# Load data
data = joblib.load('medical_data.pkl')

# Load FAISS index
idx = faiss.read_index('faiss_index.faiss')

# Load embedding model
model = SentenceTransformer(
    'embedding_model'
)

st.title('AI Medical Assistant')
#Creating a form for taking input from user 
with st.form('chat_form',clear_on_submit=True):
    Data=st.text_input('Enter Your Query')
    #Creating a submit button
    send=st.form_submit_button('Send')
#Perform Action
if(send):
    if Data.strip() == "":
        st.warning("Please enter a query")
    else:
        # Convert query to embedding
        embedding = np.array(model.encode([Data]),dtype='float32')

        # FAISS search
        distance, indices = idx.search(embedding,k=3)

        # Retrieve context
        retrieved_text = []

        for i in indices[0]:
            retrieved_text.append(data.iloc[i]['text'])

        context = '\n'.join(retrieved_text[:2])

        # Prompt
        prompt = f"""
        You are an expert AI Medical Assistant.
        Use the given context carefully.
        Context:
        {context}
        User Symptoms:
        {Data}
        Task:
        1. Identify most likely disease
        2. Explain why
        3. Give detailed precautions
        4. Suggest when to see a doctor
        5. Give one-line Hinglish summary
        Format output properly.
        """
        # Gemini response
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        st.success(response.choices[0].message.content)
st.sidebar.title('🩺 Medical Assistant')
st.sidebar.markdown(
    "<h2 style='color:red'>Thankyou for Visting here..</h2>",unsafe_allow_html=True)
st.sidebar.markdown("""<p style='background-color:#f5f5f5;padding:10px;border-radius:10px;color:black;'>
                    I am your Medical Assistant which can analyze your problem,and giving you 
                    a well suitable generated response through RAG system .
                    </p>
                    """,unsafe_allow_html=True)
st.sidebar.markdown("""
<ul style='color:purple;'>
<li>Symptom Analysis</li>
<li>Disease Prediction</li>
<li>Precautions Suggestion</li>
</ul>
""", unsafe_allow_html=True)
st.sidebar.info(
    "⏳ Please be patient. Responses may take a few seconds as the system uses a local Ollama model for generating medical answers."
)