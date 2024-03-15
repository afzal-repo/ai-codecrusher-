import time

import azurecognitive_search_AzureOpenAI_Test

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
import streamlit as st

from streamlit_chat import message
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ESG Survey Automation", page_icon=":robot:")
st.header("ESG Sustainability Survey Automation")


    #st.write(bytes_data)



memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)
def load_chain():


    prompt_template = """You are a helpful assistant for questions.
Understand user intention and provide the layman response.
Create a final answer with references ("SOURCES").
{context}
Question: {question}
Answer here:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=10)

    llm = AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        deployment_name=os.getenv("OPENAI_DEPLOYMENT_GEN_NAME"),
        model=os.getenv("OPENAI_MODEL_GEN_NAME"),
        temperature=0.7,
        openai_api_version="2023-05-15")



    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    return chain

chain = load_chain()

if 'generated' not in st.session_state:
    st.session_state['generated'] = []


if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    print(user_input)
    output = chain.run(question=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)



if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
