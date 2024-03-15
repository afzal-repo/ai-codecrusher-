import os

import streamlit as st
from azure.storage.blob import BlobServiceClient

import azurecognitive_search_AzureOpenAI_Test

st.set_page_config(page_title="ESG Survey Automation", page_icon=":robot:")
st.header("Upload Survey Documents ")


if 'clicked' not in st.session_state:
    st.session_state.clicked = False



def click_button():
    st.session_state.clicked = True



if not st.session_state.clicked:
    upload_files=st.file_uploader("Choose a file", accept_multiple_files=True)
    blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("AZURE_CONN_STRING"))
    for uploaded_file in upload_files:
        bytes_data= uploaded_file.read()
        blob_client = blob_service_client.get_blob_client(
            container=os.environ.get("CONTAINER_NAME"), blob=uploaded_file.name
        )
        try:
            response = blob_client.upload_blob(bytes_data)
        except Exception as err:
            if str(err).find("BlobAlreadyExists") != -1:
                errorText="Uploaded file already exists - " + uploaded_file.name
                st.error(errorText)
                raise
            print(f"Data uploaded Successfully - Filename {uploaded_file.name}")
            azurecognitive_search_AzureOpenAI_Test.vector_stores()
            # st.session_state.clicked = False
            st.write('Embedding is Complete! for ', uploaded_file.name)

       # blob_client.upload_blob(bytes_data)




#st.button('Run Embedding', on_click=click_button)

