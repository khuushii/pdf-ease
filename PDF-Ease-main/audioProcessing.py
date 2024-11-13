import streamlit as st
from dotenv import load_dotenv
import tempfile
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import assemblyai as aai

aai.settings.api_key = f"100db39ff5df46bdaa12f22fddeb30d6"

def get_audio_text(audio):
    load_dotenv()
    text = ""
    for clip in audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(clip.read())
            temp_file_path = temp_file.name

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(temp_file_path)
        text += transcript.text

        # Delete the temporary file
        os.remove(temp_file_path)

    # Split text into smaller chunks
    # text_chunks = get_text_chunks(text)

    return text


def get_text_chunks(text):
    # You may adjust the chunk size based on the model's token limit
    max_token_limit = 1000
    chunks = [text[i:i + max_token_limit] for i in range(0, len(text), max_token_limit)]
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="Model/instructor")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    # llm = HuggingFaceHub(repo_id="Model/flan/", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def audioSection():

    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("𝄞 Talk to Audio 𝄞")
    user_question = st.text_input("Ask a question about your Audio:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        audio_docs = st.file_uploader(
            "Upload your Audio Files here and click on 'Process'", accept_multiple_files=True, type=['.wav', '.mp3'])
        if st.button("Process"):
            with st.spinner("Processing"):
                # get audio text
                raw_text = get_audio_text(audio_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)