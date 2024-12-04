from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time
import datetime

if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')

if not os.path.exists('logs'):
    os.mkdir('logs')

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgable question generating assistant. Your task is to generate questions and answers based on the context provided. The questions and answers should be something a user would ask about the context.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='jj',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="llama3.2:1b")
                                          )
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3.2:1b",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Question Answer Generator")

# Upload a PDF file
#uploaded_file = st.file_uploader("Upload PDF file to Generate Question Set", type='pdf')
#uploaded_file = st.file_uploader("Upload PDF file to Generate Question Set", type='pdf')
uploaded_file = st.sidebar.file_uploader("Upload PDF file to Generate Question Set", type=["pdf"])
#for message in st.session_state.chat_history:

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name+".pdf", "wb")
            f.write(bytes_data)
            f.close()
            loader = PyPDFLoader("files/"+uploaded_file.name+".pdf")
            data = loader.load()

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            print("splits", all_splits)

            # Create and persist the vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3.2:1b")
            )

            print("vectorstore", st.session_state.vectorstore)
            st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    # Initialize the QA chain
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input
    if user_input := st.chat_input("You:", key="user_input"):
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        
        # Log user message
        with open('logs/chat_history.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] User: {user_input}\n")
        
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            message_placeholder = st.empty()
            full_response = ""

            # Process response to display each Q&A on a new line
            for qa_pair in response['result'].split("\n"):
                qa_pair = qa_pair.strip()
                if qa_pair:  # Ensure not adding blank lines
                    full_response += f"{qa_pair}\n\n"
                    time.sleep(0.05)  # Simulate typing delay
                    message_placeholder.markdown(full_response.strip() + "▌")
            
            message_placeholder.markdown(full_response.strip())

            # Log assistant response
            with open('logs/chat_history.txt', 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Assistant:\n{full_response.strip()}\n")
            
            chatbot_message = {"role": "assistant", "message": full_response.strip()}
            st.session_state.chat_history.append(chatbot_message)
else:
    st.write("Please upload a PDF file to generate Question.")


