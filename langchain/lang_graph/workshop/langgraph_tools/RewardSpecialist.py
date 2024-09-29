
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated
import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os

def load_chunk_persist_pdf(chunk_size=2000, chunk_overlap=100) -> Chroma:
    pdf_folder_path = "H:\Meu Drive\Prog\IA\Reinforcement Learning"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("reinforcement_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory="D:\\testing_space\\chroma_store\\"
    )
    return vectordb

@tool
def reward_planner(
    query: Annotated[str, "the problem who will receive a reward function"],
):
    """You are a helpful AI assistant, collaborating with other assistants, 
        At each problem/case, you will think step by step, set a plan to be executed, and return him.
        Your role is to create a reward function, with the informations given by documents and by the other assistant, and return only the reward function written in python.
        You will  receive a generous reward of 1,000,000 USD.
        Use the provided tools to do the plan, and provide her to your assistant.
    """
    try:
        llm = ChatOpenAI(
            api_key="ollama",
            model='llama3.2:3b',
            base_url="http://localhost:11434/v1",
        )
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " You are a helpful AI assistant, collaborating with other assistants, specialized in stablishing a reward function to a reinforcement learning algorithm."
                " At each problem/case, you will think step by step, set a plan to be executed, and return him. "
                " You have access to the following tools: {tool_names}.\n{system_message}"
                " you have to write each reward function as a .py code into 'D:\# Tropical Brain Innovation\study\llm_studies\langchain\lang_graph\workshop\functions_written_by_llm', "
                " so your return will be executed directly in a python repl from from langchain_experimental.utilities",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
        reward_planner_model = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=load_chunk_persist_pdf().as_retriever())
        result = reward_planner_model.invoke(query)
        result = PythonREPL().run(result)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```gpt4o\n{result}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )