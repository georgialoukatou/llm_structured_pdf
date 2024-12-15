from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

import os
import tempfile
import uuid
import pandas as pd
import re

# Load API Key from environment variables
API_KEY = os.getenv('OPENAI_API_KEY')

def sanitize_filename(filename):
    """
    Cleans a filename by removing patterns like "(number)".

    Parameters:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    return re.sub(r'\s\(\d+\)', '', filename)

def extract_pdf_content(uploaded_file):
    """
    Extracts content from an uploaded PDF file.

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file.

    Returns:
        list: A list of documents extracted from the PDF.
    """
    try:
        pdf_data = uploaded_file.read()

        # Create temporary file for loading
        temp_path = tempfile.NamedTemporaryFile(delete=False)
        temp_path.write(pdf_data)
        temp_path.close()

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_path.name)
        doc_list = loader.load()

        return doc_list

    finally:
        # Remove the temporary file
        os.unlink(temp_path.name)

def split_text_chunks(doc_list, size=1000, overlap=200):
    """
    Splits text documents into smaller chunks.

    Parameters:
        doc_list (list): List of documents to split.
        size (int): Maximum chunk size.
        overlap (int): Overlap between consecutive chunks.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                              chunk_overlap=overlap,
                                              length_function=len,
                                              separators=["\n\n", "\n", " "])
    return splitter.split_documents(doc_list)

def initialize_embedding_function(api_key):
    """
    Initializes the embedding function using OpenAI.

    Parameters:
        api_key (str): OpenAI API key.

    Returns:
        OpenAIEmbeddings: Embedding function.
    """
    return OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
def format_documents(doc_list):
    """
    Formats documents into a single string.

    Parameters:
        doc_list (list): List of documents.

    Returns:
        str: Formatted text from all documents.
    """
    return "\n\n".join(doc.page_content for doc in doc_list)

def build_vector_database(chunks, embedding_func, filename, storage_path="vector_store"):
    """
    Builds a vector database from text chunks.

    Parameters:
        chunks (list): List of text chunks.
        embedding_func: Embedding function.
        filename (str): Name of the source file.
        storage_path (str): Path to store the vector database.

    Returns:
        Chroma: A Chroma vector database object.
    """
    unique_ids = set()
    unique_chunks = []
    doc_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    # Filter for unique IDs and chunks
    for doc, doc_id in zip(chunks, doc_ids):
        if doc_id not in unique_ids:
            unique_ids.add(doc_id)
            unique_chunks.append(doc)

    # Create and persist the vector database
    vector_db = Chroma.from_documents(documents=unique_chunks,
                                      collection_name=sanitize_filename(filename),
                                      embedding=embedding_func,
                                      ids=list(unique_ids),
                                      persist_directory=storage_path)
    vector_db.persist()
    return vector_db


def load_vector_database(filename, api_key, storage_path="vector_store"):
    """
    Loads a saved vector database.

    Parameters:
        filename (str): File name for the database.
        api_key (str): OpenAI API key.
        storage_path (str): Path to the vector database.

    Returns:
        Chroma: A Chroma vector database object.
    """
    embedding_func = initialize_embedding_function(api_key)
    return Chroma(persist_directory=storage_path,
                  embedding_function=embedding_func,
                  collection_name=sanitize_filename(filename))

# Prompt template for question answering
QUERY_PROMPT = """
You are an assistant tasked with answering questions.
Use the context below to respond accurately. If you cannot find an answer,
state that you do not know. Do not fabricate responses.

{context}

---

Answer: {question}
"""

class QAResponse(BaseModel):
    """Response model for QA tasks."""
    answer: str = Field(description="The answer to the question.")
    sources: str = Field(description="The sources used to derive the answer.")
    reasoning: str = Field(description="Reasoning behind the answer.")

class DocumentDetails(BaseModel):
    """Extracted details of a research article."""
    title: QAResponse
    summary: QAResponse
    publication_year: QAResponse
    authors: QAResponse

def execute_query(vector_db, question, api_key):
    """
    Executes a query on the vector database.

    Parameters:
        vector_db (Chroma): The vector database.
        question (str): The query.
        api_key (str): OpenAI API key.

    Returns:
        pd.DataFrame: Results in a tabular format.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    retriever = vector_db.as_retriever(search_type="similarity")
    prompt_template = ChatPromptTemplate.from_template(QUERY_PROMPT)

    query_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt_template
        | llm.with_structured_output(DocumentDetails, strict=True)
    )

    response = query_chain.invoke(question)
    df = pd.DataFrame([response.dict()])

    # Transforming data into a readable table
    rows = {"answer": [], "source": [], "reasoning": []}
    for col in df.columns:
        rows["answer"].append(df[col][0]['answer'])
        rows["source"].append(df[col][0]['sources'])
        rows["reasoning"].append(df[col][0]['reasoning'])

    return pd.DataFrame(rows, index=["answer", "source", "reasoning"]).T

def create_vector_database_from_docs(doc_list, api_key, filename):
    """
    Creates a vector database from documents.

    Parameters:
        doc_list (list): List of documents.
        api_key (str): OpenAI API key.
        filename (str): File name for the vector database.

    Returns:
        Chroma: A Chroma vector database object.
    """
    chunks = split_text_chunks(doc_list)
    embedding_func = initialize_embedding_function(api_key)
    return build_vector_database(chunks, embedding_func, filename)