import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import os

def create_retriever_from_file(file_path: str, collection_name: str = "buildorbuy_docs"):
    """Create a retriever from a file (PDF or text)."""
    # Load the document
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Initialize in-memory Qdrant client
    client = QdrantClient(":memory:")
    
    # Create collection with OpenAI embedding dimensions
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    
    # Create vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    # Add documents
    vectorstore.add_documents(splits)
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

def extract_capability(retriever) -> dict:
    """Extract capability from user stories using RAG."""
    template = """Based on the extracted user stories, please name one capability.

    Here's the format I want you to follow:

    Context: "As a user, I want to be able to view my monthly bill, so that I can understand my charges."
    System Description: "This is a billing system that allows users to view their monthly bill."
    Capability: "Billing Management"

    Rules for the capability name:
    - Keep it short and no more than 3 words
    - Use simple, clear terms
    - Keep it business focused and not technical
    - You can use words like "Management", "System", "Module"
    - Make it concise and focused

    Now, based on these user stories, please describe the system and identify one key capability:
    {context}

    Question: {question}

    Output your response in exactly this format:
    System Description: "<system_description>"
    Capability: "<capability_name>"
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(
        model="gpt-4-1106-preview",
        temperature=0
    )
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response = rag_chain.invoke("What is the one capability that is most relevant to the user stories?")
    
    system_match = re.search(r'System Description:\s*"([^"]+)"', response)
    capability_match = re.search(r'Capability:\s*"([^"]+)"', response)
    
    if not system_match or not capability_match:
        raise ValueError("Could not extract system description or capability from response")
    
    return {
        "system_description": system_match.group(1),
        "capability": capability_match.group(1)
    }

def select_file():
    """Opens a file dialog to select a file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*")
            ]
        )
        
    except ImportError:
        file_path = input("\nEnter file path: ").strip()
    
    if not file_path:
        raise ValueError("No file selected")
    
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
        
    return file_path 