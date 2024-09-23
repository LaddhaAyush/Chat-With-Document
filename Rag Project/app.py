import os
import streamlit as st
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="PDF Query Assistant", page_icon=":books:")
st.title("Chat with PDF Documents :books:")

# Define model and embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Qdrant client configuration
url = "http://localhost:6333"  # Replace with your Qdrant server URL
client = QdrantClient(url=url, prefer_grpc=False)

# Initialize memory with conversation history
memory = ConversationBufferWindowMemory(k=5)

# Function to process PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to ingest the PDF text into Qdrant
def ingest_pdf_to_qdrant(pdf_docs):
    try:
        documents = []
        for pdf in pdf_docs:
            # Read the PDF content
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Create a Document object for each PDF content
            documents.append(Document(page_content=text, metadata={"source": pdf.name}))

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Ingest the documents into Qdrant vector DB
        qdrant = Qdrant.from_documents(
            texts,
            embeddings,
            url=url,
            prefer_grpc=False,
            collection_name="vector_db"
        )

        st.success("PDFs ingested successfully!")
        return "\n".join([doc.page_content for doc in documents])

    except Exception as e:
        st.error(f"Error ingesting PDFs: {e}")
        return None

# Function to query Qdrant and retrieve context
def query_qdrant(query):
    db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
    docs = db.similarity_search_with_score(query=query, k=2)

    retrieved_content = "\n".join([doc.page_content for doc, score in docs])
    return retrieved_content, docs

# Sidebar for PDF uploads
with st.sidebar:
    st.subheader("Upload your PDF documents")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

    if st.button("Process PDFs"):
        with st.spinner("Processing..."):
            try:
                raw_text = ingest_pdf_to_qdrant(pdf_docs)
                st.session_state.raw_text = raw_text
            except Exception as e:
                st.error(f"Error processing PDFs: {e}")

# Query input and processing
if "raw_text" in st.session_state:
    user_query = st.text_input("Ask a question about your documents:")
    if user_query:
        with st.spinner("Generating response..."):
            try:
                retrieved_content, docs = query_qdrant(user_query)

                # Define prompt template
                prompt_template = PromptTemplate(
                    input_variables=["query", "context"],
                    # template=(
                    #     "You are an intelligent assistant capable of understanding and summarizing PDF content. "
                    #     "Provide a detailed response based on the content extracted from the PDF related to the user query. "
                    #     "Context: {context}\n"
                    #     "Query: {query}\n\n"
                    #     "Your task is to generate a helpful and informative response based on the retrieved PDF content."
                    # )
                    template = (
                        "You are an intelligent assistant highly skilled in analyzing, summarizing, and explaining content from PDF documents. "
                        "Your goal is to provide a detailed, accurate, and well-structured response based on the user's query and the content extracted from the PDF. "
                        "If the query relates to specific sections or details of the document, focus on those parts, ensuring clarity and relevance. "
                        "Summarize complex information effectively, highlighting key insights or critical points. "
                        "If certain aspects of the query cannot be answered due to missing information, acknowledge this and suggest possible next steps. "
                        "In addition, provide a brief summary of the most important insights from the document to offer further value."
                        "\nContext: {context}\n"
                        "Query: {query}\n\n"
                        "Provide an insightful, clear, and comprehensive response, ensuring you address the query using relevant sections of the PDF. "
                        "Break down complex topics where necessary, and offer a concise summary of key points at the end of your response."
)
                )

                # Format input using the prompt template
                formatted_input = prompt_template.format(context=retrieved_content, query=user_query)

                # Set up Groq model for response generation
                groq_api_key = os.environ['GROQ_API_KEY']
                model = 'mixtral-8x7b-32768'  # Or any other model you prefer

                groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
                conversation = ConversationChain(llm=groq_chat, memory=memory)

                # Generate response
                response = conversation.run(formatted_input)

                # Display response
                st.write("### Response:")
                st.write(response)

            except Exception as e:
                st.error(f"Error during response generation: {e}")
else:
    st.info("Please upload and process your PDFs first.")
