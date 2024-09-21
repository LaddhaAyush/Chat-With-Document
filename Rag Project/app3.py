import os
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

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
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

# Initialize Qdrant DB
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

# Initialize memory with conversation history
memory = ConversationBufferWindowMemory(k=5)

# Input query
query = input("Enter your query: ")

if query:
    # Perform similarity search with context
    docs = db.similarity_search_with_score(query=query, k=2)

    # Display search results
    print("Search Results:")
    for i, (doc, score) in enumerate(docs):
        print(f"**Result {i+1}:**")
        print(f"  **Score:** {score}")
        print(f"  **Content:** {doc.page_content}")
        print(f"  **Metadata:** {doc.metadata}")
        print("-" * 40)

    # Prepare data for generation
    retrieved_content = "\n".join([doc.page_content for doc, score in docs])

    # Define the prompt template for PDFs
    prompt_template = PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "You are an intelligent assistant capable of understanding and summarizing PDF content. "
            "Provide a detailed response based on the content extracted from the PDF related to the user query. "
            "Context: {context}\n"
            "Query: {query}\n\n"
            "Your task is to generate a helpful and informative response based on the retrieved PDF content."
        )
    )

    # Format the input using the prompt template
    formatted_input = prompt_template.format(context=retrieved_content, query=query)

    # Set up generation model
    groq_api_key = os.environ['GROQ_API_KEY']
    model = 'mixtral-8x7b-32768'  # Or any other model you prefer

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Generate response using the formatted input
    response = conversation.run(formatted_input)

    # Print final result
    print("Generated Response:")
    print(response)
