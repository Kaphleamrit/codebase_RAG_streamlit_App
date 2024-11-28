import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import tempfile
from git import Repo
from openai import OpenAI
from pathlib import Path
from langchain.schema import Document
from pinecone import Pinecone

# Load environment variables
load_dotenv()


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def clone_repository(repo_url):
    """Clones a GitHub repository to a temporary directory.

    Args:
        repo_url: The URL of the GitHub repository.

    Returns:
        The path to the cloned repository.
    """
    # repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    # repo_path = f"/content/{repo_name}"
    # Repo.clone_from(repo_url, str(repo_path))
    repo_path = "./temp/SecureAgent"
    return str(repo_path)

def get_file_content(file_path, repo_path):
    """Get content of a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        rel_path = os.path.relpath(file_path, repo_path)
        return {"name": rel_path, "content": content}
    except Exception as e:
        st.error(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    """Get content of supported code files."""
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                          '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}
    IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                    '__pycache__', '.next', '.vscode', 'vendor'}
    files_content = []
    
    try:
        for root, _, files in os.walk(repo_path):
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)
    except Exception as e:
        st.error(f"Error reading repository: {str(e)}")
    
    return files_content




# Set the PINECONE_API_KEY as an environment variable
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Connect to your Pinecone index
pinecone_index = pc.Index("codebase-rag")

vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings())

path = clone_repository("https://github.com/CoderAgent/SecureAgent")
file_content = get_main_files_content(path)
documents = []
for file in file_content:
    doc = Document(
        page_content=f"{file['name']}\n{file['content']}",
        metadata={"source": file['name']}
    )

    documents.append(doc)


vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(),
    index_name="codebase-rag",
    namespace="https://github.com/CoderAgent/SecureAgent"
)


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

query = "How are python files parsed?"
raw_query_embedding = get_huggingface_embeddings(query)

top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=3, include_metadata=True, namespace="https://github.com/CoderAgent/SecureAgent")

contexts = [item['metadata']['text'] for item in top_matches['matches']]
augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.

Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
"""

llm_response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_query}
    ]
)

response = llm_response.choices[0].message.content


def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace="https://github.com/CoderAgent/SecureAgent")

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are a Senior Software Engineer, specializing in TypeScript.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

# Page config
st.set_page_config(
    page_title="Codebase RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Codebase RAG Assistant")
st.markdown("""
This app helps you query your codebase using AI. It uses:
- HuggingFace embeddings for semantic search
- Pinecone for vector storage
- Groq LLM for generating responses
""")

# # Sidebar for API keys
# with st.sidebar:
#     st.header("Configuration")
#     st.subheader("Repository Settings")
#     repo_url = st.text_input("GitHub Repository URL", 
#                             value="https://github.com/CoderAgent/SecureAgent",
#                             help="Enter the URL of the GitHub repository you want to analyze")
    
#     if 'loaded_repo' not in st.session_state:
#         st.session_state.loaded_repo = None
    
#     repo_name = get_repo_name(repo_url)
#     if st.button("Load Repository"):
#         if st.session_state.loaded_repo == repo_url:
#             st.info("Repository already loaded!")
#         else:
#             with st.spinner("Cloning repository..."):
#                 try:
#                     repo_path = clone_repository(repo_url)
#                     file_content = get_main_files_content(repo_path)
                    
                #     # Create documents
                #     documents = [
                #         Document(
                #             page_content=f"{file['name']}\n{file['content']}",
                #             metadata={"source": file['name']}
                #         ) for file in file_content
                #     ]

                #     st.success("Repository loaded and indexed successfully!")
                # except Exception as e:
                #     st.error(f"Error loading repository: {str(e)}")



# # Add image upload and custom prompt before the main query
# st.header("Configuration")
# cols = st.columns(2)
# with cols[0]:
#     uploaded_image = st.file_uploader("Upload an image (optional)", type=['png', 'jpg', 'jpeg'])
#     if uploaded_image:
#         st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# with cols[1]:
#     custom_prompt = st.text_area(
#         "Customize system prompt (optional):",
#         value="""You are a Senior Software Engineer, specializing in TypeScript.

# Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.
# """,
#         height=150
#     )

# Main query interface
st.header("Ask about your codebase")
query = st.text_area("Enter your question:", height=100)
groq_api_key = os.environ.get("GROQ_API_KEY")

if st.button("Get Answer") and query:
    if not (pinecone_api_key and groq_api_key):
        st.error("Please configure your API keys first!")
    else:
        try:
            with st.spinner("Searching codebase and generating response..."):
                # Get similar documents using the vectorstore
                results = perform_rag(query)                  
                # Display response
                st.markdown("### Answer:")
                st.markdown(results)
                
                # # Display relevant code snippets
                # with st.expander("View relevant code snippets"):
                #     for doc in results:
                #         st.markdown(f"**File:** `{doc.metadata.get('source', 'Unknown')}`")
                #         st.code(doc.page_content, language="python")
                        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Pinecone, and Groq LLM")

