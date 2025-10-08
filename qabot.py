# qabot.py
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import gradio as gr
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Configuration
# -----------------------------
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"

LLM_MODEL_ID = "ibm/granite-3-3-8b-instruct"
EMBEDDING_MODEL_ID = "ibm/slate-125m-english-rtrvr"

LLM_PARAMS = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
}

CHROMA_PERSIST_DIR = "./chroma_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5

# -----------------------------
# Watsonx LLM initializer
# -----------------------------
def get_llm():
    llm = WatsonxLLM(
        model_id=LLM_MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=LLM_PARAMS,
    )
    return llm

# -----------------------------
# Document loader
# -----------------------------
def document_loader(filepath: str):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs

# -----------------------------
# Text splitter
# -----------------------------
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(data)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    return chunks

# -----------------------------
# Embedding model (Watsonx embeddings)
# -----------------------------
def watsonx_embedding():
    embed_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 512
    }
    embedding_model = WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        url=WATSONX_URL,
        project_id=PROJECT_ID,
        params=embed_params,
    )
    return embedding_model

# -----------------------------
# Vector DB (Chroma)
# -----------------------------
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    persist_dir = CHROMA_PERSIST_DIR

    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
    except Exception as e:
        print(f"❌ Embedding/Vector DB creation failed: {e}")
        raise

    return vectordb

# -----------------------------
# Retriever pipeline
# -----------------------------
def retriever(file_path: str):
    docs = document_loader(file_path)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    retr = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    return retr

# -----------------------------
# QA Chain
# -----------------------------
def retriever_qa(file_path: str, query: str):
    llm = get_llm()
    retr = retriever(file_path)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retr,
        return_source_documents=True
    )

    result = qa({"query": query})

    if isinstance(result, dict):
        answer = result.get("result", "")
        sources = result.get("source_documents", [])
    else:
        answer = result
        sources = []

    sources_text = ""
    if sources:
        snippets = []
        for doc in sources:
            src = doc.metadata.get("source", "Unknown")
            excerpt = (doc.page_content[:200].strip().replace("\n", " ") + "...") if getattr(doc, "page_content", None) else ""
            snippets.append(f"- {src}: \"{excerpt}\"")
        sources_text = "\n\nSources:\n" + "\n".join(snippets)

    final = f"{answer}\n\n{sources_text}".strip()
    return final

# -----------------------------
# Gradio UI
# -----------------------------
def main_gradio():
    iface = gr.Interface(
        fn=retriever_qa,
        allow_flagging="never",
        inputs=[
            gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Answer (with sources)"),
        title="PDF Retrieval QA (Watsonx + Chroma)",
        description="Upload a PDF and ask questions about its content. Uses Watsonx embeddings and LLM to answer.",
    )
    iface.launch(server_name="127.0.0.1", server_port=7860)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    main_gradio()
