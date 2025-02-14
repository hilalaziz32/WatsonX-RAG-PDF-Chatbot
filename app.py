from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
import warnings

warnings.filterwarnings('ignore')

def get_llm():
    model_id = "mistralai/mixtral-8x7b-instruct-v01"
    parameters = {
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.5,
    }
    
    project_id = "skills-network"
    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters
    )
    return watsonx_llm

def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    loaded_doc = loader.load()
    return loaded_doc

def text_splitter(data):
    sp = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=55,
        length_function=len
    )
    chunks = sp.split_documents(data)
    return chunks

def embedding_model():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding

def vector_databases(chunks):
    embedding = embedding_model()
    vectordb = Chroma.from_documents(chunks, embedding)
    return vectordb

def analyze_document(file_path):
    """Analyzes the document: provides word count, key topics, and summary."""
    llm = get_llm()
    
    # Load and process document
    document = document_loader(file_path)
    text_content = " ".join([page.page_content for page in document])
    word_count = len(text_content.split())
    
    # Generate summary
    summary_prompt = f"Summarize the following document in 3-4 sentences: {text_content[:3000]}"
    summary = llm.invoke(summary_prompt)
    
    # Extract main topics
    topic_prompt = f"Extract the main topics covered in this document: {text_content[:3000]}"
    topics = llm.invoke(topic_prompt)
    
    return f"Document Insights:\n- **Word Count:** {word_count}\n- **Summary:** {summary}\n- **Main Topics:** {topics}"

def retriever(file_path):
    splits = document_loader(file_path)
    chunks = text_splitter(splits)
    vectordb = vector_databases(chunks)
    return vectordb.as_retriever()

def retriever_qa(file_path, query):
    llm = get_llm()
    ret_obj = retriever(file_path)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ret_obj,
        return_source_documents=False
    )
    response = qa.invoke(query)
    return response["result"]

def process_document(file_path, query):
    if not query:
        return analyze_document(file_path)
    else:
        return retriever_qa(file_path, query)

rag_application = gr.Interface(
    fn=process_document,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query (Leave blank for auto-analysis)", lines=2, placeholder="Ask a question or leave blank for insights...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Enhanced RAG Chatbot",
    description="Upload a PDF document. First, the chatbot will analyze the document and provide insights. You can then ask any question about the document.",
)

rag_application.launch(server_name="0.0.0.0", server_port=7860, share=True)
