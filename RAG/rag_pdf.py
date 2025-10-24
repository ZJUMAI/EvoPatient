import sys
from pathlib import Path
from dotenv import load_dotenv

# 允许从上一级目录导入本地工具（相对路径）
sys.path.append(str(Path.cwd().parent))
from helper_functions import *          # 需要: PyPDFLoader, RecursiveCharacterTextSplitter, OpenAIEmbeddings, FAISS, replace_t_with_space
from evaluation.evalute_rag import *    # 需要: retrieve_context_per_question, evaluate_rag

# 从 .env 读取环境变量（OPENAI_API_KEY / 可选 OPENAI_API_BASE）
load_dotenv()

# 相对路径 PDF（请将文件放在上级 data 目录，或自行调整）
PDF_PATH = Path("../data/Multi-Agent Collaboration via Cross-Team Orchestration.pdf")

def encode_pdf(path: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = OpenAIEmbeddings()  # 从环境变量读取密钥/基址
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore

chunks_vector_store = encode_pdf(PDF_PATH, chunk_size=1000, chunk_overlap=200)
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

test_query = "What is the main content of this article?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)

evaluate_rag(chunks_query_retriever)
