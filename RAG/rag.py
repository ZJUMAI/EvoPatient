# RAG function from langchain

import os
from dotenv import load_dotenv

from RAG.helper_functions import (  # 需在你的 helper_functions 中提供这些对象
    RecursiveCharacterTextSplitter,
    OpenAIEmbeddings,
    FAISS,
    retrieve_context_per_question,
)


def encode_from_string(content: str, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([content])

    for chunk in chunks:
        chunk.metadata["relevance_score"] = 1.0

    embeddings = OpenAIEmbeddings()  # 从环境变量读取 OPENAI_API_KEY / OPENAI_API_BASE
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def rag_patient(question: str, resource: str, size: int, overlap: int, top_k: int) -> str:
    # 从 .env 读取环境变量（不在代码中写入密钥）
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("未检测到 OPENAI_API_KEY，请在环境或 .env 中设置")

    content = resource
    chunks_vector_store = encode_from_string(content, chunk_size=size, chunk_overlap=overlap)
    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": top_k})

    context = retrieve_context_per_question(question, chunks_query_retriever)

    rag_info = "".join(context)
    return rag_info
