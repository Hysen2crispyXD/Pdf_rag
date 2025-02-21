import os
import re
import streamlit as st
import fitz  
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from FlagEmbedding import FlagReranker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
GOOGLE_API_KEY = "AIzaSyCLDum-DzmQXy_5kMPx7UFCMgm1UIi62vA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key="AIzaSyCLDum-DzmQXy_5kMPx7UFCMgm1UIi62vA")

chat_history = []

def clean_text(text):
    cleaned_text = re.sub(r'\bPage\s*\d+\b', '', text)  
    cleaned_text = re.sub(r'\s*\b\d+\b\s*$', '', text, flags=re.MULTILINE)  
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  
    return cleaned_text.strip()

def curves_to_edges(cs):
    edges = []
    for c in cs:
        if hasattr(c, "rect_to_edges"): 
            edges += fitz.utils.rect_to_edges(c)
        else:
            edges.append(c)  
    return edges



def load_chunks(file_path, embedding):
    documents = []           
    with fitz.open(file_path) as pdf:
        for page in pdf[3:]:  
            p = page
            tables = p.find_tables()
    
            if tables:
                drawings = p.get_drawings()
                curves = [d for d in drawings if d["type"] == "curve"]
                edges = [d for d in drawings if d["type"] == "edge"]
                
                ts = {  
                    "vertical_strategy": "explicit",
                    "horizontal_strategy": "explicit",
                    "explicit_vertical_lines": curves_to_edges(curves + edges),
                    "explicit_horizontal_lines": curves_to_edges(curves + edges),
                    "intersection_y_tolerance": 10,
                }
                bboxes = [table.bbox for table in tables]
                
                def not_within_bboxes(block):
                    """Check if the block is in any of the table's bbox."""
                    def block_in_bbox(_bbox):
                        v_mid = (block["bbox"][1] + block["bbox"][3]) / 2
                        h_mid = (block["bbox"][0] + block["bbox"][2]) / 2
                        x0, top, x1, bottom = _bbox
                        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)
                    return not any(block_in_bbox(__bbox) for __bbox in bboxes)
                
                text = p.get_text("dict")
                filtered_text = [block for block in text["blocks"] if block["type"] == 0 and not_within_bboxes(block)]
                clean_text_content = clean_text(" ".join([span["text"] for block in filtered_text for line in block["lines"] for span in line["spans"]]))
                if clean_text_content:
                    documents.append(Document(page_content=clean_text_content, metadata={
                        "type": "text",
                        "page_number": page.number
                    }))
              
                for table_index, table in enumerate(tables):
                    table_content = str(table.extract())                  
                    documents.append(Document(page_content=table_content, metadata={
                        "type": "table",
                        "table_index": table_index,
                        "columns": len(table.rows[0].cells) if table.rows else 0,
                        "rows": len(table.rows),
                        "page_number": page.number,     
                    }))
            else:
                text = p.get_text()
                if text:                
                    clean_text_content = clean_text(text)
                    documents.append(Document(page_content=clean_text_content, metadata={
                        "type": "text",
                        "page_number": page.number
                    }))
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    # text_splitter = SemanticChunker(
    #     embedding,
    #     buffer_size=1,
    #     breakpoint_threshold_type="percentile",
    #     breakpoint_threshold_amount=85,
    # )
    text_chunks = []
    for doc in documents:
        if doc.metadata["type"] == "text":
            text_chunks.extend(text_splitter.transform_documents([doc]))
        else:
            text_chunks.append(doc)
            
    return text_chunks

def vector_store(text_chunks,embedding):
    vector_store = FAISS.from_documents(text_chunks, embedding=embedding)
    vector_store.save_local("Faiss_index")


class Hybrid_search:
  def __init__(self,text_chunks, semantic_retriever, bm25_retriever, reranker):
    self.semantic_retriever = semantic_retriever
    self.bm25_retriever = bm25_retriever
    self.reranker = reranker
    self.text_chunks=text_chunks

  def __call__(self,query):
    semantic_results = self.semantic_retriever.similarity_search(
      query,
      k=10,
    )
    bm25_results = self.bm25_retriever.invoke(query)

    content = set()
    retrieval_docs = []

    for result in semantic_results:
      if result.page_content not in content:
        content.add(result.page_content)
        retrieval_docs.append(result)

    for result in bm25_results:
      if result.page_content not in content:
        content.add(result.page_content)
        retrieval_docs.append(result)

    pairs = [[query,doc.page_content] for doc in retrieval_docs]

    scores = self.reranker.compute_score(pairs,normalize = True)
    
    high_score_context=[]
    low_score_context=[]
    high_id = set()
    low_id=set()
    for i in range(len(retrieval_docs)):
        if scores[i] >= 0.6:
            doc_idx = retrieval_docs[i]
            page_number =retrieval_docs[i].metadata.get("page_number")
            if doc_idx.page_content not in high_id:
                high_id.add(doc_idx.page_content)
                high_score_context.append(doc_idx)
        
            for table_chunk in self.text_chunks:
                if (
                    table_chunk.metadata.get("type") == "table" and
                    table_chunk.metadata.get("page_number") == page_number
                ):
                    if table_chunk.page_content not in high_id:
                        high_id.add(table_chunk.page_content)
                        high_score_context.append(table_chunk)
                    
        elif scores[i] >= 0.1:
            doc_idx = retrieval_docs[i]
            page_number =retrieval_docs[i].metadata.get("page_number")
            if doc_idx.page_content not in low_id:
                low_id.add(doc_idx.page_content)
                low_score_context.append(doc_idx)
        
            for table_chunk in self.text_chunks:
                if (
                    table_chunk.metadata.get("type") == "table" and
                    table_chunk.metadata.get("page_number") == page_number
                ):
                    if table_chunk.page_content not in low_id:
                        low_id.add(table_chunk.page_content)
                        low_score_context.append(table_chunk)
                    
    if len(high_score_context) > 0:
      return high_score_context
          
    elif len(low_score_context) > 0:
      return low_score_context
    
    else:
      return []


def generate_answer(user_question,retriever):
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, max_tokens=3000)
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Do NOT answer the question, just reformulate it if needed and otherwise return it.
    You must return the question in Vietnamese.
    Chat history: {history}\n
    User question: {question}\n
    """
    
    contextualize_q_prompt = PromptTemplate(template=contextualize_q_system_prompt, input_variables=["question","history"])
    previous_question = chat_history[-1]["question"] if chat_history else None
    
    contextualize_q_chain = LLMChain(llm=llm, prompt=contextualize_q_prompt)
    query= contextualize_q_chain({"question": user_question, "history": previous_question or ""}, return_only_outputs=True)
    contexts = retriever(query["text"])
    
    qa_system_prompt = """
    You are an assistant for question-answering tasks. 
   
    Answer the question as detailed as possible from the provided contexts which can include text and tables (identified by metadata "type":) , make sure to provide all the details \
    
    Some of the tables in the context (based on metadata "page_number" is consecutive and "colummns" are equal) might highly relate to each other as parts of the same dataset then the answer should contain all of them.\
        
    If the answer is not in provided contexts just say, "Không tìm thấy thông tin liên quan", don't provide the wrong answer\
    You must answer the question in Vietnamese\n\n
    Context: \n{context}\n
    Question: \n{question}?\n
    
    Answer:
    """
    llm_answer = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1,max_tokens=3000)
    qa_prompt = PromptTemplate(template=qa_system_prompt, input_variables=["question","context"])

    # Tạo QA chain
    question_answer_chain = load_qa_chain(llm_answer,chain_type="stuff", prompt=qa_prompt)
    
    # Kết hợp retriever và QA chain
    rag_chain = question_answer_chain({"input_documents": contexts, "question": query["text"]}, return_only_outputs=True)
    
    return contexts,rag_chain

def user_input(user_question, text_chunks,embedding,reranker):
    try:
        db =  FAISS.load_local(os.path.join(os.getcwd(), "Faiss_index"), embedding,
                                  allow_dangerous_deserialization=True)
       
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return
    
    bm25_retriever = BM25Retriever.from_documents(text_chunks, k=10)
    
    retriever=Hybrid_search(text_chunks,db,bm25_retriever,reranker)
    
    contexts,chain = generate_answer(user_question,retriever)
    
    chat_history.append({"question": user_question, "answer": chain["output_text"]})
    print(contexts)



def save_cache(data, cache_path="text_chunks_cache.pkl"):
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)


def load_cache(cache_path="text_chunks_cache.pkl"):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def main():
    #embedding = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    
    col1, col2 = st.columns([7, 1.5])
    
    back_img="D:\\NLP first\\logo.png"
    with col1:
        st.title("Hệ Thống Hỏi Đáp Sinh Viên")
    with col2:
        st.image(back_img,width=100)
        
    with st.spinner("Processing..."):  
        cache_path = "text_chunks_cache.pkl"
        text_chunks = load_cache(cache_path)  # Kiểm tra cache
        if text_chunks is None:
            st.info("Đang xử lý tài liệu lần đầu...")
            text_chunks = load_chunks("D:\\NLP first\\STSV.pdf", embedding)
            save_cache(text_chunks, cache_path)  # Lưu vào cache
            st.success("Dữ liệu đã được lưu vào cache.")
            
        vector_store(text_chunks,embedding) 
        
    if "messages" not in st.session_state:
        st.session_state.messages = []   
             
    if not st.session_state.messages:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Chào mừng bạn đến với hệ thống hỏi đáp Đại Học Bách Khoa Hà Nội !"
            })
            
    for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    prompt = st.chat_input('Hãy đặt câu hỏi về Đại Học Bách Khoa Hà Nội')
    
    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)

        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.spinner('Đang tìm kiếm câu trả lời...'):
            user_input(prompt,text_chunks,embedding,reranker)  
                
        for i, pair in enumerate(chat_history, start=1):
                with st.chat_message('assistant'):
                    st.markdown(f"**Câu hỏi**: {pair['question']}")
                    st.markdown(f"**Trả lời**: {pair['answer']}")
                st.session_state.messages.append({'role': 'assistant', 'content': f"**Câu hỏi**: {pair['question']}  \n\n **Trả lời**: {pair['answer']}"})
           
    with st.sidebar:     
        if st.button("Đoạn chat mới"):
                st.session_state.messages = []  
                chat_history.clear() 
                st.markdown(" ")
                st.rerun() 
            
if __name__ == "__main__":
    main()