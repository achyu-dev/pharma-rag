

import streamlit as st
from streamlit_jupyter import StreamlitPatcher, tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import requests
import os


DB_CHROMA_PATH = "vector_stores/db_chroma"
EMBEDDINGS_MODEL = "thenlper/gte-large"
LOCAL_API_URL = "http://127.0.0.1:1234"


def get_embeddings_model(model_name=EMBEDDINGS_MODEL, device="cpu"):
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})

def load_vector_db():
    if not os.path.exists(DB_CHROMA_PATH):
        raise FileNotFoundError(f"Chroma database not found at {DB_CHROMA_PATH}")
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)

def search_tool(query):
    vectordb = load_vector_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

def set_custom_prompt():
    custom_prompt_template = """ 
    <s> [INST] You are an assistant for answering pharma-related queries.
    Use the provided context to answer the question.
    If the information is not available, state it clearly. [/INST] </s>
    [INST] Question: {question}
    Context: {context}
    Answer: [/INST]
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def call_local_model_api(prompt):
    try:
        response = requests.post(
            f"{LOCAL_API_URL}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 800, "temperature": 0}
        )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with the local API: {e}"

def router_node(query):
    if "search" in query.lower() or "find" in query.lower():
        context = search_tool(query)
        if context:
            return {"tool": "search", "result": context}
        else:
            return {"tool": "search", "result": "No relevant documents found."}
    else:
        context = search_tool(query)  
        prompt = set_custom_prompt().format(context=context, question=query)
        answer = call_local_model_api(prompt)
        return {"tool": "qa", "result": answer}


def optimized_summarizer(query: str, retriever, llm: LMStudioLLM) -> str:
    try:
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant documents found to summarize."
        combined_docs = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are an expert medical summarizer. Read the following documents and provide a concise and informative summary relevant to the query "{query}".\n\nDocuments:\n{combined_docs}\n\nSummary:"""
        summary = llm._call(prompt)
        return summary.strip()
    except Exception:
        return "Error during summarization."


def main():
    st.title("Pharma Knowledge Assistant")
    st.markdown(
        """
        This tool allows you to query pharmaceutical data using natural language.
        It leverages retrieval-augmented generation (RAG) and agent-based design.
        """
    )

    query = st.text_input("Enter your query:", "")
    action = st.radio("Choose action:", ("Search", "Summarize", "QA"))
    
    if st.button("Submit"):
        if query.strip() == "":
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Processing your query..."):
                if action == "Search":
                    response = router_node(query)
                elif action == "Summarize":
                    response = router_node(query, summarizer=True)
                else:
                    response = router_node(query)
                
                st.success("Query processed successfully!")
                st.subheader("Response")
                st.write(f"**Tool Used**: {response['tool'].capitalize()}")
                st.write(f"**Result**: {response['result']}")

if __name__ == "__main__":
    main()
