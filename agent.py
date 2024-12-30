from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import requests


DATA_PATH = "datasets/pharma_data"
DB_CHROMA_PATH = "vector_stores/db_chroma"
EMBEDDINGS_MODEL = "thenlper/gte-large"
LOCAL_API_URL = "http://127.0.0.1:1234"

def get_embeddings_model(model_name=EMBEDDINGS_MODEL, device="cpu"):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return embeddings_model

def load_vector_db():
    if not os.path.exists(DB_CHROMA_PATH):
        raise FileNotFoundError(f"Chroma database not found at {DB_CHROMA_PATH}")
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)


custom_prompt_template = """ 
<s> [INST] You are an assistant for answering pharma-related queries.
Use the provided context to answer the question.
If the information is not available, state it clearly. [/INST] </s>
[INST] Question: {question}
Context: {context}
Answer: [/INST]
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def search_tool(query):
    vectordb = load_vector_db()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])


def call_local_model_api(prompt):
    try:
        response = requests.post(
            f"{LOCAL_API_URL}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 800, "temperature": 0}
        )
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error with local API: {e}")


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
    

def pharma_query_bot():
    print("Welcome to the Pharma Query Assistant!")
    print("Type your query or 'quit' to exit.")
    
    while True:
        query = input("Your Query: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        
        result = router_node(query)
        print(f"Tool Used: {result['tool'].capitalize()}")
        print(f"Response: {result['result']}\n")

if __name__ == "__main__":
    pharma_query_bot()
    
