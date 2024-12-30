import os
import requests
from typing import Dict, Any, List
from langchain.llms.base import LLM
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import Field


class LocalLLMAdapter(LLM):
    api_url: str = Field(...)  

    def _init_(self, api_url: str):
        super()._init_(endpoint=api_url)

    def _call(self, input_text: str, stop: None = None, **kwargs: Any) -> str:
        try:
            import time
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json={
                    "model": "llama-3.2-3b-instruct",
                    "messages": [{"role": "user", "content": input_text}],
                    "max_tokens": 1000,
                },
                timeout=100  
            )
            duration = time.time() - start_time
            print(f"[DEBUG] LLM call duration: {duration:.2f} seconds")
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"Error with LLM API: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("[ERROR] Request to LLM timed out.")
        except Exception as err:
            raise Exception(f"[ERROR] Unexpected error: {str(err)}")

    @property
    def _llm_type(self) -> str:
        return "local_llm_adapter"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"api_url": self.api_url}


def generate_summary(user_query: str, doc_retriever, language_model: LLM) -> str:
    
    relevant_docs = doc_retriever.get_relevant_documents(user_query)

    if not relevant_docs:
        return "No related documents were found."

    
    all_docs_combined = "\n\n".join([doc.page_content for doc in relevant_docs])

    
    summarization_prompt = f"""You are a skilled summarizer specializing in medical content. 
Analyze the following materials and provide a clear, concise summary addressing the query "{user_query}".

Documents:
{all_docs_combined}

Summary:"""

    
    summary = language_model(summarization_prompt)

    return summary.strip()

if __name__ == "__main__":
    
    database_path = ".\chroma_db"
    server_url = "http://127.0.0.1:1234"

    
    print("[INFO] Initializing vector database...")
    embedding_model = OpenAIEmbeddings()  
    doc_store = Chroma(persist_directory=database_path, embedding_function=embedding_model)
    print("[INFO] Vector database loaded successfully.")

    
    print("[INFO] Setting up the LLM interface...")
    local_llm = LocalLLMAdapter(api_url=server_url)
    print("[INFO] LLM setup complete.")

    
    doc_retriever = doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})  

    
    sample_queries = [
        "Summarize the usage and side effects of Amoxicillin."
    ]

    for question in sample_queries:
        print(f"\n[USER QUERY]: {question}")
        result_summary = generate_summary(question, doc_retriever, local_llm)
        print(f"[SUMMARY RESULT]: {result_summary}")