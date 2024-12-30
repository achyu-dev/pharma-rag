import requests
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

EMBEDDINGS_MODEL = "thenlper/gte-large"  
DB_CHROMA_PATH = "vector_stores/db_chroma"  
LOCAL_API_URL = "http://127.0.0.1:1234"  


custom_prompt_template = """ 
<s> [INST] You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know, just say that you don't know.
Keep the answer concise. [/INST] </s>
[INST] Question: {question}
Context: {context}
Answer: [/INST]
"""


def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


'''def call_local_model_api(prompt):
    """
    Sends a request to the local API for model inference.

    :param prompt: The input prompt for the model.
    :return: Model's response as text.
    """
    try:
        response = requests.post(
            f"{LOCAL_API_URL}/v1/chat/completions",
            json={"prompt": prompt, "max_tokens": 800, "temperature": 0}
        )
        response.raise_for_status()
        return response.json().get("response", "No response received.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error communicating with the local API: {e}")
'''

def get_retriever():
    """
    After texts are ingested into vectordb, get it as a retriever.
    """
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"})

    
    if not os.path.exists(DB_CHROMA_PATH):
        raise FileNotFoundError(f"Chroma database path not found: {DB_CHROMA_PATH}")

    
    vectordb = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    return vectordb


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def QAnswer_Bot():
    try:
        
        vectordb = get_retriever()
        retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        print("Retriever initialized successfully!")

        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | set_custom_prompt()
            | call_local_model_api
            | StrOutputParser()
        )

        
        query = ""
        while query.lower() != "quit":
            query = input("Your Query: ")
            if query.lower() == "quit":
                print("Exiting QAnswer_Bot. Goodbye!")
                break

            
            retrieved_context = retriever.get_relevant_documents(query)
            formatted_context = format_docs(retrieved_context)
            prompt = custom_prompt_template.format(context=formatted_context, question=query)

            
            output = call_local_model_api(prompt)
            print(f"Answer: {output}")
    except Exception as e:
        print(f"Error: {e}")

def call_local_model_api(prompt):
    """
    Sends a request to the local API for model inference.
    """
    messages = [
        {"role": "system", "content": "You are an assistant for question-answering tasks."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = requests.post(
            f"{LOCAL_API_URL}/v1/chat/completions",
            json={"messages": messages, "max_tokens": 800, "temperature": 0}
        )
        response.raise_for_status()
        print(f"API Response: {response.text}")  
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response received.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error communicating with the local API: {e}")

if __name__ == "__main__":
    QAnswer_Bot()
