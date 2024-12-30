import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import os


EMBEDDINGS_MODEL = "thenlper/gte-large"  
DB_CHROMA_PATH = "vector_stores/db_chroma"  
LOCAL_API_URL = "http://127.0.0.1:1234"  


embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)


def get_chromadb_retriever():
    """
    Load the ChromaDB retriever from the same database used in RAG.
    """
    if not os.path.exists(DB_CHROMA_PATH):
        raise FileNotFoundError(f"Chroma database path not found: {DB_CHROMA_PATH}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"})
    vectordb = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    return vectordb


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def generate_recommendations_from_chromadb(user_input):
    """
    Generate medical recommendations based on user input using the shared ChromaDB.
    """
    try:
        
        vectordb = get_chromadb_retriever()
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})  

        
        retrieved_docs = retriever.get_relevant_documents(user_input)

        
        if not retrieved_docs:
            return "No relevant symptoms found in the database."

        context = "\n".join([doc.page_content for doc in retrieved_docs])

        
        inputs = tokenizer.encode_plus(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50
        )

        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

        
        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        medicines = []
        for doc in retrieved_docs:
            medicines.extend(json.loads(doc.metadata.get("medicines", "[]")))

        if not medicines:
            return f"Model suggests: {model_response}. However, no direct medicine recommendation found for your symptoms."
        else:
            return f"Based on your symptoms, we recommend: {', '.join(set(medicines))}. Model suggests: {model_response}"
    except Exception as e:
        return f"Error generating recommendations: {e}"


if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    print(generate_recommendations_from_chromadb(user_input))