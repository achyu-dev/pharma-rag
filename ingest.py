from langchain_community.document_loaders import ArxivLoader
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import embeddings
from openai.types import EmbeddingModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from sympy.physics.units import temperature
from tensorboard.backend.event_processing.directory_loader import DirectoryLoader

DATA_PATH = 'datasets\microlabs_usa'
DB_CHROMA_PATH = 'vector_stores/db_chroma'
EMBEDDINGS_MODEL = 'thenlper/gte-large'

def get_docs(source="arxiv"):
    """
    loads the documents from the given source
    :return:
    """
    if source=="arxiv":
        docs=ArxivLoader(query="1706.03762",load_max_docs=2).load()
    else:
        loader = DirectoryLoader(DATA_PATH, glob = "*.json",loader_cls="JSONLoader", recursive=True)
        docs= loader.load()
    return docs


def get_chunks(docs, chunk_size=512,chunk_overlap=56):
    """
    Given docs obtained by using LangChain loader , split these to chunks and return them
    :param docs: documents returned by loader that could be ArxivLoader or local directory loader
    :return: chunks
    """
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    texts=text_splitter.split_documents(docs)
    return texts


def get_embeddings_model(model_name = None , device = "cpu"):
    if model_name is None:
        model_name = EMBEDDINGS_MODEL
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={"device" : device})
    return embeddings_model

def create_vector_store(texts, embeddings,db_path,use_db="chroma"):
    """
    Given the chunks , their embeddings and path to save to the db, save and persist the data in the data store
    :param texts: chunks for which we are constructing the data store
    :param embeddings: vector embeddings for given chunks
    :param db_path: storage path
    :param use_db: type of data store to use
    :return: None
    """
    flag = True
    try:
        if use_db == "chroma":
            db = Chroma.from_documents(texts,embeddings,persist_directory= db_path)
        else:
            print("unknown db")
            db = None
            import sys
            sys.exit(-1)
        db.persist()
    except:
        flag = False
        
        print("exception when creating datastore",db_path, use_db)
    return flag

def ingest():
    """
    Ingest PDF files from given data source
    :param source : Can be "arxiv" to load from arxiv website "local" from directory loader
    :return:
    """
    
    docs = get_docs()
    

    
    texts= get_chunks(docs)
    print(len(docs),len(texts))

    
    embs_model = get_embeddings_model(device="cpu")

    
    flag = create_vector_store((texts,embs_model, DB_CHROMA_PATH, "chroma"))
    if flag:
        print("vector store created")


if __name__=='__main__ ':
    ingest()
    
