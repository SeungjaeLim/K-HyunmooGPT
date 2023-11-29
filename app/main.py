from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from tqdm import tqdm
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from pdf_data_load import pdf_file_load, split_text

app = FastAPI()

# Create a Chroma client
client = PersistentClient(path="./data")

CHUNK_SIZE = 256


# Load your Sentence Transformer model
sbert_model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
sbert_model_name = 'jhgan/ko-sroberta-multitask'
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sbert_model_name)

# Get or create the Chroma collection
collection = client.get_or_create_collection(name="chunk_store", 
                                             embedding_function=sentence_transformer_ef,
                                             metadata={"hnsw:space": "cosine"},
                                             )
kv_collection = client.get_or_create_collection(name="kv_store", 
                                                embedding_function=sentence_transformer_ef,
                                                metadata={"hnsw:space": "cosine"} ,
                                                )

class DataInput(BaseModel):
    filename: str

import os
@app.post("/data-input-txt")
async def input_data(filename: DataInput):
    # Read documents from a file and add them to the collection
    # text = pdf_file_load(filename)
    # read txt file from path
    filename = filename.filename
    # Get a list of all files in the directory
    file_names = os.listdir("./history_script/")

    # Print the list of file names
    for filename in file_names:
        print(filename)
        with open(f"./history_script/{filename}", "r") as f:
            text = f.read()
        documents = split_text(text, CHUNK_SIZE, CHUNK_SIZE//4)
        #get max document id
        document_ids = [f"{filename}_{idx}" for idx in range(len(documents))]

        collection.add(documents=documents, ids=document_ids)
        print("data count: ", len(documents))

    return {"message": "Data successfully added to the collection."}

@app.post("/data-input-pdf")
async def input_data(filename: DataInput):
    # Read documents from a file and add them to the collection
    filename = filename.filename
    print(filename)
    text = pdf_file_load(filename)

    documents = split_text(text, CHUNK_SIZE, CHUNK_SIZE*3//4)
    document_ids = [f"{filename}_{idx}" for idx in range(len(documents))]    
    collection.add(documents=documents, ids=document_ids)
    print("data count: ", len(documents))

    return {"message": "Data successfully added to the collection."}


@app.post("/data-input-kv")
async def input_data(filename: DataInput):
    # Read documents from a file and add them to the collection
    filename = filename.filename

    keys = []
    documents = []


    document_ids = [f"{filename}_{idx}" for idx in range(len(documents))]    

    kv_collection.add(
        embeddings=kv_collection.embedding_function.encode(keys),
        documents=documents,
        ids=document_ids,
    )
    # update items in a collection
    collection.update()

    return {"message": "Data successfully added to the collection."}


@app.post("/data-input-thesaurus")
async def input_data(filename: DataInput):
    # Read documents from a file and add them to the collection
    filename = filename.filename
    df = pd.read_csv(filename)
    documents = []
    keys = []
    embeddings = []
    document_ids = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc = f"{row['term_name']} {row['term_kind']} {row['term_ch']} {row['term_remark']} {row['term_attr']} {row['term_year']} {row['term_times']} {row['term_lk']} {row['term_desc']}"
        documents.append(doc)
        keys.append(row['term_name'])
        document_ids.append(f"{filename}_{idx}")    
        if len(keys) == 64:
            embeddings = sentence_transformer_ef(keys)
            kv_collection.add(documents=documents, ids=document_ids, embeddings=embeddings)
            documents = []
            keys = []
            embeddings = []
            document_ids = []
            
    embeddings = sentence_transformer_ef(keys)
    kv_collection.add(documents=documents, ids=document_ids, embeddings=embeddings)
    
    embeddings += sentence_transformer_ef(keys)
    print("data count: ", len(documents))

    return {"message": "Data successfully added to the collection."}


@app.get("/er-data-input")
async def er_input_data(filename: DataInput):
    filename = filename.filename
    print(filename)
    with open(filename, "r") as f:
        text = f.read()
    documents = split_text(text, CHUNK_SIZE, CHUNK_SIZE*3//4)
    document_ids = [f"id{idx}" for idx in range(len(documents))]

    collection.add(documents=documents, ids=document_ids)

    return {"message": "Data successfully added to the entity relation collection."}

class QueryInput(BaseModel):
    query: str
    n_results: int = 5

@app.post("/query-top-n")
async def query_top_n(data: QueryInput):
    query = data.query
    n_results = data.n_results

    try:
        # Include the source document and the Cosine Distance in the query result
        result = collection.query(query_texts=[query], n_results=n_results, include=["documents", 'distances'])

        # Extract the first (and only) list inside 'ids'
        ids = result.get('ids')[0]
        # Extract the first (and only) list inside 'documents'
        documents = result.get('documents')[0]
        # Extract the first (and only) list inside 'distances'
        distances = result.get('distances')[0]

        response_data = []

        for id_, document, distance in zip(ids, documents, distances):
            # Cosine Similarity is calculated as 1 - Cosine Distance
            similarity = 1 - distance
            response_data.append({"id": id_, "document": document, "similarity": similarity})

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




@app.post("/query-kv-top-n")
async def query_kv_top_n(data: QueryInput):
    query = data.query
    n_results = data.n_results

    try:
        # Include the source document and the Cosine Distance in the query result
        result = kv_collection.query(query_texts=[query], n_results=n_results, include=["documents", 'distances'])

        # Extract the first (and only) list inside 'ids'
        ids = result.get('ids')[0]
        # Extract the first (and only) list inside 'documents'
        documents = result.get('documents')[0]
        # Extract the first (and only) list inside 'distances'
        distances = result.get('distances')[0]

        response_data = []

        for id_, document, distance in zip(ids, documents, distances):
            # Cosine Similarity is calculated as 1 - Cosine Distance
            similarity = 1 - distance
            response_data.append({"id": id_, "document": document, "similarity": similarity})

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
