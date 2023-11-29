from chromadb import HttpClient
# from embedding_util import CustomEmbeddingFunction


# Create a Chroma client
client = HttpClient(host="localhost", port=8000)

# Testing to ensure that the chroma server is running
print('HEARTBEAT:', client.heartbeat())

# Get a collection object from an existing collection, by name. If it doesn't exist, create it.


from chromadb.utils import embedding_functions

sbert_model_name = 'jhgan/ko-sroberta-multitask'
sbert_model_name = 'snunlp/KR-SBERT-Medium-NLI-STS'
sbert_model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=sbert_model_name)

collection = client.get_or_create_collection(
    name="test_0", embedding_function=sentence_transformer_ef)


"""
Read documents from a file and add them to the collection.
"""
from app.pdf_data_load import pdf_file_load, split_text

text = pdf_file_load("test.pdf")
documents = split_text(text, 512, 128)
print(documents[0])
# Every document needs an id for Chroma
document_ids = list(map(lambda tup: f"id{tup[0]}", enumerate(documents)))

collection.add(documents=documents, ids=document_ids)

query = "고종은 누구입니까?"

# Include the source document and the Cosine Distance in the query result
result = collection.query(query_texts=[query],
                          n_results=5, include=["documents", 'distances',])

print("Query:", query)
print("Most similar sentences:")
# Extract the first (and only) list inside 'ids'
ids = result.get('ids')[0]
# Extract the first (and only) list inside 'documents'
documents = result.get('documents')[0]
# Extract the first (and only) list inside 'documents'
distances = result.get('distances')[0]

for id_, document, distance in zip(ids, documents, distances):
    # Cosine Similiarity is calculated as 1 - Cosine Distance
    print(f"ID: {id_}, Document: {document}, Similarity: {1 - distance}")
