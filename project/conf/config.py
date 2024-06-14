# vector database configs
URI = "http://localhost:19530"
COLLECTION_NAME = "test"

VECTOR_DIMENSION = 1024
METRIC = "COSINE"
VECTOR_INDEX = "IVF_FLAT"

# search range of similarity score, e.g. [0.5, 1.0] inclusive
UPPER_BOUND = 0.999
LOWER_BOUND = 0.5
