"""
This script demostrate how to build a vector database using Milvus for SLR dataset
"""
import json
from pathlib import Path

from conf.config import COLLECTION_NAME, METRIC, VECTOR_DIMENSION
from src.vector_db.milvus_pipeline import VectorDBPipeline
from tqdm import tqdm


PROJ = Path(__file__).parents[2]

vdb = VectorDBPipeline(collection_name=COLLECTION_NAME)
data_dir = PROJ / "data/slr_training_dataset_240125_1523.json"

test_data = [
    {"title": "computer science",
     "text": "Computer science is the study of computation, information, and automation.[1][2][3] Computer science \
         spans theoretical disciplines (such as algorithms, theory of computation, and information theory) to applied \
             disciplines (including the design and implementation of hardware and software)."},
    {"title": "data science",
     "text": "Data science is the study of data to extract meaningful insights for business. It is a multidisciplinary \
         approach that combines principles and practices from the fields of mathematics, statistics, \
             artificial intelligence, and computer engineering to analyze large amounts of data."},
    {"title": "urban planning",
     "text": "Urban planning includes techniques such as: predicting population growth, zoning, geographic mapping \
         and analysis, analyzing park space, surveying the water supply, identifying transportation patterns, \
             recognizing food supply demands, allocating healthcare and social services, and analyzing the impact of \
                 land use."},
]

# create collection
vdb.create(
    vector_dim=VECTOR_DIMENSION,
    metric_type=METRIC,
    drop=True
)


# build database
data = list()
for element in tqdm(test_data):
    title = element.get("title", "")
    text = element.get("text", "")
    vector = vdb.embed_text(text)
    data.append({"embeddings": vector, "text": text, "title": title})

vdb.update(data, partition_name="vdb")


# query
query = """
Spatial analysis refers to modeling location-specific problems, identifying patterns,
and assessing spatial data to make decisions."""

result = vdb.search(query=query,
                    partition_names=["vdb"],
                    k=5)

print(f"Based on text similarity, the topics related to spatial anslysis shall be: {result[-1]}")
