from dataclasses import dataclass
from typing import Union, List, Tuple

import pydash
from conf.config import (LOWER_BOUND, METRIC, UPPER_BOUND,
                         URI, VECTOR_DIMENSION, VECTOR_INDEX)
from pymilvus import DataType, MilvusClient
from sentence_transformers import SentenceTransformer
from src.vector_db.pipeline import Pipeline

MODEL = SentenceTransformer("Snowflake/snowflake-arctic-embed-l", device="cpu")


@dataclass
class VectorDBPipeline(Pipeline):
    """Vector database implemented based on Milvus database

    Args:
        Pipeline (_type_): abstract class or template
    """
    collection_name: str
    embedding_model = MODEL

    def __post_init__(self):
        # Connect to Milvus DB server
        self.client = MilvusClient(
            uri=URI
        )
        # TODO: add authen

    def update_collection_name(self, collection_name: str) -> None:
        """update collection name for the pipeline

        Args:
            collection_name (str): name
        """
        self.collection_name = collection_name

    def create(self, drop=False, **kwargs) -> None:
        """create the vector collection, which is equivalent to a table in relational DB

        Args:
            drop (bool, optional): drop the existing database if exists. Defaults to False.
        """
        if self.client.has_collection(self.collection_name):
            if drop:
                self.client.drop_collection(self.collection_name)
            else:
                return

        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        # define fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embeddings",
                         datatype=DataType.FLOAT_VECTOR,
                         dim=VECTOR_DIMENSION,
                         description="embedding")
        schema.add_field(field_name="text",
                         datatype=DataType.VARCHAR,
                         max_length=65535,
                         description="text description")
        schema.add_field(field_name="title",
                         datatype=DataType.VARCHAR,
                         max_length=65535,
                         description="topic")

        # define index
        index_params = self.client.prepare_index_params()

        # scalar index
        index_params.add_index(
            field_name="text",
            index_type="INVERTED",
        )

        # vector index
        index_params.add_index(
            field_name="embeddings",
            index_type=VECTOR_INDEX,
            metric_type=METRIC,
            params={"nlist": 128}
        )

        # Create a collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            **kwargs
        )

    def update(self,
               data,
               partition_name: str = "_default") -> None:
        """add new vector entry to the database

        Args:
            data (_type_): _description_
            collection_name (_type_): _description_
            partition_name (str, optional): recommended to partition collection for better speed.
            Defaults to "_default".
        """
        if not self.client.has_partition(
            collection_name=self.collection_name,
            partition_name=partition_name
        ):
            self.client.create_partition(
                collection_name=self.collection_name,
                partition_name=partition_name
            )

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
            partition_name=partition_name
        )

    def delete(self, ids: List[int]) -> None:
        """delete any particular data points

        Args:
            ids (list[int]):
        """
        self.client.delete(
            collection_name="quick_setup",
            filter=f"id in {ids}"
        )

    def search(
        self,
        query: Union[str, List[str]],
        partition_names: List[str],
        k: int = 3) -> Tuple[list, list, list, list]:
        """vector index search for top k similar texts

        Args:
            query (Union[str, list]): query by single query or list of query
            partition_names (list[str]): this partition can be infered by LLM by classification
            k (int, optional):  top k number of search results based on metric score. Defaults to 3.

        Returns:
            tuple[list, list, list]: returned texts
        """
        if isinstance(query, str):  # Single-vector search
            vector = self.embedding_model.encode(query)
        elif isinstance(query, list):  # Bulk-vector search
            vector = [self.embedding_model.encode(q) for q in query]

        # In normal cases, you do not need to set search parameters manually
        # Except for range searches.
        search_parameters = {
            'metric_type': METRIC,
            'params': {
                'nprobe': 10,
                'radius': LOWER_BOUND,
                'range_filter': UPPER_BOUND
            }
        }

        res = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=k,
            search_params=search_parameters,
            # group_by_field="neutral_citation",  # this does not support range
            output_fields=["text", "title"],
            partition_names=partition_names,
            # filter='color like "red%"'  # Filtered search applied on scalar attributes
        )[0]

        ids = list()
        scores = list()
        texts = list()
        title = list()

        # re-format
        for itm in res:
            ids.append(itm.get("id"))
            scores.append(itm.get("distance"))
            texts.append(pydash.get(itm, ["entity", "text"]))
            title.append(pydash.get(itm, ["entity", "title"]))

        return ids, scores, texts, title

    def embed_text(self, text: str):
        return self.embedding_model.encode(text)

    @property
    def view_all_collections(self):
        return self.client.list_collections()

    def drop_a_collection(self, name: str) -> None:
        self.client.drop_collection(collection_name=name)

    def view_a_collection(self, name: str) -> None:
        print(self.client.describe_collection(collection_name=name))

    def delete_entities(self, collection_name: str, ids: List[int]) -> None:
        self.client.delete(
            collection_name=collection_name,
            ids=ids,
            partition_name="partitionA"
        )