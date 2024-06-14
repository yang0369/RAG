from abc import ABC, abstractmethod
from dataclasses import dataclass


class Pipeline(ABC):
    """define an abstract pipeline, any database (e.g. Milvus) must be implemented with the
    required basic methods to work with this project

    Args:
        ABC (_type_): base class
    """

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def embed_text(self):
        pass

    @abstractmethod
    def search(self):
        pass


@dataclass
class KnowledgeGraphPipeline(Pipeline):
    """placeholder for knowledgegraph pipeline

    Args:
        Pipeline (_type_): _description_
    """
    def create(self):
        pass

    def update(self):
        pass

    def delete(self):
        pass

    def embed_text(self):
        pass

    def search(self):
        pass
