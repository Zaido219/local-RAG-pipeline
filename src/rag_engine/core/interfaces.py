from abc import ABC, abstractmethod

class BaseInferenceClient(ABC):
    @abstractmethod
    def generate(self,prompt:str) -> str:
        pass