import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CTIEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None

    def _initialize(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._initialize()
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=False)
        return embeddings.tolist()    
     
