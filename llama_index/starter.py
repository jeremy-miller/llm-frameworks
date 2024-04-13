import logging
import sys

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.llm = Ollama(model="mistral", request_timeout=300.0)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
