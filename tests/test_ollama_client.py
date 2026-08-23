from unittest.mock import MagicMock, patch
from rag_engine.clients.ollama_client import OllamaEmbeddingModel, OllamaInferenceClient

@patch("ollama.Client")
def test_ollama_embedding_embed_query(mock_client_cls):
    # 1. ARRANGE
    mock_client_instance = MagicMock()
    mock_client_cls.return_value = mock_client_instance
    mock_client_instance.embed.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }

    embedder = OllamaEmbeddingModel(model_name="nomic-embed-text")

    # 2. ACT
    result = embedder.embed_query("test query")

    # 3. ASSERT
    mock_client_instance.embed.assert_called_once_with(
        model="nomic-embed-text", input="test query"
    )
    assert result == [0.1, 0.2, 0.3]


@patch("ollama.Client")
def test_ollama_embedding_embed_documents(mock_client_cls):
    # 1. ARRANGE
    mock_client_instance = MagicMock()
    mock_client_cls.return_value = mock_client_instance
    mock_client_instance.embed.return_value = {
        "embeddings": [[0.1, 0.2], [0.3, 0.4]]
    }

    embedder = OllamaEmbeddingModel()

    # 2. ACT
    result = embedder.embed_documents(["doc1", "doc2"])

    # 3. ASSERT
    mock_client_instance.embed.assert_called_once_with(
        model="nomic-embed-text", input=["doc1", "doc2"]
    )
    assert len(result) == 2
    assert result == [[0.1, 0.2], [0.3, 0.4]]


@patch("ollama.Client")
def test_ollama_inference_client_generate(mock_client_cls):
    # 1. ARRANGE
    mock_client_instance = MagicMock()
    mock_client_cls.return_value = mock_client_instance
    mock_client_instance.generate.return_value = {
        "response": "This is a generated answer from Phi-3."
    }

    client = OllamaInferenceClient(model_name="phi3")

    # 2. ACT
    response = client.generate("Why is the sky blue?")

    # 3. ASSERT
    mock_client_instance.generate.assert_called_once_with(
        model="phi3", prompt="Why is the sky blue?"
    )
    assert response == "This is a generated answer from Phi-3."