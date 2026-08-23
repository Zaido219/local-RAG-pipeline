from rag_engine.services.prompt_builder import PromptBuilderService
from rag_engine.core.models import RetrievalResult, TextChunk

def test_build_prompt_with_retrieved_results():
    # 1. ARRANGE: Create mock domain objects
    chunk1 = TextChunk(
        chunk_id="c1", document_id="d1", text="Python is a programming language."
    )
    chunk2 = TextChunk(
        chunk_id="c2", document_id="d1", text="Pytest is used for automated testing."
    )

    results = [
        RetrievalResult(chunk=chunk1, score=0.1),
        RetrievalResult(chunk=chunk2, score=0.2),
    ]

    builder = PromptBuilderService()
    query = "What is Pytest used for?"

    # 2. ACT
    prompt = builder.build_prompt(query=query, results=results)

    # 3. ASSERT: Check that prompt includes system instruction, chunks, and query
    assert builder.system_instruction in prompt
    assert "[Source Chunk 1]:\nPython is a programming language." in prompt
    assert "[Source Chunk 2]:\nPytest is used for automated testing." in prompt
    assert f"USER QUESTION:\n\t\t{query}" in prompt or query in prompt

def test_build_prompt_custom_system_instruction():
    # Test that custom system instructions are respected
    custom_instruction = "Answer like a pirate."
    builder = PromptBuilderService(system_instruction=custom_instruction)

    prompt = builder.build_prompt(query="Hello", results=[])

    assert "Answer like a pirate." in prompt