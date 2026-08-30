from sandevistan_read.retrieval import EmbeddingService, tokenize


def test_chinese_tokenization_has_characters_and_bigrams():
    tokens = tokenize("孩子如何学习 Agent Runtime")
    assert "孩" in tokens
    assert "孩子" in tokens
    assert "agent" in tokens
    assert "runtime" in tokens


def test_ngram_fallback_is_deterministic():
    service = EmbeddingService()
    first = service._hash_embedding("财政部回购与弱美元")
    second = service._hash_embedding("财政部回购与弱美元")
    assert first == second
    assert any(value for value in first)
