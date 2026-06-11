from __future__ import annotations

from mataltmag_hybrid.gnn.adapter import MatAltMagEmbeddingExtractor


def test_embedding_extraction_is_deterministic_for_fixture():
    extractor = MatAltMagEmbeddingExtractor(embedding_dim=8, seed=7, fixture_mode=True)
    first = extractor.extract(["mp-1", "mp-2"])
    second = extractor.extract(["mp-1", "mp-2"])
    assert first.equals(second)
    assert list(first.columns) == ["material_id"] + [f"emb_{i:03d}" for i in range(8)]

