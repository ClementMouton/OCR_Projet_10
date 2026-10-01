from pathlib import Path

import faiss


INDEX_PATH = Path("vector_db/faiss_index.idx")
CHUNKS_PATH = Path("vector_db/document_chunks.pkl")


def test_faiss_index_exists():
    assert INDEX_PATH.exists()


def test_document_chunks_exist():
    assert CHUNKS_PATH.exists()


def test_faiss_index_not_empty():
    index = faiss.read_index(
        str(INDEX_PATH)
    )

    assert index.ntotal > 0


def test_faiss_index_expected_size():
    index = faiss.read_index(
        str(INDEX_PATH)
    )

    assert index.ntotal == 100


def test_index_and_chunks_have_same_size():
    import pickle

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        CHUNKS_PATH,
        "rb",
    ) as file:
        chunks = pickle.load(file)

    assert index.ntotal == len(chunks)


def test_chunks_are_not_empty():
    import pickle

    with open(
        CHUNKS_PATH,
        "rb",
    ) as file:
        chunks = pickle.load(file)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk is not None