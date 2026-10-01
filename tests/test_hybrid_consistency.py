import pytest

from utils.rag_pipeline import RAGPipeline


QUESTION = (
    "Quels joueurs d'OKC sont évoqués dans les commentaires "
    "et que montrent leurs statistiques ?"
)


@pytest.fixture(scope="module")
def hybrid_result():
    pipeline = RAGPipeline()
    return pipeline.ask(QUESTION)


def test_hybrid_route(hybrid_result):
    assert hybrid_result["route"] == "hybrid"


def test_hybrid_sql_contains_expected_okc_players(
    hybrid_result,
):
    players = {
        row["player"]
        for row in hybrid_result["sql_results"]
    }

    assert "Shai Gilgeous-Alexander" in players
    assert "Alex Caruso" in players
    assert "Isaiah Joe" in players


def test_haliburton_not_in_okc_sql_results(
    hybrid_result,
):
    players = {
        row["player"]
        for row in hybrid_result["sql_results"]
    }

    assert "Tyrese Haliburton" not in players


def test_haliburton_does_not_receive_isaiah_joe_stats(
    hybrid_result,
):
    answer = hybrid_result["answer"]

    if "Haliburton" not in answer:
        return

    haliburton_section = answer.split(
        "Haliburton",
        maxsplit=1,
    )[1][:1500]

    forbidden_values = [
        "755",
        "192",
        "118",
        "41.2",
        "41,2",
        "61.7",
        "61,7",
        "15.8",
        "15,8",
    ]

    for value in forbidden_values:
        assert value not in haliburton_section, (
            f"La statistique {value} d'Isaiah Joe semble avoir "
            "été attribuée à Tyrese Haliburton."
        )