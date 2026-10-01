import json
from pathlib import Path

from utils.rag_pipeline import RAGPipeline


# ============================================================
# Fichiers
# ============================================================

TEST_CASES_FILE = Path(
    "evaluation/test_cases.json"
)

OUTPUT_FILE = Path(
    "evaluation/results/hybrid_results.json"
)


# ============================================================
# Chargement des cas de test
# ============================================================

def load_test_cases() -> list[dict]:

    with open(
        TEST_CASES_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# Benchmark
# ============================================================

def main():

    print(
        "=== Évaluation du pipeline RAG + SQL + Hybrid ==="
    )

    test_cases = load_test_cases()

    print(
        f"{len(test_cases)} cas de test chargés."
    )

    pipeline = RAGPipeline()

    results = []

    # --------------------------------------------------------
    # Exécution des questions
    # --------------------------------------------------------

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        question_id = test_case.get(
            "id",
            f"Q{index:02d}",
        )

        question = test_case["question"]

        print(
            f"\n[{index}/{len(test_cases)}] "
            f"{question_id} - {question}"
        )

        try:

            result = pipeline.ask(
                question
            )

            benchmark_result = {
                "id": question_id,
                "question": question,

                # Route choisie
                "route": result["route"],

                # Réponse finale
                "answer": result["answer"],

                # Retrieval FAISS
                "contexts": result["contexts"],
                "sources": result["sources"],
                "scores": result["scores"],

                # SQL
                "sql": result["sql"],
                "sql_results": result["sql_results"],
                "sql_error": result["sql_error"],

                # Vérité terrain
                "reference": test_case.get(
                    "reference",
                    "",
                ),
            }

            results.append(
                benchmark_result
            )

            print(
                f"✓ Route : {result['route']}"
            )

            if result["sql"]:

                print(
                    f"  SQL : {result['sql']}"
                )

                print(
                    f"  Résultats SQL : "
                    f"{len(result['sql_results'])}"
                )

            if result["sources"]:

                print(
                    "  Sources FAISS : "
                    + ", ".join(
                        result["sources"]
                    )
                )

        except Exception as error:

            print(
                f"✗ Erreur : {error}"
            )

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "reference": test_case.get(
                        "reference",
                        "",
                    ),
                    "error": str(error),
                }
            )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Résumé du routage
    # --------------------------------------------------------

    print(
        "\n=== Résumé du routage ==="
    )

    route_counts = {
        "rag": 0,
        "sql": 0,
        "hybrid": 0,
        "out_of_scope": 0,
        "error": 0,
    }

    for result in results:

        if "error" in result:

            route_counts["error"] += 1

        else:

            route = result["route"]

            route_counts[route] = (
                route_counts.get(
                    route,
                    0,
                )
                + 1
            )

    print(
        f"RAG     : {route_counts['rag']}"
    )

    print(
        f"SQL     : {route_counts['sql']}"
    )

    print(
        f"HYBRID  : {route_counts['hybrid']}"
    )

    print(
        f"OUT     : {route_counts['out_of_scope']}"
    )

    print(
        f"ERREURS : {route_counts['error']}"
    )

    print(
        f"\nRésultats sauvegardés dans : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()