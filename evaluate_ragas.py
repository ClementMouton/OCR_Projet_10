import json
import logging
from pathlib import Path

from utils.rag_pipeline import RAGPipeline


TEST_CASES_FILE = Path("evaluation/test_cases.json")
RESULTS_DIR = Path("evaluation/results")
BASELINE_RESULTS_FILE = RESULTS_DIR / "baseline_results.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def run_baseline_evaluation(
    rag: RAGPipeline,
    test_cases: list[dict]
) -> list[dict]:

    results = []

    for index, test_case in enumerate(test_cases, start=1):
        question = test_case["question"]

        print(
            f"\n[{index}/{len(test_cases)}] "
            f"{test_case['id']} - {question}"
        )

        try:
            rag_result = rag.ask(question)

            result = {
                "id": test_case["id"],
                "category": test_case["category"],
                "difficulty": test_case["difficulty"],
                "question": question,
                "reference": test_case["reference"],
                "expected_source": test_case["expected_source"],
                "answer": rag_result["answer"],
                "contexts": rag_result["contexts"],
                "retrieved_sources": rag_result["sources"],
                "retrieval_scores": rag_result["scores"],
            }

            results.append(result)

            print("✓ Réponse générée")

        except Exception as error:
            logging.exception(
                "Erreur pendant l'évaluation de %s",
                test_case["id"]
            )

            results.append(
                {
                    "id": test_case["id"],
                    "category": test_case["category"],
                    "difficulty": test_case["difficulty"],
                    "question": question,
                    "reference": test_case["reference"],
                    "expected_source": test_case["expected_source"],
                    "error": str(error),
                }
            )

    return results


def save_results(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        BASELINE_RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nRésultats sauvegardés dans "
        f"{BASELINE_RESULTS_FILE}"
    )


def main():
    print("=== Évaluation de la baseline RAG ===")

    test_cases = load_test_cases()

    print(f"{len(test_cases)} cas de test chargés.")

    rag = RAGPipeline()

    results = run_baseline_evaluation(
        rag=rag,
        test_cases=test_cases
    )

    save_results(results)

    successful = sum(
        1 for result in results
        if "error" not in result
    )

    print(
        f"\nÉvaluation terminée : "
        f"{successful}/{len(results)} réponses générées."
    )


if __name__ == "__main__":
    main()