import json
from pathlib import Path

from utils.rag_pipeline import RAGPipeline


TEST_CASES_FILE = Path(
    "evaluation/test_cases.json"
)


EXPECTED_ROUTES = {
    "Q01": "rag",
    "Q02": "rag",
    "Q03": "rag",
    "Q04": "rag",
    "Q05": "rag",
    "Q06": "out_of_scope",
    "Q07": "sql",
    "Q08": "sql",
    "Q09": "sql",
    "Q10": "sql",
    "Q11": "hybrid",
    "Q12": "hybrid",
}


def main():

    with open(
        TEST_CASES_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    pipeline = RAGPipeline()

    correct = 0

    print(
        "\n=== Test du routeur ===\n"
    )

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):

        question_id = test_case.get(
            "id",
            f"Q{index:02d}",
        )

        question = test_case["question"]

        expected = EXPECTED_ROUTES[
            question_id
        ]

        predicted = pipeline.route_question(
            question
        )

        is_correct = (
            predicted == expected
        )

        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(
            f"{status} {question_id} | "
            f"attendu={expected:<12} | "
            f"obtenu={predicted}"
        )

    total = len(test_cases)

    accuracy = (
        correct / total
        if total
        else 0
    )

    print(
        "\n=============================="
    )

    print(
        f"Routage correct : "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy : "
        f"{accuracy:.1%}"
    )


if __name__ == "__main__":
    main()