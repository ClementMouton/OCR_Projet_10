import json
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
)


BASELINE_FILE = Path("evaluation/results/baseline_results.json")
OUTPUT_FILE = Path("evaluation/results/ragas_baseline_scores.csv")


def load_baseline() -> list[dict]:
    with open(BASELINE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_dataset(results: list[dict]) -> EvaluationDataset:
    samples = []

    for result in results:
        if "error" in result:
            continue

        samples.append(
            {
                "user_input": result["question"],
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": result["reference"],
            }
        )

    return EvaluationDataset.from_list(samples)


def main():
    print("=== Évaluation RAGAS de la baseline ===")

    baseline_results = load_baseline()

    print(
        f"{len(baseline_results)} réponses "
        "chargées depuis la baseline."
    )

    dataset = prepare_dataset(baseline_results)

    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=MODEL_NAME,
            api_key=MISTRAL_API_KEY,
            temperature=0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=MISTRAL_API_KEY,
        )
    )

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("\n=== Scores moyens ===")
    print(result)

    dataframe = result.to_pandas()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\nRésultats détaillés sauvegardés dans : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()