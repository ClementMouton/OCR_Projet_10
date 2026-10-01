import sys
import json
from pathlib import Path


# ============================================================
# Chemin racine du projet
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports RAGAS
# ============================================================

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)


# ============================================================
# Imports LangChain / Mistral
# ============================================================

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings,
)


# ============================================================
# Imports projet
# ============================================================

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    EMBEDDING_MODEL,
)


# ============================================================
# Fichiers
# ============================================================

HYBRID_FILE = Path(
    "evaluation/results/hybrid_results.json"
)

OUTPUT_RAG = Path(
    "evaluation/results/ragas_hybrid_rag_scores.csv"
)


# ============================================================
# Chargement des résultats
# ============================================================

def load_hybrid_results() -> list[dict]:
    """
    Charge les résultats générés par evaluate_hybrid.py.
    """

    with open(
        HYBRID_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Préparation du dataset RAGAS
# ============================================================

def prepare_rag_dataset(
    results: list[dict],
) -> EvaluationDataset:
    """
    Construit le dataset RAGAS à partir des questions
    utilisant le retrieval documentaire.

    Routes conservées :
    - RAG
    - HYBRID

    Les routes SQL sont exclues car elles n'utilisent pas
    FAISS.

    Les questions OUT_OF_SCOPE sont également exclues.
    """

    samples = []

    for result in results:

        # Ignorer les résultats en erreur
        if "error" in result:
            continue

        # Seules les routes utilisant FAISS sont évaluées
        if result["route"] not in {
            "rag",
            "hybrid",
        }:
            continue

        samples.append(
            {
                "user_input": result["question"],
                "response": result["answer"],
                "retrieved_contexts": result["contexts"],
                "reference": result["reference"],
            }
        )

    return EvaluationDataset.from_list(
        samples
    )


# ============================================================
# Programme principal
# ============================================================

def main():

    print(
        "=== Évaluation RAGAS du pipeline hybride ==="
    )

    # --------------------------------------------------------
    # Chargement du benchmark
    # --------------------------------------------------------

    results = load_hybrid_results()

    print(
        f"{len(results)} résultats chargés."
    )

    # --------------------------------------------------------
    # Préparation du dataset
    # --------------------------------------------------------

    rag_dataset = prepare_rag_dataset(
        results
    )

    print(
        f"{len(rag_dataset)} cas "
        "RAG/HYBRID à évaluer."
    )

    # --------------------------------------------------------
    # LLM utilisé comme juge par RAGAS
    # --------------------------------------------------------

    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model=MODEL_NAME,
            api_key=MISTRAL_API_KEY,
            temperature=0,
        )
    )

    # --------------------------------------------------------
    # Embeddings utilisés par RAGAS
    # --------------------------------------------------------

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=MISTRAL_API_KEY,
        )
    )

    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    # --------------------------------------------------------
    # Évaluation
    # --------------------------------------------------------

    result = evaluate(
        dataset=rag_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # --------------------------------------------------------
    # Affichage
    # --------------------------------------------------------

    print(
        "\n=== Scores RAG / HYBRID ==="
    )

    print(
        result
    )

    # --------------------------------------------------------
    # Export CSV
    # --------------------------------------------------------

    dataframe = result.to_pandas()

    OUTPUT_RAG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_RAG,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        "\n=== Évaluation terminée ==="
    )

    print(
        f"Résultats sauvegardés dans : "
        f"{OUTPUT_RAG}"
    )

    print(
        "\nRoutes évaluées avec RAGAS : "
        "RAG + HYBRID"
    )

    print(
        "Routes SQL : exclues des métriques de retrieval."
    )

    print(
        "Routes OUT_OF_SCOPE : exclues de RAGAS."
    )

    print(
        "\nNote : ResponseRelevancy n'est pas utilisée "
        "car cette métrique est incompatible avec "
        "l'intégration Mistral/RAGAS utilisée dans "
        "l'environnement actuel."
    )


# ============================================================
# Exécution
# ============================================================

if __name__ == "__main__":
    main()