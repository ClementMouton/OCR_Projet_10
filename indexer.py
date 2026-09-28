# indexer.py

import argparse
import logging
from pathlib import Path
from typing import Optional

from utils.config import INPUT_DIR
from utils.data_loader import (
    download_and_extract_zip,
    load_and_parse_files,
)
from utils.vector_store import VectorStoreManager


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================
# Filtrage des documents destinés à FAISS
# ============================================================

def filter_documents_for_faiss(
    documents: list[dict],
) -> list[dict]:
    """
    Exclut de FAISS les sources Excel structurées.

    Les statistiques NBA contenues dans les fichiers Excel
    sont désormais chargées dans SQLite et interrogées
    via le SQLTool.

    FAISS reste donc dédié aux documents textuels
    non structurés, notamment les commentaires Reddit.
    """

    documents_for_faiss = []

    for document in documents:

        source = (
            document
            .get("metadata", {})
            .get("source", "")
        )

        source_lower = source.lower()

        if ".xlsx" in source_lower:
            logging.info(
                "Source structurée exclue de FAISS : %s",
                source,
            )
            continue

        documents_for_faiss.append(document)

    return documents_for_faiss


# ============================================================
# Pipeline d'indexation
# ============================================================

def run_indexing(
    input_directory: str,
    data_url: Optional[str] = None,
) -> None:
    """
    Exécute le processus complet d'indexation FAISS.

    Étapes :
    1. Téléchargement éventuel des données.
    2. Chargement et parsing des fichiers.
    3. Exclusion des données structurées Excel.
    4. Construction de l'index FAISS.
    """

    logging.info(
        "--- Démarrage du processus d'indexation FAISS ---"
    )

    # ========================================================
    # 1. Téléchargement / extraction optionnels
    # ========================================================

    if data_url:

        logging.info(
            "Tentative de téléchargement depuis l'URL : %s",
            data_url,
        )

        success = download_and_extract_zip(
            data_url,
            input_directory,
        )

        if not success:
            logging.error(
                "Échec du téléchargement ou de l'extraction. "
                "Arrêt de l'indexation."
            )
            return

    else:

        logging.info(
            "Aucune URL fournie. "
            "Utilisation des fichiers locaux dans : %s",
            input_directory,
        )

    # ========================================================
    # 2. Chargement des documents
    # ========================================================

    logging.info(
        "Chargement et parsing des fichiers depuis : %s",
        input_directory,
    )

    documents = load_and_parse_files(
        input_directory
    )

    if not documents:

        logging.warning(
            "Aucun document n'a été chargé ou parsé."
        )

        logging.info(
            "--- Indexation terminée : aucun document ---"
        )

        return

    logging.info(
        "%s document(s) chargé(s) avant filtrage.",
        len(documents),
    )

    # ========================================================
    # 3. Séparation données structurées / non structurées
    # ========================================================

    documents_for_faiss = filter_documents_for_faiss(
        documents
    )

    excluded_documents = (
        len(documents)
        - len(documents_for_faiss)
    )

    logging.info(
        "%s document(s) structuré(s) exclu(s) de FAISS.",
        excluded_documents,
    )

    logging.info(
        "%s document(s) conservé(s) pour FAISS.",
        len(documents_for_faiss),
    )

    if not documents_for_faiss:

        logging.warning(
            "Aucun document non structuré disponible "
            "pour construire l'index FAISS."
        )

        return

    # ========================================================
    # 4. Construction de l'index
    # ========================================================

    logging.info(
        "Initialisation du VectorStoreManager..."
    )

    vector_store = VectorStoreManager()

    logging.info(
        "Construction du nouvel index FAISS..."
    )

    vector_store.build_index(
        documents_for_faiss
    )

    # ========================================================
    # 5. Contrôle du résultat
    # ========================================================

    if vector_store.index is None:

        logging.error(
            "L'index FAISS n'a pas pu être créé."
        )

        return

    logging.info(
        "--- Processus d'indexation terminé avec succès ---"
    )

    logging.info(
        "Documents indexés dans FAISS : %s",
        len(documents_for_faiss),
    )

    logging.info(
        "Nombre total de chunks/vecteurs : %s",
        vector_store.index.ntotal,
    )


# ============================================================
# Exécution CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Construction de l'index FAISS "
            "pour les documents NBA non structurés."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=(
            "Répertoire contenant les fichiers sources "
            f"(par défaut : {INPUT_DIR})"
        ),
    )

    parser.add_argument(
        "--data-url",
        type=str,
        default=None,
        help=(
            "URL optionnelle permettant de télécharger "
            "et extraire un fichier inputs.zip."
        ),
    )

    args = parser.parse_args()

    run_indexing(
        input_directory=args.input_dir,
        data_url=args.data_url,
    )