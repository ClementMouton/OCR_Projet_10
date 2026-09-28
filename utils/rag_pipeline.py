import json
import logging
from typing import Any

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from .config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
)
from .vector_store import VectorStoreManager
from .sql_tool import SQLTool


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)


# ============================================================
# Prompt de réponse
# ============================================================

SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.

Ta mission est de répondre précisément à la question de l'utilisateur
en t'appuyant uniquement sur les informations fournies dans le contexte.

Le contexte peut contenir :
- des extraits documentaires provenant des commentaires NBA ;
- des résultats statistiques provenant d'une base SQL ;
- ou les deux.

RÈGLES :
1. N'invente aucune information absente du contexte.
2. Pour les statistiques numériques, utilise en priorité les résultats SQL fournis.
3. Pour les opinions, réactions ou commentaires, utilise les extraits documentaires.
4. Si le contexte ne permet pas de répondre, indique-le clairement.
5. Ne présente pas une opinion de commentateur comme un fait statistique.

---

{context_str}

---

QUESTION DU FAN:

{question}

RÉPONSE DE L'ANALYSTE NBA:"""


# ============================================================
# Prompt de routage
# ============================================================

ROUTER_PROMPT = """Tu dois déterminer quelle source de données est nécessaire
pour répondre à une question sur la NBA.

Tu dois choisir exactement UNE valeur parmi :

rag
sql
hybrid

Définitions :

rag :
La question porte sur des commentaires, opinions, débats,
analyses textuelles ou informations présentes dans les documents.

sql :
La question demande un calcul, un classement, une statistique,
un pourcentage, un nombre de matchs ou une comparaison basée
uniquement sur les statistiques des joueurs.

hybrid :
La question nécessite à la fois des commentaires ou analyses
textuelles ET des statistiques chiffrées.

Exemples :

Question :
Quelles équipes ont impressionné les commentateurs pendant les playoffs ?
Réponse :
rag

Question :
Quel joueur ayant disputé au moins 50 matchs possède le meilleur TS% ?
Réponse :
sql

Question :
Quels joueurs des Minnesota Timberwolves sont mis en avant par
les commentateurs et que montrent leurs statistiques individuelles ?
Réponse :
hybrid

Question :
{question}

Réponds uniquement par :
rag
sql
ou
hybrid
"""


# ============================================================
# Pipeline
# ============================================================

class RAGPipeline:

    def __init__(self):

        if not MISTRAL_API_KEY:
            raise ValueError(
                "La clé API Mistral (MISTRAL_API_KEY) n'est pas définie."
            )

        self.client = MistralClient(
            api_key=MISTRAL_API_KEY
        )

        self.model = MODEL_NAME

        # ----------------------------------------------------
        # Vector Store
        # ----------------------------------------------------

        self.vector_store = VectorStoreManager()

        if (
            self.vector_store.index is None
            or not self.vector_store.document_chunks
        ):
            raise RuntimeError(
                "L'index FAISS ou les chunks n'ont pas pu être chargés."
            )

        # ----------------------------------------------------
        # SQL Tool
        # ----------------------------------------------------

        self.sql_tool = SQLTool()

        logging.info(
            "RAGPipeline initialisé avec %s vecteurs et SQLTool actif.",
            self.vector_store.index.ntotal
        )

    # ========================================================
    # Routage
    # ========================================================

    def route_question(self, question: str) -> str:

        prompt = ROUTER_PROMPT.format(
            question=question
        )

        messages = [
            ChatMessage(
                role="user",
                content=prompt
            )
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0
        )

        if not response.choices:
            raise RuntimeError(
                "Impossible de déterminer la route de la question."
            )

        route = (
            response.choices[0]
            .message
            .content
            .strip()
            .lower()
        )

        # Nettoyage au cas où le modèle ajoute du Markdown
        route = route.replace("```", "")
        route = route.strip()

        if route not in {
            "rag",
            "sql",
            "hybrid",
        }:
            logging.warning(
                "Route inattendue '%s'. Utilisation de RAG par défaut.",
                route
            )

            route = "rag"

        logging.info(
            "Route sélectionnée pour la question : %s",
            route
        )

        return route

    # ========================================================
    # Retrieval FAISS
    # ========================================================

    def retrieve(
        self,
        question: str
    ) -> list[dict[str, Any]]:

        return self.vector_store.search(
            question,
            k=SEARCH_K
        )

    # ========================================================
    # Contexte documentaire
    # ========================================================

    def format_rag_context(
        self,
        search_results: list[dict[str, Any]]
    ) -> str:

        if not search_results:
            return (
                "Aucune information documentaire pertinente "
                "n'a été trouvée."
            )

        return "\n\n---\n\n".join(
            [
                (
                    f"Source documentaire : "
                    f"{result['metadata'].get('source', 'Inconnue')} "
                    f"(Score : {result['score']:.1f}%)\n"
                    f"Contenu : {result['text']}"
                )
                for result in search_results
            ]
        )

    # ========================================================
    # Contexte SQL
    # ========================================================

    def format_sql_context(
        self,
        sql_result: dict[str, Any]
    ) -> str:

        results = sql_result.get(
            "results",
            []
        )

        if not results:
            return (
                "La requête SQL n'a retourné aucun résultat."
            )

        return (
            "Résultats statistiques issus de la base SQL :\n\n"
            f"Requête SQL exécutée :\n"
            f"{sql_result['sql']}\n\n"
            f"Résultats :\n"
            f"{json.dumps(results, ensure_ascii=False, indent=2)}"
        )

    # ========================================================
    # Génération de réponse
    # ========================================================

    def generate(
        self,
        question: str,
        context_str: str
    ) -> str:

        final_prompt = SYSTEM_PROMPT.format(
            context_str=context_str,
            question=question
        )

        messages = [
            ChatMessage(
                role="user",
                content=final_prompt
            )
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0.1
        )

        if not response.choices:
            raise RuntimeError(
                "L'API Mistral n'a retourné aucune réponse valide."
            )

        return response.choices[0].message.content

    # ========================================================
    # Pipeline complet
    # ========================================================

    def ask(
        self,
        question: str
    ) -> dict[str, Any]:

        if not question or not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        # ----------------------------------------------------
        # 1. Routage
        # ----------------------------------------------------

        route = self.route_question(
            question
        )

        search_results = []
        sql_result = None

        context_parts = []

        # ----------------------------------------------------
        # 2. Retrieval documentaire
        # ----------------------------------------------------

        if route in {
            "rag",
            "hybrid",
        }:

            search_results = self.retrieve(
                question
            )

            rag_context = self.format_rag_context(
                search_results
            )

            context_parts.append(
                "=== CONTEXTE DOCUMENTAIRE ===\n"
                + rag_context
            )

        # ----------------------------------------------------
        # 3. Retrieval SQL
        # ----------------------------------------------------

        if route in {
            "sql",
            "hybrid",
        }:

            try:

                sql_result = self.sql_tool.ask(
                    question
                )

                sql_context = self.format_sql_context(
                    sql_result
                )

                context_parts.append(
                    "=== CONTEXTE STATISTIQUE SQL ===\n"
                    + sql_context
                )

            except Exception as error:

                logging.exception(
                    "Erreur pendant l'utilisation du SQLTool."
                )

                sql_result = {
                    "question": question,
                    "sql": None,
                    "results": [],
                    "error": str(error),
                }

                context_parts.append(
                    "=== CONTEXTE STATISTIQUE SQL ===\n"
                    "Le SQL Tool n'a pas pu produire de résultat."
                )

        # ----------------------------------------------------
        # 4. Construction du contexte final
        # ----------------------------------------------------

        if context_parts:

            context_str = "\n\n".join(
                context_parts
            )

        else:

            context_str = (
                "Aucun contexte disponible pour répondre "
                "à cette question."
            )

        # ----------------------------------------------------
        # 5. Génération
        # ----------------------------------------------------

        answer = self.generate(
            question=question,
            context_str=context_str
        )

        # ----------------------------------------------------
        # 6. Résultat structuré
        # ----------------------------------------------------

        return {
            "question": question,
            "route": route,
            "answer": answer,

            # Données documentaires
            "contexts": [
                result["text"]
                for result in search_results
            ],

            "sources": [
                result["metadata"].get(
                    "source",
                    "Inconnue"
                )
                for result in search_results
            ],

            "scores": [
                result["score"]
                for result in search_results
            ],

            # Données SQL
            "sql": (
                sql_result.get("sql")
                if sql_result
                else None
            ),

            "sql_results": (
                sql_result.get("results", [])
                if sql_result
                else []
            ),

            "sql_error": (
                sql_result.get("error")
                if sql_result
                else None
            ),
        }