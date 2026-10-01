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

RÈGLES GÉNÉRALES :

1. N'invente aucune information absente du contexte.

2. Pour les statistiques numériques, utilise uniquement les résultats
SQL fournis lorsqu'ils sont présents.

3. Pour les opinions, réactions ou commentaires, utilise les extraits
documentaires.

4. Si le contexte ne permet pas de répondre, indique-le clairement.

5. Ne présente pas une opinion de commentateur comme un fait statistique.

6. N'utilise pas tes connaissances générales pour compléter une
information manquante dans le contexte.

7. Si une information n'est présente ni dans les documents ni dans
les résultats SQL, indique qu'elle n'est pas disponible.


RÈGLES STRICTES POUR LES DONNÉES HYBRIDES :

1. Les données SQL sont la seule source autorisée pour les statistiques
chiffrées des joueurs lorsque des résultats SQL sont fournis.

2. Une statistique SQL ne peut être attribuée à un joueur que si le nom
du joueur correspond exactement au champ "player" de la même ligne SQL.

3. Il est strictement interdit d'associer les statistiques d'une ligne
SQL à un autre joueur mentionné dans les documents.

Exemple interdit :

Document :
"Tyrese Haliburton est mentionné dans les commentaires."

SQL :
{{
    "player": "Isaiah Joe",
    "pts": 755,
    "net_rtg": 15.8
}}

Tu ne dois JAMAIS attribuer les 755 points ou le NETRTG de 15.8
à Tyrese Haliburton.

4. Si un joueur est mentionné dans les documents mais absent des
résultats SQL, tu peux expliquer ce que les documents disent de lui,
mais tu dois préciser qu'aucune statistique n'est disponible pour
ce joueur dans les résultats SQL fournis.

5. Si un joueur apparaît dans les résultats SQL mais pas dans les
documents, tu peux utiliser ses statistiques si elles sont pertinentes
pour répondre à la question, mais tu ne dois jamais prétendre qu'il
est mentionné par les commentateurs.

6. Ne déduis jamais qu'un joueur appartient à une équipe uniquement
parce qu'il est mentionné dans un document.

7. Ne suppose jamais qu'un joueur a été transféré, échangé ou qu'il
appartient à une autre équipe si cette information n'est pas
explicitement présente dans le contexte fourni.

8. N'utilise aucune connaissance extérieure pour expliquer une
incohérence entre les documents et les données SQL.

9. Si les documents mentionnent un joueur mais que les résultats SQL
concernant l'équipe interrogée ne contiennent pas ce joueur, conserve
strictement cette distinction.

10. Avant de produire la réponse, vérifie pour chaque joueur cité :
- si son nom apparaît dans les documents ;
- si son nom apparaît dans les résultats SQL ;
- si chaque statistique citée provient bien de sa propre ligne SQL.

11. En cas de doute sur l'association entre un joueur et une
statistique, n'attribue pas la statistique.

12. Ne fusionne jamais deux joueurs différents même si leurs noms,
leurs équipes, leurs rôles ou leur contexte documentaire semblent liés.

---

{context_str}

---

QUESTION DU FAN:

{question}

RÉPONSE DE L'ANALYSTE NBA:"""


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
        """
        Détermine quelle source doit être utilisée pour répondre.

        Routes possibles :
        - rag : documents textuels non structurés
        - sql : statistiques NBA structurées
        - hybrid : combinaison documents + statistiques
        - out_of_scope : question hors du périmètre NBA
        """

        prompt = f"""
Tu es le routeur d'un système de question-réponse spécialisé
sur la NBA.

Tu dois classer la question de l'utilisateur dans UNE SEULE
des quatre catégories suivantes :

RAG
SQL
HYBRID
OUT_OF_SCOPE


============================================================
1. RAG
============================================================

Choisis RAG lorsque la réponse doit être recherchée dans les
documents textuels du corpus.

Cela concerne notamment :

- commentaires de fans ;
- opinions ;
- débats ;
- réactions ;
- explications présentes dans les documents ;
- discussions sur les playoffs ;
- faits ou événements NBA décrits dans les documents ;
- questions historiques NBA lorsque les statistiques de la
base structurée actuelle ne permettent pas d'y répondre.

IMPORTANT :

La simple présence du nom d'un joueur ou d'une équipe ne
justifie PAS l'utilisation de SQL.

Exemples :

Question :
Quelles équipes ont impressionné les commentateurs pendant
les playoffs ?

Route :
RAG


Question :
Pourquoi certains fans pensent-ils qu'une finale entre les
Pacers et le Thunder pourrait être moins suivie ?

Route :
RAG


Question :
Pourquoi Reggie Miller est-il présenté comme une première
option particulièrement efficace en playoffs ?

Route :
RAG


Question :
Une équipe NBA a-t-elle déjà joué les trois premiers tours
sans avantage du terrain avant de l'obtenir en Finales ?

Route :
RAG


Question :
Pourquoi les Wolves ont-ils impressionné pendant les
playoffs ?

Route :
RAG


============================================================
2. SQL
============================================================

Choisis SQL UNIQUEMENT lorsque la question peut être résolue
à partir des statistiques individuelles de la saison NBA
présentes dans la base structurée.

La base contient notamment :

- joueur ;
- équipe ;
- matchs joués ;
- points ;
- rebonds ;
- passes ;
- tentatives à trois points ;
- pourcentage à trois points ;
- True Shooting Percentage ;
- NETRTG.

SQL est particulièrement adapté aux :

- classements ;
- maximums ou minimums ;
- top N ;
- filtres numériques ;
- seuils ;
- comparaisons statistiques ;
- recherche d'une valeur statistique.

Exemples :

Question :
Quel joueur ayant tenté au moins 100 tirs à 3 points possède
le meilleur pourcentage de réussite à 3 points ?

Route :
SQL


Question :
Quel joueur ayant disputé au moins 50 matchs possède le
meilleur True Shooting Percentage ?

Route :
SQL


Question :
Quels sont les cinq joueurs ayant tenté au moins 100 tirs à
3 points avec le meilleur pourcentage de réussite ?

Route :
SQL


Question :
Parmi les joueurs ayant disputé au moins 50 matchs, quels
sont les cinq joueurs ayant le meilleur NETRTG ?

Route :
SQL


============================================================
3. HYBRID
============================================================

Choisis HYBRID uniquement si répondre correctement nécessite
À LA FOIS :

1. des informations provenant des documents textuels ;
ET
2. des statistiques provenant de la base SQL.

Les deux sources doivent réellement être nécessaires.

Ne choisis PAS HYBRID simplement parce qu'un joueur ou une
équipe possède des statistiques dans la base.

Exemples :

Question :
Quels joueurs des Minnesota Timberwolves sont mis en avant
par les commentateurs et que montrent leurs statistiques
individuelles disponibles ?

Route :
HYBRID


Question :
Quels joueurs du Orlando Magic sont mis en avant par les
commentateurs et que montrent leurs statistiques
individuelles disponibles ?

Route :
HYBRID


============================================================
4. OUT_OF_SCOPE
============================================================

Choisis OUT_OF_SCOPE lorsque la question n'appartient pas au
périmètre NBA du système.

Cela inclut notamment :

- football ;
- tennis ;
- politique ;
- cinéma ;
- météo ;
- sujets sans rapport avec la NBA.

Exemple :

Question :
Quel joueur a remporté le Ballon d'Or de football en 2024 ?

Route :
OUT_OF_SCOPE


============================================================
RÈGLES IMPORTANTES
============================================================

Règle 1 :
Une question sur une opinion, un commentaire ou un débat
doit être RAG, sauf si elle demande explicitement aussi des
statistiques.

Règle 2 :
Une question statistique calculable à partir de la base doit
être SQL.

Règle 3 :
HYBRID nécessite explicitement les deux types d'information.

Règle 4 :
Une question NBA historique n'est pas automatiquement SQL.
La base SQL contient des statistiques de joueurs de la
saison actuelle et ne constitue pas une base historique
générale.

Règle 5 :
Une question extérieure à la NBA est OUT_OF_SCOPE.

Règle 6 :
Ne déduis pas qu'une question nécessite SQL uniquement parce
qu'elle contient le nom d'un joueur ou d'une équipe.

Réponds UNIQUEMENT avec l'une des quatre valeurs suivantes :

rag
sql
hybrid
out_of_scope


QUESTION :
{question}

ROUTE :
"""

        messages = [
            ChatMessage(
                role="user",
                content=prompt,
            )
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        route = (
            response.choices[0]
            .message.content
            .strip()
            .lower()
        )

        valid_routes = {
            "rag",
            "sql",
            "hybrid",
            "out_of_scope",
        }

        if route not in valid_routes:
            logging.warning(
                "Route invalide retournée par le modèle : %s. "
                "Fallback vers RAG.",
                route,
            )
            route = "rag"

        logging.info(
            "Route sélectionnée pour la question : %s",
            route,
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

        if route == "out_of_scope":

            logging.info(
                "Question hors périmètre NBA détectée."
            )

            return {
                "question": question,
                "route": route,
                "answer": (
                    "Cette question est hors du périmètre de ce système, "
                    "qui est spécialisé dans l'analyse des données et "
                    "discussions NBA disponibles dans son corpus."
                ),
                "contexts": [],
                "sources": [],
                "scores": [],
                "sql": None,
                "sql_results": [],
                "sql_error": None,
            }

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
                    "Le SQLTool n'a pas pu fournir de résultat."
                )

        # ----------------------------------------------------
        # 4. Construction du contexte final
        # ----------------------------------------------------

        if not context_parts:

            context_str = (
                "Aucun contexte exploitable n'a été trouvé."
            )

        else:

            context_str = "\n\n".join(
                context_parts
            )

        # ----------------------------------------------------
        # 5. Génération
        # ----------------------------------------------------

        answer = self.generate(
            question=question,
            context_str=context_str,
        )

        # ----------------------------------------------------
        # 6. Métadonnées documentaires
        # ----------------------------------------------------

        contexts = [
            result["text"]
            for result in search_results
        ]

        sources = [
            result["metadata"].get(
                "source",
                "Inconnue"
            )
            for result in search_results
        ]

        scores = [
            result["score"]
            for result in search_results
        ]

        # ----------------------------------------------------
        # 7. Métadonnées SQL
        # ----------------------------------------------------

        sql_query = None
        sql_results = []
        sql_error = None

        if sql_result:

            sql_query = sql_result.get(
                "sql"
            )

            sql_results = sql_result.get(
                "results",
                []
            )

            sql_error = sql_result.get(
                "error"
            )

        # ----------------------------------------------------
        # 8. Résultat final
        # ----------------------------------------------------

        return {
            "question": question,
            "route": route,
            "answer": answer,
            "contexts": contexts,
            "sources": sources,
            "scores": scores,
            "sql": sql_query,
            "sql_results": sql_results,
            "sql_error": sql_error,
        }