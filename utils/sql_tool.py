import logging
import re
import sqlite3
from typing import Any

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    NBA_DATABASE_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)


# ============================================================
# Description de la base
# ============================================================

DATABASE_SCHEMA = """
La base SQLite contient une table nommée player_stats.

Colonnes disponibles :

- player : nom du joueur
- team : code abrégé de l'équipe NBA sur 3 caractères.
    Exemples :
    MIN = Minnesota Timberwolves
    ORL = Orlando Magic
    OKC = Oklahoma City Thunder
    CLE = Cleveland Cavaliers
    CHA = Charlotte Hornets
    BOS = Boston Celtics
    MIL = Milwaukee Bucks
    ATL = Atlanta Hawks
    SAC = Sacramento Kings
- gp : nombre de matchs joués
- pts : nombre total de points
- reb : nombre total de rebonds
- ast : nombre total de passes décisives
- three_pa : nombre de tirs à 3 points tentés
- three_p_pct : pourcentage de réussite à 3 points
- ts_pct : True Shooting Percentage
- net_rtg : Net Rating

IMPORTANT :
- Les pourcentages sont stockés sur une échelle de 0 à 100.
  Exemple : 45.6 signifie 45,6 %.
- Utilise uniquement la table player_stats.
"""


# ============================================================
# Few-shot examples
# ============================================================

FEW_SHOT_EXAMPLES = """
EXEMPLE 1

Question :
Quel joueur ayant tenté au moins 100 tirs à 3 points possède
le meilleur pourcentage de réussite à 3 points ?

SQL :
SELECT player, team, three_pa, three_p_pct
FROM player_stats
WHERE three_pa >= 100
ORDER BY three_p_pct DESC
LIMIT 1;


EXEMPLE 2

Question :
Quel joueur ayant disputé au moins 50 matchs possède
le meilleur True Shooting Percentage ?

SQL :
SELECT player, team, gp, ts_pct
FROM player_stats
WHERE gp >= 50
ORDER BY ts_pct DESC
LIMIT 1;


EXEMPLE 3

Question :
Quels sont les cinq joueurs ayant tenté au moins 100 tirs
à 3 points avec le meilleur pourcentage de réussite ?

SQL :
SELECT player, team, three_pa, three_p_pct
FROM player_stats
WHERE three_pa >= 100
ORDER BY three_p_pct DESC
LIMIT 5;


EXEMPLE 4

Question :
Parmi les joueurs ayant disputé au moins 50 matchs,
quels sont les cinq joueurs ayant le meilleur NETRTG ?

SQL :
SELECT player, team, gp, net_rtg
FROM player_stats
WHERE gp >= 50
ORDER BY net_rtg DESC
LIMIT 5;


EXEMPLE 5

Question :
Quelles sont les statistiques individuelles disponibles
pour les joueurs des Minnesota Timberwolves ?

SQL :
SELECT player, team, gp, pts, reb, ast, three_pa,
       three_p_pct, ts_pct, net_rtg
FROM player_stats
WHERE team = 'MIN'
ORDER BY pts DESC;


EXEMPLE 6

Question :
Quelles sont les statistiques individuelles disponibles
pour les joueurs du Orlando Magic ?

SQL :
SELECT player, team, gp, pts, reb, ast, three_pa,
       three_p_pct, ts_pct, net_rtg
FROM player_stats
WHERE team = 'ORL'
ORDER BY pts DESC;
"""

# ============================================================
# SQL Tool
# ============================================================

class SQLTool:

    def __init__(self):

        if not MISTRAL_API_KEY:
            raise ValueError(
                "La clé API Mistral n'est pas configurée."
            )

        self.client = MistralClient(
            api_key=MISTRAL_API_KEY
        )

        self.model = MODEL_NAME
        self.database_file = NBA_DATABASE_FILE

        logging.info(
            "SQLTool initialisé avec la base : %s",
            self.database_file,
        )

    # --------------------------------------------------------
    # Génération SQL
    # --------------------------------------------------------

    def generate_sql(self, question: str) -> str:

        prompt = f"""
Tu es un expert SQLite.

Ta mission est de transformer une question utilisateur
en UNE requête SQL SQLite.

{DATABASE_SCHEMA}

Voici des exemples :

{FEW_SHOT_EXAMPLES}

RÈGLES :

1. Retourne uniquement la requête SQL.
2. N'ajoute aucune explication.
3. N'utilise jamais de Markdown.
4. Utilise uniquement SELECT.
5. Utilise uniquement la table player_stats.
6. N'invente aucune colonne.
7. Utilise LIMIT lorsque la question demande un nombre
   précis de résultats.

QUESTION :

{question}

SQL :
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

        sql = response.choices[0].message.content.strip()

        # Nettoyage au cas où le modèle renvoie malgré tout
        # un bloc Markdown.
        sql = sql.replace("```sql", "")
        sql = sql.replace("```", "")
        sql = sql.strip()

        logging.info(
            "SQL généré : %s",
            sql,
        )

        return sql

    # --------------------------------------------------------
    # Validation sécurité
    # --------------------------------------------------------

    def validate_sql(self, sql: str) -> None:

        normalized_sql = re.sub(
            r"\s+",
            " ",
            sql.strip().lower(),
        )

        # SELECT uniquement
        if not normalized_sql.startswith("select "):
            raise ValueError(
                "Requête refusée : seules les requêtes SELECT "
                "sont autorisées."
            )

        # Une seule instruction SQL
        statements = [
            statement.strip()
            for statement in normalized_sql.split(";")
            if statement.strip()
        ]

        if len(statements) != 1:
            raise ValueError(
                "Requête refusée : une seule instruction SQL "
                "est autorisée."
            )

        forbidden_keywords = [
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "replace",
            "truncate",
            "attach",
            "detach",
            "pragma",
        ]

        for keyword in forbidden_keywords:
            if re.search(
                rf"\b{keyword}\b",
                normalized_sql,
            ):
                raise ValueError(
                    f"Requête refusée : mot-clé interdit "
                    f"'{keyword}'."
                )

        # Vérification de la table interrogée
        tables = re.findall(
            r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            normalized_sql,
        )

        if not tables:
            raise ValueError(
                "Requête refusée : aucune table détectée."
            )

        if any(
            table != "player_stats"
            for table in tables
        ):
            raise ValueError(
                "Requête refusée : seule la table "
                "'player_stats' est autorisée."
            )

    # --------------------------------------------------------
    # Exécution SQL
    # --------------------------------------------------------

    def execute_sql(
        self,
        sql: str,
    ) -> list[dict[str, Any]]:

        self.validate_sql(sql)

        connection = sqlite3.connect(
            self.database_file
        )

        connection.row_factory = sqlite3.Row

        try:

            cursor = connection.cursor()

            cursor.execute(sql)

            rows = cursor.fetchall()

            results = [
                dict(row)
                for row in rows
            ]

            logging.info(
                "%s ligne(s) retournée(s).",
                len(results),
            )

            return results

        finally:
            connection.close()

    # --------------------------------------------------------
    # Question complète
    # --------------------------------------------------------

    def ask(
        self,
        question: str,
    ) -> dict[str, Any]:

        sql = self.generate_sql(question)

        results = self.execute_sql(sql)

        return {
            "question": question,
            "sql": sql,
            "results": results,
        }