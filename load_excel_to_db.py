from pathlib import Path
import sqlite3

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from utils.config import NBA_DATABASE_FILE


# ============================================================
# Configuration
# ============================================================

EXCEL_FILE = Path("inputs/regular NBA.xlsx")
SHEET_NAME = "Données NBA"


# ============================================================
# Modèle Pydantic
# ============================================================

class PlayerStats(BaseModel):
    """
    Validation minimale des données NBA nécessaires
    aux premières requêtes statistiques.
    """

    model_config = ConfigDict(populate_by_name=True)

    player: str = Field(alias="Player")
    team: str = Field(alias="Team")
    gp: int = Field(alias="GP")

    pts: float = Field(alias="PTS")
    reb: float = Field(alias="REB")
    ast: float = Field(alias="AST")

    three_pa: float = Field(alias="3PA")
    three_p_pct: float = Field(alias="3P%")

    ts_pct: float = Field(alias="TS%")
    net_rtg: float = Field(alias="NETRTG")


# ============================================================
# Chargement Excel
# ============================================================

def load_excel() -> pd.DataFrame:
    print(f"Chargement de : {EXCEL_FILE}")

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier Excel introuvable : {EXCEL_FILE}"
        )

    # La première ligne Excel contient des numéros de colonnes.
    # Les véritables noms de variables sont sur la ligne suivante.
    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=SHEET_NAME,
        header=1,
    )

    # Suppression des colonnes entièrement vides
    df = df.dropna(
        axis=1,
        how="all",
    )

    # Excel interprète "3PM" comme l'heure 15:00.
    # La colonne située entre FG% et 3PA correspond à 3PM.
    for column in df.columns:
        if str(column) == "15:00:00":
            df = df.rename(
                columns={column: "3PM"}
            )

    print(f"{len(df)} lignes chargées.")
    print(f"{len(df.columns)} colonnes détectées.")

    print("\nColonnes utilisées :")
    print(df.columns.tolist())

    return df


# ============================================================
# Validation Pydantic
# ============================================================

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    valid_rows = []
    errors = []

    for index, row in df.iterrows():

        try:
            player = PlayerStats.model_validate(
                row.to_dict()
            )

            valid_rows.append(
                player.model_dump()
            )

        except ValidationError as error:
            errors.append(
                {
                    "row": index,
                    "error": error.errors(),
                }
            )

            # Afficher le détail uniquement pour la première ligne rejetée
            if len(errors) == 1:
                print("\n=== Première erreur Pydantic ===")
                print(error)

                print("\n=== Valeurs de la ligne ===")
                print(row.to_dict())

    print(
        f"{len(valid_rows)} lignes valides / "
        f"{len(df)} lignes."
    )

    if errors:
        print(
            f"{len(errors)} lignes rejetées "
            "par la validation Pydantic."
        )

    return pd.DataFrame(valid_rows)


# ============================================================
# Création SQLite
# ============================================================

def create_database(df: pd.DataFrame) -> None:

    database_path = Path(NBA_DATABASE_FILE)

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(database_path) as connection:

        df.to_sql(
            "player_stats",
            connection,
            if_exists="replace",
            index=False,
        )

    print(
        f"Base SQLite créée : {database_path}"
    )

    print(
        f"{len(df)} lignes insérées "
        "dans player_stats."
    )


# ============================================================
# Pipeline
# ============================================================

def main():

    print(
        "=== Ingestion Excel NBA → SQLite ==="
    )

    dataframe = load_excel()

    validated_dataframe = validate_data(
        dataframe
    )

    if validated_dataframe.empty:
        raise ValueError(
            "Aucune donnée valide à insérer."
        )

    create_database(
        validated_dataframe
    )

    print(
        "=== Ingestion terminée avec succès ==="
    )


if __name__ == "__main__":
    main()