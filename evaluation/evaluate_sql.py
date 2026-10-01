import json
from pathlib import Path


# ============================================================
# Fichiers
# ============================================================

HYBRID_FILE = Path(
    "evaluation/results/hybrid_results.json"
)

OUTPUT_FILE = Path(
    "evaluation/results/sql_evaluation.json"
)


# ============================================================
# Résultats attendus
# ============================================================

EXPECTED_RESULTS = {
    "Q07": [
        {
            "player": "Seth Curry",
            "team": "CHA",
            "three_pa": 184.0,
            "three_p_pct": 45.6,
        }
    ],

    "Q08": [
        {
            "player": "Jarrett Allen",
            "team": "CLE",
            "gp": 82,
            "ts_pct": 72.4,
        }
    ],

    "Q09": [
        {
            "player": "Seth Curry",
            "team": "CHA",
            "three_pa": 184.0,
            "three_p_pct": 45.6,
        },
        {
            "player": "Zach LaVine",
            "team": "SAC",
            "three_pa": 533.0,
            "three_p_pct": 44.6,
        },
        {
            "player": "Ty Jerome",
            "team": "CLE",
            "three_pa": 252.0,
            "three_p_pct": 43.9,
        },
        {
            "player": "Taurean Prince",
            "team": "MIL",
            "three_pa": 336.0,
            "three_p_pct": 43.9,
        },
        {
            "player": "Vít Krejčí",
            "team": "ATL",
            "three_pa": 205.0,
            "three_p_pct": 43.7,
        },
    ],

    "Q10": [
        {
            "player": "Shai Gilgeous-Alexander",
            "team": "OKC",
            "gp": 76,
            "net_rtg": 16.7,
        },
        {
            "player": "Isaiah Joe",
            "team": "OKC",
            "gp": 74,
            "net_rtg": 15.8,
        },
        {
            "player": "Alex Caruso",
            "team": "OKC",
            "gp": 54,
            "net_rtg": 15.0,
        },
        {
            "player": "Luke Kornet",
            "team": "BOS",
            "gp": 73,
            "net_rtg": 14.9,
        },
        {
            "player": "Kenrich Williams",
            "team": "OKC",
            "gp": 69,
            "net_rtg": 14.5,
        },
    ],
}


# ============================================================
# Chargement
# ============================================================

def load_results() -> list[dict]:

    with open(
        HYBRID_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# Comparaison
# ============================================================

def compare_results(
    actual: list[dict],
    expected: list[dict],
) -> bool:
    """
    Compare exactement les résultats SQL obtenus
    aux résultats attendus.

    L'ordre est volontairement pris en compte,
    car Q09 et Q10 demandent un classement.
    """

    if len(actual) != len(expected):
        return False

    for actual_row, expected_row in zip(
        actual,
        expected,
    ):

        for key, expected_value in expected_row.items():

            if key not in actual_row:
                return False

            actual_value = actual_row[key]

            # Tolérance pour les nombres flottants
            if isinstance(
                expected_value,
                float,
            ):

                try:
                    if abs(
                        float(actual_value)
                        - expected_value
                    ) > 1e-6:
                        return False

                except (
                    TypeError,
                    ValueError,
                ):
                    return False

            else:

                if actual_value != expected_value:
                    return False

    return True


# ============================================================
# Évaluation
# ============================================================

def main():

    print(
        "=== Évaluation déterministe du SQLTool ===\n"
    )

    results = load_results()

    results_by_id = {
        result["id"]: result
        for result in results
        if "id" in result
    }

    evaluation_results = []

    correct = 0

    for question_id, expected in EXPECTED_RESULTS.items():

        result = results_by_id.get(
            question_id
        )

        # ----------------------------------------------------
        # Question absente
        # ----------------------------------------------------

        if result is None:

            print(
                f"✗ {question_id} - "
                "résultat absent"
            )

            evaluation_results.append(
                {
                    "id": question_id,
                    "success": False,
                    "reason": "Résultat absent",
                }
            )

            continue

        # ----------------------------------------------------
        # Mauvaise route
        # ----------------------------------------------------

        if result.get("route") != "sql":

            print(
                f"✗ {question_id} - "
                f"route={result.get('route')}"
            )

            evaluation_results.append(
                {
                    "id": question_id,
                    "success": False,
                    "reason": (
                        "Route incorrecte : "
                        f"{result.get('route')}"
                    ),
                }
            )

            continue

        # ----------------------------------------------------
        # Erreur SQL
        # ----------------------------------------------------

        if result.get("sql_error"):

            print(
                f"✗ {question_id} - "
                f"erreur SQL : "
                f"{result['sql_error']}"
            )

            evaluation_results.append(
                {
                    "id": question_id,
                    "success": False,
                    "reason": result["sql_error"],
                }
            )

            continue

        # ----------------------------------------------------
        # Comparaison
        # ----------------------------------------------------

        actual = result.get(
            "sql_results",
            [],
        )

        success = compare_results(
            actual,
            expected,
        )

        if success:

            correct += 1

            print(
                f"✓ {question_id}"
            )

        else:

            print(
                f"✗ {question_id}"
            )

            print(
                f"  Attendu : {expected}"
            )

            print(
                f"  Obtenu  : {actual}"
            )

        evaluation_results.append(
            {
                "id": question_id,
                "success": success,
                "expected": expected,
                "actual": actual,
                "sql": result.get("sql"),
            }
        )

    # ========================================================
    # Score global
    # ========================================================

    total = len(
        EXPECTED_RESULTS
    )

    accuracy = (
        correct / total
        if total
        else 0
    )

    print(
        "\n=============================="
    )

    print(
        f"Requêtes correctes : "
        f"{correct}/{total}"
    )

    print(
        f"Exactitude SQL : "
        f"{accuracy:.1%}"
    )

    # ========================================================
    # Export
    # ========================================================

    output = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "results": evaluation_results,
    }

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
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nRésultats sauvegardés dans : "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()