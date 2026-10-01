import sqlite3
from pathlib import Path


DB_PATH = Path("database/nba.db")


def execute_query(query: str):
    with sqlite3.connect(DB_PATH) as connection:
        return connection.execute(query).fetchall()


def test_database_exists():
    assert DB_PATH.exists()


def test_player_stats_not_empty():
    result = execute_query(
        "SELECT COUNT(*) FROM player_stats"
    )

    assert result[0][0] > 0


def test_q07_best_three_point_percentage():
    result = execute_query(
        """
        SELECT
            player,
            team,
            three_pa,
            three_p_pct
        FROM player_stats
        WHERE three_pa >= 100
        ORDER BY three_p_pct DESC
        LIMIT 1
        """
    )

    assert result == [
        (
            "Seth Curry",
            "CHA",
            184.0,
            45.6,
        )
    ]


def test_q08_best_true_shooting():
    result = execute_query(
        """
        SELECT
            player,
            team,
            gp,
            ts_pct
        FROM player_stats
        WHERE gp >= 50
        ORDER BY ts_pct DESC
        LIMIT 1
        """
    )

    assert result == [
        (
            "Jarrett Allen",
            "CLE",
            82,
            72.4,
        )
    ]


def test_q09_top_five_three_point_percentage():
    result = execute_query(
        """
        SELECT
            player
        FROM player_stats
        WHERE three_pa >= 100
        ORDER BY three_p_pct DESC
        LIMIT 5
        """
    )

    players = [
        row[0]
        for row in result
    ]

    assert players == [
        "Seth Curry",
        "Zach LaVine",
        "Ty Jerome",
        "Taurean Prince",
        "Vít Krejčí",
    ]


def test_q10_top_five_net_rating():
    result = execute_query(
        """
        SELECT
            player
        FROM player_stats
        WHERE gp >= 50
        ORDER BY net_rtg DESC
        LIMIT 5
        """
    )

    players = [
        row[0]
        for row in result
    ]

    assert players == [
        "Shai Gilgeous-Alexander",
        "Isaiah Joe",
        "Alex Caruso",
        "Luke Kornet",
        "Kenrich Williams",
    ]