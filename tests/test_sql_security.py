import pytest

from utils.sql_tool import SQLTool


@pytest.fixture
def sql_tool():
    """
    Instancie SQLTool sans appeler l'API Mistral.
    """
    return SQLTool()


def test_valid_select_is_accepted(sql_tool):
    sql = """
        SELECT player, team
        FROM player_stats
        LIMIT 5;
    """

    sql_tool.validate_sql(sql)


def test_insert_is_rejected(sql_tool):
    sql = """
        INSERT INTO player_stats (player)
        VALUES ('Test');
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_update_is_rejected(sql_tool):
    sql = """
        UPDATE player_stats
        SET pts = 0;
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_delete_is_rejected(sql_tool):
    sql = """
        DELETE FROM player_stats;
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_drop_is_rejected(sql_tool):
    sql = """
        DROP TABLE player_stats;
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_multiple_statements_are_rejected(sql_tool):
    sql = """
        SELECT *
        FROM player_stats;

        SELECT *
        FROM player_stats;
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_other_table_is_rejected(sql_tool):
    sql = """
        SELECT *
        FROM users;
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_join_with_other_table_is_rejected(sql_tool):
    sql = """
        SELECT *
        FROM player_stats
        JOIN users
        ON player_stats.player = users.player;
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)


def test_pragma_is_rejected(sql_tool):
    sql = """
        PRAGMA table_info(player_stats);
    """

    with pytest.raises(ValueError):
        sql_tool.validate_sql(sql)