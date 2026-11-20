from candidate import migration_sql


def test_production_index_is_non_blocking():
    sql = migration_sql().upper()
    assert "CREATE INDEX CONCURRENTLY" in sql
    assert sql.endswith(";")
