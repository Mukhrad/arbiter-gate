def migration_sql() -> str:
    return "CREATE INDEX CONCURRENTLY idx_decisions_bucket ON decisions(bucket);"
