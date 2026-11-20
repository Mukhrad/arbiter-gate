def token_is_valid(issued_at: int, now: int, ttl_seconds: int = 3600) -> bool:
    age = now - issued_at
    return 0 <= age <= ttl_seconds
