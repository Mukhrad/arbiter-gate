from candidate import token_is_valid


def test_normal_token_window():
    assert token_is_valid(1_000, 1_500)
    assert not token_is_valid(1_000, 5_000)


def test_future_issued_token_is_rejected():
    assert not token_is_valid(2_000, 1_000)
