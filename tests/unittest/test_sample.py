import pytest


def test_assert() -> None:
    result = 1
    expected_result = 1
    assert result == expected_result


def test_value_exception() -> None:
    """Test if ValueError exception is raised with an invalid value entered."""

    value = 5
    expected_values = [1, 2, 3]
    with pytest.raises(ValueError):
        if value not in expected_values:
            raise ValueError("`{}` not exists in {:}".format(value, expected_values))
