from _pytest.logging import LogCaptureFixture

from config.logging_config import get_logger, log_function_call

logger = get_logger(__name__)


@log_function_call(logger)
def sample_function() -> str:
    return "Function Called"


def test_sample_function(caplog: LogCaptureFixture) -> None:
    assert sample_function() == "Function Called"
    assert (
        "Function tests.unittest.test_logging.sample_function called by tests.unittest.test_logging.test_sample_function with args: None"
        in caplog.text
    )
    assert (
        "Function tests.unittest.test_logging.sample_function called by tests.unittest.test_logging.test_sample_function ended in"
        in caplog.text
    )


class TestClass:
    @log_function_call(logger)
    def method(self) -> str:
        return "Method Called"


def test_class_method(caplog: LogCaptureFixture) -> None:
    test_object = TestClass()
    assert test_object.method() == "Method Called"
    assert (
        "Function tests.unittest.test_logging.method called by tests.unittest.test_logging.test_class_method with args: self=TestClass"
        in caplog.text
    )
    assert (
        "Function tests.unittest.test_logging.method called by tests.unittest.test_logging.test_class_method ended in"
        in caplog.text
    )
