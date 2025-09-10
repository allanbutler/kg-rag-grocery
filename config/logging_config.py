import datetime
import inspect
import logging
import logging.config
import sys
import time
from functools import wraps
from typing import Any, Callable


def setup_logging() -> logging.Logger:
    """
    Configures the application's logging system with multiple handlers and custom settings.

    This function establishes a robust logging mechanism by setting up handlers for console output and optional file logging in both standard text and JSON formats. It is designed to work within the Databricks environment, leveraging the Databricks file system (DBFS) for log storage.

    The function also sets the root logger level to DEBUG, which can be adjusted as needed. Specific third-party loggers ('py4j', 'pyspark', 'matplotlib', 'graphviz', 'PIL') are set to WARNING level by default.

    Features:
        - Console Handler: Logs output to the console with a standard formatter.
        - File Handlers (optional): Logs output to text and JSON files stored in DBFS. File paths are dynamically generated based on the current date and time, and are stored under '/dbfs/sgs/logs/'.
        - JSON Formatter: Utilizes the python-json-logger module for JSON-formatted logs, facilitating structured logging and easier integration with log analysis tools like DataDog or AWS CloudWatch.
        - Custom Log Levels: Adjusts log levels for certain third-party libraries (e.g., 'py4j', 'pyspark') to reduce unnecessary verbosity.

    Note:
        - ENABLE_LOG_FILE_OUTPUT flag controls the creation of file handlers. If set to True, logs are written to '/dbfs/sgs/logs/' in both standard and JSON formats.
        - Since the execution time for the workflow and the timestamp used for folder creation of logs may differ in seconds, the granularity in folder name is set to days.

    Returns:
        logging.Logger: The configured root logger with the specified handlers and level settings.
    """
    # Set this to disable/enable creation of dedicated logging output files in standard & jsonified formats
    ENABLE_LOG_FILE_OUTPUT: bool = False
    # Log level: Set to "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    LOG_LEVEL: str = "DEBUG"

    log_config = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(funcName)s %(message)s",
            },
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "standard"}
        },
        "loggers": {
            "py4j": {"level": "WARNING"},
            "pyspark": {"level": "WARNING"},
            "matplotlib": {"level": "WARNING"},
            "graphviz": {"level": "WARNING"},
            "PIL": {"level": "WARNING"},
        },
        "root": {"handlers": ["console"], "level": f"{LOG_LEVEL}"},
    }

    # Check if file logging is enabled
    if ENABLE_LOG_FILE_OUTPUT:
        log_config["handlers"].update(
            {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": f"/dbfs/sgs/logs/{datetime.datetime.now().strftime('%Y-%H-%M')}/std_logs.log",
                    "formatter": "standard",
                },
                "json_file": {
                    "class": "logging.FileHandler",
                    "filename": f"/dbfs/sgs/logs/{datetime.datetime.now().strftime('%Y-%H-%M')}/json_logs.log",
                    "formatter": "json",
                },
            }
        )
        log_config["root"]["handlers"].extend(["file", "json_file"])

    logging.config.dictConfig(log_config)

    return logging.getLogger()


# Initialize logger
logger = setup_logging()


def get_logger(module_name: str) -> logging.Logger:
    """
    Retrieves a logger instance for the specified module.

    This function obtains a logger from Python's logging module based on the provided module name.
    The returned logger is already configured via the setup_logging function and is ready to use.

    Args:
        module_name (str): The name of the module for which the logger is being requested.

    Returns:
        logging.Logger: A logger instance for the specified module.
    """
    return logging.getLogger(module_name)


def log_function_call(logger: logging.Logger) -> Callable:
    """
    A decorator for logging the call details of a function.

    This decorator logs the start and end of a function call, along with the argument names, types,
    and sizes (in kilobytes). It also logs the calling module.function, called module.function
    and its execution time.

    Args:
        logger (logging.Logger): The logger object to use for logging.

    Returns:
        function: A wrapper function that adds logging to the decorated function.
    """

    def decorator_log_function_call(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get caller's frame
            caller_frame = inspect.currentframe().f_back

            # Initialize caller's details
            caller_module, caller_func_name = "Notebook Cell", "Unknown"

            # Check if caller frame exists and extract details
            if caller_frame:
                module = inspect.getmodule(caller_frame)
                if module:
                    caller_module = module.__name__
                caller_func_name = caller_frame.f_code.co_name

            # Get the module name of the function being called
            func_module = func.__module__

            # Log the start of the function call
            start_time = time.time()

            # Function to determine if the type is a basic Python data type
            def is_basic_type(obj):
                return isinstance(obj, (int, float, str, list, dict, tuple))

            # Prepare a summary of argument names, types, sizes in KB, and values (if basic type)
            arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
            arg_summary = (
                ", ".join(
                    [
                        f"{name}={type(a).__name__} (size: {sys.getsizeof(a) / 1024:.2f} KB, value: {a if is_basic_type(a) else 'Complex Type'})"
                        for name, a in zip(arg_names, args)
                    ]
                    + [
                        f"{k}={type(v).__name__} (size: {sys.getsizeof(v) / 1024:.2f} KB, value: {v if is_basic_type(v) else 'Complex Type'})"
                        for k, v in kwargs.items()
                    ]
                )
                or "None"
            )

            logger.debug(
                f"Function {func_module}.{func.__name__} called by {caller_module}.{caller_func_name} with args: {arg_summary}"
            )

            # Call the actual function
            result = func(*args, **kwargs)

            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Log the end of the function call
            logger.debug(
                f"Function {func_module}.{func.__name__} called by {caller_module}.{caller_func_name} ended in {execution_time:.4f} seconds"
            )

            return result

        return wrapper

    return decorator_log_function_call
