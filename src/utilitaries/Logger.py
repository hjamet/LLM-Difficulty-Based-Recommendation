import logging
from logging.handlers import RotatingFileHandler
import os
import inspect
import glob
from colorama import init, Fore, Style
import sys
import re

# Initialize colorama
init(autoreset=True)


def clean_old_logs(log_dir="logs", pattern="*.log"):
    """
    Remove all old log files from the specified directory.

    Args:
        log_dir (str): The directory containing log files. Defaults to "logs".
        pattern (str): The pattern to match log files. Defaults to "*.log".

    Returns:
        int: The number of log files removed.
    """
    removed_count = 0
    try:
        # Get all files matching the pattern in the log directory
        log_files = glob.glob(os.path.join(log_dir, pattern))

        # Remove each log file
        for log_file in log_files:
            try:
                os.remove(log_file)
                removed_count += 1
                logging.debug(f"Removed old log file: {log_file}")
            except Exception as e:
                logging.error(f"Error removing log file {log_file}: {str(e)}")

        logging.info(f"Cleaned {removed_count} old log files from {log_dir}")
    except Exception as e:
        logging.error(f"Error cleaning old logs: {str(e)}")

    return removed_count


# Call the function to clean old logs when the module is imported
clean_old_logs()


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to console output.
    """

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


class Logger:
    """
    A custom logger class that sets up logging with both console and file handlers.
    All log levels are written to files, but only specified levels are displayed in the console.
    Includes filename and line number in log messages.
    """

    root_logger = None

    @staticmethod
    def is_user_file(filename):
        """
        Check if the given filename is a user file (not from installed packages).

        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if it's a user file, False otherwise.
        """
        # Adjust this pattern to match your project structure
        user_file_pattern = r"^(?!.*site-packages).*\.py$"
        return re.match(user_file_pattern, filename) is not None

    class UserFileFilter(logging.Filter):
        """
        A filter that only allows records from user files.
        """

        def filter(self, record):
            return Logger.is_user_file(record.pathname)

    @classmethod
    def setup_root_logger(
        cls, log_file="app.log", console_level=logging.ERROR, file_level=logging.DEBUG
    ):
        """
        Set up the root logger for the entire application.
        """
        if cls.root_logger is None:
            cls.root_logger = logging.getLogger()
            cls.root_logger.setLevel(logging.DEBUG)

            # Remove all existing handlers
            for handler in cls.root_logger.handlers[:]:
                cls.root_logger.removeHandler(handler)

            # Create console handler with a simplified formatter
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            console_formatter = ColoredFormatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            cls.root_logger.addHandler(console_handler)

            # Create file handler with a simplified formatter
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_handler = RotatingFileHandler(
                os.path.join(log_dir, log_file),
                maxBytes=1024 * 1024,
                backupCount=5,  # 1MB
            )
            file_handler.setLevel(file_level)
            file_formatter = logging.Formatter(
                "%(asctime)s %(levelname)s - %(message)s", datefmt="%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            # Add the UserFileFilter to the file handler
            file_handler.addFilter(cls.UserFileFilter())

            cls.root_logger.addHandler(file_handler)

    def __init__(self, name):
        """
        Initialize the logger with the given name.

        Args:
            name (str): The name of the logger.
        """
        if self.root_logger is None:
            self.setup_root_logger()

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Inherit level from root logger

    def get_logger(self):
        """
        Get the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    def _log_with_context(self, level, message, *args, **kwargs):
        """
        Log a message with context information (filename and line number) for all log levels.
        Shows only the last relevant file in the call stack.
        """
        frame = inspect.currentframe().f_back
        while frame:
            filename = frame.f_code.co_filename
            if filename != __file__ and not filename.startswith(sys.prefix):
                break
            frame = frame.f_back

        if frame:
            filename = os.path.relpath(frame.f_code.co_filename, os.getcwd())
            lineno = frame.f_lineno
            context = f"[{filename}:{lineno}]"
        else:
            context = ""

        log_method = getattr(self.logger, level)
        log_method(f"{context} {message}", *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._log_with_context("debug", message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._log_with_context("info", message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._log_with_context("warning", message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._log_with_context("error", message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._log_with_context("critical", message, *args, **kwargs)


# Setup the root logger at the beginning of your application
Logger.setup_root_logger(console_level=logging.INFO)
