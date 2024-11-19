import os
from pathlib import Path
from getpass import getpass
from src.utilitaries.Logger import Logger

logger = Logger(__name__)


def setup_openai_token() -> str:
    """
    Sets up the OpenAI API token by either loading it from the scratch directory
    or prompting the user to enter it.

    Returns:
        str: The OpenAI API token
    """
    logger.debug("Setting up OpenAI token")

    # Define token file path in scratch directory
    scratch_dir = Path("scratch")
    token_file = scratch_dir / ".openai_token"

    # Create scratch directory if it doesn't exist
    scratch_dir.mkdir(exist_ok=True)

    # Try to load existing token
    if token_file.exists():
        logger.debug("Loading existing OpenAI token")
        with open(token_file, "r") as f:
            token = f.read().strip()
        if token:
            os.environ["OPENAI_API_KEY"] = token
            return token

    # If no token exists or it's empty, prompt user
    logger.info("OpenAI token not found. Please enter your OpenAI API key:")
    token = getpass("OpenAI API Key: ").strip()

    # Save token
    logger.debug("Saving OpenAI token")
    with open(token_file, "w") as f:
        f.write(token)

    # Set environment variable
    os.environ["OPENAI_API_KEY"] = token

    logger.info("OpenAI token successfully saved")
    return token
