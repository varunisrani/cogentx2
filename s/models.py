from pydantic import BaseModel, ValidationError
from dataclasses import dataclass
import os
import sys
import logging
import httpx
from dotenv import load_dotenv

@dataclass
class GitHubAgent:
    client: httpx.AsyncClient
    github_token: str | None = None

@dataclass
class SpotifyAgent:
    client: httpx.AsyncClient
    spotify_token: str | None = None

@dataclass
class GitHubDeps:
    client: httpx.AsyncClient
    github_token: str | None = None

class Config(BaseModel):
    """Configuration for Spotify and GitHub Agents"""
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    SPOTIFY_API_KEY: str | None = None
    GITHUB_PERSONAL_ACCESS_TOKEN: str | None = None

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()

        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("SPOTIFY_API_KEY"):
            missing_vars.append("SPOTIFY_API_KEY")
        if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
            missing_vars.append("GITHUB_PERSONAL_ACCESS_TOKEN")

        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
SPOTIFY_API_KEY=your_spotify_api_key
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_access_token
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)

        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            SPOTIFY_API_KEY=os.getenv("SPOTIFY_API_KEY"),
            GITHUB_PERSONAL_ACCESS_TOKEN=os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        )

def load_config() -> Config:
    """Load the configuration from environment variables"""
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["SPOTIFY_API_KEY"] = "***" if safe_config["SPOTIFY_API_KEY"] else None
        safe_config["GITHUB_PERSONAL_ACCESS_TOKEN"] = "***" if safe_config["GITHUB_PERSONAL_ACCESS_TOKEN"] else None
        logging.debug(f"Config values: {safe_config}")
        return config
    except ValidationError as e:
        logging.error("Configuration validation error:")
        for error in e.errors():
            logging.error(f"  - {error['loc'][0]}: {error['msg']}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {str(e)}")
        sys.exit(1)

def initialize_agents() -> dict:
    """Load the configuration from environment variables and initialize agents"""
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["SPOTIFY_API_KEY"] = "***" if safe_config["SPOTIFY_API_KEY"] else None
        safe_config["GITHUB_PERSONAL_ACCESS_TOKEN"] = "***" if safe_config["GITHUB_PERSONAL_ACCESS_TOKEN"] else None
        logging.debug(f"Config values: {safe_config}")

        spotify_agent = SpotifyAgent(client=httpx.AsyncClient(), spotify_token=config.SPOTIFY_API_KEY)
        github_agent = GitHubAgent(client=httpx.AsyncClient(), github_token=config.GITHUB_PERSONAL_ACCESS_TOKEN)

        return {
            "config": config,
            "spotify_agent": spotify_agent,
            "github_agent": github_agent
        }
    except ValidationError as e:
        logging.error("Configuration validation error:")
        for error in e.errors():
            logging.error(f"  - {error['loc'][0]}: {error['msg']}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {str(e)}")
        sys.exit(1)