from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import sys
import logging

class SerperSpotifyAgentConfig(BaseModel):
    """Configuration for the Serper and Spotify Integrated Agent"""
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: str
    SERPER_API_KEY: str
    SPOTIFY_API_KEY: str

    @classmethod
    def load_from_environment(cls) -> 'SerperSpotifyAgentConfig':
        """Load agent configuration from environment variables with comprehensive error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_env_vars = []
        if not os.getenv("OPENAI_API_KEY"):
            missing_env_vars.append("OPENAI_API_KEY")
        if not os.getenv("SERPER_API_KEY"):
            missing_env_vars.append("SERPER_API_KEY")
        if not os.getenv("SPOTIFY_API_KEY"):
            missing_env_vars.append("SPOTIFY_API_KEY")
            
        if missing_env_vars:
            logging.error("Missing mandatory environment variables:")
            for var in missing_env_vars:
                logging.error(f"  - {var}")
            logging.error("""
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
SPOTIFY_API_KEY=your_spotify_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            SERPER_API_KEY=os.getenv("SERPER_API_KEY"),
            SPOTIFY_API_KEY=os.getenv("SPOTIFY_API_KEY")
        )

def setup_serper_spotify_agent() -> SerperSpotifyAgentConfig:
    """Initialize the Serper and Spotify integrated agent configuration from environment variables"""
    try:
        config = SerperSpotifyAgentConfig.load_from_environment()
        logging.debug("Successfully loaded configuration for Serper and Spotify integration")
        # Obscure sensitive information in logs for security
        anonymized_config = config.dict()
        anonymized_config["OPENAI_API_KEY"] = "***" if anonymized_config["OPENAI_API_KEY"] else None
        anonymized_config["SERPER_API_KEY"] = "***" if anonymized_config["SERPER_API_KEY"] else None
        anonymized_config["SPOTIFY_API_KEY"] = "***" if anonymized_config["SPOTIFY_API_KEY"] else None
        logging.debug(f"Config values: {anonymized_config}")
        return config
    except ValidationError as e:
        logging.error("Error during configuration validation:")
        for error in e.errors():
            logging.error(f"  - {error['loc'][0]}: {error['msg']}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading configuration: {str(e)}")
        sys.exit(1)

def load_config() -> SerperSpotifyAgentConfig:
    """Load the agent configuration from environment variables.
    This is the main entry point for configuration loading used by main.py.
    
    Returns:
        SerperSpotifyAgentConfig: The loaded and validated configuration
    """
    try:
        logging.info("Loading agent configuration...")
        config = setup_serper_spotify_agent()
        logging.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise