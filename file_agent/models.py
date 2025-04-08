from pydantic import BaseModel
import os
import sys
import logging
import httpx
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class FileDeps:
    client: httpx.AsyncClient
    base_path: str | None = None

class Config(BaseModel):
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    BASE_PATH: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("BASE_PATH"):
            missing_vars.append("BASE_PATH")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
BASE_PATH=your_base_path  # Base directory for filesystem operations
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            BASE_PATH=os.getenv("BASE_PATH")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        logging.debug(f"Config values: {safe_config}")
        
        # Validate base path exists
        if not os.path.exists(config.BASE_PATH):
            logging.error(f"Base path does not exist: {config.BASE_PATH}")
            logging.error("Please provide a valid directory path in the BASE_PATH environment variable")
            sys.exit(1)
            
        return config
    except Exception as e:
        logging.error(f"Unexpected error loading configuration: {str(e)}")
        sys.exit(1)
