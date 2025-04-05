from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import sys
import logging

class Config(BaseModel):
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    SERPER_API_KEY: str

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("SERPER_API_KEY"):
            missing_vars.append("SERPER_API_KEY")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            SERPER_API_KEY=os.getenv("SERPER_API_KEY")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["SERPER_API_KEY"] = "***" if safe_config["SERPER_API_KEY"] else None
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