from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import sys
import logging

class Config(BaseModel):
    MODEL_CHOICE: str = "gpt-4o-mini"
    BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str
    FIRECRAWL_API_KEY: str
    FIRECRAWL_RETRY_MAX_ATTEMPTS: str = "5"
    FIRECRAWL_RETRY_INITIAL_DELAY: str = "2000"
    FIRECRAWL_RETRY_MAX_DELAY: str = "30000"
    FIRECRAWL_RETRY_BACKOFF_FACTOR: str = "3"
    FIRECRAWL_CREDIT_WARNING_THRESHOLD: str = "2000"
    FIRECRAWL_CREDIT_CRITICAL_THRESHOLD: str = "500"

    @classmethod
    def load_from_env(cls) -> 'Config':
        """Load configuration from environment variables with better error handling"""
        load_dotenv()
        
        # Check for required environment variables
        missing_vars = []
        if not os.getenv("LLM_API_KEY"):
            missing_vars.append("LLM_API_KEY")
        if not os.getenv("FIRECRAWL_API_KEY"):
            missing_vars.append("FIRECRAWL_API_KEY")
            
        if missing_vars:
            logging.error("Missing required environment variables:")
            for var in missing_vars:
                logging.error(f"  - {var}")
            logging.error("\nPlease create a .env file with the following content:")
            logging.error("""
LLM_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
MODEL_CHOICE=gpt-4o-mini  # optional
BASE_URL=https://api.openai.com/v1  # optional

# Optional Firecrawl settings
FIRECRAWL_RETRY_MAX_ATTEMPTS=5
FIRECRAWL_RETRY_INITIAL_DELAY=2000
FIRECRAWL_RETRY_MAX_DELAY=30000
FIRECRAWL_RETRY_BACKOFF_FACTOR=3
FIRECRAWL_CREDIT_WARNING_THRESHOLD=2000
FIRECRAWL_CREDIT_CRITICAL_THRESHOLD=500
            """)
            sys.exit(1)
            
        return cls(
            MODEL_CHOICE=os.getenv("MODEL_CHOICE", "gpt-4o-mini"),
            BASE_URL=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            LLM_API_KEY=os.getenv("LLM_API_KEY"),
            FIRECRAWL_API_KEY=os.getenv("FIRECRAWL_API_KEY"),
            FIRECRAWL_RETRY_MAX_ATTEMPTS=os.getenv("FIRECRAWL_RETRY_MAX_ATTEMPTS", "5"),
            FIRECRAWL_RETRY_INITIAL_DELAY=os.getenv("FIRECRAWL_RETRY_INITIAL_DELAY", "2000"),
            FIRECRAWL_RETRY_MAX_DELAY=os.getenv("FIRECRAWL_RETRY_MAX_DELAY", "30000"),
            FIRECRAWL_RETRY_BACKOFF_FACTOR=os.getenv("FIRECRAWL_RETRY_BACKOFF_FACTOR", "3"),
            FIRECRAWL_CREDIT_WARNING_THRESHOLD=os.getenv("FIRECRAWL_CREDIT_WARNING_THRESHOLD", "2000"),
            FIRECRAWL_CREDIT_CRITICAL_THRESHOLD=os.getenv("FIRECRAWL_CREDIT_CRITICAL_THRESHOLD", "500")
        )

def load_config() -> Config:
    try:
        config = Config.load_from_env()
        logging.debug("Configuration loaded successfully")
        # Hide sensitive information in logs
        safe_config = config.model_dump()
        safe_config["LLM_API_KEY"] = "***" if safe_config["LLM_API_KEY"] else None
        safe_config["FIRECRAWL_API_KEY"] = "***" if safe_config["FIRECRAWL_API_KEY"] else None
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