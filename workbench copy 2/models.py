# Merged models.py from templates: spotify_agent, github_agent
# Created at: 2025-04-05T12:00:27.506978

from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import httpx
import logging
import os
import sys

class Config(BaseModel):
    MODEL_CHOICE: str  = "gpt-4o-mini"
    BASE_URL: str  = "https://api.openai.com/v1"
    LLM_API_KEY: str
    SPOTIFY_API_KEY: str
    GITHUB_PERSONAL_ACCESS_TOKEN: str


class GitHubDeps:
    client: httpx.AsyncClient
    github_token: str | None = None

