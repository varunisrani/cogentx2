from pydantic import BaseModel, Field
from typing import List, Optional

class FireCrawlResult(BaseModel):
    """Model for FireCrawl search results"""
    title: str = Field(description="Title of the webpage")
    snippet: str = Field(description="Content snippet from the webpage")
    url: str = Field(description="URL of the webpage")
    timestamp: Optional[str] = Field(description="Timestamp of when the page was crawled")

    def __str__(self):
        return f"Title: {self.title}\nSnippet: {self.snippet}\nURL: {self.url}\nTimestamp: {self.timestamp}\n"

class FireCrawlResponse(BaseModel):
    """Model for the complete FireCrawl response"""
    query: str = Field(description="Original search query")
    results: List[FireCrawlResult] = Field(description="List of search results")
    summary: str = Field(description="AI-generated summary of results")

    def __str__(self):
        results_str = "\n".join(str(r) for r in self.results)
        return f"Query: {self.query}\nSummary: {self.summary}\nResults:\n{results_str}"

class FireCrawlConfig(BaseModel):
    """Configuration for the FireCrawl agent"""
    api_key: str = Field(description="API key for FireCrawl")
    model_name: str = Field(default="gpt-4o-mini", description="Model to use for the agent")
    max_retries: int = Field(default=5, description="Maximum number of retry attempts")
    initial_delay: int = Field(default=2000, description="Initial retry delay in milliseconds")
    max_delay: int = Field(default=30000, description="Maximum retry delay in milliseconds")
    backoff_factor: int = Field(default=3, description="Retry backoff factor")
    credit_warning: int = Field(default=2000, description="Credit warning threshold")
    credit_critical: int = Field(default=500, description="Credit critical threshold") 