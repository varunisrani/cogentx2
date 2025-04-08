from setuptools import setup, find_packages

setup(
    name="time_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic-ai",
        "pytz",
        "streamlit",
        "colorlog",
        "python-dotenv"
    ],
    py_modules=["mcp_server_time"],
)
