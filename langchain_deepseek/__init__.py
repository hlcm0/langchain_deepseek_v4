"""LangChain DeepSeek integration."""

from importlib import metadata

from langchain_deepseek.chat_models import ChatDeepSeek

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = ["ChatDeepSeek", "__version__"]
