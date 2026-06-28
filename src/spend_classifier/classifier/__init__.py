"""Classifier package: builds the configured code selector."""

from __future__ import annotations

from ..config import ClassifierConfig
from .base import CodeSelector, SelectionResult
from .mock_llm import MockCodeSelector
from .openai_llm import OpenAICodeSelector


def build_selector(config: ClassifierConfig) -> CodeSelector:
    if config.classifier_backend == "openai":
        return OpenAICodeSelector(config.openai_model)
    return MockCodeSelector()


__all__ = ["build_selector", "CodeSelector", "SelectionResult", "MockCodeSelector", "OpenAICodeSelector"]
