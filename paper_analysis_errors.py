"""
Shared exception classes for paper analysis pipeline.

This module defines custom exceptions used across the paper analysis
components to provide consistent error handling and diagnostics.
"""

from typing import Any, Dict, Optional


class PaperAnalysisError(Exception):
    """
    Base exception for paper analysis errors.

    Raised when the overall paper analysis process fails due to
    configuration, setup, or fundamental issues.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class ExtractionError(Exception):
    """
    Exception raised when structured extraction from LLM fails.

    Used when the LLM response cannot be parsed as valid JSON,
    does not match the expected schema, or contains invalid values.

    Args:
        message: Error message describing the failure
        diagnostics: Optional dictionary with diagnostic information
                   (e.g., attempt count, duration, error details)
    """
    def __init__(self, message: str, diagnostics: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}
