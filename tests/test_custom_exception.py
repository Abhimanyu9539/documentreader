# tests/test_custom_exception.py

import pytest
import os
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exception.custom_exception import DocumentPortalException

# =================================================================
# Test for Custom Exception (exception/custom_exception.py)
# =================================================================

def test_custom_exception_format():
    """Tests that the DocumentPortalException class formats the error message correctly."""
    try:
        # Simulate a real error to capture its details
        raise ValueError("This is the original error.")
    except ValueError as e:
        # Create the custom exception instance, passing sys for context
        custom_exc = DocumentPortalException(e, sys)
        
        # Convert the exception to its string representation
        error_message = str(custom_exc)
        
        # Assert that the formatted message contains the actual, correct parts
        assert "Error in" in error_message
        assert "at line" in error_message
        assert "Message: This is the original error." in error_message
        # Check that the name of this test file is in the error message
        assert os.path.basename(__file__) in error_message
