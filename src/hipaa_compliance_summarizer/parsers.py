from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class ParsingError(Exception):
    """Base exception for parsing-related errors."""
    pass


class FileReadError(ParsingError):
    """Raised when file cannot be read due to various issues."""
    pass


class EncodingError(ParsingError):
    """Raised when file encoding issues prevent proper text extraction."""
    pass


def _load_text(path_or_text: str, encoding: str = 'utf-8', fallback_encodings: Optional[list] = None) -> str:
    """Return the text content from a file path or raw string with robust error handling.
    
    Args:
        path_or_text: File path or text content
        encoding: Primary encoding to try (default: utf-8)
        fallback_encodings: List of fallback encodings to try if primary fails
        
    Returns:
        Text content as string
        
    Raises:
        FileReadError: If file cannot be read
        EncodingError: If text cannot be decoded with any encoding
        TypeError: If input is not a string
    """
    if not isinstance(path_or_text, str):
        raise TypeError(f"Input must be a string, got {type(path_or_text).__name__}")
    
    if not path_or_text:
        return ""
    
    # Check if it's likely a file path (has path separators, file extension, or typical path structure)
    is_likely_path = (
        ('/' in path_or_text or '\\' in path_or_text or 
         ('.' in path_or_text and len(path_or_text.split()) == 1)) and  # Single word with dot (likely filename)
        len(path_or_text) < 1000 and
        '\n' not in path_or_text  # Paths don't contain newlines
    )
    
    if is_likely_path:
        try:
            path = Path(path_or_text)
            
            # Check if path exists and is a file
            if not path.exists():
                logger.warning("File path does not exist: %s", path_or_text)
                raise FileNotFoundError(f"File not found: {path_or_text}")
            
            if path.is_dir():
                logger.error("Path is a directory, not a file: %s", path_or_text)
                raise IsADirectoryError(f"Path is a directory: {path_or_text}")
            
            if not path.is_file():
                logger.warning("Path is not a regular file: %s", path_or_text)
                # For device files like /dev/null, return empty string
                return ""
            
            # Try to read with primary encoding
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as e:
                logger.warning("Failed to decode file %s with %s: %s", path_or_text, encoding, e)
                
                # Try fallback encodings
                fallback_encodings = fallback_encodings or ['latin1', 'cp1252', 'ascii']
                
                for fallback_encoding in fallback_encodings:
                    try:
                        logger.info("Trying fallback encoding %s for %s", fallback_encoding, path_or_text)
                        content = path.read_text(encoding=fallback_encoding, errors='replace')
                        logger.info("Successfully decoded with %s encoding", fallback_encoding)
                        return content
                    except Exception as fallback_error:
                        logger.debug("Fallback encoding %s failed: %s", fallback_encoding, fallback_error)
                        continue
                
                # If all encodings fail, raise an encoding error
                raise EncodingError(f"Could not decode file {path_or_text} with any encoding") from e
            
            except (OSError, PermissionError) as e:
                logger.error("Failed to read file %s: %s", path_or_text, e)
                raise FileReadError(f"Cannot read file {path_or_text}: {e}") from e
            
        except (ValueError,) as e:
            # If path creation fails, treat as text content
            logger.debug("Path creation failed, treating as text content: %s", e)
            pass
    
    # Return as-is if it's text content or path operations failed
    return path_or_text


def parse_medical_record(data: str) -> str:
    """Extract text from a medical record file or string.
    
    Args:
        data: File path or text content
        
    Returns:
        Extracted and cleaned text content
        
    Raises:
        ParsingError: If parsing fails
        TypeError: If input is not a string
    """
    try:
        content = _load_text(data)
        return content.strip()
    except (FileReadError, EncodingError, FileNotFoundError, IsADirectoryError, PermissionError) as e:
        logger.error("Failed to parse medical record: %s", e)
        raise ParsingError(f"Medical record parsing failed: {e}") from e
    except Exception as e:
        logger.error("Unexpected error parsing medical record: %s", e)
        raise ParsingError(f"Unexpected parsing error: {e}") from e


def parse_clinical_note(data: str) -> str:
    """Extract text from a clinical note.
    
    Args:
        data: File path or text content
        
    Returns:
        Extracted and cleaned text content
        
    Raises:
        ParsingError: If parsing fails
        TypeError: If input is not a string
    """
    try:
        content = _load_text(data)
        return content.strip()
    except (FileReadError, EncodingError, FileNotFoundError, IsADirectoryError, PermissionError) as e:
        logger.error("Failed to parse clinical note: %s", e)
        raise ParsingError(f"Clinical note parsing failed: {e}") from e
    except Exception as e:
        logger.error("Unexpected error parsing clinical note: %s", e)
        raise ParsingError(f"Unexpected parsing error: {e}") from e


def parse_insurance_form(data: str) -> str:
    """Extract text from an insurance form.
    
    Args:
        data: File path or text content
        
    Returns:
        Extracted and cleaned text content
        
    Raises:
        ParsingError: If parsing fails
        TypeError: If input is not a string
    """
    try:
        content = _load_text(data)
        return content.strip()
    except (FileReadError, EncodingError, FileNotFoundError, IsADirectoryError, PermissionError) as e:
        logger.error("Failed to parse insurance form: %s", e)
        raise ParsingError(f"Insurance form parsing failed: {e}") from e
    except Exception as e:
        logger.error("Unexpected error parsing insurance form: %s", e)
        raise ParsingError(f"Unexpected parsing error: {e}") from e
