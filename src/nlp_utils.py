# Knowledge Graph Builder - NLP Utilities
# Author: Maharshi Soni | License: MIT

"""Shared NLP helpers: NLTK resource management and text pre-processing."""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import List

import nltk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data path sanitization
# ---------------------------------------------------------------------------
# On some Windows machines, NLTK's default search paths include drives that
# are not mounted or have unrecognised file systems (e.g. D:\nltk_data).
# Iterating over those paths with os.path.realpath raises OSError, so we
# proactively remove any path that cannot be resolved.

def _sanitize_nltk_paths() -> None:
    """Remove NLTK data paths that point to inaccessible volumes."""
    clean: List[str] = []
    for p in nltk.data.path:
        if not isinstance(p, str):
            continue
        try:
            os.path.realpath(p)
            clean.append(p)
        except OSError:
            logger.debug("Removing inaccessible NLTK data path: %s", p)
    nltk.data.path = clean

    # Ensure a writable fallback directory exists
    fallback = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(fallback, exist_ok=True)
    if fallback not in nltk.data.path:
        nltk.data.path.insert(0, fallback)

_sanitize_nltk_paths()

# ---------------------------------------------------------------------------
# NLTK resource bootstrapping
# ---------------------------------------------------------------------------

_REQUIRED_RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]


def ensure_nltk_resources() -> None:
    """Download required NLTK data packages if they are not already present."""
    download_dir = nltk.data.path[0] if nltk.data.path else os.path.join(os.path.expanduser("~"), "nltk_data")
    for resource in _REQUIRED_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"chunkers/{resource}")
                except LookupError:
                    try:
                        nltk.data.find(f"corpora/{resource}")
                    except LookupError:
                        logger.info("Downloading NLTK resource: %s", resource)
                        nltk.download(resource, download_dir=download_dir, quiet=True)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK's sentence tokenizer."""
    ensure_nltk_resources()
    return nltk.sent_tokenize(clean_text(text))
