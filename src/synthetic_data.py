# Knowledge Graph Builder - Synthetic Data Generator
# Author: Maharshi Soni | License: MIT

"""Generate synthetic text documents about a fictional tech company domain
for demonstrating entity and relationship extraction."""

from __future__ import annotations

from typing import List

from src.models import DocumentInput

# ---------------------------------------------------------------------------
# Synthetic corpus about "NovaTech" - a fictional tech company
# ---------------------------------------------------------------------------

SYNTHETIC_DOCUMENTS: List[str] = [
    # Document 1 - Company overview
    (
        "NovaTech is a technology company headquartered in San Francisco. "
        "Founded by Alice Chen and David Park in 2018, the company focuses on "
        "building cloud-native infrastructure tools. NovaTech employs over 500 "
        "engineers and operates offices in New York, London, and Tokyo."
    ),
    # Document 2 - Engineering team
    (
        "Alice Chen leads the engineering division at NovaTech. Her team uses "
        "Python, Kubernetes, and Docker to build the company's core platform. "
        "Senior engineer Bob Martinez works on the GraphQL API layer, while "
        "Carol Zhang manages the machine learning infrastructure team. "
        "The team recently adopted Terraform for infrastructure-as-code."
    ),
    # Document 3 - Product launch
    (
        "NovaTech launched CloudBridge, a serverless data pipeline product, in "
        "March 2024. CloudBridge uses Apache Kafka for event streaming and "
        "PostgreSQL for metadata storage. The product was developed by the "
        "platform team led by David Park. CloudBridge is built on AWS and "
        "leverages Redis for caching."
    ),
    # Document 4 - Partnership
    (
        "NovaTech partnered with DataFlow Labs to integrate real-time analytics "
        "into CloudBridge. DataFlow Labs, based in Austin, specializes in "
        "stream processing using Apache Kafka and Elasticsearch. "
        "Emily Roberts from DataFlow Labs collaborated with Carol Zhang on "
        "the integration project."
    ),
    # Document 5 - Acquisition
    (
        "In January 2025, NovaTech acquired PixelAI, a computer vision startup "
        "founded by Frank Lee in Boston. PixelAI developed deep learning models "
        "using PyTorch and TensorFlow. Frank Lee joined NovaTech as VP of "
        "AI Research after the acquisition."
    ),
    # Document 6 - Conference talk
    (
        "At the DevConnect conference in London, Alice Chen presented on "
        "microservices architecture. She discussed how NovaTech uses Docker "
        "and Kubernetes at scale. Bob Martinez gave a workshop on GraphQL "
        "best practices. The conference was attended by engineers from Google, "
        "Microsoft, and Amazon."
    ),
    # Document 7 - Open source
    (
        "NovaTech open-sourced FlowGraph, a graph processing library written in "
        "Python. Carol Zhang leads the FlowGraph project, which uses NetworkX "
        "under the hood. FlowGraph supports knowledge graph construction and "
        "integrates with Pandas and NumPy for data manipulation."
    ),
    # Document 8 - Hiring & culture
    (
        "David Park manages the hiring process at NovaTech in San Francisco. "
        "The company uses GitHub Actions for CI/CD and follows agile practices. "
        "New hires receive training on Python, Docker, and the company's "
        "internal tools. Grace Kim joined as Head of Product based in New York."
    ),
    # Document 9 - Security & compliance
    (
        "NovaTech's security team, led by Henry Wu, developed an internal tool "
        "for vulnerability scanning. The tool is built with Python and uses "
        "Elasticsearch for log analysis. Henry Wu works at the San Francisco "
        "headquarters and reports to Alice Chen."
    ),
    # Document 10 - Future roadmap
    (
        "NovaTech plans to expand CloudBridge with support for MongoDB and "
        "Apache Kafka Streams. Frank Lee is developing new computer vision "
        "features using PyTorch. The company is exploring partnerships with "
        "CloudScale Inc located in Seattle and DataFlow Labs for advanced "
        "analytics capabilities."
    ),
]


def get_synthetic_documents() -> List[DocumentInput]:
    """Return the synthetic corpus as a list of DocumentInput objects."""
    return [
        DocumentInput(text=text, doc_id=f"doc-{i+1:03d}")
        for i, text in enumerate(SYNTHETIC_DOCUMENTS)
    ]


def get_synthetic_texts() -> List[str]:
    """Return just the raw text strings."""
    return list(SYNTHETIC_DOCUMENTS)
