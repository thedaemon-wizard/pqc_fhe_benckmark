"""
PQC-FHE REST API Package
========================

FastAPI-based REST API for Post-Quantum Cryptography 
and Fully Homomorphic Encryption operations.

Usage:
    # Run server
    python -m api.server
    
    # Or with uvicorn
    uvicorn api.server:app --reload
    
    # Access Swagger UI
    http://localhost:8000/docs
    
    # Access ReDoc
    http://localhost:8000/redoc
"""

# Only import 'app' which is the FastAPI application instance
from .server import app

__all__ = ['app']
