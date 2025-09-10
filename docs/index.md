# Welcome to smart-grocery-search

Smart Grocery Search is a demo project that combines vector search, knowledge graphs, and DSPy to power smarter, attribute-aware product discovery in grocery retail.

This documentation is designed to evolve alongside the smart-grocery-search project,
ensuring up-to-date and accurate information for all users and contributors.

# Smart Grocery Search (KG-RAG + DSPy)

## Quickstart
```bash
poetry install
poetry shell
poetry run python -m sgs prepare-data
poetry run python -m sgs run-server
# in another terminal:
curl 'http://127.0.0.1:8000/search?q=nut-free%20granola'
