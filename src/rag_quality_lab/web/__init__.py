"""A lightweight local workbench: a JSON API and one static page over the
existing runner, store, budget, compare, and reporting service layer."""

from rag_quality_lab.web.app import create_app, serve

__all__ = ["create_app", "serve"]
