"""AquaQuery FastAPI application."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.middleware import (
    RateLimitMiddleware,
    RequestIdMiddleware,
    RequestLoggingMiddleware,
)
from backend.api.routes import router, set_data_loader, set_graph, set_session_store
from backend.api.session import SessionStore
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize resources on startup, clean up on shutdown."""
    settings = get_settings()
    logger.info("Starting AquaQuery API...")

    # Initialize session store
    session_store = SessionStore()
    set_session_store(session_store)

    # Initialize data loader
    from backend.data.loader import ArgoDataLoader
    data_loader = ArgoDataLoader(settings=settings)
    set_data_loader(data_loader)

    # Initialize knowledge base
    from backend.data.indexer import ArgoKnowledgeIndexer
    indexer = ArgoKnowledgeIndexer(settings=settings)
    indexer.index_knowledge_base()
    logger.info("Knowledge base indexed")

    # Build agent graph
    from backend.agents.supervisor import build_graph
    graph = build_graph(settings=settings, indexer=indexer)
    set_graph(graph)
    logger.info("Agent pipeline initialized")

    # Warn if Argo cache is empty
    cache_dir = Path(settings.argo_cache_dir)
    if not list(cache_dir.glob("*.nc")):
        logger.warning(
            "Argo cache is empty (%s). "
            "Run 'python -m backend.scripts.precache_argo' for faster queries.",
            cache_dir,
        )

    yield

    logger.info("Shutting down AquaQuery API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="AquaQuery API",
        description="Natural language interface for Argo oceanographic data",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware (order matters: outermost first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

    # Routes
    app.include_router(router)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
