"""
CARELINK AI - SQLite Database Base

Sets up SQLAlchemy engine + session factory for SQLite.
Uses aiosqlite for async support in FastAPI.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

from app.core.settings import get_settings

settings = get_settings()

# Sync engine (SQLite — file-based)
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + multi-thread
    echo=settings.DEBUG,                         # SQL logging in debug mode
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator:
    """
    FastAPI dependency that provides a transactional DB session.

    Usage:
        @router.get("/")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables defined in models. Called on application startup."""
    # Import all model modules so SQLAlchemy picks up the metadata
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
