from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.main as app_main


@pytest.fixture
def test_engine(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(test_engine):
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def client(test_engine, monkeypatch):
    def init_db_override() -> None:
        SQLModel.metadata.create_all(test_engine)

    @contextmanager
    def get_session_override():
        with Session(test_engine) as session:
            yield session

    monkeypatch.setattr(app_main, "init_db", init_db_override)
    monkeypatch.setattr(app_main, "get_session", get_session_override)

    with TestClient(app_main.app) as client:
        yield client
