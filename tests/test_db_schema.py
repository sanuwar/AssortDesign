import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from app.models import Tag


def test_tables_created(test_engine):
    inspector = inspect(test_engine)
    table_names = set(inspector.get_table_names())
    expected = {
        "document",
        "job",
        "jobattempt",
        "tag",
        "documenttag",
        "documentclue",
    }
    assert expected.issubset(table_names)


def test_tag_name_unique_constraint(test_engine):
    with Session(test_engine) as session:
        session.add(Tag(name="duplicate"))
        session.commit()
        session.add(Tag(name="duplicate"))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_document_tag_composite_pk(test_engine):
    inspector = inspect(test_engine)
    pk = inspector.get_pk_constraint("documenttag")
    columns = pk.get("constrained_columns", [])
    assert set(columns) == {"document_id", "tag_id"}
