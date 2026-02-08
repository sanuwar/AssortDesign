from __future__ import annotations

from typing import Any, Dict, List

from sqlmodel import Session, select

import app.graph as app_graph
from app.models import Job, JobAttempt


def _create_job(client, audience: str = "auto") -> int:
    response = client.post(
        "/web/documents",
        data={"input_text": "Pipeline test content", "audience": audience},
        follow_redirects=False,
    )
    location = response.headers["location"]
    return int(location.rstrip("/").split("/")[-1])


def _mock_generate(*args, **kwargs) -> Dict[str, Any]:
    return {
        "one_line_summary": "Test summary.",
        "tags": ["alpha", "beta"],
        "key_clues": ["clue a", "clue b"],
        "decision_bullets": [
            "Executive Summary: summary bullet",
            "Market Opportunity: market bullet",
            "Value Proposition: value bullet",
        ],
        "mind_map": "mindmap\n  root((Summary))\n    Executive Summary\n    Market Opportunity",
    }


def _mock_route(*args, **kwargs) -> Dict[str, Any]:
    return {"audience": "commercial", "confidence": 0.9, "reasons": ["mocked"]}


def test_run_pipeline_creates_attempt_and_updates_status(client, db_session, monkeypatch):
    monkeypatch.setattr(app_graph, "route_audience", _mock_route)
    monkeypatch.setattr(app_graph, "generate_content", _mock_generate)
    monkeypatch.setattr(
        app_graph,
        "evaluate_content",
        lambda *args, **kwargs: {
            "pass": True,
            "word_count": 10,
            "missing_sections": [],
            "fail_reasons": [],
            "fix_instructions": [],
        },
    )

    job_id = _create_job(client, audience="auto")
    response = client.post(f"/web/jobs/{job_id}/run", follow_redirects=False)
    assert response.status_code in (302, 303)

    job = db_session.get(Job, job_id)
    attempts = db_session.exec(
        select(JobAttempt).where(JobAttempt.job_id == job_id)
    ).all()
    assert job.status in ("completed", "failed")
    assert len(attempts) >= 1


def test_revision_respects_max_retries(client, db_session, monkeypatch):
    monkeypatch.setattr(app_graph, "route_audience", _mock_route)
    monkeypatch.setattr(app_graph, "generate_content", _mock_generate)
    monkeypatch.setattr(
        app_graph,
        "evaluate_content",
        lambda *args, **kwargs: {
            "pass": False,
            "word_count": 999,
            "missing_sections": ["Executive Summary"],
            "fail_reasons": ["force_fail"],
            "fix_instructions": ["revise"],
        },
    )

    job_id = _create_job(client, audience="auto")
    response = client.post(f"/web/jobs/{job_id}/run", follow_redirects=False)
    assert response.status_code in (302, 303)

    job = db_session.get(Job, job_id)
    attempts = db_session.exec(
        select(JobAttempt).where(JobAttempt.job_id == job_id)
    ).all()
    assert len(attempts) <= job.max_retries + 1
    assert job.status == "failed"


def test_selected_audience_preserved(client, db_session, monkeypatch):
    def _route_should_not_be_called(*args, **kwargs):
        raise AssertionError("route_audience should not be called when audience is preselected.")

    monkeypatch.setattr(app_graph, "route_audience", _route_should_not_be_called)
    monkeypatch.setattr(app_graph, "generate_content", _mock_generate)
    monkeypatch.setattr(
        app_graph,
        "evaluate_content",
        lambda *args, **kwargs: {
            "pass": True,
            "word_count": 10,
            "missing_sections": [],
            "fail_reasons": [],
            "fix_instructions": [],
        },
    )

    job_id = _create_job(client, audience="commercial")
    response = client.post(f"/web/jobs/{job_id}/run", follow_redirects=False)
    assert response.status_code in (302, 303)

    job = db_session.get(Job, job_id)
    assert job.audience == "commercial"
