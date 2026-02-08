def test_get_home(client):
    response = client.get("/web")
    assert response.status_code == 200


def test_create_document_and_job(client, db_session):
    response = client.post(
        "/web/documents",
        data={"input_text": "Test content", "audience": "auto"},
        follow_redirects=False,
    )
    assert response.status_code in (302, 303)
    location = response.headers.get("location")
    assert location and location.startswith("/web/jobs/")

    job_id = int(location.rstrip("/").split("/")[-1])
    job_response = client.get(f"/web/jobs/{job_id}")
    assert job_response.status_code == 200
