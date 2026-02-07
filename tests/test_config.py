from app.config import load_profiles


def test_agent_profiles_structure():
    profiles = load_profiles()

    assert isinstance(profiles, dict)
    assert "audiences" in profiles
    assert "routing" in profiles
    assert "evaluation" in profiles

    audiences = profiles["audiences"]
    assert isinstance(audiences, dict)

    expected_audiences = {
        "commercial",
        "medical_affairs",
        "r_and_d",
        "cross_functional",
    }
    assert expected_audiences.issubset(set(audiences.keys()))

    for name, audience in audiences.items():
        assert "display_name" in audience
        assert "system_prompt" in audience
        assert "required_sections" in audience
        assert "default_max_words" in audience

    routing = profiles["routing"]
    assert "low_confidence_threshold" in routing
    threshold = routing["low_confidence_threshold"]
    assert 0 <= float(threshold) <= 1

    evaluation = profiles["evaluation"]
    assert "evaluator_prompt" in evaluation
