from starlette.testclient import TestClient

from operator_console.app import create_app


def test_annotation_page_exposes_review_before_publish_workflow() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/annotations").text
        script = client.get("/assets/annotations.js").text

    assert "生成候选" in page
    assert "接受候选" in page
    assert "发布版本" in page
    assert script.index("requestEpisodeProposals") < script.index("acceptProposals")
    assert script.index("acceptProposals") < script.index("publishCurrentDraft")
    assert "候选已进入草稿，请逐段检查边界和标签" in script
    assert "正式 Episode 不允许重叠" in script


def test_annotation_page_supports_task_attempt_and_internal_phase_labels() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/annotations").text
        script = client.get("/assets/annotations.js").text

    for label in ("任务描述", "动作", "对象", "参与手", "结果", "内部阶段"):
        assert label in page
    for phase in (
        "prepare",
        "approach",
        "contact",
        "manipulate",
        "release",
        "complete",
    ):
        assert f'<option value="{phase}">' in page
    assert "phase.start_frame_index" in script
    assert "phase.end_frame_index_exclusive" in script
    assert "内部阶段不能重叠" in script
