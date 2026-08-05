from pathlib import Path

from tests.conftest import SESSION_ID
from ui.annotation.controller import AnnotationController
from ui.annotation.models import EpisodeLabels, PhaseKind
from ui.annotation.store import AnnotationStore


def test_controller_edits_undoes_saves_and_publishes(recordings_root: Path) -> None:
    ids = iter(("3" * 32, "4" * 32, "5" * 32))
    controller = AnnotationController(
        AnnotationStore(recordings_root, id_factory=lambda: next(ids)),
        id_factory=lambda: next(ids),
    )
    controller.refresh_workspace()
    controller.select_session(SESSION_ID)

    episode = controller.add_episode(30, 240)
    controller.update_labels(
        EpisodeLabels(
            task_id="place-cup",
            instruction="拿起杯子并放到托盘中",
            verb="放置",
            object="杯子",
            target="托盘",
            hand="right",
            outcome="success",
        )
    )
    controller.add_phase(
        PhaseKind.MANIPULATE,
        start_frame_index=30,
        end_frame_index_exclusive=240,
        active_hand="right",
        action_verb="拿起并放置",
        object_name="杯子",
    )

    controller.undo()
    assert controller.selected_episode is not None
    assert controller.selected_episode.phases == []
    controller.redo()
    assert len(controller.selected_episode.phases) == 1

    draft = controller.save()
    revision = controller.publish()

    assert episode.episode_id == "3" * 32
    assert draft.draft_revision == 1
    assert controller.dirty is False
    assert revision.quality.episode_count == 1
    assert revision.quality.phase_count == 1


def test_controller_accepts_fixed_window_proposals(recordings_root: Path) -> None:
    ids = iter(f"{value:x}" * 32 for value in range(3, 16))
    controller = AnnotationController(
        AnnotationStore(recordings_root),
        id_factory=lambda: next(ids),
    )
    controller.refresh_workspace()
    controller.select_session(SESSION_ID)

    proposals = controller.generate_proposals(
        "fixed_window",
        window_duration_ms=4000,
        stride_duration_ms=4000,
    )
    accepted = controller.accept_proposals()

    assert accepted == len(proposals.proposals) == 3
    assert controller.detail is not None
    assert len(controller.detail.draft.episodes) == 3
    assert controller.detail.draft.segmentation_strategy == "fixed_window"
