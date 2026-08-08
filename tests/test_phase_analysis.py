from phase_analysis import PhaseAnalysisConfig, PhaseAnalysisService, PhaseInputFrame
from schemas.phase import MotionPhase


def _frames() -> list[PhaseInputFrame]:
    frames: list[PhaseInputFrame] = []
    for index in range(80):
        if index < 20:
            position = (index * 0.02, 0.0, 0.0)
            hand_speed = 0.0
            grasping = False
        elif index < 30:
            position = (0.4, 0.0, 0.0)
            hand_speed = 0.0
            grasping = False
        elif index < 60:
            position = (0.4, 0.0, 0.0)
            hand_speed = 0.1
            grasping = index >= 40
        else:
            position = (0.4, 0.0, 0.0)
            hand_speed = 0.0
            grasping = False
        frames.append(
            PhaseInputFrame(
                clip_id="clip",
                frame_index=index,
                session_time_ns=index * 100_000_000,
                head_position_m=position,
                head_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                hand_linear_speed_m_s=hand_speed,
                grasping=grasping,
            )
        )
    return frames


def test_phase_analysis_uses_real_timestamps_and_generates_object_window() -> None:
    result = PhaseAnalysisService(
        PhaseAnalysisConfig(
            minimum_segment_frames=3,
            precontact_window_frames=5,
            minimum_object_window_frames=10,
            finished_trailing_frames=5,
        )
    ).analyze("run", _frames())

    assert result.frames[0].phase is MotionPhase.FORWARD
    assert any(frame.phase is MotionPhase.MANIPULATION for frame in result.frames)
    assert any(frame.phase is MotionPhase.FINISHED for frame in result.frames)
    assert result.object_centric_windows
    window = result.object_centric_windows[0]
    assert window.reference_frame_index < 40
    assert window.start_session_time_ns == window.start_frame_index * 100_000_000


def test_phase_analysis_keeps_clips_isolated() -> None:
    frames = _frames()
    frames.extend(
        PhaseInputFrame(
            clip_id="other",
            frame_index=index,
            session_time_ns=index * 50_000_000,
            head_position_m=(0.0, 0.0, 0.0),
            head_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            hand_linear_speed_m_s=0.0,
            grasping=False,
        )
        for index in range(10)
    )
    result = PhaseAnalysisService().analyze("run", frames)
    assert {frame.clip_id for frame in result.frames} == {"clip", "other"}
    assert all(segment.clip_id in {"clip", "other"} for segment in result.segments)
