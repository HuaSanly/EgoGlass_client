from ui.dataset_builder import EpisodeInterval, split_valid_intervals


def test_dataset_valid_interval_retention_eval() -> None:
    """At least 90% of valid synthetic frames survive deterministic splitting."""

    frames = tuple(range(300))
    output = split_valid_intervals("clip", frames, minimum_frames=10)
    retained = sum(interval.frame_count for interval in output)
    assert output == (EpisodeInterval("clip", 0, 300),)
    assert retained / len(frames) >= 0.9
