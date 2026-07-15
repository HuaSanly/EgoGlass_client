from __future__ import annotations

from collections import OrderedDict

from .webrtc_models import VideoFrameMetadata

RTP_MODULUS = 1 << 32
MAX_TIMESTAMP_ERROR_90KHZ = 90


class DuplicateMetadataError(ValueError):
    """Raised when a frame metadata message is replayed."""


class FrameMetadataMatcher:
    """Associate decoded RTP timestamps with bounded, possibly reordered metadata."""

    def __init__(self, max_pending: int = 256) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._metadata: OrderedDict[int, VideoFrameMetadata] = OrderedDict()
        self._frames: OrderedDict[int, int] = OrderedDict()
        self._recent_frame_ids: OrderedDict[int, None] = OrderedDict()
        self._metadata_origin_90khz: int | None = None
        self.matched = 0
        self.duplicates = 0
        self.dropped = 0
        self.sdk_clock_discontinuities = 0
        self.max_timestamp_match_error_90khz = 0
        self._last_sdk_timestamp_ms: int | None = None

    @property
    def pending_metadata(self) -> int:
        return len(self._metadata)

    @property
    def pending_frames(self) -> int:
        return len(self._frames)

    @property
    def metadata_origin_90khz(self) -> int | None:
        return self._metadata_origin_90khz

    def add_metadata(self, metadata: VideoFrameMetadata) -> bool:
        if metadata.frame_id in self._recent_frame_ids:
            self.duplicates += 1
            raise DuplicateMetadataError(f"duplicate frame_id {metadata.frame_id}")
        self._remember_frame_id(metadata.frame_id)

        if (
            self._last_sdk_timestamp_ms is not None
            and metadata.captured_at_rokid_sdk_ms < self._last_sdk_timestamp_ms
        ):
            self.sdk_clock_discontinuities += 1
        self._last_sdk_timestamp_ms = metadata.captured_at_rokid_sdk_ms

        if self._metadata_origin_90khz is None:
            self._metadata_origin_90khz = metadata.rtp_timestamp_90khz
        key = (metadata.rtp_timestamp_90khz - self._metadata_origin_90khz) & 0xFFFFFFFF
        if key in self._metadata:
            self.duplicates += 1
            raise DuplicateMetadataError(f"duplicate RTP timestamp {key}")
        frame_key = self._nearest_key(self._frames, key)
        if frame_key is not None:
            self._frames.pop(frame_key)
            self._record_match_error(frame_key, key)
            self.matched += 1
            return True

        self._metadata[key] = metadata
        self._trim(self._metadata)
        return False

    def add_frame(self, pts: int | None) -> bool:
        if pts is None:
            return False
        key = pts & 0xFFFFFFFF
        metadata_key = self._nearest_key(self._metadata, key)
        if metadata_key is not None:
            self._metadata.pop(metadata_key)
            self._record_match_error(key, metadata_key)
            self.matched += 1
            return True

        self._frames[key] = pts
        self._frames.move_to_end(key)
        self._trim(self._frames)
        return False

    def _remember_frame_id(self, frame_id: int) -> None:
        self._recent_frame_ids[frame_id] = None
        self._recent_frame_ids.move_to_end(frame_id)
        while len(self._recent_frame_ids) > self._max_pending * 2:
            self._recent_frame_ids.popitem(last=False)

    def _trim(self, entries: OrderedDict[int, object]) -> None:
        while len(entries) > self._max_pending:
            entries.popitem(last=False)
            self.dropped += 1

    @staticmethod
    def _signed_distance(left: int, right: int) -> int:
        return ((left - right + RTP_MODULUS // 2) % RTP_MODULUS) - RTP_MODULUS // 2

    def _nearest_key(self, entries: OrderedDict[int, object], target: int) -> int | None:
        if not entries:
            return None
        nearest = min(entries, key=lambda key: abs(self._signed_distance(key, target)))
        if abs(self._signed_distance(nearest, target)) > MAX_TIMESTAMP_ERROR_90KHZ:
            return None
        return nearest

    def _record_match_error(self, frame_key: int, metadata_key: int) -> None:
        error = abs(self._signed_distance(frame_key, metadata_key))
        self.max_timestamp_match_error_90khz = max(
            self.max_timestamp_match_error_90khz,
            error,
        )
