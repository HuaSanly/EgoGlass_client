from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import OrderedDict, deque
from dataclasses import dataclass
from statistics import median
from typing import Literal

from .webrtc_models import VideoFrameMetadata

RTP_MODULUS = 1 << 32

# Glass3 runs at 30 fps, or roughly 3,000 ticks per frame. The device-backed
# residual is at most 622 ticks, so 1,000 covers measured clock quantization
# while remaining below the 1,500-tick half-frame ambiguity boundary.
MAX_TIMESTAMP_ERROR_90KHZ = 1_000
MAX_ORDERED_GAP_MATCH_ERROR_90KHZ = 6_000
CALIBRATION_CLUSTER_RADIUS_90KHZ = MAX_TIMESTAMP_ERROR_90KHZ
CALIBRATION_MIN_CANDIDATES = 60
CALIBRATION_MIN_CLUSTER_SIZE = 30
CALIBRATION_DOMINANCE_NUMERATOR = 3
CALIBRATION_DOMINANCE_DENOMINATOR = 2
CALIBRATION_MAX_RECEIPT_DELTA_NS = 500_000_000


class DuplicateMetadataError(ValueError):
    """Raised when a frame metadata message is replayed."""


@dataclass(frozen=True)
class DecodedFrameObservation:
    pts: int
    frame_index: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    received_at_client_monotonic_ns: int | None = None


@dataclass(frozen=True)
class MetadataObservation:
    metadata: VideoFrameMetadata
    received_at_client_monotonic_ns: int | None


@dataclass(frozen=True)
class FrameMetadataMatch:
    metadata: VideoFrameMetadata
    decoded_frame_pts: int
    decoded_frame_index: int | None
    decoded_frame_time_base_num: int | None
    decoded_frame_time_base_den: int | None
    decoded_frame_received_at_client_monotonic_ns: int | None
    timestamp_match_error_90khz: int
    match_method: Literal["timestamp_anchor", "ordered_gap_fill"] = "timestamp_anchor"


@dataclass(frozen=True)
class _OffsetCandidate:
    offset_90khz: int
    receipt_delta_ns: int


class FrameMetadataMatcher:
    """Associate receiver-relative decoded PTS with raw Glass3 RTP metadata."""

    def __init__(self, max_pending: int = 256) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._metadata: OrderedDict[int, MetadataObservation] = OrderedDict()
        self._frames: OrderedDict[int, DecodedFrameObservation] = OrderedDict()
        self._ready_matches: deque[FrameMetadataMatch] = deque()
        self._recent_frame_ids: OrderedDict[int, None] = OrderedDict()
        self._metadata_origin_90khz: int | None = None
        self._calibration_support = 0
        self._requires_legacy_origin = False
        self.matched = 0
        self.anchor_matches = 0
        self.ordered_gap_matches = 0
        self.duplicates = 0
        self.dropped = 0
        self.sdk_clock_discontinuities = 0
        self.max_timestamp_match_error_90khz = 0
        self._last_sdk_timestamp_ms: int | None = None
        self._last_sdk_frame_id: int | None = None
        self._last_matched_metadata_frame_id: int | None = None
        self._last_camera_start_generation: int | None = None

    @property
    def pending_metadata(self) -> int:
        return len(self._metadata)

    @property
    def pending_frames(self) -> int:
        return len(self._frames)

    @property
    def metadata_origin_90khz(self) -> int | None:
        """Raw metadata RTP timestamp corresponding to decoded PTS zero."""

        return self._metadata_origin_90khz

    @property
    def calibrated(self) -> bool:
        return self._metadata_origin_90khz is not None

    @property
    def calibration_support(self) -> int:
        return self._calibration_support

    def add_metadata(
        self,
        metadata: VideoFrameMetadata,
        *,
        received_at_client_monotonic_ns: int | None = None,
    ) -> FrameMetadataMatch | None:
        if self._last_camera_start_generation is not None:
            if metadata.camera_start_generation < self._last_camera_start_generation:
                self.dropped += 1
                return self._pop_ready_match()
            if metadata.camera_start_generation > self._last_camera_start_generation:
                self._reset_generation()
        self._last_camera_start_generation = metadata.camera_start_generation
        if metadata.frame_id in self._recent_frame_ids:
            self.duplicates += 1
            raise DuplicateMetadataError(f"duplicate frame_id {metadata.frame_id}")
        self._remember_frame_id(metadata.frame_id)

        if self._last_sdk_frame_id is None or metadata.frame_id > self._last_sdk_frame_id:
            if (
                self._last_sdk_timestamp_ms is not None
                and metadata.captured_at_rokid_sdk_ms < self._last_sdk_timestamp_ms
            ):
                self.sdk_clock_discontinuities += 1
            self._last_sdk_frame_id = metadata.frame_id
            self._last_sdk_timestamp_ms = metadata.captured_at_rokid_sdk_ms

        if (
            self._last_matched_metadata_frame_id is not None
            and metadata.frame_id <= self._last_matched_metadata_frame_id
        ):
            self.dropped += 1
            return self._pop_ready_match()

        key = metadata.rtp_timestamp_90khz
        if key in self._metadata:
            self.duplicates += 1
            raise DuplicateMetadataError(f"duplicate RTP timestamp {key}")
        if received_at_client_monotonic_ns is None:
            self._requires_legacy_origin = True

        observation = MetadataObservation(
            metadata=metadata,
            received_at_client_monotonic_ns=received_at_client_monotonic_ns,
        )
        if self.calibrated:
            self._match_or_store_metadata(key, observation)
        else:
            self._store_metadata(key, observation)
            self._trim(self._metadata)
            self._try_calibrate()
        return self._pop_ready_match()

    def add_frame(
        self,
        pts: int | None,
        *,
        frame_index: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        received_at_client_monotonic_ns: int | None = None,
    ) -> FrameMetadataMatch | None:
        if pts is None:
            return None
        key = pts & 0xFFFFFFFF
        if received_at_client_monotonic_ns is None:
            self._requires_legacy_origin = True
        observation = DecodedFrameObservation(
            pts=pts,
            frame_index=frame_index,
            time_base_num=time_base_num,
            time_base_den=time_base_den,
            received_at_client_monotonic_ns=received_at_client_monotonic_ns,
        )
        if self.calibrated:
            self._match_or_store_frame(key, observation)
        else:
            self._frames[key] = observation
            self._frames.move_to_end(key)
            self._trim(self._frames)
            self._try_calibrate()
        return self._pop_ready_match()

    def drain_matches(self) -> tuple[FrameMetadataMatch, ...]:
        matches = tuple(self._ready_matches)
        self._ready_matches.clear()
        return matches

    def _try_calibrate(self) -> None:
        if self.calibrated or not self._metadata or not self._frames:
            return
        if self._requires_legacy_origin:
            # Compatibility for deterministic callers without receipt clocks.
            first_metadata_key = next(iter(self._metadata))
            self._set_calibration(first_metadata_key, support=1)
            return

        candidates = self._receipt_time_candidates()
        if len(candidates) < CALIBRATION_MIN_CANDIDATES:
            return
        best_offset, best_members = self._densest_cluster(candidates)
        if best_offset is None or len(best_members) < CALIBRATION_MIN_CLUSTER_SIZE:
            return
        remaining = [
            candidate
            for index, candidate in enumerate(candidates)
            if index not in best_members
        ]
        _, runner_members = self._densest_cluster(remaining)
        runner_count = len(runner_members)
        if (
            runner_count > 0
            and len(best_members) * CALIBRATION_DOMINANCE_DENOMINATOR
            < runner_count * CALIBRATION_DOMINANCE_NUMERATOR
        ):
            return
        self._set_calibration(best_offset, support=len(best_members))

    def _receipt_time_candidates(self) -> list[_OffsetCandidate]:
        timed_metadata = [
            (key, observation.received_at_client_monotonic_ns)
            for key, observation in self._metadata.items()
            if observation.received_at_client_monotonic_ns is not None
        ]
        candidates: list[_OffsetCandidate] = []
        for frame_key, frame in self._frames.items():
            received_at_ns = frame.received_at_client_monotonic_ns
            if received_at_ns is None or not timed_metadata:
                continue
            metadata_key, metadata_received_at_ns = min(
                timed_metadata,
                key=lambda item: (abs(item[1] - received_at_ns), item[0]),
            )
            receipt_delta_ns = abs(metadata_received_at_ns - received_at_ns)
            if receipt_delta_ns > CALIBRATION_MAX_RECEIPT_DELTA_NS:
                continue
            candidates.append(
                _OffsetCandidate(
                    offset_90khz=(metadata_key - frame_key) & 0xFFFFFFFF,
                    receipt_delta_ns=receipt_delta_ns,
                )
            )
        return candidates

    @classmethod
    def _densest_cluster(
        cls,
        candidates: list[_OffsetCandidate],
    ) -> tuple[int | None, set[int]]:
        if not candidates:
            return None, set()
        ordered = sorted(
            enumerate(candidates),
            key=lambda item: (item[1].offset_90khz, item[0]),
        )
        expanded = [
            (candidate.offset_90khz - RTP_MODULUS, index, candidate)
            for index, candidate in ordered
        ]
        expanded.extend(
            (candidate.offset_90khz, index, candidate)
            for index, candidate in ordered
        )
        expanded.extend(
            (candidate.offset_90khz + RTP_MODULUS, index, candidate)
            for index, candidate in ordered
        )
        expanded.sort(key=lambda item: (item[0], item[1]))
        values = [item[0] for item in expanded]

        best_score: tuple[int, int, int, int] | None = None
        best_members: set[int] = set()
        best_offset: int | None = None
        for _candidate_index, candidate in ordered:
            center = candidate.offset_90khz
            left = bisect_left(values, center - CALIBRATION_CLUSTER_RADIUS_90KHZ)
            right = bisect_right(values, center + CALIBRATION_CLUSTER_RADIUS_90KHZ)
            window = expanded[left:right]
            members = {item[1] for item in window}
            deltas = sorted(candidates[index].receipt_delta_ns for index in members)
            score = (
                len(members),
                -int(median(deltas)),
                -deltas[-1],
                -center,
            )
            if best_score is None or score > best_score:
                offsets = sorted(
                    cls._signed_distance(candidates[index].offset_90khz, center)
                    for index in members
                )
                best_score = score
                best_members = members
                best_offset = (center + int(median(offsets))) & 0xFFFFFFFF
        return best_offset, best_members

    def _set_calibration(self, offset_90khz: int, *, support: int) -> None:
        self._metadata_origin_90khz = offset_90khz & 0xFFFFFFFF
        self._calibration_support = support
        self._match_all_pending()

    def _match_all_pending(self) -> None:
        self._reconcile_pending()

    def _match_or_store_metadata(
        self,
        key: int,
        observation: MetadataObservation,
    ) -> None:
        self._store_metadata(key, observation)
        self._reconcile_pending()
        self._trim(self._metadata)

    def _store_metadata(self, key: int, observation: MetadataObservation) -> None:
        self._metadata[key] = observation
        self._metadata = OrderedDict(
            sorted(
                self._metadata.items(),
                key=lambda item: item[1].metadata.frame_id,
            )
        )

    def _match_or_store_frame(
        self,
        key: int,
        observation: DecodedFrameObservation,
    ) -> None:
        self._frames[key] = observation
        self._frames.move_to_end(key)
        self._reconcile_pending()
        self._trim(self._frames)

    def _reconcile_pending(self) -> None:
        while self._frames and self._metadata:
            anchor = self._first_timestamp_anchor()
            if anchor is None:
                return
            frame_key, metadata_key = anchor
            frame_keys = list(self._frames)
            metadata_keys = list(self._metadata)
            frame_prefix = frame_keys[: frame_keys.index(frame_key)]
            metadata_prefix = metadata_keys[: metadata_keys.index(metadata_key)]
            self._match_ordered_gap(frame_prefix, metadata_prefix)

            frame = self._frames.pop(frame_key)
            metadata = self._metadata.pop(metadata_key)
            self._record_match(
                metadata,
                frame,
                frame_key,
                metadata_key,
                match_method="timestamp_anchor",
            )

    def _first_timestamp_anchor(self) -> tuple[int, int] | None:
        for frame_key in self._frames:
            metadata_key = self._nearest_metadata_key(frame_key)
            if metadata_key is not None:
                return frame_key, metadata_key
        return None

    def _match_ordered_gap(
        self,
        frame_keys: list[int],
        metadata_keys: list[int],
    ) -> None:
        if not frame_keys and not metadata_keys:
            return
        pairs = self._minimum_skip_alignment(frame_keys, metadata_keys)
        paired_frames = {frame_index for frame_index, _ in pairs}
        paired_metadata = {metadata_index for _, metadata_index in pairs}

        frames = [self._frames.pop(key) for key in frame_keys]
        metadata = [self._metadata.pop(key) for key in metadata_keys]
        for frame_index, metadata_index in pairs:
            self._record_match(
                metadata[metadata_index],
                frames[frame_index],
                frame_keys[frame_index],
                metadata_keys[metadata_index],
                match_method="ordered_gap_fill",
            )
        self.dropped += (
            len(frame_keys)
            - len(paired_frames)
            + len(metadata_keys)
            - len(paired_metadata)
        )

    def _minimum_skip_alignment(
        self,
        frame_keys: list[int],
        metadata_keys: list[int],
    ) -> tuple[tuple[int, int], ...]:
        frame_count = len(frame_keys)
        metadata_count = len(metadata_keys)
        # Lexicographic cost first maximizes one-to-one matches between two
        # trusted anchors, then chooses the lowest timestamp and receipt error.
        infinity = (frame_count + metadata_count + 1, 0, 0)
        costs = [
            [infinity for _ in range(metadata_count + 1)]
            for _ in range(frame_count + 1)
        ]
        operations = [
            [0 for _ in range(metadata_count + 1)]
            for _ in range(frame_count + 1)
        ]
        costs[0][0] = (0, 0, 0)

        for frame_index in range(frame_count + 1):
            for metadata_index in range(metadata_count + 1):
                current = costs[frame_index][metadata_index]
                if current == infinity:
                    continue
                if frame_index < frame_count:
                    candidate = (current[0] + 1, current[1], current[2])
                    if candidate < costs[frame_index + 1][metadata_index]:
                        costs[frame_index + 1][metadata_index] = candidate
                        operations[frame_index + 1][metadata_index] = 1
                if metadata_index < metadata_count:
                    candidate = (current[0] + 1, current[1], current[2])
                    if candidate < costs[frame_index][metadata_index + 1]:
                        costs[frame_index][metadata_index + 1] = candidate
                        operations[frame_index][metadata_index + 1] = 2
                if frame_index < frame_count and metadata_index < metadata_count:
                    timestamp_error = self._timestamp_error(
                        frame_keys[frame_index],
                        metadata_keys[metadata_index],
                    )
                    if timestamp_error > MAX_ORDERED_GAP_MATCH_ERROR_90KHZ:
                        continue
                    receipt_error = self._receipt_error(
                        self._frames[frame_keys[frame_index]],
                        self._metadata[metadata_keys[metadata_index]],
                    )
                    candidate = (
                        current[0],
                        current[1] + timestamp_error,
                        current[2] + receipt_error,
                    )
                    if candidate < costs[frame_index + 1][metadata_index + 1]:
                        costs[frame_index + 1][metadata_index + 1] = candidate
                        operations[frame_index + 1][metadata_index + 1] = 3

        pairs: list[tuple[int, int]] = []
        frame_index = frame_count
        metadata_index = metadata_count
        while frame_index or metadata_index:
            operation = operations[frame_index][metadata_index]
            if operation == 3:
                frame_index -= 1
                metadata_index -= 1
                pairs.append((frame_index, metadata_index))
            elif operation == 1:
                frame_index -= 1
            elif operation == 2:
                metadata_index -= 1
            else:
                raise RuntimeError("ordered frame metadata alignment failed")
        pairs.reverse()
        return tuple(pairs)

    def _timestamp_error(self, frame_key: int, metadata_key: int) -> int:
        if self._metadata_origin_90khz is None:
            raise RuntimeError("frame metadata matcher is not calibrated")
        predicted = (frame_key + self._metadata_origin_90khz) & 0xFFFFFFFF
        return abs(self._signed_distance(predicted, metadata_key))

    @staticmethod
    def _receipt_error(
        frame: DecodedFrameObservation,
        metadata: MetadataObservation,
    ) -> int:
        frame_received = frame.received_at_client_monotonic_ns
        metadata_received = metadata.received_at_client_monotonic_ns
        if frame_received is None or metadata_received is None:
            return 0
        return abs(frame_received - metadata_received)

    def _nearest_metadata_key(self, frame_key: int) -> int | None:
        if self._metadata_origin_90khz is None:
            return None
        target = (frame_key + self._metadata_origin_90khz) & 0xFFFFFFFF
        return self._nearest_key(self._metadata, target)

    def _nearest_frame_key(self, metadata_key: int) -> int | None:
        if self._metadata_origin_90khz is None:
            return None
        target = (metadata_key - self._metadata_origin_90khz) & 0xFFFFFFFF
        return self._nearest_key(self._frames, target)

    def _record_match(
        self,
        metadata: MetadataObservation,
        frame: DecodedFrameObservation,
        frame_key: int,
        metadata_key: int,
        *,
        match_method: Literal["timestamp_anchor", "ordered_gap_fill"],
    ) -> None:
        if self._metadata_origin_90khz is None:
            raise RuntimeError("frame metadata matcher is not calibrated")
        predicted_metadata_key = (frame_key + self._metadata_origin_90khz) & 0xFFFFFFFF
        error = self._record_match_error(predicted_metadata_key, metadata_key)
        self.matched += 1
        self._last_matched_metadata_frame_id = metadata.metadata.frame_id
        if match_method == "timestamp_anchor":
            self.anchor_matches += 1
        else:
            self.ordered_gap_matches += 1
        self._ready_matches.append(
            self._match(metadata.metadata, frame, error, match_method)
        )

    def _pop_ready_match(self) -> FrameMetadataMatch | None:
        return self._ready_matches.popleft() if self._ready_matches else None

    def _reset_generation(self) -> None:
        self.dropped += len(self._metadata) + len(self._frames) + len(self._ready_matches)
        self._metadata.clear()
        self._frames.clear()
        self._ready_matches.clear()
        self._recent_frame_ids.clear()
        self._metadata_origin_90khz = None
        self._calibration_support = 0
        self._requires_legacy_origin = False
        self._last_sdk_timestamp_ms = None
        self._last_sdk_frame_id = None
        self._last_matched_metadata_frame_id = None

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

    def _record_match_error(self, frame_key: int, metadata_key: int) -> int:
        error = abs(self._signed_distance(frame_key, metadata_key))
        self.max_timestamp_match_error_90khz = max(
            self.max_timestamp_match_error_90khz,
            error,
        )
        return error

    @staticmethod
    def _match(
        metadata: VideoFrameMetadata,
        frame: DecodedFrameObservation,
        error: int,
        match_method: Literal["timestamp_anchor", "ordered_gap_fill"],
    ) -> FrameMetadataMatch:
        return FrameMetadataMatch(
            metadata=metadata,
            decoded_frame_pts=frame.pts,
            decoded_frame_index=frame.frame_index,
            decoded_frame_time_base_num=frame.time_base_num,
            decoded_frame_time_base_den=frame.time_base_den,
            decoded_frame_received_at_client_monotonic_ns=(
                frame.received_at_client_monotonic_ns
            ),
            timestamp_match_error_90khz=error,
            match_method=match_method,
        )
