from aiortc import RTCBundlePolicy

from egoglass_ingest_gateway.adapters.aiortc_peer import (
    h264_video_codecs,
    lan_rtc_configuration,
    negotiated_video_codec_from_sdp,
)


def test_lan_configuration_disables_default_public_stun_and_maximizes_bundling() -> None:
    configuration = lan_rtc_configuration()

    assert configuration.iceServers == []
    assert configuration.bundlePolicy is RTCBundlePolicy.MAX_BUNDLE


def test_negotiated_video_codec_is_parsed_from_structured_sdp() -> None:
    sdp = "\r\n".join(
        (
            "v=0",
            "o=- 1 1 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "m=video 9 UDP/TLS/RTP/SAVPF 102",
            "c=IN IP4 0.0.0.0",
            "a=rtpmap:102 H264/90000",
            "",
        )
    )

    assert negotiated_video_codec_from_sdp(sdp) == "H264"


def test_viewer_forwarding_uses_only_h264_video_codecs() -> None:
    codecs = h264_video_codecs()

    assert codecs
    assert {codec.mimeType.casefold() for codec in codecs} == {"video/h264"}
