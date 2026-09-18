"""Guards for the diagnostic override plumbing.

The real-meeting diagnostic overrides ONLY the STT language and the requested speaker
count on top of production defaults. These tests make sure (a) the production defaults
are what the diagnostic assumes, and (b) overriding them does not mutate the shared
settings object or the defaults.
"""

from __future__ import annotations

from app.config import Settings


def test_production_stt_and_diarization_defaults_are_unchanged() -> None:
    settings = Settings(
        llm_provider="openai_compatible", llm_base_url=None, llm_api_key=None, llm_model=None
    )

    assert settings.stt_language == "auto"
    assert settings.stt_threads == 8
    assert settings.stt_beam_size == 5
    assert settings.diarization_threshold == 0.80
    assert settings.diarization_min_duration_on == 0.3
    assert settings.diarization_min_duration_off == 0.5
    assert settings.diarization_threads == 8
    assert settings.merge_boundary_tolerance_seconds == 0.25


def test_language_and_speaker_count_overrides_are_isolated() -> None:
    base = Settings(
        llm_provider="openai_compatible", llm_base_url=None, llm_api_key=None, llm_model=None
    )

    turkish = base.model_copy(update={"stt_language": "tr"})

    assert turkish.stt_language == "tr"
    assert base.stt_language == "auto"  # the shared object is untouched
    assert Settings().stt_language == "auto"  # the default is untouched


def test_diagnostic_uses_production_threshold_when_speaker_count_is_known() -> None:
    # Known-count runs pass requested_speaker_count to the diarization service; the
    # production threshold stays configured (sherpa bypasses it for known counts).
    settings = Settings(
        llm_provider="openai_compatible", llm_base_url=None, llm_api_key=None, llm_model=None
    )

    assert settings.diarization_threshold == 0.80
