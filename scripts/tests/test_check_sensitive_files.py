"""Unit tests for the pre-commit safety script (standard library only).

Run with:  python3 scripts/tests/test_check_sensitive_files.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_sensitive_files import (  # noqa: E402
    content_violation,
    path_violation,
    scan,
)


class PathRuleTests(unittest.TestCase):
    def test_rejects_plain_env_files(self) -> None:
        self.assertEqual(path_violation(".env"), "env secret file")
        self.assertEqual(path_violation("benchmarks/elevenlabs-stt/.env.local"), "env secret file")
        self.assertEqual(path_violation("backend/.env.production"), "env secret file")

    def test_allows_env_examples(self) -> None:
        self.assertIsNone(path_violation(".env.example"))
        self.assertIsNone(path_violation("frontend/.env.local.example"))

    def test_rejects_audio(self) -> None:
        for name in ("rec.wav", "rec.mp3", "rec.webm", "rec.m4a"):
            self.assertEqual(path_violation(f"data/meetings/{name}"), "audio file")

    def test_rejects_runtime_databases(self) -> None:
        self.assertEqual(path_violation("data/app.db"), "runtime database")
        self.assertEqual(path_violation("data/app.db-wal"), "runtime database")
        self.assertEqual(path_violation("data/app.db-shm"), "runtime database")

    def test_rejects_model_binaries(self) -> None:
        for name in ("m.onnx", "m.bin", "m.gguf", "m.pt", "m.pth", "m.safetensors"):
            self.assertEqual(path_violation(f"models/{name}"), "model binary")

    def test_rejects_private_and_raw_directories(self) -> None:
        self.assertEqual(
            path_violation("benchmarks/elevenlabs-stt/results/private/real.txt"),
            "private artifact under private/",
        )
        self.assertEqual(
            path_violation("benchmarks/elevenlabs-stt/results/raw/real.json"),
            "private artifact under raw/",
        )
        self.assertEqual(
            path_violation("benchmarks/stt/results/matrix.json"),
            "benchmark output under results/",
        )

    def test_allows_whitelisted_safe_aggregates(self) -> None:
        self.assertIsNone(path_violation("benchmarks/stt/results/model-metadata.json"))
        self.assertIsNone(path_violation("benchmarks/stt/results/model-metadata-turbo.json"))
        self.assertIsNone(path_violation("benchmarks/diarization/results/model-metadata.json"))
        self.assertIsNone(path_violation("benchmarks/stt/results/.gitkeep"))

    def test_allows_ordinary_sources(self) -> None:
        self.assertIsNone(path_violation("backend/app/services/providers/elevenlabs.py"))
        self.assertIsNone(path_violation("frontend/src/lib/api.ts"))


class ContentRuleTests(unittest.TestCase):
    def test_rejects_fake_api_key_assignment(self) -> None:
        fake = "sk_" + "a" * 30
        reason = content_violation("config.env.local", f"ELEVENLABS_API_KEY={fake}\n")
        self.assertIsNotNone(reason)

    def test_rejects_fake_bearer_token(self) -> None:
        fake = "sk-" + "b" * 30
        reason = content_violation("notes.txt", f"Authorization: Bearer {fake}\n")
        self.assertIsNotNone(reason)

    def test_rejects_fake_xi_api_key_header(self) -> None:
        fake = "c" * 32
        reason = content_violation("notes.txt", f"xi-api-key: {fake}\n")
        self.assertIsNotNone(reason)

    def test_allows_printf_recipes_in_documentation(self) -> None:
        recipe = "read -s -p 'key: ' k && printf 'ELEVENLABS_API_KEY=%s\\n' \"$k\" > .env.local\n"
        self.assertIsNone(content_violation("benchmarks/elevenlabs-stt/README.md", recipe))

    def test_allows_placeholders(self) -> None:
        self.assertIsNone(content_violation("README.md", "ELEVENLABS_API_KEY=..."))
        # A trailing source escape next to a placeholder is still a placeholder.
        backslash = chr(92)
        key_name = "ELEVENLABS_API_KEY"
        self.assertIsNone(content_violation("docs.md", f"{key_name}=...{backslash}n"))
        self.assertIsNone(
            content_violation("compose.yaml", "ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY:-}\n")
        )
        self.assertIsNone(content_violation("README.md", "xi-api-key: <ELEVENLABS_API_KEY>\n"))

    def test_allows_source_containing_field_names(self) -> None:
        source = (
            'headers = {"xi-api-key": api_key}\n'
            'auth = {"Authorization": f"Bearer {self._api_key}"}\n'
            "alias = \"ELEVENLABS_API_KEY\"\n"
            'monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test_key_value")\n'
        )
        self.assertIsNone(content_violation("backend/tests/test_x.py", source))

    def test_allows_benchmark_harness_header_construction(self) -> None:
        harness = 'stdin_config = f\'header = "xi-api-key: {key}"\\n\'\n'
        self.assertIsNone(content_violation("benchmarks/elevenlabs-stt/run_benchmark.py", harness))


class ScanTests(unittest.TestCase):
    def test_scan_reports_path_and_reason_only(self) -> None:
        fake = "sk_" + "d" * 30
        violations = scan(
            [
                ("ok.py", "print('hello')"),
                ("data/app.db", None),
                ("notes.txt", f"ELEVENLABS_API_KEY={fake}"),
            ]
        )
        paths = [path for path, _ in violations]
        self.assertIn("data/app.db", paths)
        self.assertIn("notes.txt", paths)
        self.assertNotIn("ok.py", paths)
        rendered = "\n".join(reason for _, reason in violations)
        self.assertNotIn(fake, rendered)

    def test_scan_accepts_the_whitelisted_aggregate_content(self) -> None:
        payload = '{"quantized_sha256": "' + "a" * 64 + '"}'
        self.assertEqual(scan([("benchmarks/stt/results/model-metadata.json", payload)]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
