"""Native API contract checks with a simulated HTTP response, no network."""
import io
import json
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client import Client, ModelConfig


class NativeContract(unittest.TestCase):
    def test_native_payload_and_parser(self):
        captured = []
        def fake(request, **kwargs):
            captured.append(request)
            return io.BytesIO(json.dumps({"model": "claude-sonnet-4-5-20250929",
                "stop_reason": "tool_use", "content": [{"type": "tool_use", "name": "choose", "input": {"patient": 2}}]}).encode())
        cfg = ModelConfig("https://api.anthropic.com/v1", "test-key", "claude-sonnet-4-5-20250929", provider="anthropic")
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake):
            cli = Client(cfg)
            self.assertEqual(cli.choose("two patients"), 2)
        request = captured[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "choose"})
        self.assertNotIn("system", payload)
        self.assertNotIn("thinking", payload)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "two patients"}])
        self.assertIn("input_schema", payload["tools"][0])
        self.assertEqual(cli.provenance()[0]["model"], cfg.model)


if __name__ == "__main__":
    unittest.main()
