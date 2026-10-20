"""Offline tests for the Requesty provider profile."""

from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stubs import StubCanonicalUsage, load_plugin  # noqa: E402

MANAGED = [
    {
        "id": "claude-sonnet-5-5",
        "api": "chat",
        "supports_tool_calling": True,
        "supports_reasoning": True,
        "context_window": 1_000_000,
        "input_price": 3e-06,
        "cached_price": 3e-07,
        "caching_price": 3.75e-06,
        "output_price": 1.5e-05,
    },
    {
        "id": "claude-sonnet-5-5@eu",
        "api": "chat",
        "supports_tool_calling": True,
        "supports_reasoning": True,
    },
    {
        "id": "gpt-4o-mini@eu",
        "api": "chat",
        "supports_tool_calling": True,
        "supports_reasoning": False,
        "context_window": 128_000,
    },
    {
        "id": "laguna-m.1",
        "api": "chat",
        "supports_tool_calling": False,
        "supports_reasoning": True,
    },
    {
        "id": "tiered",
        "api": "chat",
        "supports_tool_calling": True,
        "input_price": 1e-06,
        "output_price": 2e-06,
        "pricing": [
            {"prompt_tokens_threshold": 0, "input_price": 1e-06, "output_price": 2e-06},
            {
                "prompt_tokens_threshold": 1000,
                "input_price": 2e-06,
                "output_price": 4e-06,
            },
        ],
    },
]
MODELS = [
    {
        "id": "anthropic/claude-sonnet-5-5",
        "api": "chat",
        "supports_tool_calling": True,
        "supports_reasoning": True,
    },
]


def fake_get_json(url: str, timeout: float):
    if url.endswith("/models/managed"):
        return list(MANAGED)
    if url.endswith("/models"):
        return list(MODELS)
    raise AssertionError(url)


class RequestyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.mod, self.profile = load_plugin()

    def load_catalog(self) -> None:
        with mock.patch.object(self.mod, "_get_json", side_effect=fake_get_json):
            self.assertTrue(self.mod._CATALOG.refresh(self.mod.BASE_URL))


class RegistrationTests(RequestyTestCase):
    def test_profile_fields(self) -> None:
        self.assertEqual(self.profile.name, "requesty")
        self.assertEqual(self.profile.aliases, ("rq",))
        self.assertEqual(
            self.profile.env_vars, ("REQUESTY_API_KEY", "REQUESTY_BASE_URL")
        )
        self.assertEqual(self.profile.base_url, "https://router.requesty.ai/v1")
        self.assertTrue(self.profile.supports_vision)
        self.assertIn(
            self.profile.default_aux_model,
            self.profile.fallback_models + ("gemini-3.8-flash",),
        )

    def test_no_default_headers(self) -> None:
        self.assertEqual(self.profile.default_headers, {})

    def test_registers_on_legacy_profile(self) -> None:
        _, profile = load_plugin(legacy=True)
        self.assertEqual(profile.name, "requesty")
        self.assertFalse(hasattr(profile, "hostname"))


class FetchModelsTests(RequestyTestCase):
    def test_lists_tool_capable_managed_policies(self) -> None:
        with mock.patch.object(self.mod, "_get_json", side_effect=fake_get_json):
            models = self.profile.fetch_models()
        self.assertEqual(
            models,
            ["claude-sonnet-5-5", "claude-sonnet-5-5@eu", "gpt-4o-mini@eu", "tiered"],
        )

    def test_uses_caller_base_url(self) -> None:
        with mock.patch.object(self.mod, "_get_json", side_effect=fake_get_json) as get:
            self.profile.fetch_models(base_url="https://router.eu.requesty.ai/v1/")
        get.assert_any_call("https://router.eu.requesty.ai/v1/models/managed", 8.0)

    def test_failure_returns_none(self) -> None:
        with mock.patch.object(self.mod, "_get_json", return_value=None):
            self.assertIsNone(self.profile.fetch_models())

    def test_cached_after_first_fetch(self) -> None:
        self.load_catalog()
        with mock.patch.object(self.mod, "_get_json") as get:
            self.assertTrue(self.profile.fetch_models())
        get.assert_not_called()


class ReasoningTests(RequestyTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.load_catalog()

    def extras(self, model: str, config: dict | None):
        return self.profile.build_api_kwargs_extras(
            reasoning_config=config, model=model, supports_reasoning=False
        )

    def test_effort_sent_top_level_for_reasoning_model(self) -> None:
        self.assertEqual(
            self.extras("claude-sonnet-5-5", {"effort": "high"}),
            ({}, {"reasoning_effort": "high"}),
        )
        self.assertEqual(
            self.extras("anthropic/claude-sonnet-5-5", {"effort": "low"}),
            ({}, {"reasoning_effort": "low"}),
        )

    def test_effort_mapped_to_requesty_vocabulary(self) -> None:
        cases = {
            "minimal": "min",
            "xhigh": "high",
            "ultra": "max",
            "max": "max",
            "none": "none",
        }
        for hermes, wire in cases.items():
            with self.subTest(hermes):
                self.assertEqual(
                    self.extras("claude-sonnet-5-5", {"effort": hermes})[1],
                    {"reasoning_effort": wire},
                )

    def test_disabled_sends_none(self) -> None:
        self.assertEqual(
            self.extras("claude-sonnet-5-5", {"enabled": False})[1],
            {"reasoning_effort": "none"},
        )

    def test_nothing_sent_without_config_or_support(self) -> None:
        self.assertEqual(self.extras("claude-sonnet-5-5", None), ({}, {}))
        self.assertEqual(self.extras("gpt-4o-mini@eu", {"effort": "high"}), ({}, {}))
        self.assertEqual(self.extras("policy/unknown", {"effort": "high"}), ({}, {}))

    def test_default_config_only_for_reasoning_models(self) -> None:
        self.assertEqual(
            self.profile.default_reasoning_config("claude-sonnet-5-5"),
            {"enabled": True, "effort": "medium"},
        )
        self.assertIsNone(self.profile.default_reasoning_config("gpt-4o-mini@eu"))

    def test_supported_efforts_tri_state(self) -> None:
        self.assertEqual(
            self.profile.supported_reasoning_efforts("claude-sonnet-5-5"),
            self.mod.REASONING_EFFORTS,
        )
        self.assertEqual(self.profile.supported_reasoning_efforts("gpt-4o-mini@eu"), ())
        self.assertIsNone(self.profile.supported_reasoning_efforts("policy/unknown"))


class ColdCacheTests(RequestyTestCase):
    def test_hot_path_never_blocks_and_warms_once(self) -> None:
        with mock.patch.object(self.mod.threading, "Thread") as thread:
            self.assertEqual(
                self.profile.build_api_kwargs_extras(
                    reasoning_config={"effort": "high"}, model="claude-sonnet-5-5"
                ),
                ({}, {}),
            )
            self.assertIsNone(
                self.profile.get_model_context_length("claude-sonnet-5-5")
            )
        thread.assert_called_once()
        thread.return_value.start.assert_called_once()


class RequestShapingTests(RequestyTestCase):
    def test_session_id_becomes_trace_id(self) -> None:
        self.assertEqual(
            self.profile.build_extra_body(session_id="s-1"),
            {"requesty": {"trace_id": "s-1"}},
        )
        self.assertEqual(self.profile.build_extra_body(session_id=None), {})

    def test_context_length_from_catalog(self) -> None:
        self.load_catalog()
        self.assertEqual(
            self.profile.get_model_context_length("claude-sonnet-5-5"), 1_000_000
        )
        self.assertIsNone(self.profile.get_model_context_length("claude-sonnet-5-5@eu"))


class CostTests(RequestyTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.load_catalog()

    def test_prices_cache_reads_and_writes(self) -> None:
        usage = StubCanonicalUsage(
            input_tokens=1000,
            cache_read_tokens=2000,
            cache_write_tokens=100,
            output_tokens=500,
        )
        cost = self.profile.get_usage_cost("claude-sonnet-5-5", usage)
        expected = (
            1000 * Decimal("3e-06")
            + 2000 * Decimal("3e-07")
            + 100 * Decimal("3.75e-06")
            + 500 * Decimal("1.5e-05")
        )
        self.assertEqual(cost.amount_usd, expected)
        self.assertEqual(cost.status, "estimated")

    def test_tiered_band_applies_above_threshold(self) -> None:
        small = self.profile.get_usage_cost(
            "tiered", StubCanonicalUsage(input_tokens=1000, output_tokens=10)
        )
        large = self.profile.get_usage_cost(
            "tiered", StubCanonicalUsage(input_tokens=1001, output_tokens=10)
        )
        self.assertEqual(
            small.amount_usd, 1000 * Decimal("1e-06") + 10 * Decimal("2e-06")
        )
        self.assertEqual(
            large.amount_usd, 1001 * Decimal("2e-06") + 10 * Decimal("4e-06")
        )

    def test_unknown_model_falls_through(self) -> None:
        self.assertIsNone(
            self.profile.get_usage_cost("policy/unknown", StubCanonicalUsage())
        )


if __name__ == "__main__":
    unittest.main()
