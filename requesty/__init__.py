"""Requesty provider profile for Hermes Agent.

Requesty (https://requesty.ai) is an OpenAI-compatible LLM gateway. This
profile wires Hermes to it with no core changes:

  - **Model picker** — Requesty's managed policies (``GET /v1/models/managed``):
    curated routes with failover, used as bare ids (``claude-sonnet-5-5``,
    ``gpt-6.1-sol@eu``). Any id from ``GET /v1/models`` (``anthropic/...``) or
    an org policy (``policy/...``) still works when typed explicitly.
  - **Reasoning** — sent as top-level ``reasoning_effort``, Requesty's
    documented control, and only to models the catalog marks as reasoning.
  - **Prompt caching** — handled server-side by Requesty; nothing to do here.
  - **Tracing** — the Hermes session id becomes ``requesty.trace_id`` so a
    conversation groups together in the Requesty dashboard.

Both catalogs are public and identical for every caller, so they are fetched
without credentials and cached once per process. Hooks on the request hot path
read the cache only; a cold cache warms in the background.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import urllib.request
from decimal import Decimal
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

PROVIDER_ID = "requesty"
API_KEY_ENV = "REQUESTY_API_KEY"
BASE_URL_ENV = "REQUESTY_BASE_URL"
BASE_URL = "https://router.requesty.ai/v1"

# Hermes' effort ladder is none < minimal < low < medium < high < xhigh < max < ultra.
# Requesty accepts none/min/low/medium/high/max; xhigh only on some OpenAI models.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "max")
_WIRE_EFFORTS = {"minimal": "min", "xhigh": "high", "ultra": "max"}


def _get_json(url: str, timeout: float) -> list[dict[str, Any]] | None:
    """``data`` array of a public Requesty list endpoint, or None on any failure."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", f"hermes-provider-requesty/{__version__}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode()).get("data")
    except Exception as exc:  # noqa: BLE001 - a catalog failure must never break Hermes
        logger.debug("requesty: fetching %s failed: %s", url, exc)
        return None
    if not isinstance(data, list):
        return None
    return [e for e in data if isinstance(e, dict) and isinstance(e.get("id"), str)]


class _Catalog:
    """Process-wide cache of Requesty's managed policies and model metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._warm_started = False
        self.policies: tuple[str, ...] = ()
        self.entries: dict[str, dict[str, Any]] = {}

    def refresh(self, base_url: str, timeout: float = 8.0) -> bool:
        """Fetch both catalogs synchronously. False when the managed list is unavailable."""
        base = base_url.rstrip("/")
        managed = _get_json(f"{base}/models/managed", timeout)
        if not managed:
            return False
        models = _get_json(f"{base}/models", timeout) or []

        policies = tuple(
            dict.fromkeys(
                e["id"]
                for e in managed
                if e.get("supports_tool_calling") and e.get("api", "chat") == "chat"
            )
        )
        # Swap whole objects: readers on other threads never see a partial cache.
        self.entries = {e["id"]: e for e in (*models, *managed)}
        self.policies = policies
        return True

    def entry(self, model: str | None) -> dict[str, Any] | None:
        """Cached metadata for *model*. Never blocks; a cold cache warms in the background."""
        if not self.entries:
            self._warm()
            return None
        return self.entries.get(model or "")

    def _warm(self) -> None:
        with self._lock:
            if self._warm_started:
                return
            self._warm_started = True
        threading.Thread(
            target=self.refresh,
            args=(BASE_URL,),
            name="requesty-catalog-warm",
            daemon=True,
        ).start()


_CATALOG = _Catalog()


def _supports_reasoning(model: str | None) -> bool:
    entry = _CATALOG.entry(model)
    return bool(entry and entry.get("supports_reasoning"))


def _wire_effort(reasoning_config: dict | None) -> str | None:
    """Map a Hermes reasoning config onto Requesty's ``reasoning_effort`` values."""
    if not reasoning_config:
        return None
    if reasoning_config.get("enabled", True) is False:
        return "none"
    effort = str(reasoning_config.get("effort") or "medium").strip().lower()
    return _WIRE_EFFORTS.get(effort, effort)


def _price(source: dict[str, Any], key: str, default: Decimal) -> Decimal:
    raw = source.get(key)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
        return Decimal(str(raw))
    return default


def _pricing_band(entry: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    """The tiered pricing band in effect once the prompt exceeds its threshold."""
    band: dict[str, Any] = entry
    for candidate in entry.get("pricing") or ():
        if isinstance(candidate, dict) and prompt_tokens > candidate.get(
            "prompt_tokens_threshold", 0
        ):
            band = candidate
    return band


def _supported_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop profile fields an older Hermes build does not define.

    An unknown dataclass field raises TypeError at import, which the plugin
    loader swallows: the provider would silently fail to register.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kwargs.items() if k in known}


class RequestyProfile(ProviderProfile):
    """Requesty gateway: managed-policy picker, catalog-gated reasoning, tracing."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Managed policies that support tool calling; None falls back to ``fallback_models``."""
        if not _CATALOG.policies and not _CATALOG.refresh(
            base_url or self.base_url, timeout
        ):
            return None
        return list(_CATALOG.policies) or None

    def default_reasoning_config(self, model: str | None = None) -> dict | None:
        if _supports_reasoning(model):
            return {"enabled": True, "effort": "medium"}
        return None

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        entry = _CATALOG.entry(model)
        if entry is None:
            return None
        return REASONING_EFFORTS if entry.get("supports_reasoning") else ()

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = _wire_effort(reasoning_config) if _supports_reasoning(model) else None
        return {}, ({"reasoning_effort": effort} if effort else {})

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        return {"requesty": {"trace_id": session_id}} if session_id else {}

    def get_model_context_length(self, model: str) -> int | None:
        entry = _CATALOG.entry(model) or {}
        window = entry.get("context_window")
        return window if isinstance(window, int) and window > 0 else None

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """Estimate cost from the cached catalog's per-token prices, tiered bands included."""
        entry = _CATALOG.entry(model)
        if entry is None:
            return None
        try:
            from agent.usage_pricing import CostResult

            uncached = int(usage.input_tokens)
            cache_read = int(usage.cache_read_tokens)
            cache_write = int(usage.cache_write_tokens)
            output = int(usage.output_tokens)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            logger.debug("requesty: cannot price usage for %s: %s", model, exc)
            return None

        band = _pricing_band(entry, uncached + cache_read + cache_write)
        input_price = _price(band, "input_price", Decimal(0))
        amount = (
            uncached * input_price
            + cache_read * _price(band, "cached_price", input_price)
            + cache_write * _price(band, "caching_price", input_price)
            + output * _price(band, "output_price", Decimal(0))
        )
        return CostResult(
            amount_usd=amount,
            status="estimated",
            source="provider_models_api",
            label=f"~${amount:.4f}",
            notes=(
                "Requesty catalog price; see the Requesty dashboard for billed cost.",
            ),
        )


PROFILE_FIELDS: dict[str, Any] = {
    "name": PROVIDER_ID,
    "aliases": ("rq",),
    "display_name": "Requesty",
    "description": "Requesty — LLM gateway with managed routing policies (OpenAI-compatible)",
    "signup_url": "https://app.requesty.ai/api-keys",
    "env_vars": (API_KEY_ENV, BASE_URL_ENV),
    "base_url": BASE_URL,
    "models_url": f"{BASE_URL}/models/managed",
    "hostname": "router.requesty.ai",
    "auth_type": "api_key",
    "api_mode": "chat_completions",
    "supports_vision": True,
    "default_aux_model": "gemini-3.8-flash",
    # Shown only when the managed-policy list is unreachable. All verified live.
    "fallback_models": (
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "gpt-6.1-sol",
        "gemini-3.8-flash",
        "grok-4.7",
        "kimi-k3",
        "glm-5.3",
    ),
}

requesty = RequestyProfile(**_supported_kwargs(RequestyProfile, PROFILE_FIELDS))

register_provider(requesty)
