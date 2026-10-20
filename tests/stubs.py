"""Minimal stand-ins for the Hermes runtime, so the plugin is testable alone.

The plugin imports ``providers`` / ``providers.base`` from the Hermes tree and
``agent.usage_pricing`` when pricing. None exist in a bare checkout, so tests
install stubs that mirror the real contracts.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

PLUGIN_INIT = Path(__file__).resolve().parent.parent / "requesty" / "__init__.py"


@dataclass
class StubProviderProfile:
    """Mirrors the fields of ``providers.base.ProviderProfile`` the plugin sets."""

    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    signup_url: str = ""
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""
    auth_type: str = "api_key"
    supports_vision: bool = False
    fallback_models: tuple = ()
    hostname: str = ""
    default_headers: dict = field(default_factory=dict)
    default_aux_model: str = ""


@dataclass
class LegacyStubProviderProfile:
    """An older ``ProviderProfile`` without ``supports_vision`` / ``hostname``."""

    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()
    display_name: str = ""
    description: str = ""
    signup_url: str = ""
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""
    auth_type: str = "api_key"
    fallback_models: tuple = ()
    default_aux_model: str = ""


@dataclass(frozen=True)
class StubCanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True)
class StubCostResult:
    amount_usd: Decimal | None
    status: str
    source: str
    label: str
    notes: tuple = ()


def load_plugin(*, legacy: bool = False):
    """Install stubs, import the plugin fresh, return (module, registered_profile)."""
    registry: dict[str, Any] = {}
    profile_cls = LegacyStubProviderProfile if legacy else StubProviderProfile

    base_mod = types.ModuleType("providers.base")
    base_mod.ProviderProfile = profile_cls
    providers_mod = types.ModuleType("providers")
    providers_mod.register_provider = lambda profile: registry.__setitem__(
        profile.name, profile
    )
    providers_mod.base = base_mod

    pricing_mod = types.ModuleType("agent.usage_pricing")
    pricing_mod.CostResult = StubCostResult
    agent_mod = types.ModuleType("agent")
    agent_mod.usage_pricing = pricing_mod

    sys.modules.update(
        {
            "providers": providers_mod,
            "providers.base": base_mod,
            "agent": agent_mod,
            "agent.usage_pricing": pricing_mod,
        }
    )

    spec = importlib.util.spec_from_file_location("requesty_under_test", PLUGIN_INIT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, registry["requesty"]
