"""Read-time resolution of an ``auxiliary.<task>`` config block.

Built-in slot defaults come from ``DEFAULT_CONFIG``; plugin-registered tasks
(``PluginContext.register_auxiliary_task``) layer their declared ``defaults``
under the user's block and may inherit another slot via ``inherit_from``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from hermes_constants import get_default_hermes_root, get_hermes_home

logger = logging.getLogger(__name__)


def _get_auxiliary_task_config(task: str, _seen: frozenset = frozenset()) -> Dict[str, Any]:
    """Config dict for auxiliary.<task>, or {} when unavailable. Plugin-registered tasks get their
    declared defaults layered under user config (user wins); built-in defaults live in DEFAULT_CONFIG.
    A task registered with ``inherit_from`` is resolved here, at read time, over the base task's
    effective config, so it follows the base's current settings (and the active profile's config)
    until the user pins a route on the task itself. Under a profile process the profile's
    ``auxiliary.<task>`` block is layered over the GLOBAL config's (see
    ``_profile_auxiliary_override``). ``_seen`` guards re-registration cycles."""
    if not task:
        return {}
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except ImportError:
        return {}
    aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
    task_config = aux.get(task, {}) if isinstance(aux, dict) else {}
    if not isinstance(task_config, dict):
        task_config = {}
    task_config = _profile_auxiliary_override(task, task_config)
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        for _entry in get_plugin_auxiliary_tasks():
            if _entry.get("key") == task:
                _defaults = _entry.get("defaults") or {}
                if isinstance(_defaults, dict):
                    _inherit = _entry.get("inherit_from")
                    if _inherit and task in _seen:
                        logger.warning("Auxiliary task %r has a circular inherit_from chain — "
                                       "ignoring inheritance", task)
                    if not _inherit or task in _seen:
                        return {**_defaults, **task_config}
                    base = _get_auxiliary_task_config(_inherit, _seen | {task})
                    return _layer_over_inherited({**base, **_defaults}, task_config)
                break
    except Exception:  # health: allow BLE001 -- plugin discovery must never break aux config reads
        logger.debug("plugin auxiliary task lookup failed for %r", task, exc_info=True)
    return task_config


def _profile_auxiliary_override(task: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
    """Layer the ACTIVE PROFILE's ``auxiliary.<task>`` block over the GLOBAL config's.

    A profile process reads its own ``~/.hermes/profiles/<name>/config.yaml`` (``HERMES_HOME``
    points there), but profile setups scaffold every ``auxiliary`` task with ``provider: auto``
    / ``model: ''`` placeholders, which would shadow the global route (compaction would follow
    the profile primary instead of the configured compaction model). This resolves the GLOBAL
    ``auxiliary.<task>`` and applies the profile block with the same layering semantics as
    ``_layer_over_inherited``: placeholders mean "follow the base", a pinned route comes wholly
    from the profile block, and non-route keys override per key. The global read uses a
    context-local ``HERMES_HOME`` override so no ``os.environ`` mutation leaks to concurrent
    readers; anything that fails leaves *task_config* untouched."""
    profile_home = get_hermes_home()
    try:
        default_root = get_default_hermes_root()
        if profile_home.resolve() == default_root.resolve():
            return task_config  # default profile: HERMES_HOME already IS the global home
    except Exception:
        return task_config

    try:
        from contextlib import contextmanager

        @contextmanager
        def _global_home():
            from hermes_constants import set_hermes_home_override, reset_hermes_home_override
            token = set_hermes_home_override(str(default_root))
            try:
                yield
            finally:
                reset_hermes_home_override(token)

        from hermes_cli.config import load_config_readonly
        with _global_home():
            global_config = load_config_readonly()
        global_aux = global_config.get("auxiliary", {}) if isinstance(global_config, dict) else {}
        global_task = global_aux.get(task, {}) if isinstance(global_aux, dict) else {}
        if not isinstance(global_task, dict):
            global_task = {}
    except Exception:  # global read failed: keep the profile-effective block, never break the call
        logger.debug("global auxiliary.<%s> read failed; using profile block only", task, exc_info=True)
        return task_config

    if global_task == task_config:
        return task_config
    return _layer_over_inherited(global_task, task_config)


# The fields that together pick WHERE a call goes. They travel as one unit: a provider pinned on an
# inheriting task must never pick up the base's base_url/api_key (that would send one vendor's key
# to another's endpoint).
_AUX_ROUTE_KEYS = frozenset({"provider", "model", "base_url", "api_key", "api_mode", "key_env",
                             "api_key_env", "reasoning_effort"})


def _layer_over_inherited(inherited: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a task's own ``auxiliary.<task>`` block over its inherited base.

    The picker, "reset to auto" and the dashboard persist ``provider: auto`` plus ``""`` for
    model/base_url/api_key/reasoning_effort when the operator expresses no preference, so those
    placeholders mean "follow the base", not "override it with nothing". Once the operator pins a
    route (a non-auto provider, a model or a base_url) the whole route comes from the task's own
    block. Non-route keys (timeout, extra_body, ...) override per key; only ``""`` is dropped,
    never other falsy values (``reasoning_effort: false`` is an explicit choice)."""
    provider = str(user.get("provider") or "").strip().lower()
    pinned = (provider not in ("", "auto") or bool(str(user.get("model") or "").strip())
              or bool(str(user.get("base_url") or "").strip()))
    merged = {k: v for k, v in inherited.items() if not (pinned and k in _AUX_ROUTE_KEYS)}
    for key, value in user.items():
        if value == "" and not (pinned and key in _AUX_ROUTE_KEYS):
            continue
        if key == "provider" and not pinned:
            continue
        merged[key] = value
    return merged
