# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pipeline-scoped object owner for split-field data transport.

Background
==========
When a worker actor calls ``ray.put(value)``, the resulting object is *owned*
by that actor. ``ray.put`` objects have no lineage, so when the owning actor is
torn down (e.g. ``ActorPool.stop()`` between/at the end of stages), the backing
Plasma object is garbage-collected and any downstream ``ray.get`` fails with
``OwnerDiedError``.

This breaks the "split-field" optimization, where a producing stage pushes a
large payload into Plasma once and only a tiny ``ObjectRef`` (48 bytes) rides
through the intermediate pass-through stages until a consumer resolves it.

Fix
===
Create a single lightweight, pipeline-scoped ``ObjectOwner`` actor that lives
for the entire pipeline (created before any stage worker, destroyed only after
the final outputs are collected). Producers then call
``ray.put(value, _owner=owner_handle)`` so ownership is assigned to this
long-lived actor instead of the ephemeral worker. The payload still lands in the
producer's *local* Plasma store (no routing through the owner, so no cross-node
bottleneck); only the ownership/ref-counting metadata is held by the owner.

Wiring
======
The owner is registered as a Ray named actor (``OWNER_ACTOR_NAME``) in the
pipeline's namespace. Any worker in the same Ray job can therefore discover it
via :func:`get_current_owner`, which performs a cached ``ray.get_actor`` lookup.
No handle plumbing through ``ActorPool`` / ``StageWorker`` is required.

Gating
======
Split-field is opt-in and STREAMING-only:

* The owner is created only when :func:`split_field_enabled` returns True, and
  only by the STREAMING engine. The BATCH engine never creates it, so
  split-field naturally no-ops there (consumers fall back to inline transport).
* :func:`split_field_enabled` reads ``SPLIT_FIELD_ENV_VAR`` on the *driver*.
  Workers do not need the env var: they simply follow owner presence via
  :func:`get_current_owner`.
"""

import os
import threading
import typing

import ray

from cosmos_xenna.utils import python_log as logger

if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

# Ray named-actor identity used to publish/discover the pipeline-scoped owner.
OWNER_ACTOR_NAME = "cosmos_xenna_object_owner"

# Driver-side env var that opts the pipeline into split-field transport.
SPLIT_FIELD_ENV_VAR = "CC_LAZYDATA_SPLIT_FIELD"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Per-process cache of the discovered owner handle. Only *positive* results are
# cached so a transient early lookup miss is retried rather than pinned to None.
_lock = threading.Lock()
_cached_owner: "ActorHandle | None" = None


def split_field_enabled() -> bool:
    """Return whether split-field transport is enabled via the driver env var."""
    return os.environ.get(SPLIT_FIELD_ENV_VAR, "").strip().lower() in _TRUTHY


@ray.remote(num_cpus=0)
class ObjectOwner:
    """A do-nothing actor whose sole purpose is to *own* Plasma objects.

    It holds no state and does no work; it simply needs to outlive the stage
    workers so that objects assigned to it via ``ray.put(..., _owner=self)``
    survive individual worker teardown.
    """

    def ping(self) -> bool:
        """Liveness probe used to confirm the actor has started."""
        return True


def create_object_owner() -> "ActorHandle":
    """Create (or fetch) the pipeline-scoped owner named actor.

    Uses ``get_if_exists=True`` for idempotency across retries and nested runs.
    Blocks briefly on a ``ping`` so callers know the actor is schedulable before
    the pipeline starts producing owner-assigned objects.
    """
    global _cached_owner
    handle = ObjectOwner.options(
        name=OWNER_ACTOR_NAME,
        get_if_exists=True,
        num_cpus=0,
    ).remote()
    # Confirm liveness; surfaces scheduling failures eagerly instead of at the
    # first ray.put(_owner=...) deep inside a worker.
    ray.get(handle.ping.remote())
    with _lock:
        _cached_owner = handle
    logger.debug(f"Created pipeline-scoped ObjectOwner actor '{OWNER_ACTOR_NAME}'")
    return handle


def destroy_object_owner(handle: "ActorHandle | None") -> None:
    """Kill the owner actor. Safe to call with None or a dead handle."""
    global _cached_owner
    with _lock:
        _cached_owner = None
    if handle is None:
        return
    try:
        ray.kill(handle)
        logger.debug(f"Destroyed pipeline-scoped ObjectOwner actor '{OWNER_ACTOR_NAME}'")
    except Exception as e:  # noqa: BLE001 - best-effort teardown, never raise from cleanup
        logger.debug(f"Failed to kill ObjectOwner actor (already gone?): {e!r}")


def get_current_owner() -> "ActorHandle | None":
    """Return the pipeline-scoped owner handle for this process, or None.

    Performs a cached, best-effort ``ray.get_actor`` lookup by name. Returns
    None when Ray is not initialized or the owner does not exist (e.g. BATCH
    mode, or split-field disabled), signalling callers to use inline transport.
    Only successful lookups are cached, so an early miss does not permanently
    disable split-field for the process.
    """
    global _cached_owner
    with _lock:
        if _cached_owner is not None:
            return _cached_owner

    if not ray.is_initialized():
        return None
    try:
        handle = ray.get_actor(OWNER_ACTOR_NAME)
    except ValueError:
        return None
    with _lock:
        _cached_owner = handle
    return handle


def reset_owner_cache() -> None:
    """Clear the per-process cached handle. Intended for tests."""
    global _cached_owner
    with _lock:
        _cached_owner = None
