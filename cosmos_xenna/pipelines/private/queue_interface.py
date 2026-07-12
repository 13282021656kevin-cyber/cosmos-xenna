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

"""Queue abstraction that decouples the streaming scheduler from queue internals.

The streaming executor (``streaming.py``) moves work between stages through
per-stage queues. Historically it reached directly into the concrete queue's
internals (``by_node_id`` deques, ``ray.ObjectRef`` handling, locality-aware
batching). That coupling made it impossible to swap the queue implementation --
e.g. to a persistent/durable backend -- without editing the scheduler.

This module defines the seam:

- :class:`Batch` -- the opaque unit of work that flows between stages. Its
  ``items`` are *handles*, not real payloads. What a handle is depends on the
  backend (``ray.ObjectRef`` for the in-memory backend, a payload-store key for
  a persistent backend). The scheduler never inspects a handle directly; it only
  materialises payloads at the boundary via a :class:`PayloadCodec`.

- :class:`StageQueue` -- the Protocol the scheduler depends on. Any backend that
  satisfies it can be dropped in without touching the scheduling loop.

- :class:`PayloadCodec` -- the materialisation boundary between a real sample and
  the handle that travels inside a queue. This is the *only* place the scheduler
  turns a payload into a handle (``encode``) or back (``decode``).

Design notes
------------
- ``Batch`` is deliberately independent of ``ray_utils.actor_pool.Task`` so the
  queue layer does not depend on the scheduling/Ray layer. The scheduler
  converts between the two at its own boundary.
- ``ack``/``nack`` are optional capabilities. The in-memory backend treats a
  read as destructive ("take == consume") and implements them as no-ops. A
  durable backend can use them to implement claim/lease semantics
  (PENDING -> CLAIMED -> ACKED) without changing the scheduler's call sites.
  Callers gate on :attr:`StageQueue.supports_ack`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class Batch:
    """A unit of work flowing between pipeline stages.

    Attributes:
        items: Opaque handles to the samples in this batch. The concrete type is
            backend-specific and MUST NOT be interpreted by the scheduler --
            only by the owning queue and the matching :class:`PayloadCodec`.
            For the in-memory backend these are ``ray.ObjectRef``s.
        origin_node_id: Optional hint for where this batch's data resides, used
            for locality-aware scheduling. Backends without a notion of locality
            may ignore it (and readers may receive ``None``).
        lease: Optional backend-owned token identifying a claim/lease, populated
            by durable backends on read so a later ``ack``/``nack`` can refer to
            the exact claim. Opaque to the scheduler.
    """

    items: list[Any]
    origin_node_id: Optional[str] = None
    lease: Any = field(default=None)

    def __len__(self) -> int:
        return len(self.items)


@runtime_checkable
class StageQueue(Protocol):
    """The queue contract the streaming scheduler depends on.

    Depth is always expressed in *samples* (individual items), not batches, so
    back-pressure and autoscaling signals are comparable across stages with
    different batch sizes.
    """

    # --- capability flags -------------------------------------------------
    # Whether ``ack``/``nack`` are meaningful. A destructive-read (in-memory)
    # backend sets this False; a claim/lease backend sets it True.
    supports_ack: bool

    # --- depth / emptiness ------------------------------------------------
    def __len__(self) -> int:
        """Total number of samples currently queued (readable + not-yet-acked)."""
        ...

    def __bool__(self) -> bool:
        """True iff the queue holds at least one sample."""
        ...

    # --- writes -----------------------------------------------------------
    def put(self, batch: Batch) -> None:
        """Append a batch of handles.

        Unifies the three historical write paths:
          - seeding offline input:  ``put(Batch(input_data, None))``
          - serving ingestion:      ``put(Batch([item], None))``
          - inter-stage transfer:   ``put(Batch(task.items, task.origin_node_id))``
        """
        ...

    # --- reads ------------------------------------------------------------
    def try_get_batch(self, size: int) -> Optional[Batch]:
        """Pull up to ``size`` samples as one batch.

        Returns ``None`` when the queue cannot currently satisfy the request
        (e.g. fewer than ``size`` samples available and the caller wants a full
        batch). Locality/ordering policy is entirely the implementation's
        choice. For durable backends this is a *claim*: the returned batch's
        ``lease`` must be passed to a later ``ack``/``nack``.
        """
        ...

    # --- back-pressure statistics ----------------------------------------
    def avg_samples_per_batch(self) -> Optional[float]:
        """Rolling average samples-per-batch over recent writes, or ``None``.

        Used by the scheduler to translate a queue's sample depth into an
        approximate task count for back-pressure accounting.
        """
        ...

    # --- teardown ---------------------------------------------------------
    def drain(self) -> list[Any]:
        """Remove and return all remaining handles. Empties the queue."""
        ...

    # --- optional claim/lease semantics (no-ops for destructive backends) --
    def ack(self, batch: Batch) -> None:
        """Confirm a batch obtained from :meth:`try_get_batch` is fully processed.

        No-op for destructive-read backends. Durable backends move the claim
        from CLAIMED to ACKED.
        """
        ...

    def nack(self, batch: Batch) -> None:
        """Return a claimed batch for reprocessing.

        No-op for destructive-read backends. Durable backends move the claim
        back to PENDING.
        """
        ...


@runtime_checkable
class PayloadCodec(Protocol):
    """Materialisation boundary between a real sample and a queue handle.

    This is the single place the scheduler converts a payload into the opaque
    handle a :class:`StageQueue` stores, and back. Swapping this out (together
    with a matching queue backend) is what makes payload storage pluggable --
    e.g. Ray object store vs. an on-disk/NVMe payload store -- without the
    scheduler knowing where payloads live.
    """

    def encode(self, sample: Any) -> Any:
        """Turn a real sample into a queue handle (e.g. ``ray.put``)."""
        ...

    def decode(self, handle: Any) -> Any:
        """Resolve a queue handle back into a real sample (e.g. ``ray.get``)."""
        ...
