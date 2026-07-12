# Persistent Queue Backend for Streaming / Serving

_Design document — February 2026_

---

## Status

**Status**: 📝 DRAFT — FOR REVIEW
**Scope**: Streaming executor (`pipelines/private/streaming.py`) queue layer
**Depends on**: the `StageQueue` / `PayloadCodec` seam already landed in
`pipelines/private/queue_interface.py`

This document specifies a **durable queue backend** that lets a streaming /
serving pipeline **resume mid-flight after a driver (scheduler) restart without
reprocessing already-completed stages**. It deliberately does **not** persist
payloads; only the lightweight per-stage queue metadata is made durable.

---

## Decision Summary (state as of this review)

**Current state: interface seam landed, no persistent backend implemented — by
deliberate choice.** What exists today is the decoupling layer only; whether/how
to add durability is deferred until a concrete need justifies the cost.

**Landed (verified, behaviour-preserving pure refactor):**
- `queue_interface.py`: `Batch`, `StageQueue`, `PayloadCodec` protocols, incl.
  `ack`/`nack`/`supports_ack` extension points for a future lease backend.
- `InMemoryStageQueue` + `RayObjectStoreCodec`: built-in defaults, byte-for-byte
  equivalent to the original `Queue`.
- Scheduler depends only on the protocol; payload materialisation is isolated in
  the codec.
- `StreamingSpecificSpec.queue_factory` / `payload_codec_factory`: backend is
  swappable from config without touching the scheduler.

**The core trade-off (decides any future backend):** the deciding variable is
**per-item processing cost**, i.e. how expensive it is to re-run one item end to
end.

| Option | Recovery granularity | Cost | Best when |
|---|---|---|---|
| **sink-done** (source LEFT JOIN done; the sink/its keys *are* the done set) | Whole pipeline re-runs the in-flight item | Cheapest: 1 persistent write per item, no atomic cross-stage transfer, almost no scheduler change | **Stages are cheap.** Result is exactly-once (idempotent sink); processing is at-least-once. |
| **per-stage durable (Anvil/SlateDB)** | Resume from the exact stage reached | Write amplification (one persistent write *per stage boundary* per item) + S3-latency per op; needs `ack_and_forward` atomic transfer | **Individual stages are expensive** (e.g. large-model inference) and re-running them is real money. |
| **selective checkpoint** | Only the expensive stage's output survives | Write amplification on *one* boundary only | **Only one/few stages are expensive**; cheap stages re-run freely. |

**Key constraints established:**
- **Payload lifecycle bounds recovery.** Because handles are Ray `ObjectRef`s
  (cluster-scoped), any of these only survive a *driver* restart while the Ray
  cluster lives; a full cluster bounce loses in-flight payloads unless payloads
  themselves are persisted. See §2.
- **"sink-done" saves result duplication, not compute duplication.** An item
  that crashes just before the sink write re-runs the whole pipeline; the
  idempotent sink only prevents a duplicate *result*, not the wasted compute.
- **done-set read cost scales with id *count*, not metadata size.** Store only
  ids. ≤10M: load once into memory (RoaringBitmap for int ids). ~100M+: push the
  LEFT JOIN into the store with keyset-paginated streaming reads. Best: let the
  sink itself be the done set (upsert / deterministic-key existence check).
- **SlateDB = embedded LSM KV storing SSTs on object storage (S3).** Gives
  compute/storage separation (broker can move machines without losing queue
  state) at the price of S3-latency per read/write — which is exactly why the
  Anvil/SlateDB path only pays off for expensive-stage workloads, and is a poor
  fit for low-latency/high-QPS serving.
- **A "many stateless consumers competing on an S3-backed queue" serving design
  is viable but is a *different consumption model*** than Xenna's single-driver
  loop (it's essentially nurion's Anvil + stage_worker). Adopting it means either
  reusing nurion or building a new competing-consumer execution mode in Xenna —
  not just swapping a `StageQueue` backend. Deferred.

**Recommendation if/when durability is pursued:** start with **sink-done**
(covers "don't reprocess what's already finished" at minimal cost); escalate to
**selective checkpoint** for any individually expensive stage; only adopt the
full **Anvil/SlateDB** path if most stages are expensive *and* the workload can
absorb S3-latency — and in that case, prefer reusing nurion over rebuilding.

---

## Table of Contents

1. [Goal & Non-Goals](#1-goal--non-goals)
2. [The Hard Constraint: What "Restart" Can Recover](#2-the-hard-constraint-what-restart-can-recover)
3. [Architecture Overview](#3-architecture-overview)
4. [Record Model](#4-record-model)
5. [Message State Machine & Atomic Stage Advance](#5-message-state-machine--atomic-stage-advance)
6. [Interface Changes](#6-interface-changes)
7. [Scheduler Loop Changes](#7-scheduler-loop-changes)
8. [Recovery Flow](#8-recovery-flow)
9. [Backend Selection](#9-backend-selection)
10. [Failure Scenarios](#10-failure-scenarios)
11. [Rollout Plan](#11-rollout-plan)
12. [Open Questions](#12-open-questions)

---

## 1. Goal & Non-Goals

### Goal

Save processing resources on restart. When the pipeline is processing a stream
and the **driver** dies (code update, OOM, scheduler bug, deploy), work that has
already advanced to stage *K* must resume from stage *K* — stages `0..K-1` must
**not** run again for that item.

### Non-Goals

- **Payload durability.** Real sample data continues to live in the Ray object
  store (zero-copy, fast). We do not copy payloads to S3/NVMe here.
- **Surviving a full Ray-cluster restart.** See §2 — this is a physical
  consequence of not persisting payloads, and is called out explicitly rather
  than hidden.
- **Exactly-once.** The durable backend is at-least-once; duplicates on recovery
  are possible and handled by deterministic keys / idempotent sinks (§10).

---

## 2. The Hard Constraint: What "Restart" Can Recover

Recovery capability is bounded by **whether the Ray cluster (raylets + object
store) survives**, because the durable queue stores an `ObjectRef` *handle*, and
an `ObjectRef` is a **cluster-scoped pointer**.

| Restart scenario | ObjectRef validity | Can resume mid-pipeline? |
|---|---|---|
| **Driver/scheduler only** restarts; Ray cluster alive (code update, driver OOM, deploy, scheduler crash) | Still valid | ✅ **Yes — the target scenario.** Metadata + object-store data both survive; resume from each item's recorded stage. |
| **Whole Ray cluster** restarts | All invalid | ❌ Intermediate payloads vanish with the object store. Metadata survives but points at nothing. |

**This design targets the first row**, which covers the overwhelming majority of
real "the process restarted" events. The second row is out of scope and requires
also persisting stage outputs (payload durability) — a separate, heavier effort.

> ⚠️ **Must be stated to operators:** a full cluster bounce loses in-flight
> payloads. Do not market this as general crash tolerance.

---

## 3. Architecture Overview

```
             ┌──────────────────────── Ray cluster (survives driver restart) ─────────────────────────┐
             │                                                                                          │
  serving    │   ┌────────┐        ┌────────┐        ┌────────┐                                         │
  source ───▶│   │Stage 0 │──────▶ │Stage 1 │──────▶ │Stage 2 │ ───▶ sink                               │
             │   │ pool   │        │ pool   │        │ pool   │                                          │
             │   └───┬────┘        └───┬────┘        └───┬────┘                                          │
             │       │ handles=ObjectRef into Ray object store (payload, NOT persisted)                 │
             └───────┼─────────────────┼─────────────────┼──────────────────────────────────────────────┘
                     │                 │                 │
        ┌────────────▼─────────────────▼─────────────────▼────────────┐
        │            Durable queue backend (survives driver restart)   │
        │   q_input   q_stage0_out   q_stage1_out   q_stage2_out        │
        │   records = {msg_id, stage_idx, handle, origin_node_id, ...}  │  ← lightweight metadata only
        │   atomic ack_and_forward across adjacent queues (WriteBatch)  │
        └──────────────────────────────────────────────────────────────┘
```

- **Payload** = big (image/tensor/frame), stays in Ray object store, referenced
  by an `ObjectRef` handle. Not persisted.
- **Queue record** = tiny metadata (tens of bytes + a serialized handle),
  persisted in the durable backend.
- The durable backend is a `StageQueue` implementation, selected via the
  existing `StreamingSpecificSpec.queue_factory` — the scheduler stays agnostic.

---

## 4. Record Model

Each durable queue record:

```python
@dataclass
class DurableRecord:
    msg_id: str            # unique; used for ack, dedup, lease tracking
    stage_idx: int         # which stage's INPUT this record sits in (recovery anchor)
    handle: bytes          # serialized ObjectRef (valid only while cluster lives)
    origin_node_id: str | None   # locality hint, preserved across persist
    # backend-managed: state (PENDING/CLAIMED/ACKED), claimed_at, lease_id, attempt
```

- `stage_idx` is the **recovery anchor**: on restart, a record is re-inserted
  into stage `stage_idx`'s input, so stages before it never re-run.
- `handle` is opaque to the backend; it round-trips the `ObjectRef` the same way
  the in-memory backend keeps it in a deque.
- Records map onto the existing `Batch`: `Batch.items` are handles,
  `Batch.origin_node_id` is preserved, and `Batch.lease` carries the backend's
  claim token for a later `ack`/`nack`.

Payload key note: because payloads are not persisted, there is no
deterministic-key requirement here. If a future variant *does* persist payloads,
the handle should become a deterministic payload key so re-execution overwrites
in place (idempotent) — see §10.

---

## 5. Message State Machine & Atomic Stage Advance

Borrowed from the claim/ack (lease) model already proven in `nurion` Anvil.

```
PENDING ──claim()──▶ CLAIMED ──ack_and_forward()──▶ (removed here, PENDING in next queue)
   ▲                    │
   │                    │ nack() / claim timeout
   └────────────────────┘
```

### Atomic stage advance — the core primitive

When stage *K* finishes an item, the backend performs, **in one atomic
transaction**:

```
atomic {
    delete msg from q_stage(K)      # ack upstream
    insert msg' into q_stage(K+1)   # forward downstream, new handle + stage_idx=K+1
}
```

Why atomicity is non-negotiable:

| Order without atomicity | Crash between the two ops | Result |
|---|---|---|
| forward, then ack | after forward | downstream has it, upstream still has it → **duplicate** reprocessing |
| ack, then forward | after ack | upstream deleted, downstream never got it → **lost** item |
| **atomic ack_and_forward** | — | either both or neither → **always consistent** |

This is exactly `AnvilQueueClient.ack_and_forward` (SlateDB `WriteBatch`). The
last stage variant is `ack` only (or `ack` + push-to-sink), with no downstream
queue.

### Claim timeout / lease

A CLAIMED record whose owner never acks (worker died mid-process, or driver
died holding claims) returns to PENDING after `claim_timeout`, so a fresh run
re-claims it. This is what makes "resume the un-processed remainder" work: on
restart, in-flight-but-un-acked items time out and get re-claimed at their
recorded `stage_idx`.

---

## 6. Interface Changes

Extends the existing `StageQueue` protocol. **In-memory backend is unaffected**
(destructive read, `supports_ack=False`, `ack_and_forward` not required).

```python
class StageQueue(Protocol):
    supports_ack: bool          # in-memory: False, durable: True
    # existing:
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def put(self, batch: Batch) -> None: ...
    def try_get_batch(self, size: int) -> Optional[Batch]: ...   # durable: a CLAIM
    def avg_samples_per_batch(self) -> Optional[float]: ...
    def drain(self) -> list[Any]: ...
    def ack(self, batch: Batch) -> None: ...
    def nack(self, batch: Batch) -> None: ...

    # NEW — only meaningful when supports_ack is True:
    def ack_and_forward(self, batch: Batch, downstream: "StageQueue",
                        downstream_batch: Batch) -> None:
        """Atomically ack `batch` here and put `downstream_batch` into `downstream`.

        In-memory backend does NOT implement this; the scheduler only calls it
        when supports_ack is True. Durable backends implement it as a single
        transaction spanning both queues (they must share one broker/txn domain).
        """
```

A durable `StageQueue` also needs a factory-time hook to **rebuild from
persisted state** on startup (see §8). Proposed: a module-level
`recover(queue_names) -> dict[str, StageQueue]` on the backend, called by
`run_pipeline` before the main loop when `supports_ack` is True.

`recover()` reads the current queue depths (in samples) so the same length-based
back-pressure / autoscaling signals (`_upstream_queue_lens`) work unchanged.

---

## 7. Scheduler Loop Changes

Two edits in `streaming.py`, both gated on `queue.supports_ack` so the in-memory
path is byte-for-byte unchanged.

### 7.1 Pull → claim (already a no-op-compatible call)

`try_get_batch` already returns a `Batch`; for durable backends it additionally
marks CLAIMED and stamps `Batch.lease`. No call-site change — the semantics
differ inside the backend.

### 7.2 Transfer → atomic advance

Current inter-stage transfer (streaming.py, the reversed stage loop):

```python
for task in completed_tasks:
    output_queue.put(Batch(task.task_data, task.origin_node_id))
```

Durable variant:

```python
if upstream_queue.supports_ack:
    for task in completed_tasks:
        upstream_queue.ack_and_forward(
            batch=task.source_batch,                       # the claimed input batch
            downstream=output_queue,
            downstream_batch=Batch(task.task_data, task.origin_node_id),
        )
else:
    for task in completed_tasks:
        output_queue.put(Batch(task.task_data, task.origin_node_id))
```

This requires threading the **originating claimed batch** through the actor pool
so a completed task can be matched back to the upstream record it must ack.
Today `actor_pool.Task` carries `task_data` + `origin_node_id`; we add an
optional `source_lease` (opaque) that the scheduler round-trips. Failed tasks
call `nack` instead.

Last stage: `ack` (serving sink push happens after ack succeeds) — never
`ack_and_forward`.

### 7.3 SERVING ingestion & termination

- Source poll → `input_queue.put(Batch([item], None))` unchanged; durable
  backend persists it as a stage-0 PENDING record.
- The `None` termination sentinel is **not** persisted (it is a control signal,
  not work).

---

## 8. Recovery Flow (driver restart, cluster alive)

```
1. run_pipeline starts; queue_factory backend is durable (supports_ack=True).
2. backend.recover([q_input, q_stage0_out, ..., q_stageN_out]) reconnects to the
   broker; persisted records are still there.
3. For each queue, records keep their stage_idx; the scheduler rebuilds its
   per-stage input structures from them.
4. In-flight-but-CLAIMED records from the dead driver time out (claim_timeout)
   and return to PENDING, re-claimable at their recorded stage_idx.
5. Main loop runs normally. An item last acked into q_stage2 is claimed by
   stage 2 — stages 0 and 1 never touch it again. ✅ resource saved.
```

Cold start (no persisted state) is identical to today: all queues empty, seed
from source.

---

## 9. Backend Selection

The **atomic cross-stage transfer** requirement is the deciding constraint.

| Backend | Atomic ack_and_forward across two queues | Claim/lease | Locality | Verdict |
|---|---|---|---|---|
| **nurion Anvil** | ✅ native (`ack_and_forward`, SlateDB WriteBatch) | ✅ | ✅ QueueGroup | **Recommended** — reuse, don't rebuild |
| Redis Streams | ⚠️ only via Lua across streams, awkward | ✅ (XCLAIM) | ❌ | Possible but fights the model |
| Kafka | ❌ cross-topic txn heavy/complex; partition-worker coupling already rejected by nurion | ⚠️ | ❌ | Not recommended |

**Recommendation: implement `AnvilStageQueue` wrapping the existing
`nurion/lib/anvil-rs` broker.** It provides `ack_and_forward`, claim/nack
leases, and node-locality partitions out of the box, and we already own it.
Payload stays in the Ray object store; Anvil stores only the `DurableRecord`
metadata.

Mapping:

| StageQueue | AnvilQueueClient |
|---|---|
| `put` | `push` |
| `try_get_batch` | `claim` (batch_size) |
| `ack` | `ack` |
| `nack` | `nack` |
| `ack_and_forward` | `ack_and_forward` |
| `__len__` | `get_pending_count` (+ claimed, per policy) |
| `recover` | reconnect broker + `get_stats` per queue |

---

## 10. Failure Scenarios

| Scenario | Behavior |
|---|---|
| Driver dies holding claims | CLAIMED records time out → PENDING → re-claimed at recorded stage. No stage re-run before that point. |
| Worker dies mid-process | Its claim times out; item re-claimed at same stage (one wasted partial compute, no duplicate downstream — downstream only appears after atomic ack_and_forward). |
| Crash *between* ack and forward | Impossible to observe: they are one atomic txn. |
| Duplicate on re-claim (worker produced output, died before ack) | At-least-once: downstream may get the item twice on the *next* stage boundary. Mitigate with idempotent sink / deterministic keys. In-cluster object-store payloads make this rare (only the final sink push is externally visible). |
| Full Ray cluster restart | Out of scope (§2): handles invalid, payloads gone. Backend still holds metadata but cannot resume; operator must restart the stream from source. |

---

## 11. Rollout Plan

Incremental; each step independently verifiable, in-memory path never regresses.

1. **Interface**: add `ack_and_forward` to the `StageQueue` protocol (default:
   in-memory does not implement; scheduler gates on `supports_ack`). Add
   `source_lease` passthrough to `actor_pool.Task`. Land + test with in-memory
   backend still green.
2. **Scheduler gating**: add the `supports_ack` branch for transfer
   (ack_and_forward) and failure (nack). Behavior identical for in-memory.
3. **AnvilStageQueue**: implement the wrapper + `recover()`. Unit-test against a
   local Anvil broker (push/claim/ack_and_forward/timeout).
4. **Integration**: a serving pipeline with `queue_factory=AnvilStageQueue`;
   kill the driver mid-stream, restart, assert earlier stages do not re-run
   (e.g. per-stage processed-counter actors).
5. **Docs/ops**: document the cluster-restart boundary (§2) prominently.

---

## 12. Open Questions

1. **Cross-queue transaction domain.** `ack_and_forward` requires the two
   queues to share one broker/txn domain. Confirm all per-stage Anvil queues
   live under one broker instance (they do in nurion's single-broker model).
2. **Claim timeout tuning.** Too short → healthy-but-slow stages get their items
   re-claimed (wasted compute); too long → slow recovery. nurion's
   heartbeat + dual-timeout (lease vs claim-age) solves this; do we adopt the
   heartbeat, or start with a single conservative `claim_timeout`?
3. **Back-pressure semantics.** Anvil's `create_queue(max_pending)` vs the
   current `max_queued` gate — reconcile so both bound memory consistently.
4. **`avg_samples_per_batch`.** Durable backend must track rolling batch sizes
   too (needed by the back-pressure math); confirm Anvil exposes enough stats or
   compute client-side.
5. **Metadata GC.** ACKED records retention (Anvil `acked_retention_secs`) vs.
   the pipeline's own lifetime — pick a retention that survives a driver restart
   window but doesn't grow unbounded.
