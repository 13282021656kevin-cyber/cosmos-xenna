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


import collections
import random
from typing import Any, List, Optional

import pytest

from cosmos_xenna.pipelines.private.queue_interface import Batch
from cosmos_xenna.pipelines.private.streaming import InMemoryStageQueue


# Helper to build a Batch of handles for the queue under test.
# The items represent ray.ObjectRef for the purpose of these tests. There is no
# need to create real ObjectRefs, since InMemoryStageQueue only stores and
# retrieves them opaquely.
def _create_batch(data: List[Any], node_id: Optional[str] = None) -> Batch:
    return Batch(items=data, origin_node_id=node_id)


class TestInMemoryStageQueue:
    def test_initialization(self):
        q_custom = InMemoryStageQueue(samples_per_batch_window=50)
        assert q_custom._samples_per_batch.maxlen == 50
        q_default = InMemoryStageQueue()
        assert q_default._samples_per_batch.maxlen == 100

    def test_len_and_bool_empty(self):
        q = InMemoryStageQueue()
        assert len(q) == 0
        assert not q
        assert bool(q) is False

    def test_put_single_node(self):
        q = InMemoryStageQueue()
        batch1_data = ["obj1", "obj2", "obj3"]
        q.put(_create_batch(batch1_data, "nodeA"))

        assert len(q) == 3
        assert q._by_node_id["nodeA"] == collections.deque(batch1_data)
        assert q._samples_per_batch == collections.deque([3])

        batch2_data = ["obj4", "obj5"]
        q.put(_create_batch(batch2_data, "nodeA"))
        assert len(q) == 5
        assert q._by_node_id["nodeA"] == collections.deque(batch1_data + batch2_data)
        assert q._samples_per_batch == collections.deque([3, 2])

    def test_put_multiple_nodes(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2], "node1"))
        q.put(_create_batch([10, 20], "node2"))
        q.put(_create_batch([3], "node1"))

        assert len(q) == 5  # 2 + 2 + 1
        assert q._by_node_id["node1"] == collections.deque([1, 2, 3])
        assert q._by_node_id["node2"] == collections.deque([10, 20])
        assert list(q._samples_per_batch) == [2, 2, 1]  # Order of addition

    def test_put_no_data(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([], "node1"))

        assert len(q) == 0
        # Node might exist with an empty deque, but put with no data doesn't
        # append to _samples_per_batch.
        assert not q._by_node_id.get("node1") or not q._by_node_id["node1"]
        assert not q._samples_per_batch  # "Only record if samples were actually added"

        q.put(_create_batch([1], "node1"))
        assert len(q) == 1
        assert q._samples_per_batch == collections.deque([1])

    def test_len_and_bool_with_items(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2], "node1"))
        assert len(q) == 2
        assert q
        assert bool(q) is True

    def test_avg_samples_per_batch_empty(self):
        q = InMemoryStageQueue()
        assert q.avg_samples_per_batch() is None

    def test_avg_samples_per_batch_single_batch(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2, 3], "node1"))
        assert q.avg_samples_per_batch() == 3.0

    def test_avg_samples_per_batch_multiple_batches(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2], "node1"))
        q.put(_create_batch([10, 20, 30, 40], "node2"))
        assert q.avg_samples_per_batch() == (2 + 4) / 2.0  # 3.0

    def test_avg_samples_per_batch_with_empty_batches(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2], "node1"))
        q.put(_create_batch([], "node2"))
        q.put(_create_batch([1, 2, 3, 4], "node1"))
        assert q.avg_samples_per_batch() == (2 + 4) / 2.0  # Still 3.0

    def test_avg_samples_per_batch_exceeds_window(self):
        q = InMemoryStageQueue(samples_per_batch_window=2)
        q.put(_create_batch([1] * 1, "n1"))
        assert q.avg_samples_per_batch() == 1.0
        q.put(_create_batch([1] * 2, "n1"))
        assert q.avg_samples_per_batch() == 1.5
        q.put(_create_batch([1] * 3, "n1"))
        assert q.avg_samples_per_batch() == 2.5
        q.put(_create_batch([1] * 4, "n1"))
        assert q.avg_samples_per_batch() == 3.5

    def test_try_get_batch_empty_queue(self):
        q = InMemoryStageQueue()
        assert q.try_get_batch(5) is None

    def test_try_get_batch_not_enough_items(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2], "node1"))
        assert q.try_get_batch(3) is None
        assert len(q) == 2  # Items should remain

    def test_try_get_batch_exact_size_single_node(self):
        q = InMemoryStageQueue()
        items = [1, 2, 3]
        q.put(_create_batch(items, "node1"))
        batch = q.try_get_batch(3)
        assert batch is not None
        assert batch.items == items
        assert batch.origin_node_id == "node1"
        assert len(q) == 0

    def test_try_get_batch_partial_size_single_node(self):
        q = InMemoryStageQueue()
        items = [1, 2, 3, 4, 5]
        q.put(_create_batch(items, "node1"))
        batch = q.try_get_batch(3)
        assert batch is not None
        assert batch.items == [1, 2, 3]  # FIFO from deque
        assert batch.origin_node_id == "node1"
        assert len(q) == 2
        assert list(q._by_node_id["node1"]) == [4, 5]

    def test_try_get_batch_exact_size_multiple_nodes_and_distribution(self):
        q = InMemoryStageQueue()
        items_n1 = [1, 2, 3]  # 3 items
        items_n2 = [11, 12]  # 2 items
        items_n3 = [21]  # 1 item
        q.put(_create_batch(items_n1, "node1"))
        q.put(_create_batch(items_n2, "node2"))
        q.put(_create_batch(items_n3, "node3"))
        # Total 3+2+1 = 6 items

        random.seed(42)  # Control shuffle for test repeatability

        batch = q.try_get_batch(4)  # Request 4 items
        assert batch is not None
        assert len(batch.items) == 4
        assert len(q) == 2  # 6 - 4 = 2 items left

        # Check that items are from the original set
        original_items = set(items_n1 + items_n2 + items_n3)
        for item in batch.items:
            assert item in original_items

    def test_try_get_batch_invalid_batch_size(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2, 3], "node1"))
        with pytest.raises(ValueError, match="Batch size must be greater than 0"):
            q.try_get_batch(0)
        with pytest.raises(ValueError, match="Batch size must be greater than 0"):
            q.try_get_batch(-1)
        assert len(q) == 3  # No change to queue

    def test_try_get_batch_multiple_calls_drain_queue(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1, 2], "node1"))
        q.put(_create_batch([3, 4], "node2"))

        batch1 = q.try_get_batch(2)
        assert batch1 is not None
        # Test that the queue has fully drained from one node
        assert q._by_node_id[batch1.origin_node_id] == collections.deque()
        batch2 = q.try_get_batch(2)
        assert batch2 is not None
        assert q.try_get_batch(1) is None  # Queue is empty

        all_pulled_items = batch1.items + batch2.items
        assert set(all_pulled_items) == {1, 2, 3, 4}

    def test_try_get_batch_node_empties_during_batch(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1], "node1"))
        q.put(_create_batch([11, 12, 13], "node2"))
        batch = q.try_get_batch(3)  # Request 3 items
        assert batch is not None
        assert len(batch.items) == 3
        assert len(q) == 1

    def test_drain_empty_queue(self):
        q = InMemoryStageQueue()
        assert q.drain() == []
        assert not q._by_node_id  # Should be empty

    def test_drain_single_node(self):
        q = InMemoryStageQueue()
        items = [1, 2, 3]
        q.put(_create_batch(items, "node1"))
        all_samples = q.drain()
        # Order from list(queue) then extend
        assert all_samples == items
        assert len(q) == 0
        assert not q._by_node_id.get("node1")  # Node entry should be removed

    def test_drain_multiple_nodes(self):
        q = InMemoryStageQueue()
        items_n1 = [1, 2]
        items_n2 = [10, 20]
        q.put(_create_batch(items_n1, "node1"))
        q.put(_create_batch(items_n2, "node2"))

        all_samples = q.drain()
        # Order depends on dict iteration order of _by_node_id.keys() then list(queue).
        # So, check for content and size.
        assert len(all_samples) == 4
        assert set(all_samples) == {1, 2, 10, 20}
        assert len(q) == 0
        assert not q._by_node_id  # Check that nodes are cleaned up

    def test_drain_clears_queue_and_empties_nodes(self):
        q = InMemoryStageQueue()
        q.put(_create_batch([1], "node1"))
        q.put(_create_batch([2], "node2"))
        q.drain()
        assert len(q) == 0
        assert not q  # bool is False
        assert not q._by_node_id
        # _samples_per_batch is not cleared by drain
        assert q._samples_per_batch == collections.deque([1, 1])

    def test_ack_nack_are_noops(self):
        # Destructive-read backend: ack/nack must be no-ops and not raise.
        q = InMemoryStageQueue()
        assert q.supports_ack is False
        q.put(_create_batch([1, 2], "node1"))
        batch = q.try_get_batch(2)
        assert batch is not None
        q.ack(batch)  # no-op
        q.nack(batch)  # no-op
        assert len(q) == 0  # take was already destructive

    def test_integration_put_batch_drain(self):
        q = InMemoryStageQueue()
        # Put batches
        q.put(_create_batch([1, 2, 3], "N1"))
        q.put(_create_batch([4, 5], "N2"))
        q.put(_create_batch([6], "N1"))
        # State: N1: [1,2,3,6], N2: [4,5]. Total = 6.
        # _samples_per_batch: [3,2,1] -> avg (3+2+1)/3 = 2
        assert len(q) == 6
        assert q.avg_samples_per_batch() == 2.0

        random.seed(10)  # N2 then N1
        # Batch 1 (size 3):
        # Pull N2 (4), N1 (1), N2 (5) -> Batch [4,1,5], Nodes [N2,N1,N2]. Most common: N2
        # Remaining: N1: [2,3,6], N2: []
        batch1 = q.try_get_batch(3)
        assert batch1 is not None
        assert len(batch1.items) == 3
        assert set(batch1.items) == {1, 4, 5}
        assert batch1.origin_node_id == "N2"
        assert len(q) == 3
        assert q.avg_samples_per_batch() == 2.0  # Unchanged by get

        # Remaining: N1: [2,3,6] (node N2 is empty and removed from active consideration)
        # Batch 2 (size 2):
        # Pull N1 (2), N1 (3) -> Batch [2,3]. Nodes [N1,N1]. Most common: N1
        batch2 = q.try_get_batch(2)
        assert batch2 is not None
        assert len(batch2.items) == 2
        assert set(batch2.items) == {2, 3}
        assert batch2.origin_node_id == "N1"
        assert len(q) == 1

        # Drain remaining (should be [6] from N1)
        remaining_samples = q.drain()
        assert remaining_samples == [6]
        assert len(q) == 0
        assert not q

        all_retrieved = batch1.items + batch2.items + remaining_samples
        assert set(all_retrieved) == {1, 2, 3, 4, 5, 6}
        assert q.avg_samples_per_batch() == 2.0  # Still unchanged


class TestBackendConfig:
    """Config-level pluggability: queue/payload backends resolve from spec."""

    def test_conformance_and_factory_defaults(self):
        from cosmos_xenna.pipelines.private.queue_interface import PayloadCodec, StageQueue
        from cosmos_xenna.pipelines.private.specs import StreamingSpecificSpec
        from cosmos_xenna.pipelines.private.streaming import InMemoryStageQueue, RayObjectStoreCodec

        # Built-ins satisfy the runtime-checkable Protocols.
        assert isinstance(InMemoryStageQueue(), StageQueue)
        assert isinstance(RayObjectStoreCodec(), PayloadCodec)

        # Factories default to None so run_pipeline falls back to the built-ins.
        spec = StreamingSpecificSpec()
        assert spec.queue_factory is None
        assert spec.payload_codec_factory is None

    def test_custom_backend_satisfies_protocol_and_is_selectable(self):
        from cosmos_xenna.pipelines.private.queue_interface import Batch, StageQueue
        from cosmos_xenna.pipelines.private.specs import StreamingSpecificSpec

        class CountingQueue:
            """Minimal alternative backend used to prove the seam is real."""

            supports_ack = False

            def __init__(self) -> None:
                self._items: list[Any] = []

            def __len__(self) -> int:
                return len(self._items)

            def __bool__(self) -> bool:
                return bool(self._items)

            def put(self, batch: Batch) -> None:
                self._items.extend(batch.items)

            def try_get_batch(self, size: int):
                if len(self._items) < size:
                    return None
                taken, self._items = self._items[:size], self._items[size:]
                return Batch(taken, None)

            def avg_samples_per_batch(self):
                return None

            def drain(self) -> list[Any]:
                out, self._items = self._items, []
                return out

            def ack(self, batch: Batch) -> None:
                del batch

            def nack(self, batch: Batch) -> None:
                del batch

        assert isinstance(CountingQueue(), StageQueue)

        # A factory is what run_pipeline consumes; verify it yields a working queue.
        spec = StreamingSpecificSpec(queue_factory=CountingQueue)
        q = spec.queue_factory()
        q.put(Batch([1, 2, 3], None))
        assert len(q) == 3
        assert q.try_get_batch(2).items == [1, 2]
        assert q.drain() == [3]

