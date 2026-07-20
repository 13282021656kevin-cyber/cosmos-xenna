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

"""Unit tests for the pipeline-scoped ObjectOwner registry."""

import numpy as np
import pytest
import ray

from cosmos_xenna.ray_utils import object_owner


class TestSplitFieldEnabled:
    """Env-var gate parsing (no Ray required)."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "  On "])
    def test_truthy(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Recognized truthy env values enable split-field."""
        monkeypatch.setenv(object_owner.SPLIT_FIELD_ENV_VAR, value)
        assert object_owner.split_field_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "nope"])
    def test_falsy(self, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-truthy env values keep split-field disabled."""
        monkeypatch.setenv(object_owner.SPLIT_FIELD_ENV_VAR, value)
        assert object_owner.split_field_enabled() is False


def test_get_current_owner_none_when_ray_not_initialized() -> None:
    """Owner lookup returns None (not raise) when Ray is not running."""
    object_owner.reset_owner_cache()
    if not ray.is_initialized():
        assert object_owner.get_current_owner() is None


@pytest.fixture(scope="module")
def _ray_local() -> object:
    """Provide a small local Ray instance for the module."""
    started = False
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        started = True
    yield
    if started:
        ray.shutdown()


@pytest.fixture(autouse=True)
def _reset_cache() -> object:
    """Clear the per-process owner cache around every test."""
    object_owner.reset_owner_cache()
    yield
    object_owner.reset_owner_cache()


@pytest.mark.usefixtures("_ray_local")
class TestOwnerLifecycle:
    """Create / discover / destroy lifecycle plus ownership survival."""

    def test_create_ping_find_destroy(self) -> None:
        """Owner is live, discoverable by name, and gone after destroy."""
        handle = object_owner.create_object_owner()
        try:
            assert ray.get(handle.ping.remote()) is True
            assert object_owner.get_current_owner() is not None
        finally:
            object_owner.destroy_object_owner(handle)
        object_owner.reset_owner_cache()
        assert object_owner.get_current_owner() is None

    def test_create_is_idempotent(self) -> None:
        """get_if_exists means a second create returns the same named actor."""
        h1 = object_owner.create_object_owner()
        try:
            object_owner.reset_owner_cache()
            h2 = object_owner.create_object_owner()
            assert ray.get(h2.ping.remote()) is True
        finally:
            object_owner.destroy_object_owner(h1)

    def test_owner_assigned_object_survives_producer_death(self) -> None:
        """ray.put(_owner=owner) survives the producing actor's death."""

        @ray.remote(num_cpus=0)
        class _Producer:
            def put(self, owner: object, n: int) -> tuple:
                arr = np.frombuffer(bytes(n), dtype=np.uint8)
                return (ray.put(arr, _owner=owner),)

        owner = object_owner.create_object_owner()
        try:
            producer = _Producer.remote()
            (ref,) = ray.get(producer.put.remote(owner, 1_000_000))
            ray.kill(producer)
            # Owner still alive -> object must remain resolvable.
            assert ray.get(ref).nbytes == 1_000_000
        finally:
            object_owner.destroy_object_owner(owner)

    def test_destroy_is_safe_with_none(self) -> None:
        """destroy_object_owner tolerates a None handle."""
        object_owner.destroy_object_owner(None)
