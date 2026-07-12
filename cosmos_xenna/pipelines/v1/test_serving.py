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


import logging
import multiprocessing
import queue
import threading
import time

import pytest
import ray

from cosmos_xenna.pipelines import v1 as pipelines_v1

logger = logging.getLogger(__name__)


class Stage(pipelines_v1.Stage):
    def __init__(self, run_time_s: float):
        self._run_time_s = run_time_s

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int]:
        time.sleep(self._run_time_s)
        return [x * 2 for x in in_data]


def _start_serving_pipeline(pipeline_spec: pipelines_v1.PipelineSpec) -> None:
    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    except Exception as e:
        raise e
    finally:
        ray.shutdown()


def test_serving() -> None:
    input_queue = multiprocessing.Queue(maxsize=4)
    output_queue = multiprocessing.Queue(maxsize=2)
    serving_queues = pipelines_v1.ServingQueues(source=input_queue, sink=output_queue)

    stages = [
        Stage(0.1),
        Stage(0.4),
        Stage(0.2),
    ]

    spec = pipelines_v1.PipelineSpec(
        [],
        stages,
        pipelines_v1.PipelineConfig(execution_mode=pipelines_v1.ExecutionMode.SERVING),
        serving_queues=serving_queues,
    )

    try:
        # launch the pipeline in a separate thread/process
        pipeline_thread = threading.Thread(target=_start_serving_pipeline, args=(spec,))
        pipeline_thread.start()

        # send one request to the pipeline
        input_queue.put(1)
        output_result = output_queue.get(block=True, timeout=10)
        assert output_result == 8

        # send bursty requests to test back-pressure
        num_requests = 10
        for i in range(num_requests):
            input_queue.put(i)
        time.sleep(1)
        results = []
        while True and len(results) < num_requests:
            try:
                results.append(output_queue.get(block=True, timeout=10))
            except queue.Empty:
                break
        assert len(results) == num_requests
        assert sorted(results) == [x * 8 for x in range(num_requests)]

        # assert that the pipeline is still running
        assert pipeline_thread.is_alive()

    except Exception as e:
        raise e

    finally:
        input_queue.put(None)
        pipeline_thread.join()


class _SlowStage(pipelines_v1.Stage):
    """A deliberately slow stage used as the pipeline bottleneck.

    With ``slots_per_actor`` concurrent slots, a *single* worker can complete at
    most ``slots_per_actor / run_time_s`` items per second. When requests arrive
    faster than that, the autoscaler adds more workers to this stage to keep
    throughput up (as long as the cluster has spare CPUs). We assert on the
    observed end-to-end throughput rather than on an internal worker count: if
    sustained throughput exceeds what one worker could possibly deliver, the
    stage must have scaled out.
    """

    def __init__(self, run_time_s: float):
        self._run_time_s = run_time_s

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=1.0)

    def process_data(self, in_data: list[int]) -> list[int]:
        time.sleep(self._run_time_s)
        return [x * 2 for x in in_data]


@pytest.mark.slow
def test_serving_autoscales_bottleneck_stage() -> None:
    """SERVING shares the streaming executor, so its stages autoscale.

    Strategy:
    - Build a head -> bottleneck -> tail pipeline where the middle stage is far
      slower (0.5s/item vs 0.01s/item). Leave every stage free to autoscale:
      the Rust ``FragmentationBasedAutoscaler`` re-balances workers *between
      autoscaling stages*, so pinning the fast stages would actually prevent it
      from growing the bottleneck (learned the hard way).
    - Shrink ``autoscale_interval_s`` and the speed-estimation window so the
      autoscaler actually ticks within the test (defaults are 180s each, far
      longer than any reasonable test run).
    - Give the queues plenty of headroom and continuously feed + drain so the
      bottleneck's backlog signal isn't masked by back-pressure.
    - Assert on *throughput*: a single bottleneck worker can finish at most
      ``slots_per_actor / 0.5s`` items/s. If the pipeline sustains well above
      that, the bottleneck stage must have scaled out to more than one worker.

    We deliberately do NOT create a Ray named-actor counter and pre-``ray.init``
    the driver: doing so starts Ray in a way that breaks the executor's
    dashboard-state monitoring (``ray.util.state.list_actors`` -> 500) and
    crashes the run thread. Letting ``run_pipeline`` own Ray init keeps the
    monitoring API healthy.

    NOTE: heavy integration test (starts Ray, spawns real workers, runs for tens
    of seconds). Marked ``slow`` so the default ``-m 'not slow'`` addopts skips
    it; run explicitly with ``-m slow``.
    """
    # Roomy queues so back-pressure doesn't hide the bottleneck's backlog.
    input_queue = multiprocessing.Queue(maxsize=200)
    output_queue = multiprocessing.Queue(maxsize=200)
    serving_queues = pipelines_v1.ServingQueues(source=input_queue, sink=output_queue)

    bottleneck_run_time_s = 0.5
    slots_per_actor = 2  # PipelineConfig default; a single worker => this many concurrent slots.

    # All stages free to autoscale (num_workers left as None).
    stages = [
        Stage(0.01),
        _SlowStage(bottleneck_run_time_s),
        Stage(0.01),
    ]

    # Speed the autoscaler way up and make it responsive on a short window.
    streaming_spec = pipelines_v1.StreamingSpecificSpec(
        autoscale_interval_s=3.0,
        autoscale_speed_estimation_window_duration_s=4.0,
        autoscale_speed_estimation_min_data_points=1,
        # Don't tear a freshly-started worker down on the very next tick.
        scale_down_grace_after_ready_s=0.0,
        autoscaler_verbosity_level=pipelines_v1.VerbosityLevel.INFO,
    )

    spec = pipelines_v1.PipelineSpec(
        [],
        stages,
        pipelines_v1.PipelineConfig(
            execution_mode=pipelines_v1.ExecutionMode.SERVING,
            mode_specific=streaming_spec,
            slots_per_actor=slots_per_actor,
            logging_interval_s=3.0,
        ),
        serving_queues=serving_queues,
    )

    stop_feeding = threading.Event()
    completed = 0  # number of results drained from the sink
    completed_lock = threading.Lock()

    def _feeder() -> None:
        """Continuously push work so the bottleneck stays saturated."""
        i = 0
        while not stop_feeding.is_set():
            try:
                input_queue.put(i, block=True, timeout=1)
                i += 1
            except queue.Full:
                # Back-pressure is expected; just keep trying until told to stop.
                continue

    def _drainer() -> None:
        """Drain the sink and count completed items to measure throughput."""
        nonlocal completed
        while not stop_feeding.is_set():
            try:
                output_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            with completed_lock:
                completed += 1

    pipeline_thread = threading.Thread(target=_start_serving_pipeline, args=(spec,))
    feeder_thread = threading.Thread(target=_feeder)
    drainer_thread = threading.Thread(target=_drainer)

    try:
        pipeline_thread.start()
        feeder_thread.start()
        drainer_thread.start()

        # Let workers start and the autoscaler settle before measuring, so the
        # throughput window reflects the scaled-up steady state, not cold start.
        warmup_s = 20.0
        logger.info("Warming up for %.0fs while the autoscaler scales the bottleneck...", warmup_s)
        time.sleep(warmup_s)

        with completed_lock:
            start_count = completed
        measure_s = 10.0
        time.sleep(measure_s)
        with completed_lock:
            end_count = completed

        throughput = (end_count - start_count) / measure_s
        single_worker_max = slots_per_actor / bottleneck_run_time_s  # = 4 items/s here
        logger.info(
            "Measured throughput=%.1f items/s over %.0fs; a single bottleneck worker caps at %.1f items/s.",
            throughput,
            measure_s,
            single_worker_max,
        )

        # Core assertion: sustained throughput above one worker's ceiling proves
        # the bottleneck stage autoscaled to multiple workers. Use a margin so a
        # little measurement jitter can't produce a false positive.
        assert throughput > single_worker_max * 1.5, (
            f"expected throughput > {single_worker_max * 1.5:.1f} items/s (proving the bottleneck "
            f"scaled beyond one worker), but measured {throughput:.1f} items/s"
        )

        # And the pipeline should still be serving.
        assert pipeline_thread.is_alive()

    finally:
        # Stop feeders/drainers first, then signal the pipeline to shut down.
        stop_feeding.set()
        feeder_thread.join(timeout=5)
        drainer_thread.join(timeout=5)
        input_queue.put(None)
        pipeline_thread.join()


if __name__ == "__main__":
    test_serving()
    test_serving_autoscales_bottleneck_stage()
