# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""


import asyncio
import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import torch
import websockets

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Convert address formatting (e.g., host:port) into a valid WebSocket address
        clean_addr = config.server_address.replace("http://", "").replace("https://", "")
        if ":" in clean_addr and clean_addr.endswith(":443"):
            clean_addr = clean_addr.split(":")[0]

        self.server_address = f"wss://{clean_addr}"

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )

        self.shutdown_event = threading.Event()

        # Local variables and locks
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []

        # Sync the websocket thread loop and the main execution control loop
        self.start_barrier = threading.Barrier(2)

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.must_go = threading.Event()
        self.must_go.set()

        # Shared queue for thread-safe cross-communication between control loop and WebSocket thread
        self.outbound_network_queue = Queue()
        self.websocket_connection = None

        self.logger.info("Robot connected and ready")


    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Prepares state for the communication threads."""
        self.shutdown_event.clear()
        self.logger.info(f"Targeting Policy Server network route: {self.server_address}")
        return True

    def stop(self):
        """Stop the robot client and kill loops."""
        self.shutdown_event.set()
        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

    def send_observation(
        self,
        obs: TimedObservation
    ) -> bool:
        """Enqueues an observation frame to the outward WebSocket buffer thread."""
        if not self.running:
            raise RuntimeError("Client not running.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        # Routing identifier prepended: 0x02 tells our WebSocket server this data is an observation frame
        payload = b"\x02" + observation_bytes
        self.outbound_network_queue.put(payload)
        return True

    def _network_loop_worker(self, verbose: bool):
        """Synchronous wrapper execution target for our asyncio engine thread loop."""
        asyncio.run(self._async_network_manager(verbose))

    async def _async_network_manager(self, verbose: bool):
        """Asynchronous execution controller managing the pipeline life cycle."""
        self.logger.info(f"Establishing WebSocket socket line to: {self.server_address} ...")

        try:
            async with websockets.connect(self.server_address, max_size=2**26) as websocket:
                self.websocket_connection = websocket
                self.logger.info("Connection Handshake established with Proxy Gateway.")

                # Replicating SendPolicyInstructions logic
                policy_config_bytes = pickle.dumps(self.policy_config)
                # Routing identifier prepended: 0x01 tells server this is initialization data
                setup_payload = b"\x01" + policy_config_bytes
                await websocket.send(setup_payload)
                self.logger.info("Policy instructions transmitted successfully.")

                # Release thread synchronization barrier to unblock main loop execution
                self.start_barrier.wait()

                # Concurrent tracking blocks
                async def sender_task():
                    while self.running:
                        # Non-blocking checkout from the outbound queue block
                        try:
                            payload = self.outbound_network_queue.get_nowait()
                            await websocket.send(payload)
                        except Empty:
                            await asyncio.sleep(0.001)

                async def receiver_task():
                    while self.running:
                        try:
                            message = await websocket.recv()
                            if isinstance(message, bytes):
                                receive_time = time.time()

                                deserialize_start = time.perf_counter()
                                timed_actions = pickle.loads(message)  # nosec
                                deserialize_time = time.perf_counter() - deserialize_start

                                if len(timed_actions) > 0:
                                    received_device = timed_actions[0].get_action().device.type
                                    self.logger.debug(f"Received actions on device: {received_device}")

                                client_device = self.config.client_device
                                if client_device != "cpu":
                                    for timed_action in timed_actions:
                                        if timed_action.get_action().device.type != client_device:
                                            timed_action.action = timed_action.get_action().to(client_device)

                                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                                if len(timed_actions) > 0 and verbose:
                                    with self.latest_action_lock:
                                        latest_action = self.latest_action
                                    incoming_timesteps = [a.get_timestep() for a in timed_actions]
                                    first_action_timestep = timed_actions[0].get_timestep()
                                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                                    self.logger.info(
                                        f"Received action chunk for step #{first_action_timestep} | "
                                        f"Latest action: #{latest_action} | "
                                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                                        f"Network latency (server->client): {server_to_client_latency:.2f}ms"
                                    )

                                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                                self.must_go.set()

                        except websockets.exceptions.ConnectionClosed:
                            self.logger.error("Remote endpoint connection dropped abruptly.")
                            break

                await asyncio.gather(sender_task(), receiver_task())

        except Exception as e:
            self.logger.error(f"WebSocket execution lifecycle error: {e}")
            # Ensure safety fallback release if initialization fails
            try:
                self.start_barrier.wait()
            except threading.BrokenBarrierError:
                pass

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2
        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue
        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}
        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action
            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue
            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue
            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )
        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def actions_available(self):
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        _performed_action = self.robot.send_action(
            self._action_tensor_to_action_dict(timed_action.get_action())
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        return _performed_action

    def _ready_to_send_observation(self):
        with self.action_queue_lock:
            if self.action_chunk_size <= 0:
                return True
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            start_time = time.perf_counter()
            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )
            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            self.send_observation(observation)
            if observation.must_go:
                self.must_go.clear()
            return raw_observation
        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        self.start_barrier.wait()
        self.logger.info("Control loop thread actively running execution routines.")
        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))
        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Spawning core WebSocket background pipeline thread...")

        # We target the custom network loop orchestrating inbound/outbound packets asynchronously
        network_thread = threading.Thread(
            target=client._network_loop_worker,
            args=(cfg.debug_visualize_queue_size,),
            daemon=True
        )
        network_thread.start()

        try:
            # The main execution thread runs the hardware robot kinematic loops uninterrupted
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            network_thread.join(timeout=1.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client cleanly terminated.")


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
