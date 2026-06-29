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
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import asyncio
import logging
import pickle  # nosec
import threading
import time
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import torch
import websockets

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.msgpack_numpy import packb, unpackb, Packer

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer:
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self._action_packer = Packer()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes set by client initialization
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def handle_ready(self, client_id: str):
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

    def handle_policy_instructions(self, client_id: str, data: bytes):
        """Receive policy instructions from the robot client"""
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return

        # policy_specs = pickle.loads(data)  # nosec
        raw_specs = unpackb(data)
        specs_dict = {k.decode() if isinstance(k, bytes) else k: v for k, v in raw_specs.items()}
        policy_specs = RemotePolicyConfig(**specs_dict)

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()
        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

    def handle_observations(self, client_id: str, data: bytes):
        """Receive observations from the robot client"""
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()
        start_deserialize = time.perf_counter()

        # timed_observation = pickle.loads(data)  # nosec
        # deserialize_time = time.perf_counter() - start_deserialize
        obs_dict = unpackb(data)
        deserialize_time = time.perf_counter() - start_deserialize

        timed_observation = TimedObservation(
            timestamp=obs_dict[b"timestamp"],
            timestep=obs_dict[b"timestep"],
            observation=obs_dict[b"observation"],
            must_go=obs_dict.get(b"must_go", False)
        )

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

    def execute_inference_cycle(self, client_id: str) -> bytes | None:
        """Executes one pass of inference from the queue, replicating GetActions."""
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            action_data = [
                {
                    "timestamp": action.timestamp,
                    "timestep": action.timestep,
                    "action": action.action.detach().cpu().numpy() # Send raw NumPy data
                }
                for action in action_chunk
            ]

            actions_bytes = self._action_packer.pack(action_data)
            serialize_time = time.perf_counter() - start_time

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            # sleep controls inference latency
            elapsed = time.perf_counter() - getactions_starts
            sleep_needed = max(0.0, self.config.inference_latency - elapsed)
            if sleep_needed > 0:
                time.sleep(sleep_needed)

            return actions_bytes

        except Empty:
            return None
        except Exception as e:
            self.logger.error(f"Error in inference cycle: {e}")
            return None

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps
        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False
        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False
        return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (obs.must_go or self.last_processed_obs is None or self._obs_sanity_checks(obs, self.last_processed_obs)):
            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True
        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """

        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)
        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""

        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )

        """2. Apply preprocessor"""
        observation = self.preprocessor(observation)

        """3. Get action chunk"""
        self.last_processed_obs = observation_t
        action_tensor = self._get_action_chunk(observation)

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually

        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()
        return self._time_action_chunk(observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep())

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


async def connection_handler(websocket, server_instance: PolicyServer):
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    server_instance.logger.info(f"New WebSocket connection established from {client_id}")

    server_instance.handle_ready(client_id)

    async def inbound_loop():
        """Listens for payload messages from the client."""
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # We look at the first byte to route the payload type
                    # 0x01 = Setup Instructions, 0x02 = Observation Frame
                    payload_type = message[0]
                    payload_body = message[1:]

                    if payload_type == 0x01:
                        server_instance.handle_policy_instructions(client_id, payload_body)
                    elif payload_type == 0x02:
                        server_instance.handle_observations(client_id, payload_body)
                else:
                    server_instance.logger.warning("Received unsupported text frame.")
        except websockets.exceptions.ConnectionClosed:
            server_instance.logger.info(f"Client connection closed reader side: {client_id}")

    async def outbound_loop():
        """Continuously runs local inference loops and pushes downstream action bytes."""
        try:
            while server_instance.running:
                # Offload heavy ML tensor math to thread pool so it doesn't block async event loop
                action_bytes = await asyncio.to_thread(server_instance.execute_inference_cycle, client_id)
                if action_bytes:
                    await websocket.send(action_bytes)
                else:
                    await asyncio.sleep(0.001)  # Yield loop control if queue was empty
        except websockets.exceptions.ConnectionClosed:
            server_instance.logger.info(f"Client connection closed writer side: {client_id}")

    # Co-routines execution tree
    await asyncio.gather(inbound_loop(), outbound_loop())


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = PolicyServer(cfg)

    # Initialize standard WebSocket infrastructure
    async def main():
        async with websockets.serve(
            lambda ws: connection_handler(ws, policy_server),
            cfg.host,
            cfg.port,
            max_size=2**26 # 64MB buffer safety limit for handling vision tensors
        ):
            policy_server.logger.info(f"WebSocket PolicyServer started on ws://{cfg.host}:{cfg.port}")
            await asyncio.Future()  # keeps the async loop alive indefinitely

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        policy_server.stop()
        policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
