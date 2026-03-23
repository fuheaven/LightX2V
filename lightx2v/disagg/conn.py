from __future__ import annotations

import logging
import os
import struct
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zmq

from lightx2v.disagg.mooncake import MooncakeTransferEngine

logger = logging.getLogger(__name__)


class DisaggregationPhase(Enum):
    NULL = "null"
    PHASE1 = "phase1"
    PHASE2 = "phase2"


class DisaggregationMode(Enum):
    NULL = "null"
    ENCODE = "encode"
    TRANSFORMER = "transformer"
    DECODE = "decode"


def group_concurrent_contiguous(src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    src_groups = []
    dst_groups = []
    current_src = [src_indices[0]]
    current_dst = [dst_indices[0]]

    for i in range(1, len(src_indices)):
        src_contiguous = src_indices[i] == src_indices[i - 1] + 1
        dst_contiguous = dst_indices[i] == dst_indices[i - 1] + 1
        if src_contiguous and dst_contiguous:
            current_src.append(src_indices[i])
            current_dst.append(dst_indices[i])
        else:
            src_groups.append(current_src)
            dst_groups.append(current_dst)
            current_src = [src_indices[i]]
            current_dst = [dst_indices[i]]

    src_groups.append(current_src)
    dst_groups.append(current_dst)

    return src_groups, dst_groups


@dataclass
class DataArgs:
    sender_engine_rank: int
    receiver_engine_rank: int
    data_ptrs: list[int]
    data_lens: list[int]
    data_item_lens: list[int]
    ib_device: Optional[str] = None


class DataPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


RequestPoolType = Dict[int, List[int]]
WaitingPoolType = Dict[int, Tuple[str, list[int]]]


def _disagg_zmq_port_offset() -> int:
    """Shift all disagg ZMQ bind/connect ports on one host when running multiple replicas.

    Set LIGHTX2V_DISAGG_PORT_OFFSET (integer) identically for encoder, transformer, and decoder
    processes of the same replica; use a different offset per replica (e.g. 0, 100, 200).
    """
    raw = os.environ.get("LIGHTX2V_DISAGG_PORT_OFFSET", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


_ZMQ_OFF = _disagg_zmq_port_offset()
REQUEST_POLLING_PORT = 12788 + _ZMQ_OFF
DATASENDER_POLLING_PORT = 17788 + _ZMQ_OFF
DATARECEIVER_POLLING_PORT = 27788 + _ZMQ_OFF


class DataManager:
    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, disaggregation_phase: DisaggregationPhase, disaggregation_mode: DisaggregationMode):
        self.engine = MooncakeTransferEngine()
        self.data_args: Dict[int, DataArgs] = {}
        self.room_threads: Dict[int, List[threading.Thread]] = {}
        self.room_stop_events: Dict[int, threading.Event] = {}
        self.transfer_events: Dict[int, threading.Event] = {}
        self.disaggregation_phase = disaggregation_phase
        self.disaggregation_mode = disaggregation_mode
        self.request_pool: RequestPoolType = {}
        self.request_status: Dict[int, DataPoll] = {}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.server_socket.setsockopt(zmq.RCVTIMEO, 200)
        # When initializing multiple bootstrap rooms inside the same process,
        # some sockets (bound only by sender/receiver engine rank) must be bound once.
        self._bound_ports: set[str] = set()
        # Phase1 encoder: avoid multiple recv_multipart() readers on the same socket.
        # Each (sender_engine_rank -> DATASENDER_POLLING_PORT + rank) bind addr should have only one reader thread.
        self._p1_encode_reader_stop_events: Dict[str, threading.Event] = {}
        self._p1_encode_reader_threads: Dict[str, threading.Thread] = {}
        self._p1_encode_reader_refcount: Dict[str, int] = {}
        # Mooncake doesn't support re-registering an already registered (overlapping)
        # memory region. In this codebase, dynamic room rebind may reuse the same
        # RDMA buffer pointers, so we must de-duplicate registrations per-process.
        self._registered_ptrs: set[int] = set()
        if self.disaggregation_phase == DisaggregationPhase.PHASE1:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self.waiting_pool: WaitingPoolType = {}
            elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                pass
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        elif self.disaggregation_phase == DisaggregationPhase.PHASE2:
            if self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.waiting_pool: WaitingPoolType = {}
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                pass
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        else:
            raise ValueError(f"Unsupported DisaggregationPhase: {self.disaggregation_phase}")

        # When multiple rooms use the same receiver/sender rank port,
        # we must not start multiple threads doing recv_multipart() on the same socket,
        # otherwise messages get consumed by the "wrong" thread.
        self._p2_decode_thread_started_ports: set[str] = set()

    def init(self, args: DataArgs, room: int):
        self.data_args[room] = args
        self.register_buffer_to_engine(room)
        if self.disaggregation_phase == DisaggregationPhase.PHASE1:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self.transfer_events[room] = threading.Event()
                self.start_phase1_encode_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.start_phase1_transformer_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        elif self.disaggregation_phase == DisaggregationPhase.PHASE2:
            if self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.transfer_events[room] = threading.Event()
                self.start_phase2_transformer_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.start_phase2_decode_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        else:
            raise ValueError(f"Unsupported DisaggregationPhase: {self.disaggregation_phase}")

    def release(self, room: int):
        if self.disaggregation_phase == DisaggregationPhase.PHASE1:
            if self.disaggregation_mode == DisaggregationMode.ENCODE:
                self.end_phase1_encode_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.end_phase1_transformer_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        elif self.disaggregation_phase == DisaggregationPhase.PHASE2:
            if self.disaggregation_mode == DisaggregationMode.TRANSFORMER:
                self.end_phase2_transformer_thread(room)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.end_phase2_decode_thread(room)
            else:
                raise ValueError(f"Unsupported DisaggregationMode in this phase: {self.disaggregation_phase}, {self.disaggregation_mode}")
        else:
            raise ValueError(f"Unsupported DisaggregationPhase: {self.disaggregation_phase}")

        # Recycle room-scoped mappings.
        args = self.data_args.pop(room, None)
        if args is not None:
            for data_ptr in args.data_ptrs:
                self.engine.deregister(data_ptr)

        self.request_pool.pop(room, None)
        self.request_status.pop(room, None)
        if hasattr(self, "waiting_pool"):
            self.waiting_pool.pop(room, None)

    def register_buffer_to_engine(self, room: int):
        args = self.data_args[room]
        for data_ptr, data_len in zip(args.data_ptrs, args.data_lens):
            # Skip duplicate registration attempts for the same pointer.
            if int(data_ptr) in self._registered_ptrs:
                continue
            self.engine.register(data_ptr, data_len)
            self._registered_ptrs.add(int(data_ptr))

    def prepare_room_threads(self, room: int):
        self.room_stop_events[room] = threading.Event()
        self.room_threads[room] = []

    def register_room_thread(self, room: int, thread: threading.Thread):
        self.room_threads.setdefault(room, []).append(thread)

    def end_room_threads(self, room: int):
        stop_event = self.room_stop_events.get(room)
        if stop_event is not None:
            stop_event.set()
        transfer_event = self.transfer_events.get(room)
        if transfer_event is not None:
            transfer_event.set()
        threads = self.room_threads.get(room, [])
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self.room_threads.pop(room, None)
        self.room_stop_events.pop(room, None)
        self.transfer_events.pop(room, None)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_data(
        self,
        room: int,
        mooncake_session_id: str,
        sender_data_ptrs: List[int],
        receiver_ptrs: list[int],
    ):
        # TODO: transfer data in batch if there are many tensors or large tensors, instead of sending one by one.
        args = self.data_args[room]
        tensor_num = int(len(args.data_ptrs))
        for tensor_id in range(tensor_num):
            sender_addr = sender_data_ptrs[tensor_id]
            item_len = args.data_item_lens[tensor_id]
            receiver_addr = receiver_ptrs[tensor_id]

            # TODO: mooncake transfer engine can do async transfer. Do async later
            status = self.engine.transfer_sync(
                mooncake_session_id,
                sender_addr,
                receiver_addr,
                item_len,
            )
            if status != 0:
                return status
        return 0

    def sync_status_to_transformer_endpoint(self, remote: str, room: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        receiver_rank = self.data_args[room].receiver_engine_rank
        self._connect("tcp://" + remote + ":" + str(DATARECEIVER_POLLING_PORT + receiver_rank)).send_multipart(
            [
                str(room).encode("ascii"),
                str(self.request_status[room]).encode("ascii"),
            ]
        )

    def start_phase1_encode_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        transfer_event = self.transfer_events[room]
        sender_rank_port = DATASENDER_POLLING_PORT + self.data_args[room].sender_engine_rank
        logger.info("Encoder sender_rank_port=%s", sender_rank_port)
        bind_addr = "tcp://*:" + str(sender_rank_port)
        if bind_addr not in self._bound_ports:
            self.server_socket.bind(bind_addr)
            self._bound_ports.add(bind_addr)

        # Start (or reuse) a single socket reader thread for this bind_addr.
        if bind_addr not in self._p1_encode_reader_threads:
            shared_stop = threading.Event()
            self._p1_encode_reader_stop_events[bind_addr] = shared_stop

            def encode_socket_reader():
                while not shared_stop.is_set():
                    try:
                        frames = self.server_socket.recv_multipart()
                    except zmq.Again:
                        continue

                    if len(frames) != 4:
                        frame_lens = [len(f) for f in frames]
                        logger.error(
                            "Encoder recv_multipart frame count mismatch: got %s frames, lens=%s",
                            len(frames),
                            frame_lens,
                        )
                        continue

                    endpoint, mooncake_session_id, bootstrap_room, transformer_ptrs = frames
                    if bootstrap_room.decode("ascii") == "None":
                        continue

                    endpoint = endpoint.decode("ascii")
                    mooncake_session_id = mooncake_session_id.decode("ascii")
                    bootstrap_room_int = int(bootstrap_room.decode("ascii"))
                    transformer_ptrs = list(
                        struct.unpack(f"{len(transformer_ptrs) // 8}Q", transformer_ptrs)
                    )

                    self.waiting_pool[bootstrap_room_int] = (
                        endpoint,
                        mooncake_session_id,
                        transformer_ptrs,
                    )

                    # Do NOT wake transfer_thread here.
                    # Waiting pointers can arrive during initialization (before any request
                    # is enqueued). Waking early can lead to timeouts because request_pool
                    # for that room may never be produced for that transformer.
                    # transfer_thread should be woken only by DataSender.enqueue_request
                    # (which sets request_pool + transfer_event for the same room).

            t = threading.Thread(target=encode_socket_reader, daemon=True)
            t.start()
            self._p1_encode_reader_threads[bind_addr] = t
            self._p1_encode_reader_refcount[bind_addr] = 0

        self._p1_encode_reader_refcount[bind_addr] = self._p1_encode_reader_refcount.get(bind_addr, 0) + 1

        # encode_socket_reader is shared for this bind_addr; do not register it per room.

        room_id = room
        def transfer_thread():
            while not stop_event.is_set():
                transfer_event.wait()
                if stop_event.is_set():
                    break
                transfer_event.clear()
                # transfer_event may be set by DataSender.enqueue_request before
                # the transformer has sent its ZMQ pointer message (waiting_pool).
                # We must wait until both request_pool and waiting_pool are ready.
                timeout_s = int(os.environ.get("LIGHTX2V_DISAGG_WAIT_READY_TIMEOUT_S", "60"))
                t0 = time.time()
                endpoint = None
                mooncake_session_id = None
                transformer_ptrs = None
                debug_room = os.environ.get("LIGHTX2V_DISAGG_DEBUG_ROOM")
                try:
                    debug_room_int = int(debug_room) if debug_room is not None else None
                except ValueError:
                    debug_room_int = None
                if debug_room_int is not None and room_id == debug_room_int:
                    logger.warning(
                        "[DisaggDebug] transfer_event fired for room=%s (req_ready=%s wait_ready=%s)",
                        room_id,
                        room_id in self.request_pool,
                        room_id in self.waiting_pool,
                    )
                while not stop_event.is_set():
                    if room_id in self.request_pool and room_id in self.waiting_pool:
                        (endpoint, mooncake_session_id, transformer_ptrs) = self.waiting_pool.pop(room_id)
                        encode_data_ptrs = self.request_pool.pop(room_id)
                        break
                    if time.time() - t0 > timeout_s:
                        status = DataPoll.Failed
                        self.request_status[room_id] = status
                        # No endpoint: can't notify transformer reliably.
                        logger.error(
                            "Encoder transfer wait timeout: room=%s req_ready=%s wait_ready=%s",
                            room_id,
                            room_id in self.request_pool,
                            room_id in self.waiting_pool,
                        )
                        break
                    time.sleep(0.001)

                if endpoint is None:
                    continue

                status = DataPoll.Transferring
                self.request_status[room_id] = status
                self.sync_status_to_transformer_endpoint(endpoint, room_id)

                ret = self.send_data(room_id, mooncake_session_id, encode_data_ptrs, transformer_ptrs)
                if ret != 0:
                    status = DataPoll.Failed
                    self.request_status[room_id] = status
                    self.sync_status_to_transformer_endpoint(endpoint, room_id)
                    continue

                status = DataPoll.Success
                self.request_status[room_id] = status
                self.sync_status_to_transformer_endpoint(endpoint, room_id)

        transfer_worker = threading.Thread(target=transfer_thread)
        transfer_worker.start()
        self.register_room_thread(room, transfer_worker)

    def end_phase1_encode_thread(self, room: int):
        # stop shared socket reader only when last room for this bind_addr exits
        sender_rank_port = DATASENDER_POLLING_PORT + self.data_args[room].sender_engine_rank
        bind_addr = "tcp://*:" + str(sender_rank_port)

        self.end_room_threads(room)

        if bind_addr in self._p1_encode_reader_refcount:
            self._p1_encode_reader_refcount[bind_addr] -= 1
            if self._p1_encode_reader_refcount[bind_addr] <= 0:
                shared_stop = self._p1_encode_reader_stop_events.get(bind_addr)
                if shared_stop is not None:
                    shared_stop.set()
                t = self._p1_encode_reader_threads.get(bind_addr)
                if t is not None and t.is_alive():
                    t.join(timeout=1.0)
                self._p1_encode_reader_refcount.pop(bind_addr, None)
                self._p1_encode_reader_stop_events.pop(bind_addr, None)
                self._p1_encode_reader_threads.pop(bind_addr, None)

    def start_phase1_transformer_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        receiver_rank_port = DATARECEIVER_POLLING_PORT + self.data_args[room].receiver_engine_rank
        self.server_socket.bind("tcp://*:" + str(receiver_rank_port))

        def transformer_thread():
            while not stop_event.is_set():
                try:
                    (bootstrap_room, status) = self.server_socket.recv_multipart()
                except zmq.Again:
                    continue
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        transformer_worker = threading.Thread(target=transformer_thread)
        transformer_worker.start()
        self.register_room_thread(room, transformer_worker)

    def end_phase1_transformer_thread(self, room: int):
        self.end_room_threads(room)

    def start_phase2_transformer_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        transfer_event = self.transfer_events[room]
        sender_rank_port = DATASENDER_POLLING_PORT + self.data_args[room].sender_engine_rank
        logger.info("Transformer sender_rank_port=%s", sender_rank_port)
        self.server_socket.bind("tcp://*:" + str(sender_rank_port))

        def transformer_thread():
            while not stop_event.is_set():
                try:
                    (
                        endpoint,
                        mooncake_session_id,
                        bootstrap_room,
                        decode_ptrs,
                    ) = self.server_socket.recv_multipart()
                except zmq.Again:
                    continue
                if bootstrap_room.decode("ascii") == "None":
                    continue
                endpoint = endpoint.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                decode_ptrs = list(struct.unpack(f"{len(decode_ptrs) // 8}Q", decode_ptrs))
                logger.info(
                    "Transformer received ZMQ: endpoint=%s session_id=%s room=%s decode_ptrs=%s",
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    decode_ptrs,
                )
                self.waiting_pool[bootstrap_room] = (
                    endpoint,
                    mooncake_session_id,
                    decode_ptrs,
                )
                target_event = self.transfer_events.get(bootstrap_room, transfer_event)
                target_event.set()

        transformer_worker = threading.Thread(target=transformer_thread)
        transformer_worker.start()
        self.register_room_thread(room, transformer_worker)

        def transfer_thread():
            while not stop_event.is_set():
                transfer_event.wait()
                if stop_event.is_set():
                    break
                transfer_event.clear()
                bootstrap_room_ready = self.request_pool.keys()
                bootstrap_room_request = self.waiting_pool.keys()
                for room in list(bootstrap_room_request):
                    if room not in list(bootstrap_room_ready):
                        continue
                    status = DataPoll.Transferring
                    self.request_status[room] = status
                    (
                        endpoint,
                        mooncake_session_id,
                        decode_ptrs,
                    ) = self.waiting_pool.pop(room)
                    self.sync_status_to_transformer_endpoint(endpoint, room)
                    transformer_data_ptrs = self.request_pool.pop(room)
                    ret = self.send_data(
                        room,
                        mooncake_session_id,
                        transformer_data_ptrs,
                        decode_ptrs,
                    )
                    if ret != 0:
                        status = DataPoll.Failed
                        self.sync_status_to_transformer_endpoint(endpoint, room)
                        continue
                    status = DataPoll.Success
                    self.request_status[room] = status
                    self.sync_status_to_transformer_endpoint(endpoint, room)

        transfer_worker = threading.Thread(target=transfer_thread)
        transfer_worker.start()
        self.register_room_thread(room, transfer_worker)

    def end_phase2_transformer_thread(self, room: int):
        self.end_room_threads(room)

    def start_phase2_decode_thread(self, room: int):
        self.prepare_room_threads(room)
        stop_event = self.room_stop_events[room]
        receiver_rank_port = DATARECEIVER_POLLING_PORT + self.data_args[room].receiver_engine_rank
        bind_addr = "tcp://*:" + str(receiver_rank_port)
        already_started = bind_addr in self._p2_decode_thread_started_ports
        if bind_addr not in self._bound_ports:
            self.server_socket.bind(bind_addr)
            self._bound_ports.add(bind_addr)

        if already_started:
            # Reuse the existing socket reader thread created for this receiver port.
            # Multiple threads calling recv_multipart() on the same socket will race
            # and can cause missed status updates.
            return

        def decode_thread():
            while not stop_event.is_set():
                try:
                    (bootstrap_room, status) = self.server_socket.recv_multipart()
                except zmq.Again:
                    continue
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        decode_worker = threading.Thread(target=decode_thread)
        decode_worker.start()
        self._p2_decode_thread_started_ports.add(bind_addr)
        self.register_room_thread(room, decode_worker)

    def end_phase2_decode_thread(self, room: int):
        self.end_room_threads(room)

    def enqueue_request(
        self,
        bootstrap_room: int,
        data_ptrs: List[int],
    ):
        debug_room = os.environ.get("LIGHTX2V_DISAGG_DEBUG_ROOM")
        if debug_room is not None:
            try:
                debug_room_int = int(debug_room)
            except ValueError:
                debug_room_int = None
        else:
            debug_room_int = None
        if debug_room_int is not None and bootstrap_room == debug_room_int:
            args = self.data_args.get(bootstrap_room)
            logger.warning(
                "[DisaggDebug] enqueue_request room=%s sender_rank=%s receiver_rank=%s ptrs=%s",
                bootstrap_room,
                args.sender_engine_rank if args else None,
                args.receiver_engine_rank if args else None,
                len(data_ptrs),
            )

        self.request_pool[bootstrap_room] = data_ptrs
        self.request_status[bootstrap_room] = DataPoll.WaitingForInput
        if (
            self.disaggregation_phase == DisaggregationPhase.PHASE1
            and self.disaggregation_mode == DisaggregationMode.ENCODE
            or self.disaggregation_phase == DisaggregationPhase.PHASE2
            and self.disaggregation_mode == DisaggregationMode.TRANSFORMER
        ):
            transfer_event = self.transfer_events.get(bootstrap_room)
            if transfer_event is not None:
                transfer_event.set()

    def check_status(self, bootstrap_room: int):
        if (
            self.disaggregation_phase == DisaggregationPhase.PHASE1
            and self.disaggregation_mode == DisaggregationMode.TRANSFORMER
            or self.disaggregation_phase == DisaggregationPhase.PHASE2
            and self.disaggregation_mode == DisaggregationMode.DECODE
        ) and self.request_status[bootstrap_room] == DataPoll.Success:
            if bootstrap_room in self.request_pool:
                self.request_pool.pop(bootstrap_room)

        return self.request_status[bootstrap_room]

    def set_status(self, bootstrap_room: int, status: DataPoll):
        self.request_status[bootstrap_room] = status

    def get_localhost(self):
        return self.engine.get_localhost()

    def get_session_id(self):
        return self.engine.get_session_id()


class DataSender:
    def __init__(self, mgr: DataManager, bootstrap_addr: str, bootstrap_room: int):
        self.data_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.data_mgr.set_status(bootstrap_room, DataPoll.WaitingForInput)

    def init(self, num_data_indices: int):
        self.num_data_indices = num_data_indices

    def send(self, data_ptrs: List[int]):
        self.data_mgr.enqueue_request(self.bootstrap_room, data_ptrs)

    def poll(self) -> DataPoll:
        return self.data_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake DataSender Exception")


class DataReceiver:
    def __init__(self, mgr: DataManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.data_mgr = mgr
        if self.bootstrap_room is None:
            raise ValueError("bootstrap_room is required for DataReceiver")
        args = self.data_mgr.data_args[self.bootstrap_room]
        self.sender_server_url = bootstrap_addr.split(":")[0] + ":" + str(DATASENDER_POLLING_PORT + args.sender_engine_rank)
        logger.info("DataReceiver sender_server_url=%s", self.sender_server_url)
        self.receiver_ip = self.data_mgr.get_localhost()
        self.session_id = self.data_mgr.get_session_id()
        self.data_mgr.set_status(bootstrap_room, DataPoll.WaitingForInput)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self):
        args = self.data_mgr.data_args[self.bootstrap_room]
        packed_data_ptrs = b"".join(struct.pack("Q", ptr) for ptr in args.data_ptrs)
        self.data_mgr.enqueue_request(self.bootstrap_room, packed_data_ptrs)
        self._connect("tcp://" + self.sender_server_url).send_multipart(
            [
                self.receiver_ip.encode("ascii"),
                self.session_id.encode("ascii"),
                str(self.bootstrap_room).encode("ascii"),
                packed_data_ptrs,
            ]
        )

    def poll(self) -> DataPoll:
        return self.data_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake DataReceiver Exception")


class ReqManager:
    def __init__(self):
        self.context = zmq.Context.instance()
        self.push_sockets: Dict[str, zmq.Socket] = {}
        self.pull_sockets: Dict[int, zmq.Socket] = {}

    def send(self, ip: str, port: int, config: Any):
        def _to_builtin(value: Any):
            if isinstance(value, Mapping):
                return {k: _to_builtin(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_builtin(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_to_builtin(v) for v in value)
            return value

        endpoint = f"tcp://{ip}:{port}"
        socket = self.push_sockets.get(endpoint)
        if socket is None:
            socket = self.context.socket(zmq.PUSH)
            socket.connect(endpoint)
            self.push_sockets[endpoint] = socket
        socket.send_pyobj(_to_builtin(config))

    def receive(self, port: int):
        socket = self.pull_sockets.get(port)
        if socket is None:
            socket = self.context.socket(zmq.PULL)
            socket.bind(f"tcp://*:{port}")
            self.pull_sockets[port] = socket
        return socket.recv_pyobj()

    def receive_non_block(self, port: int):
        socket = self.pull_sockets.get(port)
        if socket is None:
            socket = self.context.socket(zmq.PULL)
            socket.bind(f"tcp://*:{port}")
            self.pull_sockets[port] = socket
        try:
            return socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
