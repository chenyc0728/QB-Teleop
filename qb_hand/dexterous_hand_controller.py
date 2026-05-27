from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 pyserial。请先执行: pip install pyserial"
    ) from exc


DOF_ORDER: Tuple[str, ...] = (
    "thumb_rotate",
    "thumb_bend",
    "index",
    "middle",
    "ring",
    "little",
)

DOF_ALIAS: Dict[str, str] = {
    "tr": "thumb_rotate",
    "thumb_rotate": "thumb_rotate",
    "thumb_rot": "thumb_rotate",
    "thumb_bend": "thumb_bend",
    "tb": "thumb_bend",
    "index": "index",
    "i": "index",
    "middle": "middle",
    "m": "middle",
    "ring": "ring",
    "r": "ring",
    "little": "little",
    "pinky": "little",
    "l": "little",
}

FINGER_ID_TO_NAME = {
    1: "thumb",
    2: "index",
    3: "middle",
    4: "ring",
    5: "little",
}

DOF_DESCRIPTIONS: Tuple[Tuple[str, str], ...] = (
    ("thumb_rotate", "拇指旋转/内旋，范围 0~90°"),
    ("thumb_bend", "拇指弯曲，范围 0~90°"),
    ("index", "食指弯曲，范围 0~90°"),
    ("middle", "中指弯曲，范围 0~90°"),
    ("ring", "无名指弯曲，范围 0~90°"),
    ("little", "小指弯曲，范围 0~90°"),
)

SUPPORTED_ACTIONS: Tuple[str, ...] = (
    "open",
    "thumb_open",
    "grasp",
    "one",
    "move",
    "stop",
    "resume",
    "angles",
    "force_demo",
    "force_official_demo",
    "force_start",
    "force_stop",
    "force_status",
    "force_set",
    "force_official_set",
    "tactile_once",
    "force_grasp_sequence",
    "force_start_sequence",
    "force_official_start_sequence",
    "motion_demo_sequence",
    "force_control_example",
)


class HandProtocolError(RuntimeError):
    """Raised when a received frame does not satisfy the documented protocol."""


class HandTimeoutError(TimeoutError):
    """Raised when the device does not reply within the timeout window."""


@dataclass
class ForceDOFParam:
    enable: bool = True
    speed: int = 100
    initial_angle: int = 0
    max_angle: int = 90

    def to_bytes(self) -> List[int]:
        return [
            1 if self.enable else 0,
            _clip_u8(self.speed, "speed"),
            _clip_angle(self.initial_angle, "initial_angle"),
            _clip_angle(self.max_angle, "max_angle"),
        ]


@dataclass
class ForceFingerParam(ForceDOFParam):
    threshold: int = 10

    def to_bytes(self) -> List[int]:
        return super().to_bytes() + _u16_be(self.threshold, "threshold")


@dataclass
class ForceControlConfig:
    thumb_rotate: ForceDOFParam = field(default_factory=ForceDOFParam)
    thumb_bend: ForceFingerParam = field(default_factory=ForceFingerParam)
    index: ForceFingerParam = field(default_factory=ForceFingerParam)
    middle: ForceFingerParam = field(default_factory=ForceFingerParam)
    ring: ForceFingerParam = field(default_factory=ForceFingerParam)
    little: ForceFingerParam = field(default_factory=ForceFingerParam)

    def to_payload(self) -> List[int]:
        # Serial/WIFI/Ethernet 0x40 payload layout from the manual:
        # D3 direction + thumb_rotate(4B) + thumb_bend(6B) + index(6B) + middle(6B)
        # + ring(6B) + little(6B)
        return (
            [0x00]
            + self.thumb_rotate.to_bytes()
            + self.thumb_bend.to_bytes()
            + self.index.to_bytes()
            + self.middle.to_bytes()
            + self.ring.to_bytes()
            + self.little.to_bytes()
        )


@dataclass
class HandProgramConfig:
    """Configuration edited in main instead of typed through a console."""

    port: str = "COM3"
    baudrate: int = 115200
    timeout: float = 0.2
    action: str = "open"
    move_angles: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    dofs: Optional[Tuple[str, ...]] = None
    tactile_timeout: float = 2.0
    sequence_delay: float = 1.0
    force_wait_timeout: float = 8.0
    wait_force_ack: bool = False
    use_force_status_query: bool = False
    force_run_seconds: float = 3.0
    init_before_force: bool = True
    use_official_force_config: bool = False


class DexterousHand:
    """
    Serial controller for the dexterous hand using the protocol in
    《灵巧手-通信协议-V1.5》 (serial/WIFI/Ethernet section).

    Notes
    -----
    1. This implementation targets the serial protocol used in the official example.
    2. 0x10 / 0x20 / 0x30 are one-way commands and do not return acknowledgements.
    3. 0x40 / 0x4A / 0x4B / 0xF1 do return replies according to the manual.
    """

    START = 0x5A
    END = 0x5D
    CMD_MOVE = 0x10
    CMD_STOP = 0x20
    CMD_RESUME = 0x30
    CMD_FORCE_SET = 0x40
    CMD_FORCE_SWITCH = 0x4A
    CMD_FORCE_STATUS = 0x4B
    CMD_ANGLE_QUERY = 0xF1

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.2,
        write_timeout: float = 0.2,
        auto_open: bool = True,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.ser: Optional[serial.Serial] = None
        if auto_open:
            self.open()

    def open(self) -> None:
        if self.ser and self.ser.is_open:
            return
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        self.flush()

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "DexterousHand":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------- Low-level helpers -------------------------
    def flush(self) -> None:
        self._ensure_open()
        assert self.ser is not None
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def _ensure_open(self) -> None:
        if self.ser is None or not self.ser.is_open:
            raise RuntimeError("串口未打开。")

    def _build_frame(self, cmd: int, payload: Sequence[int]) -> bytes:
        body = [cmd & 0xFF, (len(payload) + 5) & 0xFF, *[x & 0xFF for x in payload]]
        checksum = sum(body) & 0xFF
        return bytes([self.START, *body, checksum, self.END])

    def _write_frame(self, frame: bytes) -> None:
        self._ensure_open()
        assert self.ser is not None
        self.ser.write(frame)
        self.ser.flush()

    def _read_exact(self, n: int, timeout: Optional[float] = None) -> bytes:
        self._ensure_open()
        assert self.ser is not None

        deadline = None if timeout is None else time.time() + timeout
        chunks = bytearray()
        while len(chunks) < n:
            if deadline is not None:
                remain = deadline - time.time()
                if remain <= 0:
                    raise HandTimeoutError(f"读取超时，期望 {n} 字节，已收到 {len(chunks)} 字节。")
                old_timeout = self.ser.timeout
                self.ser.timeout = min(remain, 0.05)
            else:
                old_timeout = self.ser.timeout

            piece = self.ser.read(n - len(chunks))
            self.ser.timeout = old_timeout
            if piece:
                chunks.extend(piece)
            else:
                if deadline is not None and time.time() >= deadline:
                    raise HandTimeoutError(f"读取超时，期望 {n} 字节，已收到 {len(chunks)} 字节。")
        return bytes(chunks)

    def _read_next_frame(self, timeout: Optional[float] = None) -> bytes:
        self._ensure_open()
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if deadline is not None and time.time() >= deadline:
                raise HandTimeoutError("等待帧头超时。")

            head = self._read_exact(1, timeout=None if deadline is None else max(0.0, deadline - time.time()))
            if head[0] != self.START:
                continue

            hdr = self._read_exact(2, timeout=None if deadline is None else max(0.0, deadline - time.time()))
            total_len = hdr[1]
            if total_len < 5:
                continue
            remaining = self._read_exact(total_len - 3, timeout=None if deadline is None else max(0.0, deadline - time.time()))
            frame = bytes([head[0], hdr[0], hdr[1], *remaining])
            if frame[-1] == self.END:
                return frame

            # Some firmware/manual combinations report an incorrect D2 length for
            # query replies, especially force-status 0x4B. Keep reading to the
            # frame tail so these frames can still be validated by checksum.
            tail = bytearray(remaining)
            while len(tail) < 61:
                try:
                    extra = self._read_exact(1, timeout=None if deadline is None else max(0.0, deadline - time.time()))
                except HandTimeoutError:
                    break
                tail.extend(extra)
                if extra[0] == self.END:
                    return bytes([head[0], hdr[0], hdr[1], *tail])
            continue

    @staticmethod
    def _verify_general_checksum(frame: bytes) -> bool:
        if len(frame) < 5:
            return False
        return (sum(frame[1:-2]) & 0xFF) == frame[-2]

    @staticmethod
    def _verify_tactile_checksum(frame: bytes) -> bool:
        if len(frame) != 64:
            return False
        # manual: D3~D61 cumulative low 8 bits == D62
        return (sum(frame[3:62]) & 0xFF) == frame[62]

    def _recv_matching_frame(
        self,
        expected_cmd: int,
        timeout: float = 1.0,
    ) -> bytes:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._read_next_frame(timeout=max(0.0, deadline - time.time()))
            if len(frame) == 64 and len(frame) >= 4 and frame[3] == 0x83:
                # serial tactile stream can be uploaded actively. Ignore it here.
                continue
            if not self._verify_general_checksum(frame):
                continue
            if frame[1] == expected_cmd:
                return frame
        raise HandTimeoutError(f"等待命令 0x{expected_cmd:02X} 回复超时。")

    def dump_frame_hex(self, frame: bytes) -> str:
        return " ".join(f"{b:02X}" for b in frame)

    # ------------------------- Motion control -------------------------
    def move(self, angles: Sequence[int], enables: Optional[Sequence[bool]] = None) -> bytes:
        if len(angles) != 6:
            raise ValueError("angles 必须提供 6 个角度，顺序为 thumb_rotate, thumb_bend, index, middle, ring, little")
        if enables is None:
            enables = [True] * 6
        if len(enables) != 6:
            raise ValueError("enables 必须提供 6 个布尔值。")

        payload: List[int] = []
        for en, ang in zip(enables, angles):
            payload.extend([1 if en else 0, _clip_angle(ang, "angle")])
        frame = self._build_frame(self.CMD_MOVE, payload)
        self._write_frame(frame)
        return frame

    def _move_raw_positions(self, positions: Sequence[int], enables: Optional[Sequence[bool]] = None) -> bytes:
        if len(positions) != 6:
            raise ValueError("positions 必须提供 6 个原始位置字节。")
        if enables is None:
            enables = [True] * 6
        if len(enables) != 6:
            raise ValueError("enables 必须提供 6 个布尔值。")

        payload: List[int] = []
        for en, pos in zip(enables, positions):
            payload.extend([1 if en else 0, _clip_u8(pos, "position")])
        frame = self._build_frame(self.CMD_MOVE, payload)
        self._write_frame(frame)
        return frame

    def move_named(self, **angles: int) -> bytes:
        current = {name: 0 for name in DOF_ORDER}
        for k, v in angles.items():
            key = normalize_dof_name(k)
            current[key] = v
        return self.move([current[name] for name in DOF_ORDER])

    def emergency_stop(self, dofs: Optional[Iterable[str]] = None) -> bytes:
        enables = self._dofs_to_enable_mask(dofs)
        payload: List[int] = []
        for en in enables:
            payload.extend([1 if en else 0, 0x00])
        frame = self._build_frame(self.CMD_STOP, payload)
        self._write_frame(frame)
        return frame

    def emergency_resume(self, dofs: Optional[Iterable[str]] = None) -> bytes:
        enables = self._dofs_to_enable_mask(dofs)
        payload: List[int] = []
        for en in enables:
            payload.extend([1 if en else 0, 0x00])
        frame = self._build_frame(self.CMD_RESUME, payload)
        self._write_frame(frame)
        return frame

    def preset_open(self) -> bytes:
        # Official example: full open / initial state.
        return self.move([0, 0, 0, 0, 0, 0])

    def preset_thumb_open(self) -> bytes:
        # Official example HandOpen uses raw byte 0x75.
        return self._move_raw_positions([0x75, 0, 0, 0, 0, 0])

    def preset_grasp(self) -> bytes:
        # Safer than the official full-grasp example; use small angles first to avoid collision.
        return self.move([45, 25, 35, 35, 30, 30])

    def preset_one(self) -> bytes:
        # Official example HandOne uses raw byte 0x55.
        return self._move_raw_positions([0x55, 0x55, 0, 0x55, 0x55, 0x55])

    def _dofs_to_enable_mask(self, dofs: Optional[Iterable[str]]) -> List[bool]:
        if dofs is None:
            return [True] * 6
        selected = {normalize_dof_name(x) for x in dofs}
        return [name in selected for name in DOF_ORDER]

    # ------------------------- Force control -------------------------
    def set_force_control(self, config: ForceControlConfig, wait_ack: bool = False) -> Dict[str, object]:
        frame = self._build_frame(self.CMD_FORCE_SET, config.to_payload())
        self._write_frame(frame)
        result: Dict[str, object] = {
            "cmd": self.CMD_FORCE_SET,
            "tx_hex": self.dump_frame_hex(frame),
            "tx_length": len(frame),
        }
        if not wait_ack:
            result["ack_waited"] = False
            return result
        result.update(self._parse_ack(self._recv_matching_frame(self.CMD_FORCE_SET)))
        result["ack_waited"] = True
        return result

    def force_start(self, wait_ack: bool = False) -> Dict[str, object]:
        # Serial protocol: control code 0x01 = start force control.
        frame = self._build_frame(self.CMD_FORCE_SWITCH, [0x00, 0x01])
        self._write_frame(frame)
        result: Dict[str, object] = {
            "cmd": self.CMD_FORCE_SWITCH,
            "control_code": 1,
            "tx_hex": self.dump_frame_hex(frame),
        }
        if not wait_ack:
            result["ack_waited"] = False
            return result
        result.update(self._parse_ack(self._recv_matching_frame(self.CMD_FORCE_SWITCH)))
        result["ack_waited"] = True
        return result

    def force_stop(self, wait_ack: bool = False) -> Dict[str, object]:
        # Serial protocol: control code 0x00 = exit force control and return to 0 degree.
        frame = self._build_frame(self.CMD_FORCE_SWITCH, [0x00, 0x00])
        self._write_frame(frame)
        result: Dict[str, object] = {
            "cmd": self.CMD_FORCE_SWITCH,
            "control_code": 0,
            "tx_hex": self.dump_frame_hex(frame),
        }
        if not wait_ack:
            result["ack_waited"] = False
            return result
        result.update(self._parse_ack(self._recv_matching_frame(self.CMD_FORCE_SWITCH)))
        result["ack_waited"] = True
        return result

    def query_force_status(self) -> Dict[str, object]:
        frame = self._build_frame(self.CMD_FORCE_STATUS, [0x00, 0x00])
        self._write_frame(frame)
        reply = self._recv_matching_frame(self.CMD_FORCE_STATUS)
        if not self._verify_general_checksum(reply):
            raise HandProtocolError("力控状态回复校验失败。")
        if len(reply) < 12:
            raise HandProtocolError(f"力控状态回复长度异常: {len(reply)}")
        state_bytes = reply[4:10]
        return {
            "raw_hex": self.dump_frame_hex(reply),
            "is_executing": dict(zip(DOF_ORDER, [bool(x) for x in state_bytes])),
            "state_bytes": list(state_bytes),
        }

    def query_angles(self) -> Dict[str, object]:
        frame = self._build_frame(self.CMD_ANGLE_QUERY, [0x00, 0x00])
        self._write_frame(frame)
        reply = self._recv_matching_frame(self.CMD_ANGLE_QUERY)
        if not self._verify_general_checksum(reply):
            raise HandProtocolError("角度查询回复校验失败。")
        if len(reply) < 12:
            raise HandProtocolError(f"角度查询回复长度异常: {len(reply)}")
        angles = reply[4:10]
        return {
            "raw_hex": self.dump_frame_hex(reply),
            "angles": dict(zip(DOF_ORDER, [int(x) for x in angles])),
            "angle_bytes": list(angles),
        }

    def wait_until_force_idle(
        self,
        timeout: float = 5.0,
        poll_interval: float = 0.1,
        initial_delay: float = 0.2,
    ) -> Dict[str, object]:
        if initial_delay > 0:
            time.sleep(initial_delay)
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            last = self.query_force_status()
            states = last["is_executing"]
            assert isinstance(states, dict)
            if not any(states.values()):
                return last
            time.sleep(poll_interval)
        raise HandTimeoutError(f"等待力控执行完成超时，最后一次状态: {last}")

    # ------------------------- Tactile stream -------------------------
    def read_tactile_frame(self, timeout: float = 1.0) -> Dict[str, object]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._read_next_frame(timeout=max(0.0, deadline - time.time()))
            if len(frame) != 64:
                continue
            if frame[3] != 0x83:
                continue
            if not self._verify_tactile_checksum(frame):
                continue
            finger_id = frame[1]
            values = []
            for i in range(4, 36, 2):
                values.append((frame[i] << 8) | frame[i + 1])
            return {
                "raw_hex": self.dump_frame_hex(frame),
                "finger_id": finger_id,
                "finger_name": FINGER_ID_TO_NAME.get(finger_id, f"unknown_{finger_id}"),
                "points": values,
                "reserved": list(frame[36:62]),
            }
        raise HandTimeoutError("未收到有效触觉上传帧。")

    # ------------------------- Parsers -------------------------
    def _parse_ack(self, reply: bytes) -> Dict[str, object]:
        if not self._verify_general_checksum(reply):
            raise HandProtocolError("应答校验失败。")
        if len(reply) < 7:
            raise HandProtocolError(f"应答长度异常: {len(reply)}")
        status = reply[4]
        return {
            "cmd": reply[1],
            "length": reply[2],
            "direction": reply[3],
            "status": status,
            "ok": status == 0,
            "raw_hex": self.dump_frame_hex(reply),
        }


# ------------------------- Presets and examples -------------------------
def default_force_grasp_config() -> ForceControlConfig:
    """
    A conservative force-control example derived from the manual + official example.

    Notes
    -----
    - The manual says typical speed range is 20~200 and angle range is 0~90.
    - Angles are intentionally conservative to reduce collision/interference risk.
    - The manual text says threshold range is 20~1000, but the manual's own example
      and the official code both use threshold=10. Therefore this implementation does
      not hard-reject threshold=10.
    """
    return ForceControlConfig(
        thumb_rotate=ForceDOFParam(enable=True, speed=80, initial_angle=35, max_angle=35),
        thumb_bend=ForceFingerParam(enable=True, speed=80, initial_angle=0, max_angle=25, threshold=10),
        index=ForceFingerParam(enable=True, speed=80, initial_angle=0, max_angle=30, threshold=10),
        middle=ForceFingerParam(enable=True, speed=80, initial_angle=0, max_angle=30, threshold=10),
        ring=ForceFingerParam(enable=True, speed=80, initial_angle=0, max_angle=25, threshold=10),
        little=ForceFingerParam(enable=True, speed=80, initial_angle=0, max_angle=25, threshold=10),
    )


def official_force_grasp_config() -> ForceControlConfig:
    """Force-control parameters decoded from the official hand.py example."""

    return ForceControlConfig(
        thumb_rotate=ForceDOFParam(enable=True, speed=100, initial_angle=75, max_angle=75),
        thumb_bend=ForceFingerParam(enable=True, speed=100, initial_angle=0, max_angle=40, threshold=10),
        index=ForceFingerParam(enable=True, speed=100, initial_angle=0, max_angle=40, threshold=10),
        middle=ForceFingerParam(enable=True, speed=100, initial_angle=0, max_angle=90, threshold=10),
        ring=ForceFingerParam(enable=True, speed=100, initial_angle=0, max_angle=90, threshold=10),
        little=ForceFingerParam(enable=True, speed=100, initial_angle=0, max_angle=90, threshold=10),
    )


# ------------------------- Main-function action selection -------------------------
PROGRAM_CONFIG = HandProgramConfig(
    port="COM3",
    baudrate=115200,
    timeout=0.2,
    # Change this value in the main file to choose a function.
    # Supported values are listed in SUPPORTED_ACTIONS.
    action="force_control_example",
    # Used only when action="move".
    move_angles=(0, 0, 0, 0, 0, 0),
    # Used only when action="stop" or action="resume".
    # None means all DOFs; examples: ("index", "middle") or ("tr", "tb").
    dofs=None,
    tactile_timeout=2.0,
    sequence_delay=1.0,
    force_wait_timeout=8.0,
    wait_force_ack=False,
    use_force_status_query=False,
    force_run_seconds=3.0,
    init_before_force=True,
    use_official_force_config=False,
)


def run_action(hand: DexterousHand, config: HandProgramConfig) -> object:
    action = config.action.strip().lower()
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"未知功能: {config.action}。可选功能: {', '.join(SUPPORTED_ACTIONS)}")

    if action == "open":
        return hand.preset_open()
    if action == "thumb_open":
        return hand.preset_thumb_open()
    if action == "grasp":
        return hand.preset_grasp()
    if action == "one":
        return hand.preset_one()
    if action == "move":
        return hand.move(config.move_angles)
    if action == "stop":
        return hand.emergency_stop(config.dofs)
    if action == "resume":
        return hand.emergency_resume(config.dofs)
    if action == "angles":
        return hand.query_angles()
    if action == "force_status":
        return hand.query_force_status()
    if action in {"force_demo", "force_set"}:
        return hand.set_force_control(default_force_grasp_config(), wait_ack=config.wait_force_ack)
    if action in {"force_official_demo", "force_official_set"}:
        return hand.set_force_control(official_force_grasp_config(), wait_ack=config.wait_force_ack)
    if action == "force_start":
        return hand.force_start(wait_ack=config.wait_force_ack)
    if action == "force_stop":
        return hand.force_stop(wait_ack=config.wait_force_ack)
    if action == "tactile_once":
        return hand.read_tactile_frame(timeout=config.tactile_timeout)
    if action == "force_grasp_sequence":
        return run_force_grasp_sequence(
            hand,
            wait_timeout=config.force_wait_timeout,
            wait_ack=config.wait_force_ack,
        )
    if action == "force_start_sequence":
        return run_force_start_sequence(
            hand,
            config=default_force_grasp_config(),
            wait_ack=config.wait_force_ack,
            delay=config.sequence_delay,
            init_first=config.init_before_force,
        )
    if action == "force_official_start_sequence":
        return run_force_start_sequence(
            hand,
            config=official_force_grasp_config(),
            wait_ack=config.wait_force_ack,
            delay=config.sequence_delay,
            init_first=config.init_before_force,
        )
    if action == "motion_demo_sequence":
        return run_motion_demo_sequence(hand, delay=config.sequence_delay)
    if action == "force_control_example":
        return run_force_control_example(
            hand,
            wait_timeout=config.force_wait_timeout,
            wait_ack=config.wait_force_ack,
            use_status_query=config.use_force_status_query,
            run_seconds=config.force_run_seconds,
            init_first=config.init_before_force,
            force_config=official_force_grasp_config() if config.use_official_force_config else default_force_grasp_config(),
        )
    raise AssertionError(f"未处理的功能: {action}")


def run_motion_demo_sequence(hand: DexterousHand, delay: float = 1.0) -> List[Dict[str, object]]:
    """连续使用多个普通功能的示例：张开、小角度运动、急停、恢复、回零。"""

    results: List[Dict[str, object]] = []

    frame = hand.preset_open()
    results.append({"step": "open", "tx": hand.dump_frame_hex(frame)})
    time.sleep(delay)

    frame = hand.move([20, 0, 0, 0, 0, 0])
    results.append({"step": "move_thumb_rotate_20", "tx": hand.dump_frame_hex(frame)})
    time.sleep(delay)

    frame = hand.move([20, 10, 20, 20, 10, 10])
    results.append({"step": "move_small_grasp", "tx": hand.dump_frame_hex(frame)})
    time.sleep(delay)

    frame = hand.emergency_stop(("index", "middle"))
    results.append({"step": "stop_index_middle", "tx": hand.dump_frame_hex(frame)})
    time.sleep(delay)

    frame = hand.emergency_resume(("index", "middle"))
    results.append({"step": "resume_index_middle", "tx": hand.dump_frame_hex(frame)})
    time.sleep(delay)

    frame = hand.preset_open()
    results.append({"step": "open_end", "tx": hand.dump_frame_hex(frame)})
    return results


def run_force_grasp_sequence(
    hand: DexterousHand,
    wait_timeout: float = 8.0,
    wait_ack: bool = False,
) -> List[object]:
    """Common force-control flow from the protocol manual."""

    results: List[object] = []
    results.append(hand.set_force_control(default_force_grasp_config(), wait_ack=wait_ack))
    results.append(hand.force_start(wait_ack=wait_ack))
    results.append(hand.wait_until_force_idle(timeout=wait_timeout))
    results.append(hand.force_stop(wait_ack=wait_ack))
    return results


def run_force_start_sequence(
    hand: DexterousHand,
    config: ForceControlConfig,
    wait_ack: bool = False,
    delay: float = 0.2,
    init_first: bool = True,
) -> List[Dict[str, object]]:
    """Send force parameters and start in one serial session, like official hand.py."""

    results: List[Dict[str, object]] = []
    if init_first:
        frame = hand.preset_open()
        results.append({"step": "init_open", "tx": hand.dump_frame_hex(frame)})
        if delay > 0:
            time.sleep(delay)
    results.append({
        "step": "set_force_control",
        "reply": hand.set_force_control(config, wait_ack=wait_ack),
    })
    if delay > 0:
        time.sleep(delay)
    results.append({"step": "force_start", "reply": hand.force_start(wait_ack=wait_ack)})
    return results


def run_force_control_example(
    hand: DexterousHand,
    wait_timeout: float = 8.0,
    wait_ack: bool = False,
    use_status_query: bool = False,
    run_seconds: float = 3.0,
    init_first: bool = True,
    force_config: Optional[ForceControlConfig] = None,
) -> List[Dict[str, object]]:
    """力控模式示例。默认不依赖 4B 状态查询，兼容 official hand.py behavior."""

    results: List[Dict[str, object]] = []
    if init_first:
        frame = hand.preset_open()
        results.append({"step": "init_open", "tx": hand.dump_frame_hex(frame)})
        time.sleep(0.2)
    if force_config is None:
        force_config = default_force_grasp_config()
    results.append({
        "step": "set_force_control",
        "reply": hand.set_force_control(force_config, wait_ack=wait_ack),
    })
    results.append({"step": "force_start", "reply": hand.force_start(wait_ack=wait_ack)})
    if use_status_query:
        results.append({"step": "wait_until_idle", "reply": hand.wait_until_force_idle(timeout=wait_timeout)})
        results.append({"step": "query_angles", "reply": hand.query_angles()})
    else:
        if run_seconds > 0:
            time.sleep(run_seconds)
        results.append({
            "step": "skip_status_query",
            "reply": f"未查询 4B 状态，已等待 {run_seconds:.1f}s。",
        })
    results.append({"step": "force_stop", "reply": hand.force_stop(wait_ack=wait_ack)})
    return results


def print_result(hand: DexterousHand, result: object) -> None:
    if isinstance(result, bytes):
        print("TX:", hand.dump_frame_hex(result))
        return
    if isinstance(result, list):
        for index, item in enumerate(result, start=1):
            print(f"[{index}] {item}")
        return
    print(result)


def main(config: HandProgramConfig = PROGRAM_CONFIG) -> int:
    print(f"串口: {config.port}, 波特率: {config.baudrate}, 功能: {config.action}")
    with DexterousHand(config.port, baudrate=config.baudrate, timeout=config.timeout) as hand:
        result = run_action(hand, config)
        print_result(hand, result)
    return 0


# ------------------------- Utility functions -------------------------
def normalize_dof_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DOF_ALIAS:
        raise ValueError(f"未知自由度名称: {name}")
    return DOF_ALIAS[key]


def _clip_u8(value: int, field_name: str) -> int:
    if not (0 <= int(value) <= 255):
        raise ValueError(f"{field_name} 必须在 [0, 255] 内，当前={value}")
    return int(value)


def _clip_angle(value: int, field_name: str) -> int:
    if not (0 <= int(value) <= 90):
        raise ValueError(f"{field_name} 必须在 [0, 90] 度内，当前={value}")
    return int(value)


def _u16_be(value: int, field_name: str) -> List[int]:
    ivalue = int(value)
    if not (0 <= ivalue <= 65535):
        raise ValueError(f"{field_name} 必须在 [0, 65535] 内，当前={value}")
    return [(ivalue >> 8) & 0xFF, ivalue & 0xFF]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"程序异常: {exc}", file=sys.stderr)
        raise
