import argparse
import json
import socket
import time
from typing import Any, Dict

from real_robot_utils import TwoPinGPIOGripper, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Jetson GPIO 夹爪代理服务")
    parser.add_argument("--config", required=True, help="代理服务 JSON 配置文件路径")
    return parser.parse_args()


def resolve_proxy_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    server_cfg = payload.get("server", payload.get("gripper_proxy_server", {}))
    if "gripper" in payload:
        gripper_cfg = payload["gripper"]
        default_fraction = float(payload.get("default_gripper_fraction", 0.0))
    else:
        robot_cfg = payload.get("robot", {})
        gripper_cfg = robot_cfg.get("gripper", {})
        default_fraction = float(robot_cfg.get("default_gripper_fraction", 0.0))
    if not gripper_cfg.get("enabled", True):
        raise ValueError("代理服务配置里的 gripper.enabled 不能为 false")
    gripper_type = str(gripper_cfg.get("type", "twopin_gpio")).lower()
    if gripper_type not in {"twopin_gpio", "two_pin_gpio"}:
        raise ValueError("夹爪代理服务只支持 twopin_gpio 本地 GPIO 模式")
    return {
        "host": str(server_cfg.get("host", "0.0.0.0")),
        "port": int(server_cfg.get("port", 8765)),
        "poll_hz": max(float(server_cfg.get("poll_hz", 50.0)), 1.0),
        "gripper_cfg": gripper_cfg,
        "default_fraction": default_fraction,
    }


def make_state_response(gripper: TwoPinGPIOGripper) -> Dict[str, Any]:
    fraction = float(gripper.get_fraction())
    return {
        "ok": True,
        "fraction": fraction,
        "timestamp": time.time(),
    }


def handle_command(gripper: TwoPinGPIOGripper, request: Dict[str, Any]) -> Dict[str, Any]:
    command = str(request.get("command", "state")).lower()
    if command == "ping":
        return {"ok": True, "message": "pong", "timestamp": time.time()}
    if command == "state":
        return make_state_response(gripper)
    if command == "open":
        gripper.open(wait=bool(request.get("wait", False)))
        return make_state_response(gripper)
    if command == "close":
        gripper.close_gripper(wait=bool(request.get("wait", False)))
        return make_state_response(gripper)
    if command == "stop":
        gripper.stop()
        return make_state_response(gripper)
    if command == "set_fraction":
        fraction = float(request.get("fraction", 0.0))
        gripper.set_fraction(fraction, wait=bool(request.get("wait", False)))
        return make_state_response(gripper)
    raise ValueError(f"不支持的命令: {command}")


def serve_forever(host: str, port: int, gripper: TwoPinGPIOGripper, poll_hz: float) -> None:
    poll_timeout_s = 1.0 / poll_hz
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(8)
    server.settimeout(poll_timeout_s)
    print(f"[gripper-proxy] listening on {host}:{port}")

    try:
        while True:
            gripper.update()
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
            with conn:
                conn.settimeout(2.0)
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                raw = data.split(b"\n", 1)[0].decode("utf-8").strip()
                if not raw:
                    continue
                try:
                    request = json.loads(raw)
                    response = handle_command(gripper, request)
                except Exception as exc:
                    response = {
                        "ok": False,
                        "error": str(exc),
                        "timestamp": time.time(),
                    }
                conn.sendall((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                print(f"[gripper-proxy] {addr[0]}:{addr[1]} -> {response}")
    finally:
        try:
            server.close()
        except Exception:
            pass
        gripper.disconnect()


def main():
    args = parse_args()
    payload = load_json(args.config)
    proxy_cfg = resolve_proxy_config(payload)
    gripper = TwoPinGPIOGripper(
        proxy_cfg["gripper_cfg"],
        default_fraction=proxy_cfg["default_fraction"],
    )
    gripper.connect()
    serve_forever(
        host=proxy_cfg["host"],
        port=proxy_cfg["port"],
        gripper=gripper,
        poll_hz=proxy_cfg["poll_hz"],
    )


if __name__ == "__main__":
    main()
