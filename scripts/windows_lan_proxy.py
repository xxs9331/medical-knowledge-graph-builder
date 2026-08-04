"""Temporary user-space TCP forwarding for a WSL LAN demo."""

from __future__ import annotations

import argparse
import socket
import threading


BUFFER_SIZE = 64 * 1024


def _copy(source: socket.socket, destination: socket.socket) -> None:
    try:
        while data := source.recv(BUFFER_SIZE):
            destination.sendall(data)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _forward(client: socket.socket, target_host: str, target_port: int) -> None:
    with client:
        try:
            upstream = socket.create_connection((target_host, target_port), timeout=10)
        except OSError:
            return
        with upstream:
            upstream.settimeout(None)
            client.settimeout(None)
            request = threading.Thread(target=_copy, args=(client, upstream), daemon=True)
            response = threading.Thread(target=_copy, args=(upstream, client), daemon=True)
            request.start()
            response.start()
            request.join()
            response.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.listen_host, args.listen_port))
        listener.listen(32)
        while True:
            client, _ = listener.accept()
            threading.Thread(
                target=_forward,
                args=(client, args.target_host, args.target_port),
                daemon=True,
            ).start()


if __name__ == "__main__":
    main()
