#!/usr/bin/env python3
"""SSH ProxyCommand transport through the caller's HTTP CONNECT proxy."""
import base64
import os
import selectors
import socket
import sys
from urllib.parse import unquote, urlparse


def main():
    host, port = sys.argv[1:3]
    proxy = urlparse(os.environ.get('https_proxy') or os.environ.get('http_proxy') or '')
    if proxy.scheme != 'http' or not proxy.hostname:
        raise ValueError('Set an HTTP CONNECT proxy in https_proxy or http_proxy')
    tunnel = socket.create_connection((proxy.hostname, proxy.port or 80), timeout=15)
    headers = [f'CONNECT {host}:{port} HTTP/1.1', f'Host: {host}:{port}']
    if proxy.username is not None:
        credential = (unquote(proxy.username) + ':' + unquote(proxy.password or '')).encode()
        headers.append('Proxy-Authorization: Basic ' + base64.b64encode(credential).decode())
    tunnel.sendall(('\r\n'.join(headers) + '\r\n\r\n').encode())
    response = bytearray()
    while not response.endswith(b'\r\n\r\n'):
        chunk = tunnel.recv(1)
        if not chunk or len(response) > 65536:
            raise OSError('Incomplete HTTP CONNECT response')
        response.extend(chunk)
    if response.split(b'\r\n', 1)[0].split()[1] != b'200':
        raise OSError('HTTP CONNECT proxy rejected the tunnel')
    tunnel.settimeout(None)
    with selectors.DefaultSelector() as selector:
        selector.register(tunnel, selectors.EVENT_READ, 'network')
        selector.register(sys.stdin, selectors.EVENT_READ, 'input')
        while True:
            for key, _ in selector.select():
                if key.data == 'input':
                    chunk = os.read(sys.stdin.fileno(), 65536)
                    if chunk:
                        tunnel.sendall(chunk)
                    else:
                        selector.unregister(sys.stdin)
                        tunnel.shutdown(socket.SHUT_WR)
                else:
                    chunk = tunnel.recv(65536)
                    if not chunk:
                        return
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError):
        print('SSH HTTP proxy tunnel failed', file=sys.stderr)
        raise SystemExit(1)
