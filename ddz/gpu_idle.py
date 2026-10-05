"""User-requested idle GPU occupation, stopped before real local work."""
import argparse
import os
import signal
import time


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float)
    ap.add_argument('--size', type=int, default=16384)
    args = ap.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one local CUDA device required')
    if 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected the local A100')
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    a = torch.randn((args.size, args.size), device='cuda', dtype=torch.float16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    start = time.monotonic()
    print(f'ready pid={os.getpid()} device={torch.cuda.get_device_name(0)} size={args.size}', flush=True)
    iterations = 0
    while not stop[0] and (args.seconds is None or time.monotonic()-start < args.seconds):
        for _ in range(16):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize()
        iterations += 16
    print(f'stopped iterations={iterations} seconds={time.monotonic()-start:.2f}', flush=True)


if __name__ == '__main__':
    main()
