#!/usr/bin/env python3
import time
from cuda_stream_player import CUDAStreamPlayer, test_cuda_available

# RTSPS source from your README
SRC = 'rtsps://192.168.0.199:7441/Ep4rnzfdW2oGzpNp?enableSrtp'

print('Testing hardware acceleration availability:')
print(test_cuda_available())

player = CUDAStreamPlayer(source=SRC, width=1280, height=720, fps=30, hwaccel='auto', buffer_size=2)

ok = player.start()
print('player.start() ->', ok)

if not ok:
    print('Start failed; see prior FFmpeg stderr output above.')
else:
    try:
        t0 = time.time()
        while time.time() - t0 < 8.0:
            ret, frame = player.read()
            if ret:
                stats = player.get_stats()
                print(f"Got frame — fps={stats['fps']:.1f}, queue={stats['queue_size']}, drop_rate={stats['drop_rate']:.1f}%")
            else:
                print('No frame available (queue empty)')
            time.sleep(0.25)
    finally:
        player.stop()
        print('Stopped player')
