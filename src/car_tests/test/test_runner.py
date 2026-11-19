"""Exercise signal cleanup on real subprocesses without ROS or a motor."""
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_tests.runner import parser, start, stop_process


def test_cli_options_are_forwarded_without_ros():
    args = parser().parse_args(['straight', '--targets', '1.5', '2.0', '--cruise-s', '.5'])
    assert args.targets == [1.5, 2.] and args.cruise_s == .5


def test_process_interrupt_preserves_zero_tail(tmp_path):
    # Stand in for the installed ROS console script. No ROS imported here.
    path = tmp_path/'tail.txt'
    code = '''import signal, sys, time
from pathlib import Path
done = []
signal.signal(signal.SIGINT, lambda *_: done.append(True))
p = Path(sys.argv[1])
p.write_text("ready\\n")
while not done: time.sleep(.01)
for _ in range(26):
    with p.open("a") as f: f.write("zero\\n")
    time.sleep(.02)
'''
    proc = start([sys.executable, '-c', code, str(path)])
    try:
        end = time.monotonic()+3
        while not path.exists() and time.monotonic() < end:
            time.sleep(.01)
        assert path.exists()
        before = time.monotonic()
        stop_process(proc)
        assert proc.returncode == 0
        assert time.monotonic()-before >= .5
        assert path.read_text().count('zero') == 26
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
