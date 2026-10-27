import subprocess
import sys
import unittest
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[1]

# Runs in a child whose stdin is a pipe held open by the test, as the studio does.
CHILD = r"""
import os, sys, threading, time
from image_engine.__main__ import watch_stdin

closed = threading.Event()
watch_stdin(closed.set)
time.sleep(0.3)

if sys.platform == "win32":
    import ctypes, msvcrt
    answered = threading.Event()
    def query():
        info = (ctypes.c_byte * 64)()
        ctypes.WinDLL("kernel32").GetFileInformationByHandle(
            ctypes.c_void_p(msvcrt.get_osfhandle(0)), info)
        answered.set()
    threading.Thread(target=query, daemon=True).start()
    print("query", "answered" if answered.wait(5) else "blocked", flush=True)

print("eof", "seen" if closed.wait(10) else "missed", flush=True)
os._exit(0)
"""


class ParentWatchTests(unittest.TestCase):
    def run_child(self):
        child = subprocess.Popen([sys.executable, "-c", CHILD], cwd=ENGINE_ROOT, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(child.stdout.close)
        self.addCleanup(child.kill)
        lines = []
        if sys.platform == "win32":
            lines.append(child.stdout.readline().strip())
        child.stdin.close()  # what the studio's exit looks like from here
        lines.append(child.stdout.readline().strip())
        child.wait(15)
        return lines

    @unittest.skipUnless(sys.platform == "win32", "the blocking-read hazard is Windows-specific")
    def test_watching_does_not_block_queries_on_stdin(self):
        # A blocking read here once froze `import torch` inside the engine for good.
        self.assertEqual(self.run_child()[0], "query answered")

    def test_closing_stdin_is_noticed(self):
        self.assertEqual(self.run_child()[-1], "eof seen")


if __name__ == "__main__":
    unittest.main()
