#!/usr/bin/env python3
import os
import pathlib
import pty
import select
import subprocess
import tempfile
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
MERGES = ROOT / "howru.merges"
CORPUS = ROOT / "howru.txt"
PROMPTS = [
    "how r u",
    "i miss leo",
]


def compile_c():
    c_bin = pathlib.Path("/tmp/howru_generation_regression")
    proc = subprocess.run(
        ["cc", "howru.c", "-O2", "-std=c11", "-lm", "-lsqlite3", "-o", str(c_bin)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(f"C compile failed\n{proc.stderr}")
    return c_bin


def run_prompt(cmd, prompt, cwd, timeout=30):
    master, slave = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        text=False,
        close_fds=True,
    )
    os.close(slave)
    sent = False
    buf = ""
    started = time.time()
    try:
        while time.time() - started < timeout:
            r, _, _ = select.select([master], [], [], 0.5)
            if not r:
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(master, 4096).decode("utf-8", errors="replace")
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            buf += chunk
            if not sent and "HUMAN:" in buf:
                os.write(master, (prompt + "\n").encode("utf-8"))
                sent = True
            if sent and "/RESONATING:" in buf and len(buf.split("/RESONATING:", 1)[1].strip()) > 16:
                os.write(master, b"quit\n")
                proc.terminate()
                return buf
        proc.terminate()
        raise AssertionError(f"timeout waiting for Howru output\n{buf[:4000]}")
    finally:
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        os.close(master)


def main():
    c_bin = compile_c()
    workspace = pathlib.Path(tempfile.mkdtemp(prefix="howru_regression_"))
    try:
        (workspace / "howru.merges").write_bytes(MERGES.read_bytes())
        (workspace / "howru.txt").write_bytes(CORPUS.read_bytes())
        for prompt in PROMPTS:
            full = run_prompt([str(c_bin), "howru.merges", "howru.txt"], prompt, workspace)
            if "bad Howru weight magic" in full:
                raise AssertionError(full[:4000])
            print(f"[ok] c-meta :: {prompt}")
        return 0
    finally:
        for path in workspace.glob("*"):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            workspace.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
