"""Run CI build-step shell commands with Docker output staged at its bind mount."""

import os
from pathlib import Path
import subprocess
import sys

import yaml


def test_init_build_steps_leave_artifacts_where_checks_read_them(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((repo / ".github/workflows/ci.yml").read_text())
    steps = workflow["jobs"]["device-firmware-build"]["steps"]
    tools = tmp_path / "tools"
    tools.mkdir()
    docker = tools / "docker"
    # Simulate only the compiler's output file, mapping /emos to host emos/.
    # The actual workflow's mkdir/mv commands still run and must find it.
    docker.write_text(f"#!{sys.executable}\n" + '''
import pathlib, shlex, sys
args = shlex.split(sys.argv[-1])
output = pathlib.Path("emos") / args[args.index("-o") + 1]
output.write_bytes(b"test init")
''')
    docker.chmod(0o755)
    env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])
    builds = [step for step in steps if step.get("name", "").startswith("Build the emOS init (")]
    assert len(builds) == 2
    for step in builds:
        result = subprocess.run(["bash", "-e", "-c", step["run"]],
                                cwd=tmp_path, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    for name in ("init", "init32", "init.init_radar"):
        assert (tmp_path / "emos/build" / name).read_bytes() == b"test init"
