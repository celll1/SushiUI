import os
import subprocess
import sys


API_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"
)
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)

from restart_backend_helper import (  # noqa: E402
    build_helper_command,
    launch_backend_after_parent_exit,
)


def test_helper_command_uses_explicit_argv_and_same_python(tmp_path):
    python_executable = tmp_path / "venv" / "Scripts" / "python.exe"
    helper_path = tmp_path / "backend" / "api" / "restart_backend_helper.py"
    main_path = tmp_path / "backend" / "main.py"
    backend_dir = tmp_path / "backend"

    command = build_helper_command(
        python_executable=str(python_executable),
        helper_path=str(helper_path),
        parent_pid=1234,
        main_path=str(main_path),
        backend_dir=str(backend_dir),
    )

    assert command == [
        os.path.abspath(python_executable),
        os.path.abspath(helper_path),
        "--parent-pid",
        "1234",
        "--python-executable",
        os.path.abspath(python_executable),
        "--main-path",
        os.path.abspath(main_path),
        "--backend-dir",
        os.path.abspath(backend_dir),
    ]


def test_replacement_waits_for_parent_before_launch(tmp_path):
    events = []
    python_executable = tmp_path / "venv" / "Scripts" / "python.exe"
    main_path = tmp_path / "backend" / "main.py"
    backend_dir = tmp_path / "backend"
    sentinel = object()

    def fake_wait(pid):
        events.append(("wait", pid))

    def fake_popen(argv, **kwargs):
        events.append(("popen", argv, kwargs))
        return sentinel

    result = launch_backend_after_parent_exit(
        parent_pid=4321,
        python_executable=str(python_executable),
        main_path=str(main_path),
        backend_dir=str(backend_dir),
        wait_fn=fake_wait,
        popen=fake_popen,
    )

    assert result is sentinel
    assert events[0] == ("wait", 4321)
    assert events[1][0] == "popen"
    assert events[1][1] == [
        os.path.abspath(python_executable),
        os.path.abspath(main_path),
    ]
    assert events[1][2] == {
        "cwd": os.path.abspath(backend_dir),
        "creationflags": subprocess.CREATE_NEW_CONSOLE,
        "close_fds": True,
    }
