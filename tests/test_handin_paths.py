import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
GRADED = (
    REPO / "REPORT.md",
    REPO / "submission" / "README.md",
    REPO / "submission" / "report" / "report.typ",
)


def root_assigned_by(script: pathlib.Path) -> pathlib.Path:
    setup = ["import pathlib"]
    for line in script.read_text().splitlines():
        if line.startswith("ROOT = "):
            setup.append(line)
            break
    else:
        raise AssertionError(f"{script} has no ROOT assignment")
    namespace = {"__file__": str(script)}
    exec("\n".join(setup), namespace)
    return namespace["ROOT"]


def test_archive_comparison_root_contains_hedging():
    script = REPO / "archive" / "scripts" / "comparison_study.py"
    root = root_assigned_by(script)
    assert (root / "hedging" / "__init__.py").is_file(), root


def test_submission_code_loads_frozen_band(monkeypatch):
    code = REPO / "submission" / "code"
    monkeypatch.chdir(code)
    monkeypatch.syspath_prepend(str(code))
    stale = [name for name in sys.modules if name == "hedging" or name.startswith("hedging.")]
    saved = {name: sys.modules.pop(name) for name in stale}
    try:
        from hedging.bench import load_classical_best

        best = load_classical_best()
    finally:
        for name in list(sys.modules):
            if name == "hedging" or name.startswith("hedging."):
                del sys.modules[name]
        sys.modules.update(saved)
    assert best["strategy"] == "band"
    assert best["delta_band"] == 0.1


def _python_commands(text: str) -> list[str]:
    logical = text.replace("\\\n", "")
    paths = []
    marker = "uv run python "
    for line in logical.splitlines():
        start = 0
        while True:
            found = line.find(marker, start)
            if found < 0:
                break
            token = line[found + len(marker):].strip().split()[0].strip("'\"")
            paths.append(token)
            start = found + len(marker)
    return paths


def test_graded_python_commands_name_existing_files():
    missing = []
    seen = []
    for path in GRADED:
        for command in _python_commands(path.read_text()):
            seen.append(command)
            if not (REPO / command).is_file():
                missing.append(f"{path.relative_to(REPO)}: {command}")
    assert seen, "graded docs name no uv run python commands"
    assert missing == [], missing
