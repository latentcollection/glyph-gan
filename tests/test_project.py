"""Repository invariants: the notebook and the dependency lists stay honest."""

import json
import re
from pathlib import Path

import glyphgan as gg

ROOT = Path(__file__).resolve().parent.parent


def _notebook():
    return json.loads((ROOT / "glyphgan.ipynb").read_text())


def _source(nb):
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_notebook_is_valid():
    nb = _notebook()
    assert nb["nbformat"] == 4, nb["nbformat"]
    assert nb["cells"], "notebook has no cells"
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown"), cell["cell_type"]
        assert isinstance(cell["source"], list)


def test_notebook_api_calls_resolve():
    """The notebook is the interface; renaming the module must not break it."""
    missing = [
        name
        for name in sorted(set(re.findall(r"gg\.(\w+)", _source(_notebook()))))
        if not hasattr(gg, name)
    ]
    assert not missing, f"notebook calls names the module no longer has: {missing}"


def test_notebook_carries_no_dead_setup():
    """Kaggle downloads and hardcoded .cuda() were removed; keep them gone."""
    src = _source(_notebook()).lower()
    for dead in ("kaggle", ".cuda()", "drive.mount"):
        assert dead not in src, f"notebook still references {dead!r}"


def _top_level(spec_lines):
    names = set()
    for line in spec_lines:
        line = line.split("#")[0].strip().strip('",')
        if not line:
            continue
        name = re.split(r"[<>=;\[ ]", line)[0].strip().lower()
        if name:
            names.add(name.replace("_", "-"))
    return names


def test_requirements_matches_pyproject():
    """requirements.txt is a hand-kept mirror for pip-only environments.

    Nothing else enforces that the two agree, and a package added to one and
    not the other fails only on someone else's machine.
    """
    pyproject = (ROOT / "pyproject.toml").read_text()
    block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    declared = _top_level(block.splitlines())
    mirrored = _top_level((ROOT / "requirements.txt").read_text().splitlines())
    assert declared == mirrored, (
        f"only in pyproject: {sorted(declared - mirrored)}; "
        f"only in requirements: {sorted(mirrored - declared)}"
    )
