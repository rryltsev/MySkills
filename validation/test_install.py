"""Check portable installation, conflict preflight and preservation of changed skills."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.dest = self.base / "new machine" / ".agents" / "skills"

    def invoke(self, *args, expected=0):
        run = subprocess.run([sys.executable, str(ROOT / "tools/install.py"), "--dest",
                              str(self.dest), *args], capture_output=True, text=True)
        self.assertEqual(run.returncode, expected, (run.stdout, run.stderr))
        return run.stdout

    def test_clean_install_and_update(self):
        self.invoke("--check", expected=1)
        self.assertFalse(self.dest.exists())
        self.invoke()
        self.invoke("--check")
        for skill in (ROOT / "skills").glob("*/SKILL.md"):
            target = self.dest / skill.parent.name
            self.assertEqual((target / "SKILL.md").read_bytes(), skill.read_bytes())
            pointer = (target / "collection-root.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(Path(pointer), ROOT)
        copy = self.dest / "lammps" / "my-notes.txt"
        copy.write_text("local edits", encoding="utf-8")
        self.invoke("--check", "lammps", expected=1)
        self.invoke("lammps", expected=2)
        self.invoke("--update", "lammps")
        self.assertFalse(copy.exists())
        saved = list((self.dest.parent / "myskills-backups").rglob("my-notes.txt"))
        self.assertEqual([p.read_text() for p in saved], ["local edits"])
        self.invoke("--check")

    def test_unmanaged_lookalike_preserved(self):
        target = self.dest / "mlip-training"
        target.mkdir(parents=True)
        (target / "collection-root.txt").write_text("my unrelated notes")
        (target / ".myskills-install.json").write_text("{}")
        self.invoke("--update", "lammps", "mlip-training", expected=2)
        self.assertFalse((self.dest / "lammps").exists())
        self.assertEqual((target / "collection-root.txt").read_text(), "my unrelated notes")


if __name__ == "__main__":
    unittest.main()
