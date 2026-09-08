from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent

pairs = [
    (
        ROOT / "config" / "generation.example.json",
        ROOT / "config" / "generation.json",
    ),
    (
        ROOT / "workflows" / "anime.example.json",
        ROOT / "workflows" / "anime.json",
    ),
]

for source, target in pairs:
    if not target.exists():
        shutil.copy2(source, target)
        print(f"Created: {target}")

for folder in [
    ROOT / "data",
    ROOT / "data" / "generated",
    ROOT / "data" / "uploads",
    ROOT / "data" / "runs",
    ROOT / "data" / "exports",
    ROOT / "data" / "backups",
]:
    folder.mkdir(parents=True, exist_ok=True)

print("Local project files are ready.")
