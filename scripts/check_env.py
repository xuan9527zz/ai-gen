from pathlib import Path
import json
import sys
import requests

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "generation.json"

print("=" * 64)
print("ILLUSTRIOUS STUDIO ENVIRONMENT CHECK")
print("=" * 64)

if not CONFIG.exists():
    print("[FAIL] config/generation.json is missing")
    raise SystemExit(1)

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))

def resolve(raw):
    p = Path(str(raw))
    return p if p.is_absolute() else (ROOT / p).resolve()

workflow = resolve(cfg["workflow_path"])

print(f"[{'OK' if workflow.exists() else 'FAIL'}] Workflow: {workflow}")

for name, url in [
    ("ComfyUI", cfg.get("comfy_url", "http://127.0.0.1:8188")),
    ("Ollama", "http://127.0.0.1:11434"),
]:
    try:
        endpoint = (
            f"{url.rstrip('/')}/system_stats"
            if name == "ComfyUI"
            else f"{url.rstrip('/')}/api/tags"
        )
        r = requests.get(endpoint, timeout=5)
        r.raise_for_status()
        print(f"[OK] {name}: {url}")
    except Exception as exc:
        print(f"[WARN] {name}: {url} ({type(exc).__name__})")

print()
print("Warnings do not prevent Studio from starting, but analysis/generation")
print("will need the corresponding local service.")
