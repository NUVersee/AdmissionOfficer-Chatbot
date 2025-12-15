import importlib
import traceback
import sys
import os

# Ensure project root is on sys.path so `import src.*` works when running from tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

modules = [
    "src.utils",
    "src.ollama_client",
    "src.ingest",
    "src.query",
]

all_ok = True
for m in modules:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception:
        print(m, "FAILED")
        traceback.print_exc()
        all_ok = False

if not all_ok:
    print("Smoke test detected failures. Install dependencies and re-run the test.")
    sys.exit(2)

print("Smoke test passed: basic imports OK")
