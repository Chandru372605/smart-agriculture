"""
Package AgroSense project into a distributable zip.
Excludes: venv/, __pycache__/, *.pyc, .git/, *.keras (large TF model)
Includes: all source code, trained .pkl models, datasets CSVs, templates, static files.
"""
import os
import zipfile
from pathlib import Path

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)
ZIP_PATH = OUT_DIR / 'agrosense_project.zip'

# Patterns to skip
SKIP_DIRS  = {'venv', '__pycache__', '.git', '.gemini', 'outputs', 'disease_images'}
SKIP_EXTS  = {'.pyc', '.pyo', '.log', '.tmp'}
SKIP_FILES = {'agrosense_project.zip'}

# Large model files — skip to keep zip under ~50MB
# Remove 'market_lstm.keras' from skip if you want to include it
SKIP_LARGE = {'market_lstm.keras'}

included = []
skipped  = []

print("Packaging AgroSense project...")
print(f"Output: {ZIP_PATH}\n")

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for path in sorted(ROOT.rglob('*')):
        # Skip directories themselves (only add files)
        if path.is_dir():
            continue

        rel = path.relative_to(ROOT)
        parts = rel.parts

        # Skip by directory name
        if any(p in SKIP_DIRS for p in parts):
            skipped.append(str(rel))
            continue

        # Skip by extension
        if path.suffix in SKIP_EXTS:
            skipped.append(str(rel))
            continue

        # Skip by filename
        if path.name in SKIP_FILES or path.name in SKIP_LARGE:
            skipped.append(str(rel))
            continue

        zf.write(path, rel)
        size_kb = path.stat().st_size / 1024
        included.append((str(rel), size_kb))
        print(f"  + {rel}  ({size_kb:.1f} KB)")

zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"\n{'='*55}")
print(f"  Included : {len(included)} files")
print(f"  Skipped  : {len(skipped)} files  (venv, cache, etc.)")
print(f"  ZIP size : {zip_size_mb:.2f} MB")
print(f"  Saved to : {ZIP_PATH}")
print('='*55)
