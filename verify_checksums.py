"""Verify archive file SHA-256 manifest, without executing untrusted model files."""
from pathlib import Path
import json,hashlib,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'CHECKSUMS_SHA256.json').read_text())
errors=[]
for file,expected in manifest['files'].items():
    path=root/file
    if not path.is_file():errors.append(f'Missing: {file}')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append(f'Mismatch: {file}')
print(f"Files verified: {len(manifest['files'])-len(errors)}/{len(manifest['files'])}")
if errors:
    print('\n'.join(errors));sys.exit(1)
