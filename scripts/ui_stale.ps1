# Exit 1 when the built UI (frontend/dist) is older than any UI source, else 0.
# Used by start.bat to decide whether to rebuild.
$root = Split-Path -Parent $PSScriptRoot
$built = Join-Path $root 'frontend\dist\index.html'
if (-not (Test-Path $built)) { exit 1 }
$stamp = (Get-Item $built).LastWriteTime
$sources = @('src', 'public', 'next.config.ts', 'package.json') | ForEach-Object { Join-Path $root "frontend\$_" }
$newer = Get-ChildItem $sources -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.LastWriteTime -gt $stamp } |
  Select-Object -First 1
if ($newer) { exit 1 }
exit 0
