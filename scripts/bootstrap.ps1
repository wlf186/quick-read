$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:SANDEVISTAN_PROJECT_ROOT = $ProjectRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "runtime\cache\uv"
$env:COREPACK_HOME = Join-Path $ProjectRoot "runtime\cache\corepack"
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot "runtime\cache"
$env:TEMP = Join-Path $ProjectRoot "runtime\tmp"
$ToolDir = Join-Path $ProjectRoot ".tools\bin"
New-Item -ItemType Directory -Force $ToolDir,$env:TEMP | Out-Null
$Uv = Join-Path $ToolDir "uv.exe"
if (-not (Test-Path $Uv)) { Invoke-WebRequest https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip -OutFile "$env:TEMP\uv.zip"; Expand-Archive "$env:TEMP\uv.zip" $ToolDir -Force }
& $Uv sync --extra ai --extra dev
& (Join-Path $ProjectRoot ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "install-models.py")
if ($env:SANDEVISTAN_SKIP_TOOLS -ne "1") { & (Join-Path $PSScriptRoot "install-tools.ps1") }
Push-Location (Join-Path $ProjectRoot "frontend"); corepack pnpm install --store-dir (Join-Path $ProjectRoot "runtime\cache\pnpm-store"); corepack pnpm build; Pop-Location
Write-Host "Ready. Run scripts\start.ps1"
