$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:SANDEVISTAN_PROJECT_ROOT = $ProjectRoot
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot ".venv"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "runtime\cache\uv"
$env:COREPACK_HOME = Join-Path $ProjectRoot "runtime\cache\corepack"
$env:XDG_CACHE_HOME = Join-Path $ProjectRoot "runtime\cache"
$env:TEMP = Join-Path $ProjectRoot "runtime\tmp"
$env:TMP = $env:TEMP
$ToolDir = Join-Path $ProjectRoot ".tools\bin"
New-Item -ItemType Directory -Force $ToolDir,$env:TEMP | Out-Null

function Assert-NativeSuccess($Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE" }
}

function Invoke-Download($Url, $Destination) {
    $Partial = "$Destination.part"
    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        try {
            Remove-Item $Partial -Force -ErrorAction SilentlyContinue
            Invoke-WebRequest -UseBasicParsing $Url -OutFile $Partial
            Move-Item $Partial $Destination -Force
            return
        } catch {
            Remove-Item $Partial -Force -ErrorAction SilentlyContinue
            if ($Attempt -eq 3) { throw }
            Write-Warning "Download failed (attempt $Attempt/3): $($_.Exception.Message)"
        }
    }
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js is required. Install Node.js 24 LTS, reopen PowerShell, and run this script again."
}
$NodeVersion = (& node --version).Trim()
Assert-NativeSuccess "node --version"
$NodeMajor = [int](($NodeVersion -replace '^v', '').Split('.')[0])
if ($NodeMajor -lt 20) { throw "Node.js 20 or newer is required; detected $NodeVersion. Node.js 24 LTS is recommended." }
if (-not (Get-Command corepack -ErrorAction SilentlyContinue)) {
    throw "Corepack is required. Run 'npm install -g corepack', reopen PowerShell, and try again."
}
& corepack --version | Out-Null
Assert-NativeSuccess "corepack --version"

if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    Write-Warning "Windows ARM64 support is experimental. The bootstrap currently uses the x64 uv build under Windows emulation."
}

$Uv = Join-Path $ToolDir "uv.exe"
if (-not (Test-Path $Uv)) {
    $UvArchive = Join-Path $env:TEMP "uv-x86_64-pc-windows-msvc.zip"
    Write-Host "[uv] downloading the Python environment manager"
    Invoke-Download "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" $UvArchive
    Expand-Archive $UvArchive $ToolDir -Force
}

& $Uv sync --extra ai --extra dev --frozen
Assert-NativeSuccess "uv sync"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python (Join-Path $PSScriptRoot "install-models.py")
Assert-NativeSuccess "model installation"
if ($env:SANDEVISTAN_SKIP_TOOLS -ne "1") {
    & (Join-Path $PSScriptRoot "install-tools.ps1")
    Assert-NativeSuccess "media tool installation"
}

Push-Location (Join-Path $ProjectRoot "frontend")
try {
    & corepack pnpm install --store-dir (Join-Path $ProjectRoot "runtime\cache\pnpm-store") --frozen-lockfile
    Assert-NativeSuccess "frontend dependency installation"
    & corepack pnpm build
    Assert-NativeSuccess "frontend build"
} finally {
    Pop-Location
}
Write-Host "Ready. Run .\scripts\start.ps1"
