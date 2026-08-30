$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:SANDEVISTAN_PROJECT_ROOT = $ProjectRoot
$RunDir = Join-Path $ProjectRoot "runtime\run"; $LogDir = Join-Path $ProjectRoot "runtime\logs"
New-Item -ItemType Directory -Force $RunDir,$LogDir | Out-Null
$Process = Start-Process -FilePath (Join-Path $ProjectRoot ".venv\Scripts\sandevistan-read.exe") -RedirectStandardOutput (Join-Path $LogDir "server.log") -RedirectStandardError (Join-Path $LogDir "server-error.log") -PassThru -WindowStyle Hidden
$Process.Id | Set-Content (Join-Path $RunDir "server.pid")
Write-Host "Sandevistan-Read started with runtime/config.toml (local access defaults to http://127.0.0.1:20830)"
