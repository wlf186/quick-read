$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PidFile = Join-Path $ProjectRoot "runtime\run\server.pid"
$Executable = Join-Path $ProjectRoot ".venv\Scripts\sandevistan-read.exe"
if (-not (Test-Path $PidFile)) {
    Write-Host "Sandevistan-Read is not running (no PID file)."
    exit 0
}

$ProcessId = 0
if (-not [int]::TryParse((Get-Content $PidFile -Raw).Trim(), [ref]$ProcessId)) {
    Remove-Item $PidFile -Force
    throw "Removed an invalid PID file; no process was stopped."
}
$Process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if (-not $Process) {
    Remove-Item $PidFile -Force
    Write-Host "Removed a stale PID file; Sandevistan-Read was not running."
    exit 0
}
if ($Process.Path -ne $Executable) {
    Remove-Item $PidFile -Force
    throw "PID $ProcessId belongs to another executable; it was not stopped."
}

Stop-Process -Id $ProcessId
for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
    if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) { break }
    Start-Sleep -Milliseconds 200
}
if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
    throw "Process $ProcessId did not stop. Use Stop-Process -Id $ProcessId -Force after verifying the PID."
}
Remove-Item $PidFile -Force
Write-Host "Sandevistan-Read stopped."
