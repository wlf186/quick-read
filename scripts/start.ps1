$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:SANDEVISTAN_PROJECT_ROOT = $ProjectRoot
$RunDir = Join-Path $ProjectRoot "runtime\run"
$LogDir = Join-Path $ProjectRoot "runtime\logs"
$PidFile = Join-Path $RunDir "server.pid"
$Executable = Join-Path $ProjectRoot ".venv\Scripts\sandevistan-read.exe"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$StdoutLog = Join-Path $LogDir "server.log"
$StderrLog = Join-Path $LogDir "server-error.log"
New-Item -ItemType Directory -Force $RunDir,$LogDir | Out-Null
if (-not (Test-Path $Executable) -or -not (Test-Path $Python)) {
    throw "Application environment is missing. Run .\scripts\bootstrap.ps1 first."
}

if (Test-Path $PidFile) {
    $ExistingId = 0
    if ([int]::TryParse((Get-Content $PidFile -Raw).Trim(), [ref]$ExistingId)) {
        $Existing = Get-Process -Id $ExistingId -ErrorAction SilentlyContinue
        if ($Existing -and $Existing.Path -eq $Executable) {
            Write-Host "Sandevistan-Read is already running (PID $ExistingId)."
            exit 0
        }
    }
    Remove-Item $PidFile -Force
}

$Address = & $Python -c "from sandevistan_read.config import load_config; c=load_config(); print(c.server.host); print(c.server.port)"
if ($LASTEXITCODE -ne 0 -or $Address.Count -lt 2) { throw "Could not read runtime/config.toml" }
$BindHost = $Address[0].Trim()
$BindPort = [int]$Address[1].Trim()
$HealthUrl = "http://127.0.0.1:$BindPort/auth/status"

$Process = Start-Process -FilePath $Executable -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -PassThru -WindowStyle Hidden
$Process.Id | Set-Content $PidFile
$ReadyDeadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $ReadyDeadline) {
    if ($Process.HasExited) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        $Details = if (Test-Path $StderrLog) { (Get-Content $StderrLog -Tail 30) -join [Environment]::NewLine } else { "No error log was written." }
        throw "Server exited before becoming ready (code $($Process.ExitCode)).`n$Details"
    }
    try {
        Invoke-RestMethod -UseBasicParsing -Uri $HealthUrl -TimeoutSec 2 | Out-Null
        Write-Host "Sandevistan-Read is ready at http://${BindHost}:$BindPort (PID $($Process.Id))."
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}

Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
throw "Server did not become ready within 60 seconds. Check $StderrLog"
