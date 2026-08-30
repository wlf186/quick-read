$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path; $PidFile = Join-Path $ProjectRoot "runtime\run\server.pid"
if (Test-Path $PidFile) { Stop-Process -Id ([int](Get-Content $PidFile)) -ErrorAction SilentlyContinue; Remove-Item $PidFile }
Write-Host "Sandevistan-Read stopped"
