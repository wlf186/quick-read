$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Lock = Get-Content (Join-Path $PSScriptRoot "tools.lock.json") -Raw | ConvertFrom-Json
$DownloadDir = Join-Path $ProjectRoot "runtime\cache\downloads"
$TempRoot = Join-Path $ProjectRoot "runtime\tmp"
$ToolsRoot = Join-Path $ProjectRoot ".tools"
New-Item -ItemType Directory -Force $DownloadDir,$TempRoot,$ToolsRoot | Out-Null

$Architecture = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x86_64" }
$Platform = "windows-$Architecture"

function Get-VerifiedArchive($Tool) {
    $Entry = $Lock.$Tool.$Platform
    if (-not $Entry) { throw "Unsupported Windows platform: $Platform" }
    $Name = Split-Path $Entry.url -Leaf
    $Destination = Join-Path $DownloadDir $Name
    if (Test-Path $Destination) {
        $Actual = (Get-FileHash $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $Entry.sha256) { Remove-Item $Destination -Force }
    }
    if (-not (Test-Path $Destination)) {
        Write-Host "[$Tool] downloading $Name"
        Invoke-WebRequest $Entry.url -OutFile $Destination
    }
    $Actual = (Get-FileHash $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Entry.sha256) { throw "[$Tool] SHA-256 verification failed" }
    return $Destination
}

$FfmpegExe = Join-Path $ToolsRoot "ffmpeg\bin\ffmpeg.exe"
if (-not (Test-Path $FfmpegExe)) {
    $Archive = Get-VerifiedArchive "ffmpeg"
    $Stage = Join-Path $TempRoot ("ffmpeg-install-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force $Stage | Out-Null
    try {
        Expand-Archive $Archive $Stage -Force
        $Source = Get-ChildItem $Stage -Recurse -Filter ffmpeg.exe | Where-Object { $_.Directory.Name -eq "bin" } | Select-Object -First 1
        $Probe = Get-ChildItem $Stage -Recurse -Filter ffprobe.exe | Where-Object { $_.Directory.Name -eq "bin" } | Select-Object -First 1
        if (-not $Source -or -not $Probe) { throw "FFmpeg archive layout is invalid" }
        $Destination = Join-Path $ToolsRoot "ffmpeg\bin"
        New-Item -ItemType Directory -Force $Destination | Out-Null
        Copy-Item $Source.FullName (Join-Path $Destination "ffmpeg.exe")
        Copy-Item $Probe.FullName (Join-Path $Destination "ffprobe.exe")
    } finally { if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force } }
}

$SofficeExe = Join-Path $ToolsRoot "libreoffice\program\soffice.exe"
if (-not (Test-Path $SofficeExe)) {
    $Archive = Get-VerifiedArchive "libreoffice"
    $Stage = Join-Path $TempRoot ("libreoffice-install-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force $Stage | Out-Null
    try {
        $Arguments = @("/a", "`"$Archive`"", "/qn", "TARGETDIR=`"$Stage`"")
        $Process = Start-Process msiexec.exe -ArgumentList $Arguments -Wait -PassThru
        if ($Process.ExitCode -ne 0) { throw "LibreOffice administrative extraction failed: $($Process.ExitCode)" }
        $Source = Get-ChildItem $Stage -Recurse -Filter soffice.exe | Where-Object { $_.Directory.Name -eq "program" } | Select-Object -First 1
        if (-not $Source) { throw "LibreOffice MSI layout is invalid" }
        $InstallRoot = $Source.Directory.Parent.FullName
        Move-Item $InstallRoot (Join-Path $ToolsRoot "libreoffice")
    } finally { if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force } }
}

& $FfmpegExe -version | Select-Object -First 1
$env:SAL_USE_VCLPLUGIN = "svp"
& $SofficeExe --headless --version
Write-Host "Project-local media tools are ready."
