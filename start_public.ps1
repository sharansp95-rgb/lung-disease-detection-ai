<#
.SYNOPSIS
    Runs the whole app on this laptop and publishes it through an ngrok tunnel.

.DESCRIPTION
    1. Builds the frontend in "public" mode (reads frontend/.env.public.local).
    2. Starts the FastAPI backend on 127.0.0.1:8000 (never exposed directly).
    3. Waits for GET http://127.0.0.1:8000/health to return 200. Startup
       takes over a minute because of the model reproduction checks.
    4. Starts `vite preview` on 127.0.0.1:4173, which serves the built app and
       proxies /api/* to the backend.
    5. Runs `ngrok http 127.0.0.1:4173 --url https://<domain>` in this window.

    Press Ctrl+C to stop. ngrok, vite preview and the backend are all shut
    down. If anything is left running (e.g. the window was force-closed), run:
        .\start_public.ps1 -Stop

    The ngrok domain comes from $env:NGROK_DOMAIN if set, otherwise from the
    first entry of FRONTEND_ALLOWED_HOSTS in frontend/.env.public.local.
    See docs/DEPLOYMENT_PUBLIC.md.
#>
[CmdletBinding()]
param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 4173,
    [int]$HealthTimeoutSeconds = 600,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$BackendDir = Join-Path $Root 'backend'
$FrontendDir = Join-Path $Root 'frontend'
$LogDir = Join-Path $Root 'logs'
$FrontendEnvFile = Join-Path $FrontendDir '.env.public.local'
$BackendEnvFile = Join-Path $BackendDir '.env'
$ViteBin = 'node_modules/vite/bin/vite.js'  # relative to $FrontendDir

# Reads a single KEY=value from a dotenv file. Only the requested key is
# returned and nothing is printed, so secrets in the same file stay private.
function Get-EnvFileValue([string]$Path, [string]$Name) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $pattern = '^\s*' + [regex]::Escape($Name) + '\s*=\s*(.*?)\s*$'
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match $pattern) { return $Matches[1].Trim('"', "'") }
    }
    return $null
}

function Stop-ProcessTree($Process) {
    if ($Process -and -not $Process.HasExited) {
        & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null
    }
}

function Stop-PortListener([int]$Port) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($procId in ($conns | Select-Object -ExpandProperty OwningProcess -Unique)) {
        Write-Host "Stopping process $procId listening on port $Port"
        & taskkill.exe /PID $procId /T /F 2>$null | Out-Null
    }
}

function Test-PortInUse([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if ($Stop) {
    Get-Process -Name ngrok -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "Stopping ngrok (PID $($_.Id))"
        Stop-Process -Id $_.Id -Force
    }
    Stop-PortListener $FrontendPort
    Stop-PortListener $BackendPort
    Write-Host 'Stopped. The public URL is offline.'
    return
}

# ----------------------------------------------------------- preflight --

foreach ($cmd in @('python', 'node', 'ngrok')) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "'$cmd' was not found on PATH. See docs/DEPLOYMENT_PUBLIC.md (one-time setup)."
    }
}
$Python = (Get-Command python).Source
$Node = (Get-Command node).Source

if (-not (Test-Path -LiteralPath (Join-Path $FrontendDir $ViteBin))) {
    throw "Vite is not installed. Run 'npm install' in the frontend folder first."
}
if (-not (Test-Path -LiteralPath $FrontendEnvFile)) {
    throw "Missing frontend/.env.public.local. Copy frontend/.env.public.example and fill in your ngrok domain."
}

$Domain = $env:NGROK_DOMAIN
if (-not $Domain) {
    $allowedHosts = Get-EnvFileValue $FrontendEnvFile 'FRONTEND_ALLOWED_HOSTS'
    if ($allowedHosts) { $Domain = ($allowedHosts -split ',')[0] }
}
if (-not $Domain) {
    throw 'No ngrok domain configured. Set FRONTEND_ALLOWED_HOSTS in frontend/.env.public.local (or $env:NGROK_DOMAIN).'
}
$Domain = ($Domain.Trim() -replace '^https?://', '') -replace '/.*$', ''
$PublicUrl = "https://$Domain"

$apiBase = Get-EnvFileValue $FrontendEnvFile 'VITE_API_BASE_URL'
if ($apiBase -ne '/api') {
    throw "frontend/.env.public.local must set VITE_API_BASE_URL=/api (the public app talks to the backend only through the proxy)."
}

# URLs are not secrets; compare them so a stale backend/.env is caught
# before emails and Google sign-in quietly point at localhost.
$expected = @{ 'FRONTEND_URL' = $PublicUrl; 'BACKEND_URL' = "$PublicUrl/api" }
foreach ($name in $expected.Keys) {
    $actual = Get-EnvFileValue $BackendEnvFile $name
    if (-not $actual -or $actual.TrimEnd('/') -ne $expected[$name]) {
        Write-Warning "backend/.env: $name should be $($expected[$name]) for the public URL (currently '$actual'). Email links and Google sign-in will be wrong until fixed."
    }
}

foreach ($port in @($BackendPort, $FrontendPort)) {
    if (Test-PortInUse $port) {
        throw "Port $port is already in use. Stop whatever is using it (or run .\start_public.ps1 -Stop) and try again."
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$BackendOut = Join-Path $LogDir 'public_backend.out.log'
$BackendErr = Join-Path $LogDir 'public_backend.err.log'
$PreviewOut = Join-Path $LogDir 'public_preview.out.log'
$PreviewErr = Join-Path $LogDir 'public_preview.err.log'

$backend = $null
$preview = $null

try {
    # ------------------------------------------------------------- build --
    Write-Host "==> Building frontend for $PublicUrl"
    Push-Location $FrontendDir
    try {
        & $Node $ViteBin build --mode public
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
    } finally {
        Pop-Location
    }

    # ----------------------------------------------------------- backend --
    Write-Host "==> Starting backend on 127.0.0.1:$BackendPort (logs: $BackendErr)"
    $env:PYTHONUNBUFFERED = '1'
    # --no-proxy-headers: uvicorn must not rewrite the client address
    # itself; rate_limit.py decides when X-Forwarded-For can be trusted.
    $backend = Start-Process -FilePath $Python -WorkingDirectory $BackendDir -PassThru -NoNewWindow `
        -ArgumentList @('-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', "$BackendPort", '--no-proxy-headers') `
        -RedirectStandardOutput $BackendOut -RedirectStandardError $BackendErr
    $null = $backend.Handle  # keeps ExitCode readable after exit

    Write-Host "==> Waiting for backend health check (model reproduction checks take a minute or more)"
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    while ($true) {
        if ($backend.HasExited) {
            throw "Backend exited during startup (exit code $($backend.ExitCode)). See $BackendErr"
        }
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/health" -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -eq 200) { break }
        } catch {
            # not up yet
        }
        if ((Get-Date) -gt $deadline) {
            throw "Backend did not become healthy within $HealthTimeoutSeconds seconds. See $BackendErr"
        }
        Write-Host -NoNewline '.'
        Start-Sleep -Seconds 3
    }
    Write-Host ''
    Write-Host '    Backend is healthy.'

    # ----------------------------------------------------------- preview --
    Write-Host "==> Starting vite preview on 127.0.0.1:$FrontendPort (logs: $PreviewOut)"
    $preview = Start-Process -FilePath $Node -WorkingDirectory $FrontendDir -PassThru -NoNewWindow `
        -ArgumentList @($ViteBin, 'preview', '--mode', 'public', '--host', '127.0.0.1', '--port', "$FrontendPort", '--strictPort') `
        -RedirectStandardOutput $PreviewOut -RedirectStandardError $PreviewErr
    $null = $preview.Handle

    $deadline = (Get-Date).AddSeconds(60)
    while ($true) {
        if ($preview.HasExited) {
            throw "vite preview exited during startup (exit code $($preview.ExitCode)). See $PreviewErr"
        }
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort/" -UseBasicParsing -TimeoutSec 5
            if ($resp.StatusCode -eq 200) { break }
        } catch {
            # not up yet
        }
        if ((Get-Date) -gt $deadline) { throw "vite preview did not start within 60 seconds. See $PreviewErr" }
        Start-Sleep -Seconds 1
    }
    Write-Host '    Frontend preview is up.'

    # ------------------------------------------------------------- ngrok --
    Write-Host ''
    Write-Host "==> Public URL: $PublicUrl"
    Write-Host '    Starting ngrok. Press Ctrl+C to stop everything.'
    Write-Host ''
    & ngrok http "127.0.0.1:$FrontendPort" --url $PublicUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "ngrok exited with code $LASTEXITCODE (check the authtoken and domain; see docs/DEPLOYMENT_PUBLIC.md)."
    }
} finally {
    Write-Host ''
    Write-Host '==> Shutting down vite preview and backend'
    Stop-ProcessTree $preview
    Stop-ProcessTree $backend
    Write-Host '    Stopped. The public URL is offline.'
}
