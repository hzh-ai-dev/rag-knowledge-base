$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$pythonPath = Join-Path $PSScriptRoot '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.11 -m venv .venv311
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 with the Windows Python launcher first.' }
}
& $pythonPath -m pip install -r requirements-local.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network access and retry.' }
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
Write-Host 'Setup complete. Fill your API key in .env, then run .\start.ps1.'
