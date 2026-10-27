$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = '1'
$pythonPath = Join-Path $PSScriptRoot '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Python environment missing. Run .\setup.ps1 first; see README.md.'
}
& $pythonPath -m scripts.check_setup
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonPath -m app.main
exit $LASTEXITCODE
