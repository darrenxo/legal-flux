$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_bot_main.ps1"
$mainLedger = Join-Path $projectRoot "runs\legal_bot\diagnostic\generations.jsonl"
$beforeLength = if (Test-Path -LiteralPath $mainLedger) {
    (Get-Item -LiteralPath $mainLedger).Length
}
else {
    $null
}

$output = & powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File $launcher -DryRun 2>&1

if ($LASTEXITCODE -ne 0) {
    throw "BoT launcher dry run failed: $($output -join [Environment]::NewLine)"
}
if (($output -join "`n") -notmatch '"jobs": 384') {
    throw "BoT launcher did not plan the expected 384 jobs."
}
if (($output -join "`n") -notmatch '"dry_run": true') {
    throw "BoT launcher did not execute a dry run."
}
$afterLength = if (Test-Path -LiteralPath $mainLedger) {
    (Get-Item -LiteralPath $mainLedger).Length
}
else {
    $null
}
if ($afterLength -ne $beforeLength) {
    throw "BoT launcher dry run unexpectedly changed the main ledger."
}
