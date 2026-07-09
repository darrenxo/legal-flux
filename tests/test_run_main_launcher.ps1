$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot "scripts\run_main.ps1"
$mainLedger = Join-Path $projectRoot "runs\legalhk_only\diagnostic\generations.jsonl"
$beforeLength = if (Test-Path -LiteralPath $mainLedger) {
    (Get-Item -LiteralPath $mainLedger).Length
}
else {
    $null
}

$output = & powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File $launcher -DryRun 2>&1

if ($LASTEXITCODE -ne 0) {
    throw "Launcher dry run failed: $($output -join [Environment]::NewLine)"
}
if (($output -join "`n") -notmatch '"dry_run": true') {
    throw "Launcher did not execute generation dry run."
}
$afterLength = if (Test-Path -LiteralPath $mainLedger) {
    (Get-Item -LiteralPath $mainLedger).Length
}
else {
    $null
}
if ($afterLength -ne $beforeLength) {
    throw "Launcher dry run unexpectedly changed the main ledger."
}
