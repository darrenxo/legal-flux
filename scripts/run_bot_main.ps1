param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-codex\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
}
$config = Join-Path $projectRoot "configs\legal_bot.yaml"
$freeze = Join-Path `
    $projectRoot "data\processed\legalhk_only\bot_frozen_manifest.json"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found at $python"
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $freeze)) {
    throw "BoT phase 2 is not frozen. Run: python -m legal_pilot bot-freeze"
}

Set-Location -LiteralPath $projectRoot

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class LegalBotPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
[LegalBotPower]::SetThreadExecutionState(
    [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
) | Out-Null

try {
    if ($DryRun) {
        & $python -m legal_pilot --config $config bot-generate --dry-run
    }
    else {
        & $python -m legal_pilot --config $config bot-generate
    }
    $runExitCode = $LASTEXITCODE
}
finally {
    [LegalBotPower]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}

exit $runExitCode
