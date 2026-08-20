[CmdletBinding()]
param(
    [string]$DeltaLogin = "ychen129@login.delta.ncsa.illinois.edu",
    [string]$RemoteWorkRoot = "/work/hdd/bfua/ychen129/legal_nlp"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$preparedRoot = Join-Path $repositoryRoot "data\processed\legal_benchmarks"
$requiredDatasets = @(
    "annocaselaw",
    "realistic_ljp_facts",
    "il_tur_cjpe"
)

foreach ($dataset in $requiredDatasets) {
    $casesPath = Join-Path $preparedRoot "$dataset\cases.jsonl"
    $pilotPath = Join-Path $preparedRoot "$dataset\pilot_case_ids.json"
    if (-not (Test-Path -LiteralPath $casesPath -PathType Leaf)) {
        throw "Prepared cases missing for $dataset at $casesPath"
    }
    if (-not (Test-Path -LiteralPath $pilotPath -PathType Leaf)) {
        throw "Prepared pilot IDs missing for $dataset at $pilotPath"
    }
}

$remoteParent = "$RemoteWorkRoot/data/processed"
& ssh $DeltaLogin "mkdir -p '$remoteParent'"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to create the remote prepared-data directory."
}

& scp -r $preparedRoot "${DeltaLogin}:$remoteParent/"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to upload prepared legal benchmark data."
}

Write-Host "Uploaded prepared benchmarks to $RemoteWorkRoot/data/processed/legal_benchmarks"
