$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$cases = @(
    @{
        Script = "run_semantic_main.ps1"
        ExpectedJobs = 256
    },
    @{
        Script = "run_frontier_main.ps1"
        ExpectedJobs = 320
    }
)

foreach ($case in $cases) {
    $launcher = Join-Path $projectRoot "scripts\$($case.Script)"
    $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass `
        -File $launcher -DryRun 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "$($case.Script) failed: $($output -join [Environment]::NewLine)"
    }
    $rendered = $output -join "`n"
    if ($rendered -notmatch "`"jobs`": $($case.ExpectedJobs)") {
        throw "$($case.Script) did not plan $($case.ExpectedJobs) jobs."
    }
    if ($rendered -notmatch '"dry_run": true') {
        throw "$($case.Script) did not execute a dry run."
    }
}
