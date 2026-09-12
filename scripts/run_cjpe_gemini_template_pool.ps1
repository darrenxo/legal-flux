param(
    [string]$ProjectId = "legalflux-gemini",
    [int]$MaximumAttemptsPerStage = 3
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv-codex\Scripts\python.exe"
$config = Join-Path $repoRoot "configs\cjpe_template_pool.yaml"
$credentials = Join-Path $env:APPDATA "gcloud\application_default_credentials.json"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $credentials)) {
    throw "Google Application Default Credentials not found: $credentials"
}

$env:GOOGLE_CLOUD_PROJECT = $ProjectId
$env:GOOGLE_CLOUD_LOCATION = "global"
$env:GOOGLE_GENAI_USE_VERTEXAI = "true"
$env:GOOGLE_APPLICATION_CREDENTIALS = $credentials
$env:PYTHONUTF8 = "1"

function Invoke-ResumableStage {
    param(
        [string]$Name,
        [string[]]$CommandArguments
    )

    for ($attempt = 1; $attempt -le $MaximumAttemptsPerStage; $attempt++) {
        Write-Output "[$(Get-Date -Format o)] Starting $Name (attempt $attempt/$MaximumAttemptsPerStage)."
        & $python @CommandArguments
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-Output "[$(Get-Date -Format o)] Completed $Name."
            return
        }
        Write-Warning "$Name exited with code $exitCode on attempt $attempt."
        if ($attempt -lt $MaximumAttemptsPerStage) {
            Start-Sleep -Seconds (15 * $attempt)
        }
    }
    throw "$Name failed after $MaximumAttemptsPerStage attempts."
}

Set-Location -LiteralPath $repoRoot
$baseArguments = @(
    "-m", "legal_pilot",
    "--config", $config,
    "flux-gemini-templates"
)

# The completed one-batch smoke result is reused automatically. Existing parsed
# batch outputs make both candidate generation and coverage audit resumable.
Invoke-ResumableStage "candidate generation" ($baseArguments + @("--stage", "candidates"))
Invoke-ResumableStage "global consolidation" ($baseArguments + @("--stage", "merge"))
Invoke-ResumableStage "one-batch coverage-audit smoke" ($baseArguments + @("--stage", "audit", "--limit", "1"))
Invoke-ResumableStage "full coverage audit and gap adjudication" ($baseArguments + @("--stage", "audit"))
Invoke-ResumableStage "local similarity audit" ($baseArguments + @("--stage", "similarity-audit"))

Write-Output "[$(Get-Date -Format o)] CJPE Gemini template-pool workflow completed."
