param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputFile = Join-Path $RootDir "data_0428.xlsx"
$OutputRoot = Join-Path $RootDir "rerun_0428_outputs\$Mode"
$ManifestPath = Join-Path $OutputRoot "rerun_manifest.json"
$LogDir = Join-Path $OutputRoot "logs"

function Get-IsoNow {
    return (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] $Value
    )
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)] [string]$Path)
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Update-ManifestStep {
    param(
        [Parameter(Mandatory = $true)] [string]$StepId,
        [Parameter(Mandatory = $true)] [ValidateSet("pending", "running", "done", "failed")] [string]$Status
    )
    $manifest = Read-JsonFile $ManifestPath
    $now = Get-IsoNow
    $manifest.updated_at = $now
    foreach ($step in $manifest.steps) {
        if ($step.id -eq $StepId) {
            $step.status = $Status
            if ($Status -eq "running") {
                $step.started_at = $now
            }
            if ($Status -eq "done" -or $Status -eq "failed") {
                $step.ended_at = $now
            }
        }
    }
    if ($Status -eq "failed") {
        $manifest.status = "failed"
    } elseif (($manifest.steps | Where-Object { $_.status -ne "done" }).Count -eq 0) {
        $manifest.status = "done"
    }
    Write-JsonFile -Path $ManifestPath -Value $manifest
}

function Invoke-RerunStep {
    param(
        [Parameter(Mandatory = $true)] [string]$StepId,
        [Parameter(Mandatory = $true)] [string]$Label,
        [Parameter(Mandatory = $true)] [string[]]$PythonArgs
    )

    $logFile = Join-Path $LogDir "$StepId.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $StepId - $Label" | Tee-Object -FilePath $logFile
    Update-ManifestStep -StepId $StepId -Status "running"

    try {
        & python @PythonArgs *>> $logFile
        if ($LASTEXITCODE -ne 0) {
            throw "python exited with code $LASTEXITCODE"
        }
        Update-ManifestStep -StepId $StepId -Status "done"
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $StepId" | Tee-Object -FilePath $logFile -Append
    } catch {
        Update-ManifestStep -StepId $StepId -Status "failed"
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED ${StepId}: $_" | Tee-Object -FilePath $logFile -Append
        throw
    }
}

if (-not (Test-Path -LiteralPath $InputFile)) {
    throw "Input file not found: $InputFile"
}

$pythonVersion = (& python -c "import sys; print(sys.executable)") 2>$null
if ($LASTEXITCODE -ne 0 -or -not $pythonVersion) {
    throw "Cannot run python. Please run this from an activated arr_rf PowerShell prompt: conda activate arr_rf"
}
if ($env:CONDA_DEFAULT_ENV -and $env:CONDA_DEFAULT_ENV -ne "arr_rf") {
    Write-Warning "Current conda env is '$env:CONDA_DEFAULT_ENV', expected 'arr_rf'. Continuing with current python: $pythonVersion"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$steps = @()
$steps += [pscustomobject]@{
        id = "01_ensemble";
        label = "ensemble_baseline";
        status = "pending";
        output_dir = (Join-Path $OutputRoot "01_ensemble");
        log_file = (Join-Path $LogDir "01_ensemble.log");
        started_at = $null;
        ended_at = $null;
}
$steps += [pscustomobject]@{
        id = "02_scm_v2";
        label = "scm_v2_mainline";
        status = "pending";
        output_dir = (Join-Path $OutputRoot "02_scm_v2");
        log_file = (Join-Path $LogDir "02_scm_v2.log");
        started_at = $null;
        ended_at = $null;
}
$steps += [pscustomobject]@{
        id = "03_scm_v3";
        label = "scm_v3_best_candidate";
        status = "pending";
        output_dir = (Join-Path $OutputRoot "03_scm_v3");
        log_file = (Join-Path $LogDir "03_scm_v3.log");
        started_at = $null;
        ended_at = $null;
}
$steps += [pscustomobject]@{
        id = "04_v31";
        label = "v31_calibration";
        status = "pending";
        output_dir = (Join-Path $OutputRoot "04_v31");
        log_file = (Join-Path $LogDir "04_v31.log");
        started_at = $null;
        ended_at = $null;
}

$now = Get-IsoNow
$manifest = [pscustomobject]@{
    mode = $Mode
    status = "running"
    input_file = $InputFile
    output_root = $OutputRoot
    started_at = $now
    updated_at = $now
    steps = $steps
}
Write-JsonFile -Path $ManifestPath -Value $manifest

if ($Mode -eq "smoke") {
    $ensembleExtra = @("--n-trials", "1", "--n-splits", "2")
    $scmV2Extra = @("--smoke-test", "--phase1-only")
    $scmV3Extra = @("--smoke-test", "--best-only")
    $v31Extra = @("--smoke-test")
} else {
    $ensembleExtra = @("--n-trials", "6", "--n-splits", "4")
    $scmV2Extra = @()
    $scmV3Extra = @("--best-only")
    $v31Extra = @()
}

$ensembleArgs = @(
    (Join-Path $RootDir "multiclass_ensemble_experiment.py"),
    "--input", $InputFile,
    "--output-dir", (Join-Path $OutputRoot "01_ensemble")
) + $ensembleExtra
Invoke-RerunStep -StepId "01_ensemble" -Label "ensemble_baseline" -PythonArgs $ensembleArgs

$scmV2Args = @(
    (Join-Path $RootDir "frontier_scm_v2_experiment.py"),
    "--input", $InputFile,
    "--output-dir", (Join-Path $OutputRoot "02_scm_v2"),
    "--skip-tabpfn"
) + $scmV2Extra
Invoke-RerunStep -StepId "02_scm_v2" -Label "scm_v2_mainline" -PythonArgs $scmV2Args

$scmV3Args = @(
    (Join-Path $RootDir "frontier_scm_v3_experiment.py"),
    "--input", $InputFile,
    "--output-dir", (Join-Path $OutputRoot "03_scm_v3"),
    "--skip-tabpfn"
) + $scmV3Extra
Invoke-RerunStep -StepId "03_scm_v3" -Label "scm_v3_best_candidate" -PythonArgs $scmV3Args

$v31Args = @(
    (Join-Path $RootDir "scm_v31_experiment.py"),
    "--input", $InputFile,
    "--output-dir", (Join-Path $OutputRoot "04_v31"),
    "--skip-tabddpm"
) + $v31Extra
Invoke-RerunStep -StepId "04_v31" -Label "v31_calibration" -PythonArgs $v31Args

Write-Host "All rerun steps completed. Output: $OutputRoot"
