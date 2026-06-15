param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InputFile = Join-Path $RootDir "data_0428.xlsx"
$OutputBase = Join-Path $RootDir "rerun_0428_outputs\screening_constrained_0428"
$OutputRoot = Join-Path $OutputBase $Mode
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
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding UTF8
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

function Invoke-ConstrainedStep {
    param(
        [Parameter(Mandatory = $true)] [string]$StepId,
        [Parameter(Mandatory = $true)] [string]$Label,
        [Parameter(Mandatory = $true)] [string[]]$PythonArgs
    )

    $logFile = Join-Path $LogDir "$StepId.log"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $StepId - $Label" | Tee-Object -FilePath $logFile
    Update-ManifestStep -StepId $StepId -Status "running"

    try {
        $pythonExe = $script:PythonRunner[0]
        $pythonPrefixArgs = @()
        if ($script:PythonRunner.Count -gt 1) {
            $pythonPrefixArgs = $script:PythonRunner[1..($script:PythonRunner.Count - 1)]
        }
        & $pythonExe @pythonPrefixArgs @PythonArgs *>> $logFile
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

$script:PythonRunner = @("python")
$pythonPath = (& python -c "import sys; print(sys.executable)") 2>$null
if ($LASTEXITCODE -ne 0 -or -not $pythonPath) {
    throw "Cannot run python. Please run this from PowerShell with Python or conda available."
}
if (-not $env:CONDA_DEFAULT_ENV -or $env:CONDA_DEFAULT_ENV -ne "arr_rf") {
    $condaVersion = (& conda --version) 2>$null
    if ($LASTEXITCODE -eq 0 -and $condaVersion) {
        Write-Warning "Current conda env is '$env:CONDA_DEFAULT_ENV', expected 'arr_rf'. Using: conda run -n arr_rf python"
        $script:PythonRunner = @("conda", "run", "-n", "arr_rf", "python")
    } else {
        Write-Warning "Current conda env is '$env:CONDA_DEFAULT_ENV', expected 'arr_rf', and conda was not found. Continuing with current python: $pythonPath"
    }
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$steps = @()
$steps += [pscustomobject]@{
    id = "01_constrained";
    label = "screening_constrained_0428";
    status = "pending";
    output_dir = $OutputRoot;
    log_file = (Join-Path $LogDir "01_constrained.log");
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
    $extraArgs = @(
        "--smoke-test",
        "--seeds", "42"
    )
} else {
    $extraArgs = @(
        "--seeds", "42", "2024", "2025", "2026", "2027",
        "--folds", "5"
    )
}

$pythonArgs = @(
    (Join-Path $RootDir "screening_constrained_0428_experiment.py"),
    "--input", $InputFile,
    "--output-dir", $OutputBase
) + $extraArgs

Invoke-ConstrainedStep -StepId "01_constrained" -Label "screening_constrained_0428" -PythonArgs $pythonArgs

Write-Host "Constrained screening experiment completed. Output: $OutputRoot"
