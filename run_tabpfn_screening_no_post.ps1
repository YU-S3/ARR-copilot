param(
    [ValidateSet("smoke", "seed", "full")]
    [string]$Mode = "smoke",

    [ValidateSet("core", "extended")]
    [string]$VariantSet = "core",

    [string]$Device = "cpu",

    [switch]$AuditPolicies
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "D:\Download\anaconda\envs\arr_rf\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    $PythonExe = "python"
}

$OutputRoot = Join-Path $RootDir "rerun_0428_outputs\tabpfn_screening_no_post\$Mode"
$ArgsList = @(
    (Join-Path $RootDir "tabpfn_screening_no_post_experiment.py"),
    "--input", (Join-Path $RootDir "data_0428.xlsx"),
    "--tabpfn-dir", (Join-Path $RootDir "TabPFN"),
    "--output-dir", $OutputRoot,
    "--task-mode", "both",
    "--variant-set", $VariantSet,
    "--device", $Device
)

if ($AuditPolicies) {
    $ArgsList += "--include-audit-policies"
}

if ($Mode -eq "smoke") {
    $ArgsList += @("--smoke-test", "--n-estimators", "1")
} elseif ($Mode -eq "seed") {
    $ArgsList += @("--n-estimators", "8", "--skip-cv")
} else {
    $ArgsList += @("--n-estimators", "8")
}

& $PythonExe @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "TabPFN screening experiment failed with exit code $LASTEXITCODE"
}

Write-Host "TabPFN screening experiment completed. Output: $OutputRoot"
