param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointDir,

    [int[]]$Iterations = @(20000, 24000, 26000),
    [int[]]$Seeds = @(1),
    [double[]]$Difficulties = @(1.0),
    [int]$NumEnvs = 64,
    [bool]$Deterministic = $true,
    [string]$OutputDir = "logs/cmoe_benchmark",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# 必须与 config/g1/__init__.py 和 cmoe_evaluate_benchmark.py 完全一致。
$Task = "DualGate-CMoE-G1-Benchmark"
$Agent = "rsl_rl_cfg_entry_point"
$Terrains = @(
    "slope",
    "stairs_up",
    "stairs_down",
    "discrete",
    "gap",
    "hurdle",
    "mix1",
    "mix2"
)

# 本文件位于 <仓库根目录>/data/cmoe_evaluation.ps1。
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BenchmarkScript = Join-Path $RepoRoot "scripts\rsl_rl\cmoe_evaluate_benchmark.py"

if (-not (Test-Path -LiteralPath $BenchmarkScript -PathType Leaf)) {
    throw "找不到 Benchmark 脚本: $BenchmarkScript"
}
if (-not (Test-Path -LiteralPath $CheckpointDir -PathType Container)) {
    throw "CheckpointDir 不存在或不是文件夹: $CheckpointDir"
}

$CheckpointDir = (Resolve-Path -LiteralPath $CheckpointDir).Path

# 相对输出目录统一按仓库根目录解释，避免从不同终端目录运行时结果散落。
if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $ResolvedOutputDir = $OutputDir
}
else {
    $ResolvedOutputDir = Join-Path $RepoRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $ResolvedOutputDir | Out-Null

$TotalRuns = $Iterations.Count * $Seeds.Count * $Difficulties.Count * $Terrains.Count
$CurrentRun = 0
$ExecutedRuns = 0

Push-Location $RepoRoot
try {
    foreach ($Iteration in $Iterations) {
        $Checkpoint = Join-Path $CheckpointDir "model_$Iteration.pt"

        if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
            Write-Warning "Checkpoint 不存在，跳过: $Checkpoint"
            continue
        }

        foreach ($Seed in $Seeds) {
            foreach ($Difficulty in $Difficulties) {
                if (($Difficulty -lt 0.0) -or ($Difficulty -gt 1.0)) {
                    throw "Difficulty 必须位于 [0, 1]，当前值: $Difficulty"
                }

                $DifficultyText = $Difficulty.ToString(
                    [System.Globalization.CultureInfo]::InvariantCulture
                )

                foreach ($Terrain in $Terrains) {
                    $CurrentRun++
                    Write-Host ""
                    Write-Host "[$CurrentRun/$TotalRuns] model_$Iteration | $Terrain | difficulty=$DifficultyText | seed=$Seed"

                    $Arguments = @(
                        $BenchmarkScript,
                        "--task", $Task,
                        "--agent", $Agent,
                        "--checkpoint", $Checkpoint,
                        "--terrain", $Terrain,
                        "--difficulty", $DifficultyText,
                        "--velocity", "0.8",
                        "--num_envs", "$NumEnvs",
                        "--seed", "$Seed",
                        "--output_dir", $ResolvedOutputDir,
                        "--headless"
                    )

                    if ($Deterministic) {
                        $Arguments += "--deterministic"
                    }

                    & $PythonExe @Arguments
                    if ($LASTEXITCODE -ne 0) {
                        throw "评测失败: model_$Iteration | $Terrain | difficulty=$DifficultyText | seed=$Seed | exit=$LASTEXITCODE"
                    }
                    $ExecutedRuns++
                }
            }
        }
    }
}
finally {
    Pop-Location
}

if ($ExecutedRuns -eq 0) {
    throw "没有执行任何评测。请检查 CheckpointDir 和 Iterations。"
}

Write-Host ""
Write-Host "================ CMoE Benchmark finished ================"
Write-Host "成功完成: $ExecutedRuns / $TotalRuns"
Write-Host "结果目录: $ResolvedOutputDir"
