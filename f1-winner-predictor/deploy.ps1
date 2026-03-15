#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script de despliegue del predictor F1.

.DESCRIPTION
    Automatiza los pasos necesarios para entrenar, subir modelos a S3,
    desplegar la imagen Docker en AWS Lambda (ECR) y lanzar predicciones.

.PARAMETER Mode
    Que parte del pipeline ejecutar:
      lambda   -- solo rebuild + push ECR + update Lambda (sin reentrenar)
      models   -- solo reentrenar y subir modelos a S3 (sin rebuild Lambda)
      all      -- todo: reentrenar + rebuild Lambda + deploy
      predict  -- lanza prediccion XGBoost (Lambda) + TabNet y muestra resultados

.PARAMETER Round
    Numero de ronda del GP (requerido con -Mode predict).

.PARAMETER Year
    Temporada (por defecto: ano actual). Usado con -Mode predict.

.PARAMETER Refresh
    Pasa --refresh-data a train.py para redescargar los datos de FastF1.

.PARAMETER Optimize
    Pasa --optimize a train.py para buscar hiperparametros con Optuna.

.EXAMPLE
    .\deploy.ps1 -Mode lambda
    .\deploy.ps1 -Mode models
    .\deploy.ps1 -Mode all -Refresh
    .\deploy.ps1 -Mode all -Refresh -Optimize
    .\deploy.ps1 -Mode predict -Round 2 -Year 2026
#>

param(
    [ValidateSet("lambda", "models", "all", "predict")]
    [string]$Mode = "all",

    [int]$Round = 0,
    [int]$Year  = (Get-Date).Year,

    [switch]$Refresh,
    [switch]$Optimize
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Configuracion -----------------------------------------------------------
$AWS_REGION      = "eu-west-1"
$AWS_ACCOUNT_ID  = "606756239522"
$ECR_REPO        = "f1-winner-predictor"
$LAMBDA_FUNCTION = "f1-winner-predictor"
$ECR_URI         = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${ECR_REPO}:latest"

$ScriptDir  = $PSScriptRoot
$ProjectDir = Split-Path $ScriptDir -Parent

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "------------------------------------------" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "------------------------------------------" -ForegroundColor Cyan
}

function Assert-ExitCode([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Fallo: $step (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- PASO 1: Reentrenar modelos ----------------------------------------------
if ($Mode -in "models", "all") {
    Write-Step "Entrenando modelos (XGBoost + TabNet) y subiendo a S3"

    $trainArgs = @("--model", "all", "--upload-s3")
    if ($Refresh)  { $trainArgs += "--refresh-data" }
    if ($Optimize) { $trainArgs += "--optimize" }

    Push-Location $ProjectDir
    docker compose run --rm `
        -v "${ScriptDir}/train.py:/app/train.py" `
        -v "${ScriptDir}/config.py:/app/config.py" `
        -v "${ScriptDir}/src/aws_utils.py:/app/src/aws_utils.py" `
        -v "${ScriptDir}/src/data_collection.py:/app/src/data_collection.py" `
        -v "${ScriptDir}/src/feature_engineering.py:/app/src/feature_engineering.py" `
        trainer python train.py @trainArgs
    Assert-ExitCode "Entrenamiento"
    Pop-Location

    Write-Host "[OK] Modelos entrenados y subidos a S3." -ForegroundColor Green
}

# --- PASO 2: Build imagen Lambda ---------------------------------------------
if ($Mode -in "lambda", "all") {
    Write-Step "Construyendo imagen Docker para Lambda"

    Push-Location $ScriptDir
    docker build -f Dockerfile.lambda -t $ECR_REPO .
    Assert-ExitCode "docker build Lambda"
    Pop-Location

    Write-Host "[OK] Imagen Lambda construida." -ForegroundColor Green

    # --- PASO 3: Push a ECR --------------------------------------------------
    Write-Step "Autenticando en ECR y subiendo imagen"

    aws ecr get-login-password --region $AWS_REGION |
        docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    Assert-ExitCode "ECR login"

    docker tag "${ECR_REPO}:latest" $ECR_URI
    docker push $ECR_URI
    Assert-ExitCode "docker push ECR"

    Write-Host "[OK] Imagen subida a ECR." -ForegroundColor Green

    # --- PASO 4: Actualizar funcion Lambda -----------------------------------
    Write-Step "Actualizando funcion Lambda"

    aws lambda update-function-code `
        --function-name $LAMBDA_FUNCTION `
        --image-uri $ECR_URI `
        --region $AWS_REGION | Out-Null
    Assert-ExitCode "lambda update-function-code"

    Write-Host "Esperando a que Lambda termine de actualizarse..." -ForegroundColor Yellow
    aws lambda wait function-updated `
        --function-name $LAMBDA_FUNCTION `
        --region $AWS_REGION
    Assert-ExitCode "lambda wait"

    Write-Host "[OK] Lambda actualizada: $LAMBDA_FUNCTION" -ForegroundColor Green
}

# --- PASO predict: lanzar predicciones XGBoost + TabNet ---------------------
if ($Mode -eq "predict") {
    if ($Round -eq 0) {
        Write-Host "[ERROR] Debes especificar -Round N (numero de ronda del GP)." -ForegroundColor Red
        exit 1
    }

    # -- XGBoost via Lambda ---------------------------------------------------
    Write-Step "Prediccion XGBoost (Lambda) -- $Year Round $Round"

    $payload      = (@{ year = $Year; round = $Round } | ConvertTo-Json -Compress)
    $payloadFile  = Join-Path $env:TEMP "f1_payload_${Year}_${Round}.json"
    $responseFile = Join-Path $env:TEMP "f1_response_${Year}_${Round}.json"
    Set-Content -Path $payloadFile -Value $payload -Encoding utf8

    aws lambda invoke `
        --function-name $LAMBDA_FUNCTION `
        --payload "fileb://$payloadFile" `
        $responseFile `
        --region $AWS_REGION | Out-Null
    Assert-ExitCode "lambda invoke"

    $xgbResult = Get-Content $responseFile -Raw | ConvertFrom-Json
    if ($xgbResult.statusCode -ne 200) {
        Write-Host "[ERROR] Lambda devolvio error:" -ForegroundColor Red
        Write-Host ($xgbResult | ConvertTo-Json -Depth 5)
        exit 1
    }
    $body       = $xgbResult.body
    $xgbWinner  = $body.predicted_winner
    $xgbProb    = $body.win_probability

    # -- TabNet via Docker ----------------------------------------------------
    Write-Step "Prediccion TabNet (local) -- $Year Round $Round"

    $tabOutputFile = Join-Path $env:TEMP "f1_tabnet_${Year}_${Round}.txt"
    Push-Location $ProjectDir
    docker compose run --rm `
        -v "${ScriptDir}/predict.py:/app/predict.py" `
        -v "${ScriptDir}/src/predict_lambda.py:/app/src/predict_lambda.py" `
        trainer python predict.py --round $Round --year $Year | Tee-Object -FilePath $tabOutputFile
    $LASTEXITCODE_TAB = $LASTEXITCODE
    Pop-Location
    if ($LASTEXITCODE_TAB -ne 0) {
        Write-Host "[ERROR] TabNet predict fallo (exit $LASTEXITCODE_TAB)" -ForegroundColor Red
        exit $LASTEXITCODE_TAB
    }

    $tabLines = Get-Content $tabOutputFile -ErrorAction SilentlyContinue
    $tabWinner = "N/A"
    $tabProb   = "N/A"
    $winLine  = $tabLines | Where-Object { $_ -match "Ganador predicho:" } | Select-Object -Last 1
    $probLine = $tabLines | Where-Object { $_ -match "Probabilidad:" }     | Select-Object -Last 1
    if ($winLine  -match "Ganador predicho:\s+(\S+)") { $tabWinner = $Matches[1] }
    if ($probLine -match "Probabilidad:\s+([\d.]+%)") { $tabProb   = $Matches[1] }

    # -- Resumen final --------------------------------------------------------
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  RESUMEN -- GP $Year  Round $Round" -ForegroundColor Cyan
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "  XGBoost  ->  $xgbWinner   ($xgbProb)" -ForegroundColor Yellow
    Write-Host "  TabNet   ->  $tabWinner   ($tabProb)" -ForegroundColor Magenta
    Write-Host "================================================" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "[OK] Completado (mode=$Mode)." -ForegroundColor Green
