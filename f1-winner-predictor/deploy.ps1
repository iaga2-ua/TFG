#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script de despliegue del predictor F1.

.DESCRIPTION
    Automatiza los pasos necesarios para entrenar, subir modelos a S3 y
    desplegar la imagen Docker en AWS Lambda (ECR).

.PARAMETER Mode
    Qué parte del pipeline ejecutar:
      lambda   — solo rebuild + push ECR + update Lambda (sin reentrenar)
      models   — solo reentrenar y subir modelos a S3 (sin rebuild Lambda)
      all      — todo: reentrenar + rebuild Lambda + deploy

.PARAMETER Refresh
    Pasa --refresh-data a train.py para redescargar los datos de FastF1.

.PARAMETER Optimize
    Pasa --optimize a train.py para buscar hiperparámetros con Optuna.

.EXAMPLE
    # Solo redesplegar Lambda (código cambiado, no el modelo):
    .\deploy.ps1 -Mode lambda

    # Reentrenar ambos modelos y subir a S3 (sin rebuild Lambda):
    .\deploy.ps1 -Mode models

    # Pipeline completo con datos frescos:
    .\deploy.ps1 -Mode all -Refresh

    # Pipeline completo con optimización de hiperparámetros:
    .\deploy.ps1 -Mode all -Refresh -Optimize
#>

param(
    [ValidateSet("lambda", "models", "all")]
    [string]$Mode = "all",

    [switch]$Refresh,
    [switch]$Optimize
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Configuración ────────────────────────────────────────────────────────────
$AWS_REGION      = "eu-west-1"
$AWS_ACCOUNT_ID  = "606756239522"
$ECR_REPO        = "f1-winner-predictor"
$LAMBDA_FUNCTION = "f1-winner-predictor"
$ECR_URI         = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${ECR_REPO}:latest"

# Rutas relativas al directorio del script
$ScriptDir  = $PSScriptRoot
$ProjectDir = Split-Path $ScriptDir -Parent   # TFG/

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "──────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "──────────────────────────────────────────" -ForegroundColor Cyan
}

function Assert-ExitCode([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Falló: $step (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ─── PASO 1: Reentrenar modelos ───────────────────────────────────────────────
if ($Mode -in "models", "all") {
    Write-Step "Entrenando modelos (XGBoost + TabNet) y subiendo a S3"

    $trainArgs = "python train.py --model all --upload-s3"
    if ($Refresh)  { $trainArgs += " --refresh-data" }
    if ($Optimize) { $trainArgs += " --optimize" }

    # Montamos los .py como volúmenes para que los cambios locales sean inmediatos
    # sin necesidad de rebuild de la imagen Docker del trainer.
    Push-Location $ProjectDir
    docker compose run --rm `
        -v "${ScriptDir}/train.py:/app/train.py" `
        -v "${ScriptDir}/config.py:/app/config.py" `
        -v "${ScriptDir}/src/aws_utils.py:/app/src/aws_utils.py" `
        -v "${ScriptDir}/src/data_collection.py:/app/src/data_collection.py" `
        -v "${ScriptDir}/src/feature_engineering.py:/app/src/feature_engineering.py" `
        trainer $trainArgs
    Assert-ExitCode "Entrenamiento"
    Pop-Location

    Write-Host "✅ Modelos entrenados y subidos a S3." -ForegroundColor Green
}

# ─── PASO 2: Build imagen Lambda ──────────────────────────────────────────────
if ($Mode -in "lambda", "all") {
    Write-Step "Construyendo imagen Docker para Lambda"

    Push-Location $ScriptDir
    docker build -f Dockerfile.lambda -t $ECR_REPO .
    Assert-ExitCode "docker build Lambda"
    Pop-Location

    Write-Host "✅ Imagen Lambda construida." -ForegroundColor Green

    # ─── PASO 3: Push a ECR ───────────────────────────────────────────────────
    Write-Step "Autenticando en ECR y subiendo imagen"

    aws ecr get-login-password --region $AWS_REGION |
        docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    Assert-ExitCode "ECR login"

    docker tag "${ECR_REPO}:latest" $ECR_URI
    docker push $ECR_URI
    Assert-ExitCode "docker push ECR"

    Write-Host "✅ Imagen subida a ECR." -ForegroundColor Green

    # ─── PASO 4: Actualizar función Lambda ────────────────────────────────────
    Write-Step "Actualizando función Lambda"

    aws lambda update-function-code `
        --function-name $LAMBDA_FUNCTION `
        --image-uri $ECR_URI `
        --region $AWS_REGION | Out-Null
    Assert-ExitCode "lambda update-function-code"

    Write-Host "⏳ Esperando a que Lambda termine de actualizarse..." -ForegroundColor Yellow
    aws lambda wait function-updated `
        --function-name $LAMBDA_FUNCTION `
        --region $AWS_REGION
    Assert-ExitCode "lambda wait"

    Write-Host "✅ Lambda actualizada: $LAMBDA_FUNCTION" -ForegroundColor Green
}

Write-Host ""
Write-Host "🏁 Despliegue completado (mode=$Mode)." -ForegroundColor Green
