# Imagen base de Python optimizada
FROM python:3.11-slim

# Evita archivos .pyc y permite ver logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Copiar requirements primero para aprovechar la caché de capas de Docker
# (pytorch-tabnet y torch se instalan como wheels pre-compilados, sin deps del sistema)
COPY f1-winner-predictor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del predictor TabNet
COPY f1-winner-predictor/ .

# Por defecto muestra la ayuda; para inferencia real: --round N [--no-upload]
CMD ["python", "predict.py", "--help"]