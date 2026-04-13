# ========================
# Create venv with Python 3.10
# ========================

Write-Host "Creating virtual environment with Python 3.10..."

py -3.10 -m venv pycaret-env

# Activate
Write-Host "Activating environment..."
& .\pycaret-env\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
Write-Host "Installing PyCaret and dependencies..."
pip install pycaret

# Optional: freeze environment
pip freeze > requirements.txt

Write-Host "Setup complete."