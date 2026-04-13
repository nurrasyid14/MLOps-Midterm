# ========================
# Activate environment
# ========================

& .\pycaret-env\Scripts\Activate.ps1

# ========================
# Run your pipeline
# ========================

Write-Host "Running ML pipeline..."

python <run_your_pipeline.py>

Write-Host "Done."