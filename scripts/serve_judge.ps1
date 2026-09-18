# Start the local judge / teacher: Qwen3.5-9B (UD-Q4_K_XL) on llama-server, port 8080.
# Run from Windows PowerShell. Stop with: Get-Process llama-server | Stop-Process
param(
    [string]$Model = "$env:USERPROFILE\models\judge\Qwen3.5-9B-UD-Q4_K_XL.gguf",
    [int]$Port = 8080,
    [int]$Parallel = 4,
    [int]$Ctx = 16384
)
$log = "$env:USERPROFILE\models\judge\server.log"
Start-Process -FilePath llama-server -WindowStyle Hidden -RedirectStandardError $log -ArgumentList @(
    "-m", $Model, "-ngl", "99", "-c", "$Ctx", "-np", "$Parallel", "--port", "$Port", "--host", "0.0.0.0",
    "--jinja", "--alias", "judge", "-fa", "on"
)
Start-Sleep -Seconds 20
try { (Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 5).Content }
catch { Write-Host "not up yet; tail the log: Get-Content $log -Tail 20" }
