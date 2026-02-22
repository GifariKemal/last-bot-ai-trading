# Live Optimization Monitor
# Runs in a loop showing real-time progress

$logFile = "C:\Users\Administrator\Videos\Last Bot AI Trading\xauusd_trading_bot\logs\optimization_v2_20260217_055324.log"
$botLog = "C:\Users\Administrator\Videos\Last Bot AI Trading\xauusd_trading_bot\logs\bot_activity\bot_" + (Get-Date -Format "yyyy-MM-dd") + ".log"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host "XAUUSD Bot Optimization Monitor" -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host ""

while ($true) {
    Clear-Host

    $currentTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host "========================================"  -ForegroundColor Cyan
    Write-Host "Update: $currentTime" -ForegroundColor Yellow
    Write-Host "========================================"  -ForegroundColor Cyan
    Write-Host ""

    # Check processes
    Write-Host "--- PROCESS STATUS ---" -ForegroundColor Green
    $liveBotProcess = Get-Process | Where-Object {$_.CommandLine -like "*main.py --mode live*"} -ErrorAction SilentlyContinue
    $optProcess = Get-Process | Where-Object {$_.CommandLine -like "*run_optimization_v2.py*"} -ErrorAction SilentlyContinue

    if ($liveBotProcess) {
        Write-Host "[OK] Live Bot Running (PID: $($liveBotProcess.Id))" -ForegroundColor Green
    } else {
        Write-Host "[!] Live Bot NOT Running" -ForegroundColor Red
    }

    if ($optProcess) {
        Write-Host "[OK] Optimization Running (PID: $($optProcess.Id))" -ForegroundColor Green
    } else {
        Write-Host "[!] Optimization NOT Running" -ForegroundColor Red
    }
    Write-Host ""

    # System resources
    Write-Host "--- SYSTEM RESOURCES ---" -ForegroundColor Green
    $cpu = Get-WmiObject Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average
    $mem = Get-WmiObject Win32_OperatingSystem
    $memUsed = [math]::Round(($mem.TotalVisibleMemorySize - $mem.FreePhysicalMemory) / $mem.TotalVisibleMemorySize * 100, 1)
    $memFreeGB = [math]::Round($mem.FreePhysicalMemory / 1MB, 1)

    Write-Host "CPU: $cpu% | RAM: $memUsed% used ($memFreeGB GB free)" -ForegroundColor White
    Write-Host ""

    # Optimization progress
    Write-Host "--- OPTIMIZATION PROGRESS ---" -ForegroundColor Green
    if (Test-Path $logFile) {
        $trials = Select-String -Path $logFile -Pattern "Trial \d+: Score=" | Select-Object -Last 10

        if ($trials.Count -gt 0) {
            Write-Host "Recent Trials:" -ForegroundColor Yellow
            $trials | ForEach-Object {
                $line = $_.Line
                if ($line -match "Trial (\d+): Score=([0-9.]+).*PF=([0-9.]+).*WR=([0-9.]+)%.*RR=([0-9.]+).*DD=([0-9.]+)%.*Trades=(\d+)") {
                    $trialNum = $matches[1]
                    $score = $matches[2]
                    $pf = $matches[3]
                    $wr = $matches[4]
                    $rr = $matches[5]
                    $dd = $matches[6]
                    $trades = $matches[7]

                    Write-Host "  #$trialNum | Score=$score | PF=$pf | WR=$wr% | RR=$rr | DD=$dd% | Trades=$trades" -ForegroundColor White
                }
            }

            # Best score so far
            $bestScore = Select-String -Path $logFile -Pattern "Best Score So Far:" | Select-Object -Last 1
            if ($bestScore) {
                Write-Host ""
                Write-Host "$($bestScore.Line)" -ForegroundColor Cyan
            }
        } else {
            Write-Host "Waiting for first trial to complete..." -ForegroundColor Yellow
            $backtest = Select-String -Path $logFile -Pattern "Progress: [0-9.]+%" | Select-Object -Last 1
            if ($backtest) {
                Write-Host "$($backtest.Line)" -ForegroundColor Gray
            }
        }
    } else {
        Write-Host "Log file not found" -ForegroundColor Red
    }
    Write-Host ""

    # Live bot status
    Write-Host "--- LIVE BOT STATUS ---" -ForegroundColor Green
    if (Test-Path $botLog) {
        $lastSignal = Select-String -Path $botLog -Pattern "CANDLE ANALYSIS|SIGNAL FOUND|Signal check" | Select-Object -Last 1
        if ($lastSignal) {
            $lastLine = $lastSignal.Line -replace '\[32m|\[0m|\[1m|\[36m', ''
            Write-Host $lastLine -ForegroundColor Gray
        }
    }
    Write-Host ""

    Write-Host "========================================"  -ForegroundColor Cyan
    Write-Host "Refreshing in 30 seconds... (Ctrl+C to exit)" -ForegroundColor Yellow
    Write-Host "========================================"  -ForegroundColor Cyan

    Start-Sleep -Seconds 30
}
