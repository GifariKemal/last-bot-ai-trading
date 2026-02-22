$env:CLAUDECODE = ''
$env:CLAUDE_CODE_SESSION = ''
$prompt = 'Respond ONLY with this exact JSON, nothing else: {"decision":"NO_TRADE","confidence":0.5,"reason":"test ok","sl":0,"tp":0}'
Write-Host "Sending prompt to claude..."
$result = $prompt | & 'C:\Users\Administrator\AppData\Roaming\npm\claude.cmd' --model claude-haiku-4-5-20251001 --dangerously-skip-permissions -p
Write-Host "Response:"
Write-Host $result
