<#
Run the ROS sim with an RL policy on Windows, in the dev container, with RViz as a normal
window. This hands off to tools/sim_gru_linux.sh in a WSL 2 distro, which starts the
container through Docker Desktop and shows RViz through WSLg.

Needs: Windows 11 (or Windows 10 with WSL from the Microsoft Store); a Linux distro in WSL 2
(wsl --install -d Ubuntu-22.04); Docker Desktop with the WSL 2 engine and that distro turned
on under Settings > Resources > WSL integration. Once, so PowerShell runs repo scripts:
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

    .\tools\sim_gru_windows.ps1                                      # gru_u02050.npz on Duct1077, until Ctrl-C
    .\tools\sim_gru_windows.ps1 mlp3_u02050.npz 90                   # another policy; lap monitor for 90 s, then stop
    .\tools\sim_gru_windows.ps1 gru_u02050.npz 90 -Map EvalDuct3012  # a map never used in training
    .\tools\sim_gru_windows.ps1 -Maps                                # list the maps
    .\tools\sim_gru_windows.ps1 -Distro Ubuntu-22.04                 # pick the WSL distro (default: WSL's default)

The first run builds the image, which takes 10-20 minutes. Ctrl-C stops everything.
#>
param(
    [Parameter(Position = 0)][string]$Policy = "gru_u02050.npz",
    [Parameter(Position = 1)][string]$Seconds = "",
    [string]$Map = "",
    [string]$Distro = "",
    [switch]$Maps
)

function Fail([string]$Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

$Repo = Split-Path -Parent $PSScriptRoot

if ($Maps) {
    Write-Host "Duct1077 (default: held-out, used for checkpoint selection)"
    Get-ChildItem -Path (Join-Path $Repo "sim") -Filter "sim_rl_*.yaml" | Sort-Object Name |
        ForEach-Object { Write-Host ("  " + ($_.BaseName -replace "^sim_rl_", "")) }
    Write-Host "start one with  .\tools\sim_gru_windows.ps1 [policy.npz] [seconds] -Map <name>"
    exit 0
}

# bash can't run scripts checked out with Windows (CRLF) line endings.
foreach ($f in @("start_sim.sh", "tools/sim_gru.sh", "tools/sim_gru_linux.sh")) {
    $path = Join-Path $Repo $f
    if (-not (Test-Path $path)) { Fail "$path is missing: git pull" }
    if ([IO.File]::ReadAllText($path).Contains("`r`n")) {
        Fail ("$f has Windows (CRLF) line endings, which bash can't run. New clones get LF " +
              "(the repo's .gitattributes). To refresh this clone, commit or stash your changes, then:`n" +
              "    git rm -r -q --cached .`n    git reset -q --hard")
    }
}

# A repo cloned inside WSL (\\wsl.localhost\<distro>\home\...) runs in that distro, at its Linux path.
$CdPath = $Repo
if ($Repo -match '^\\\\wsl(\.localhost|\$)\\([^\\]+)(\\.*)$') {
    $Distro = $Matches[2]
    $CdPath = $Matches[3] -replace '\\', '/'
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Fail "WSL is not installed: in an admin PowerShell run  wsl --install -d Ubuntu-22.04  and reboot"
}
$env:WSL_UTF8 = "1"   # wsl.exe prints UTF-8, not UTF-16 (older WSL ignores it; NULs are stripped below)
$listing = @(wsl.exe --list --verbose 2>$null)
if ($LASTEXITCODE -ne 0) { Fail "WSL has no Linux distro: wsl --install -d Ubuntu-22.04" }
$found = @()
$default = ""
foreach ($line in ($listing | Select-Object -Skip 1)) {
    $t = ($line -replace "`0", "").Trim()
    if (-not $t) { continue }
    $isDefault = $t.StartsWith("*")
    $cols = @($t.TrimStart("*").Trim() -split "\s+")
    if ($cols.Count -lt 3 -or $cols[0] -like "docker-desktop*") { continue }
    $found += [pscustomobject]@{ Name = $cols[0]; Version = $cols[-1] }
    if ($isDefault) { $default = $cols[0] }
}
if ($Distro -eq "") {
    if ($found.Count -eq 0) { Fail "no Linux distro in WSL (Docker Desktop's own don't count): wsl --install -d Ubuntu-22.04" }
    if ($default) { $Distro = $default } else { $Distro = $found[0].Name }
}
$entry = $found | Where-Object { $_.Name -eq $Distro } | Select-Object -First 1
if (-not $entry) { Fail "no WSL distro named $Distro (wsl --list --verbose lists them)" }
if ($entry.Version -ne "2") { Fail "$Distro runs on WSL 1, and Docker Desktop needs WSL 2:  wsl --set-version $Distro 2" }

if (Get-Command docker.exe -ErrorAction SilentlyContinue) {
    $osType = docker.exe info --format "{{.OSType}}" 2>$null
    if ($LASTEXITCODE -ne 0) { Fail "Docker Desktop is not running: start it and wait for 'Engine running'" }
    if ("$osType".Trim() -ne "linux") { Fail "Docker Desktop is on Windows containers: right-click its tray icon > Switch to Linux containers" }
}

$linuxArgs = @($Policy)
if ($Seconds) { $linuxArgs += $Seconds }
Write-Host "WSL distro: $Distro   repo: $CdPath"
wsl.exe --distribution $Distro --cd $CdPath --exec env "MAP=$Map" bash tools/sim_gru_linux.sh @linuxArgs
exit $LASTEXITCODE
