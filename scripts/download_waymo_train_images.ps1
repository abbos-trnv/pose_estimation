# Download camera_image + camera_box for GP4 train segments. camera_hkp already local.
# Usage: powershell -File scripts/download_waymo_train_images.ps1

$ErrorActionPreference = "Stop"
$Gcloud = "D:\MissingML\tools\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path $Gcloud)) { throw "gcloud not found at $Gcloud" }

$Root = Join-Path $PSScriptRoot "..\data\waymo_v2\training"
$Img = Join-Path $Root "camera_image"
$Box = Join-Path $Root "camera_box"
New-Item -ItemType Directory -Force -Path $Img, $Box | Out-Null

$Yaml = Join-Path $PSScriptRoot "..\configs\data\gp4_segments.yaml"
$segs = Select-String -Path $Yaml -Pattern '^\s+-\s+"([^"]+)"' | ForEach-Object { $_.Matches.Groups[1].Value }
if (-not $segs) { throw "no train_segments in $Yaml" }

$bucket = "gs://waymo_open_dataset_v_2_0_1/training"
foreach ($s in $segs) {
    $ip = Join-Path $Img "$s.parquet"
    $bp = Join-Path $Box "$s.parquet"
    if (-not (Test-Path $ip)) {
        Write-Host "image $s"
        & $Gcloud storage cp "$bucket/camera_image/$s.parquet" $ip
    } else { Write-Host "skip image $s" }
    if (-not (Test-Path $bp)) {
        Write-Host "box $s"
        & $Gcloud storage cp "$bucket/camera_box/$s.parquet" $bp
    } else { Write-Host "skip box $s" }
}
Write-Host "done" $segs.Count "train segments"
