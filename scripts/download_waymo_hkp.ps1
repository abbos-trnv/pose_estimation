# Download Waymo Perception v2 camera keypoints (~45 MB). No admin required.
# First time: this script will open a browser for `gcloud auth login`
# (use the same Google account that accepted the Waymo license).

$ErrorActionPreference = "Stop"
$Gcloud = "D:\MissingML\tools\google-cloud-sdk\bin\gcloud.cmd"
if (-not (Test-Path $Gcloud)) {
    throw "gcloud not found at $Gcloud"
}

$Dest = Join-Path $PSScriptRoot "..\data\waymo_v2\training\camera_hkp"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

Write-Host "If a browser opens, log in with the Waymo Google account, then return here."
& $Gcloud auth login --brief
& $Gcloud storage cp --recursive `
  "gs://waymo_open_dataset_v_2_0_1/training/camera_hkp" `
  $Dest
