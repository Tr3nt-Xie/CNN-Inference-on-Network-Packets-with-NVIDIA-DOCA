#!/usr/bin/env bash
# Step 7.3 + 7.6, part 1 (once per node, BEFORE the reboot). Run on node0 and node1 in parallel:
#   bash part7_cloudlab/setup_node.sh 2>&1 | tee ~/setup_node.log
# Installs: base packages, NVIDIA open driver 580 branch, CUDA 13.0 toolkit, Docker + NVIDIA
# Container Toolkit with data-root on /mydata, builds the open-source GPUNetIO (CUDA_ARCH=80),
# and turns the IOMMU off on the kernel command line. Then reboot.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
mountpoint -q /mydata || { echo "/mydata is not mounted - check the profile's blockstore"; exit 1; }
sudo systemctl stop unattended-upgrades 2>/dev/null || true
sudo apt-get update
sudo apt-get install -y build-essential git "linux-headers-$(uname -r)" rdma-core ibverbs-utils \
    libibverbs-dev perftest pciutils jq curl gpg docker.io python3

# NVIDIA CUDA repository for Ubuntu 24.04
if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
  wget -q -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
fi
# Open kernel module driver, 580 branch (pinning package from NVIDIA's guide; Ubuntu's
# nvidia-driver-580-open as a fallback).
if ! dpkg -l | grep -qE '^ii\s+(nvidia-open|nvidia-driver-580-open)'; then
  if sudo apt-get install -y nvidia-driver-pinning-580; then
    sudo apt-get install -y nvidia-open
  else
    sudo apt-get install -y nvidia-driver-580-open
  fi
fi
sudo apt-get install -y cuda-toolkit-13-0
sudo ln -sfn /usr/local/cuda-13.0 /usr/local/cuda

# Docker data on /mydata, NVIDIA runtime
if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
fi
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl stop docker docker.socket || true
sudo mkdir -p /mydata/docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) else {}
d["data-root"] = "/mydata/docker"
json.dump(d, open(p, "w"), indent=2)
print(open(p).read())
PY
sudo systemctl start docker
sudo usermod -aG docker "$USER" || true

# Open-source GPUNetIO (examples need it at run time). A30 = sm_80 = default CUDA_ARCH.
[ -d ~/gpunetio ] || git clone -q https://github.com/NVIDIA-DOCA/gpunetio ~/gpunetio
( cd ~/gpunetio && PATH=/usr/local/cuda/bin:$PATH make install install_examples PREFIX=$HOME/gpunetio/install CUDA_ARCH=80 )
ls ~/gpunetio/install/examples

# Step 7.6: IOMMU off (takes effect after the reboot)
if ! grep -q 'amd_iommu=off iommu=off' /etc/default/grub; then
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&amd_iommu=off iommu=off /' /etc/default/grub
fi
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub
sudo update-grub
grep -qxF 'export PATH=/usr/local/cuda/bin:$PATH' ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo; echo "SETUP PART 1 DONE on $(hostname -s). Now: sudo reboot   (then run after_reboot.sh)"
