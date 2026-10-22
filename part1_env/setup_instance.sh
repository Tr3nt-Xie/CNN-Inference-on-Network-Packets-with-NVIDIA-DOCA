#!/usr/bin/env bash
# Part 1.2: install the packages for Parts 1 to 5 on an Ubuntu 22.04 GPU instance.
# Run as a user with sudo:  bash part1_env/setup_instance.sh
# Safe to run again: each step skips work that is already done.
#
# The NVIDIA driver (580 or newer) must already be installed. On AWS we use the
# "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)", which ships it.
set -euo pipefail

need_driver=580
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ -z "$driver" || ${driver%%.*} -lt $need_driver ]]; then
    echo "NVIDIA driver $need_driver+ required, found '${driver:-none}'" >&2
    exit 1
fi
echo "driver $driver"

# Ubuntu's automatic updater holds the dpkg lock right after boot (Step 1.4).
sudo systemctl stop unattended-upgrades 2>/dev/null || true
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "waiting for dpkg lock"; sleep 5
done

sudo apt-get update
sudo apt-get install -y build-essential pkg-config git python3-venv \
    pciutils dpdk dpdk-dev libdpdk-dev libpcap-dev \
    rdma-core libibverbs-dev ibverbs-utils

# CUDA Toolkit 12.6 only (not the "cuda" metapackage, which brings its own driver).
if ! dpkg -s cuda-toolkit-12-6 >/dev/null 2>&1; then
    if ! apt-cache policy cuda-toolkit-12-6 | grep -q 'Candidate: [0-9]'; then
        tmp=$(mktemp -d)
        wget -q -O "$tmp/cuda-keyring.deb" \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i "$tmp/cuda-keyring.deb"
        rm -rf "$tmp"
        sudo apt-get update
    fi
    sudo apt-get install -y cuda-toolkit-12-6
fi
# The AMI may point /usr/local/cuda at a different toolkit; point it at 12.6.
if [[ -L /usr/local/cuda || ! -e /usr/local/cuda ]]; then
    sudo ln -sfn /usr/local/cuda-12.6 /usr/local/cuda
else
    echo "/usr/local/cuda is a real directory; leaving it alone" >&2
fi

line='export PATH=/usr/local/cuda/bin:$PATH'
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
# The AWS Deep Learning AMI sets LD_LIBRARY_PATH to its own CUDA 13.2 in
# /etc/profile.d/dlami.sh. Its libnvrtc shadows the one in the pip wheels, and
# PyTorch's cuDNN then fails with CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
line='unset LD_LIBRARY_PATH  # see part1_env/setup_instance.sh'
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
grep -qxF 'source ~/venv/bin/activate' ~/.bashrc || echo 'source ~/venv/bin/activate' >> ~/.bashrc

if [[ ! -x ~/venv/bin/python ]]; then
    python3 -m venv ~/venv
fi
# shellcheck disable=SC1090
source ~/venv/bin/activate
pip install --upgrade pip
pip install torch torchvision onnx onnxscript tensorrt scapy numpy
pip freeze > ~/venv-freeze.txt
echo "done; open a new shell or run: source ~/.bashrc"
