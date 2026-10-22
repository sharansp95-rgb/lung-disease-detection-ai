#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 EC2 host for docker-compose.yml.
# Safe to re-run: every step checks before it changes anything.
#
#   ssh -i KEY.pem ubuntu@SERVER_IP 'sudo bash -s' < scripts/server_setup.sh
#
# Installs Docker Engine + the compose plugin (Docker's own apt repo), adds a
# 4 GB swap file, enables unattended security upgrades, caps Docker's log
# size, and creates /opt/lung-app owned by the SSH user.

set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run as root (sudo bash -s)" >&2; exit 1; }

APP_DIR=/opt/lung-app
APP_USER="${SUDO_USER:-ubuntu}"
export DEBIAN_FRONTEND=noninteractive

echo "==> packages"
apt-get update -q
apt-get install -y -q ca-certificates curl rsync unattended-upgrades

echo "==> Docker Engine + compose plugin"
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
if [[ ! -f /etc/docker/daemon.json ]]; then
  # Default for every container; docker-compose.yml sets the same per service.
  printf '{\n  "log-driver": "json-file",\n  "log-opts": {"max-size": "10m", "max-file": "3"}\n}\n' > /etc/docker/daemon.json
  systemctl restart docker
fi
systemctl enable --now docker >/dev/null
docker --version
docker compose version

echo "==> 4 GB swap"
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
  [[ -f /swapfile ]] || fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
echo 'vm.swappiness=10' > /etc/sysctl.d/99-swappiness.conf
sysctl -q -p /etc/sysctl.d/99-swappiness.conf
swapon --show

echo "==> unattended security upgrades"
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF
systemctl enable --now unattended-upgrades >/dev/null

echo "==> $APP_DIR"
install -d -m 0750 -o "$APP_USER" -g "$APP_USER" "$APP_DIR"
ls -ld "$APP_DIR"

echo "Server ready."
