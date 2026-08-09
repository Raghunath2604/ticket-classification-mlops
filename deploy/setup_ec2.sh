#!/usr/bin/env bash
# One-time setup for a fresh EC2 instance (Ubuntu 22.04/24.04) that will
# host the API container. Run this once via SSH before the first CI/CD
# deploy; the GitHub Actions workflow only pulls + restarts the
# container afterwards (see .github/workflows/ci-cd.yml, job deploy-ec2).
#
# Usage (from your machine):
#   scp deploy/setup_ec2.sh ubuntu@<ec2-host>:~
#   ssh ubuntu@<ec2-host> "bash setup_ec2.sh"
set -euo pipefail

echo "== Installing Docker =="
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "== Allowing current user to run docker without sudo =="
sudo usermod -aG docker "$USER"

echo "== Opening port 8000 (adjust security group instead for production) =="
sudo ufw allow 8000/tcp || true
sudo ufw allow 22/tcp || true

echo "== Done. Log out and back in for docker group membership to take effect. =="
echo "The CI/CD pipeline will now be able to SSH in and run:"
echo "  docker pull ghcr.io/<org>/<repo>/ticket-classifier:<sha>"
echo "  docker run -d -p 8000:8000 --name ticket-classifier-api ..."
