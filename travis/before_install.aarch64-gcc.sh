#!/bin/bash
set -ev
export PATH=$PATH:/usr/bin
dpkg --add-architecture arm64
cat /etc/apt/sources.list | sed -e 's/^deb /deb \[arch=amd64\] /g' -e 's/\[arch=amd64\] \[arch=amd64\]/\[arch=amd64\]/g' > /tmp/sources.list
. /etc/os-release
cat <<EOF | sed -e s/CODENAME/"$UBUNTU_CODENAME"/g >> /tmp/sources.list
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
EOF
mv /tmp/sources.list /etc/apt/sources.list
apt-get -qq update
apt-get install -y git cmake gcc-aarch64-linux-gnu libc6-arm64-cross libc6:arm64 libmpfr-dev:arm64 libgomp1:arm64 libmpfr-dev binfmt-support qemu qemu-user-static libfftw3-dev:arm64 libssl-dev libssl-dev:arm64 ninja-build
