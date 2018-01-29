#!/bin/bash
cd /build
dpkg --add-architecture arm64
cat /etc/apt/sources.list | sed -e 's/^deb /deb \[arch=amd64\] /g' -e 's/\[arch=amd64\] \[arch=amd64\]/\[arch=amd64\]/g' > /tmp/sources.list
cat <<EOF | sed -e 's/CODENAME/'`lsb_release -s -c`'/g' >> /tmp/sources.list
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
apt-get install cmake gcc-aarch64-linux-gnu libc6-arm64-cross binfmt-support libc6:arm64 libmpfr-dev:arm64 libgomp1:arm64 qemu-user-static
