#!/bin/bash
dpkg --add-architecture arm64
cat /etc/apt/sources.list | sed -e 's/^deb /deb \[arch=amd64\] /g' > /tmp/sources.list
cat <<EOF | sed -e 's/CODENAME/'`lsb_release -s -c`'/g' >> /tmp/sources.list
deb [arch=arm64] http://archive.ubuntu.com/ubuntu/ CODENAME main restricted
deb-src http://archive.ubuntu.com/ubuntu/ CODENAME main restricted
deb [arch=arm64] http://archive.ubuntu.com/ubuntu/ CODENAME-updates main restricted
deb-src http://archive.ubuntu.com/ubuntu/ CODENAME-updates main restricted
deb [arch=arm64] http://archive.ubuntu.com/ubuntu/ CODENAME universe
deb-src http://archive.ubuntu.com/ubuntu/ CODENAME universe
deb [arch=arm64] http://archive.ubuntu.com/ubuntu/ CODENAME-updates universe
deb-src http://archive.ubuntu.com/ubuntu/ CODENAME-updates universe
deb [arch=arm64] http://archive.ubuntu.com/ubuntu/ CODENAME-security main restricted
deb-src http://archive.ubuntu.com/ubuntu/ CODENAME-security main restricted
deb [arch=arm64] http://archive.ubuntu.com/ubuntu/ CODENAME-security universe
deb-src http://archive.ubuntu.com/ubuntu/ CODENAME-security universe
EOF
mv /tmp/sources.list /etc/apt/sources.list
cat /etc/apt/sources.list
apt-get -qq update
apt-get install cmake gcc-aarch64-linux-gnu libc6-arm64-cross binfmt-support libc6:arm64 libmpfr-dev:arm64 libgomp1:arm64 qemu-user-static
