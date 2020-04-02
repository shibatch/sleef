#!/bin/bash
set -ev
export PATH=$PATH:/usr/bin
dpkg --add-architecture armhf
cat /etc/apt/sources.list | sed -e 's/^deb /deb \[arch=amd64\] /g' -e 's/\[arch=amd64\] \[arch=amd64\]/\[arch=amd64\]/g' > /tmp/sources.list
. /etc/os-release
cat <<EOF | sed -e s/CODENAME/"$UBUNTU_CODENAME"/g >> /tmp/sources.list
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
EOF
mv /tmp/sources.list /etc/apt/sources.list
apt-get -qq update
apt-get install -y git cmake gcc-arm-linux-gnueabihf libc6-armhf-cross libc6:armhf libmpfr-dev:armhf libgomp1:armhf libmpfr-dev binfmt-support qemu qemu-user-static libssl-dev libssl-dev:armhf ninja-build
