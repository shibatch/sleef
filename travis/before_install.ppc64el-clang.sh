#!/bin/bash
set -ev
export PATH=$PATH:/usr/bin
#apt-get -qq update
#apt-get install -y wget
dpkg --add-architecture ppc64el
cat /etc/apt/sources.list | sed -e 's/^deb /deb \[arch=amd64\] /g' -e 's/\[arch=amd64\] \[arch=amd64\]/\[arch=amd64\]/g' > /tmp/sources.list
cat <<EOF | sed -e 's/CODENAME/xenial/g' >> /tmp/sources.list
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb [arch=ppc64el] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
#deb http://apt.llvm.org/xenial/ llvm-toolchain-CODENAME-6.0 main
EOF
mv /tmp/sources.list /etc/apt/sources.list
#wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
apt-get -qq update
apt-get install -y git cmake clang-5.0 gcc-powerpc64le-linux-gnu libc6-ppc64el-cross libc6-dev:ppc64el libmpfr-dev:ppc64el libgomp1:ppc64el libmpfr-dev binfmt-support qemu qemu-user-static libfftw3-dev:ppc64el
