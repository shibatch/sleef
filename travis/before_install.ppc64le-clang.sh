#!/bin/bash
set -ev
export PATH=$PATH:/usr/bin
dpkg --add-architecture ppc64le
cat /etc/apt/sources.list | sed -e 's/^deb /deb \[arch=ppc64le\] /g' -e 's/\[arch=ppc64le\] \[arch=ppc64le\]/\[arch=ppc64le\]/g' > /tmp/sources.list
cat <<EOF | sed -e 's/CODENAME/xenial/g' >> /tmp/sources.list
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME main restricted
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates main restricted
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME universe
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-updates universe
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-backports main restricted
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security main restricted
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security universe
deb [arch=ppc64le] http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
deb-src http://ports.ubuntu.com/ubuntu-ports/ CODENAME-security multiverse
EOF
mv /tmp/sources.list /etc/apt/sources.list
apt-get -qq update
apt-get install -y git cmake clang-5.0 libc6-ppc64le-cross libc6:ppc64le libmpfr-dev:ppc64le libgomp1:ppc64le libmpfr-dev binfmt-support qemu qemu-user-static libfftw3-dev:ppc64le
