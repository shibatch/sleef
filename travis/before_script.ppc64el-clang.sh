#!/bin/bash
set -ev
cd /build
mkdir build-native
cd build-native
cmake ..
make -j 2 all
cd /build
mkdir bin
cat <<EOF > /build/bin/ppc64el-cc
#!/bin/sh
clang-6.0 -target ppc64le-linux-gnu -mvsx -fuse-ld=lld-6.0 \$*
EOF
chmod +x /build/bin/ppc64el-cc
export PATH=$PATH:/build/bin
mkdir build-cross
cd build-cross
cmake -DCMAKE_TOOLCHAIN_FILE=../travis/toolchain-ppc64el.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DEMULATOR=qemu-ppc64el-static -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
