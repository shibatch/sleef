pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('ppc64le gcc') {
            	     agent { label 'ppc64' }
            	     steps {
	    	     	 sh '''
                	 echo "ppc64 gcc on" `hostname`
			 export CC=gcc-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VSX=TRUE -DENFORCE_VSX3=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('ppc64le clang') {
            	     agent { label 'ppc64' }
            	     steps {
	    	     	 sh '''
                	 echo "ppc64 clang on" `hostname`
			 export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VSX=TRUE -DENFORCE_VSX3=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('s390x gcc') {
            	     agent { label 's390x' }
            	     steps {
	    	     	 sh '''
                	 echo "s390x gcc on" `hostname`
			 export CC=gcc
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -G Ninja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VXE=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('s390x clang') {
            	     agent { label 's390x' }
            	     steps {
	    	     	 sh '''
                	 echo "s390x clang on" `hostname`
			 export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -G Ninja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VXE=TRUE -DENFORCE_VXE2=TRUE -DCMAKE_BUILD_TYPE=Debug ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('Armclang') {
            	     agent { label 'armclang' }
            	     steps {
	    	     	 sh '''
                	 echo "armclang+SVE on" `hostname`
			 export CC=armclang
			 export QEMU_CPU=max,sve-max-vq=1
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64 ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('Armclang AAVPCS') {
            	     agent { label 'armclang' }
            	     steps {
	    	     	 sh '''
                	 echo "armclang+SVE+AAVPCS on" `hostname`
			 export CC=armclang
			 export QEMU_CPU=max,sve-max-vq=1
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DFORCE_AAVPCS=On -DENABLE_GNUABI=On -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64 ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('aarch64 gcc-10 and cuda') {
            	     agent { label 'aarch64 && cuda && gcc-10' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch64 gcc-10 and cuda on" `hostname`
			 export CC=gcc-10.2.0
			 export QEMU_CPU=max,sve-max-vq=1
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=TRUE -DENFORCE_TESTER3=TRUE -DENABLE_GNUABI=On -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENABLE_CUDA=TRUE -DENFORCE_CUDA=TRUE -DENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64 ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 7
		         ninja install
			 '''
            	     }
                }

                stage('aarch64 clang-11') {
            	     agent { label 'aarch64 && clang-11' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch64 clang-11 on" `hostname`
			 export CC=clang-11
			 export PATH=$PATH:/opt/clang+llvm-11.0.0-aarch64-linux-gnu/bin
			 export QEMU_CPU=max,sve-max-vq=1
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DDISABLE_OPENMP=TRUE -DBUILD_SHARED_LIBS=TRUE -DENFORCE_TESTER3=TRUE -DENABLE_GNUABI=On -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64 ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('aarch32 gcc') {
            	     agent { label 'docker-aarch32' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch32 gcc on" `hostname`
			 tar cfz /tmp/builddir.tgz .
			 docker start jenkins || true
			 docker exec jenkins apt-get update
			 docker exec jenkins apt-get -y dist-upgrade
			 docker exec jenkins rm -rf /build
			 docker exec jenkins mkdir /build
			 docker cp /tmp/builddir.tgz jenkins:/tmp/
			 docker exec jenkins tar xf /tmp/builddir.tgz -C /build
			 docker exec jenkins rm -f /tmp/builddir.tgz
			 rm -f /tmp/builddir.tgz
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=gcc;export PATH=/opt/bin:$PATH;cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_TESTER3=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }

                stage('aarch32 clang') {
            	     agent { label 'docker-aarch32' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch32 clang on" `hostname`
			 tar cfz /tmp/builddir.tgz .
			 docker start jenkins || true
			 docker exec jenkins apt-get update
			 docker exec jenkins apt-get -y dist-upgrade
			 docker exec jenkins rm -rf /build
			 docker exec jenkins mkdir /build
			 docker cp /tmp/builddir.tgz jenkins:/tmp/
			 docker exec jenkins tar xf /tmp/builddir.tgz -C /build
			 docker exec jenkins rm -f /tmp/builddir.tgz
			 rm -f /tmp/builddir.tgz
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=clang-10;export PATH=/opt/bin:$PATH;cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_TESTER3=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }

                stage('Intel Compiler') {
            	     agent { label 'icc' }
            	     steps {
	    	     	 sh '''
                	 echo "Intel Compiler on" `hostname`
		         export CC=icc
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('Arm Mac') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "Cross compiling for Arm Mac with clang on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 export CC=clang
			 (brew update && brew upgrade) || true
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_OSX_ARCHITECTURES=arm64 -DSLEEF_SHOW_CONFIG=1 -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_TESTS=FALSE ..
			 ninja
			 '''
            	     }
                }

                stage('iOS') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "Cross compiling for iOS on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 cmake -GNinja -DBUILD_QUAD=TRUE ..
			 ninja
			 cd ..
			 rm -rf build-cross
			 mkdir build-cross
			 cd build-cross
			 # You need to use ios.toolchain.cmake at https://github.com/leetal/ios-cmake
			 cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=~/ios.toolchain.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DDISABLE_MPFR=TRUE -DDISABLE_SSL=TRUE ..
			 ninja
			 '''
            	     }
                }

                stage('Mac') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "Mac with clang on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 export CC=clang
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `sysctl -n hw.logicalcpu`
		         ninja install
			 '''
            	     }
                }

                stage('Static libs on mac') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "Mac with gcc on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 export CC=gcc-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `sysctl -n hw.logicalcpu`
		         ninja install
			 '''
            	     }
                }

                stage('FreeBSD') {
            	     agent { label 'freebsd' }
            	     steps {
	    	     	 sh '''
                	 echo "FreeBSD on" `hostname`
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 2
		         ninja install
			 '''
            	     }
                }

                stage('Android') {
            	     agent { label 'android-ndk' }
            	     steps {
	    	     	 sh '''
                	 echo "Android on" `hostname`
			 rm -rf build-native
			 mkdir build-native
			 cd build-native
			 /usr/bin/cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE ..
			 ninja
			 cd ..
			 rm -rf build-cross
 			 mkdir build-cross
			 cd build-cross
			 /usr/bin/cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r21d/build/cmake/android.toolchain.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DANDROID_ABI=arm64-v8a -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE ..
			 ninja
			 '''
            	     }
                }

                stage('atom gcc-10') {
            	     agent { label 'atom' }
            	     steps {
	    	     	 sh '''
                	 echo "Atom with gcc-10 on" `hostname`
			 export PATH=/usr/bin:$PATH
		         export CC=gcc-10
		         export CXX=g++-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DENABLE_CXX=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=FALSE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE ..
			 ninja
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('i386 gcc') {
            	     agent { label 'docker-i386' }
            	     steps {
	    	     	 sh '''
                	 echo "i386 gcc on" `hostname`
			 tar cfz /tmp/builddir.tgz .
			 docker start jenkins || true
			 docker exec jenkins apt-get update
			 docker exec jenkins apt-get -y dist-upgrade
			 docker exec jenkins rm -rf /build
			 docker exec jenkins mkdir /build
			 docker cp /tmp/builddir.tgz jenkins:/tmp/
			 docker exec jenkins tar xf /tmp/builddir.tgz -C /build
			 docker exec jenkins rm -f /tmp/builddir.tgz
			 rm -f /tmp/builddir.tgz
			 docker exec jenkins bash -c "set -ev;export PATH=/opt/bin:$PATH;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=gcc;cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }

                stage('i386 clang') {
            	     agent { label 'docker-i386' }
            	     steps {
	    	     	 sh '''
                	 echo "i386 clang on" `hostname`
			 tar cfz /tmp/builddir.tgz .
			 docker start jenkins || true
			 docker exec jenkins apt-get update
			 docker exec jenkins apt-get -y dist-upgrade
			 docker exec jenkins rm -rf /build
			 docker exec jenkins mkdir /build
			 docker cp /tmp/builddir.tgz jenkins:/tmp/
			 docker exec jenkins tar xf /tmp/builddir.tgz -C /build
			 docker exec jenkins rm -f /tmp/builddir.tgz
			 rm -f /tmp/builddir.tgz
			 docker exec jenkins bash -c "set -ev;export PATH=/opt/bin:$PATH;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=clang;cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }

                stage('gcc-4.8 and cmake-3.5.1') {
            	     agent { label 'x86 && gcc-4' }
            	     steps {
	    	     	 sh '''
                	 echo "gcc-4.8 and cmake-3.5.1 on" `hostname`
		         export CC=gcc-4.8.5
			 BUILD_DIR=`pwd`
			 cd ..
			 mv $BUILD_DIR $BUILD_DIR.tmp
			 mkdir $BUILD_DIR
			 mv $BUILD_DIR.tmp $BUILD_DIR/sleef
			 cd $BUILD_DIR
			 cp sleef/CMakeLists.txt.nested ./CMakeLists.txt
			 cp sleef/doc/html/hellox86.c sleef/doc/html/tutorial.c .
			 rm -rf build
 			 mkdir build
			 cd build
			 /usr/local/bin/cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE ..
			 ninja
			 '''
            	     }
                }

                stage('clang-6.0') {
            	     agent { label 'x86 && clang-6' }
            	     steps {
	    	     	 sh '''
                	 echo "clang-6.0 on" `hostname`
		         export CC=clang-6.0
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 cmake -GNinja -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('LTO with gcc') {
            	     agent { label 'x86 && gcc-10' }
            	     steps {
	    	     	 sh '''
                	 echo "LTO with gcc on" `hostname`
			 export PATH=/usr/bin:$PATH:/opt/sde-external-8.56.0-2020-07-05-lin
		         export CC=gcc-10
		         export CXX=g++-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DENABLE_CXX=TRUE -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SHARED_LIBS=FALSE -DENABLE_LTO=TRUE -DDISABLE_FMA4=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('LTO with clang') {
            	     agent { label 'x86 && clang-10' }
            	     steps {
	    	     	 sh '''
                	 echo "LTO with clang on" `hostname`
			 export PATH=/usr/bin:$PATH:/opt/sde-external-8.56.0-2020-07-05-lin
		         export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SHARED_LIBS=FALSE -DENABLE_LTO=TRUE -DLLVM_AR_COMMAND=llvm-ar-10 -DDISABLE_FMA4=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('gcc x86') {
            	     agent { label 'x86 && gcc-10' }
            	     steps {
	    	     	 sh '''
                	 echo "gcc x86_64 on" `hostname`
			 export PATH=$PATH:/opt/sde-external-8.56.0-2020-07-05-lin
		         export CC=gcc-10
		         export CXX=g++-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DENABLE_CXX=TRUE -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SHARED_LIBS=FALSE -DDISABLE_FMA4=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('clang x86') {
            	     agent { label 'x86 && clang-10' }
            	     steps {
	    	     	 sh '''
                	 echo "clang x86_64 on" `hostname`
			 export PATH=$PATH:/opt/sde-external-8.56.0-2020-07-05-lin
		         export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_SCALAR_LIB=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SHARED_LIBS=FALSE -DDISABLE_FMA4=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }
            }
        }
    }
}
