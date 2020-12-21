pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
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
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64 ..
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
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DFORCE_AAVPCS=On -DENABLE_GNUABI=On -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64 ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('aarch64 gcc-10 static and cuda') {
            	     agent { label 'cuda' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch64 gcc-10 and cuda on" `hostname`
			 export CC=gcc-10.2.0
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE -DENFORCE_TESTER3=TRUE -DFORCE_AAVPCS=On -DENABLE_GNUABI=On -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DDISABLE_SVE=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENABLE_CUDA=TRUE -DENFORCE_CUDA=TRUE ..
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
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=gcc;export PATH=/opt/bin:$PATH;cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_TESTER3=TRUE ..;ninja;ctest -j `nproc`"
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
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=clang;export PATH=/opt/bin:$PATH;cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_TESTER3=TRUE ..;ninja;ctest -j `nproc`"
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
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
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
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
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
			 export CC=gcc-9
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
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
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 2
		         ninja install
			 '''
            	     }
                }

                stage('LTO with gcc') {
            	     agent { label 'gcc-10' }
            	     steps {
	    	     	 sh '''
                	 echo "LTO with gcc on" `hostname`
			 export PATH=$PATH:/opt/sde-external-8.56.0-2020-07-05-lin
		         export CC=gcc-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SHARED_LIBS=FALSE -DENABLE_LTO=TRUE -DDISABLE_FMA4=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('LTO with clang') {
            	     agent { label 'clang-10' }
            	     steps {
	    	     	 sh '''
                	 echo "LTO with clang on" `hostname`
			 export PATH=$PATH:/opt/sde-external-8.56.0-2020-07-05-lin
		         export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DBUILD_SHARED_LIBS=FALSE -DENABLE_LTO=TRUE -DLLVM_AR_COMMAND=llvm-ar-10 -DDISABLE_FMA4=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
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
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=gcc;cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..;ninja;ctest -j `nproc`"
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
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=clang;cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }

                stage('gcc-4.8 and cmake-3.5.1') {
            	     agent { label 'gcc-4' }
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
			 /usr/local/bin/cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE ..
			 ninja
			 '''
            	     }
                }

                stage('clang-6.0') {
            	     agent { label 'clang-6' }
            	     steps {
	    	     	 sh '''
                	 echo "clang-6.0 on" `hostname`
		         export CC=clang-6.0
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
			 cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 ninja
		         ctest -j `nproc`
			 '''
            	     }
                }
            }
        }
    }
}
