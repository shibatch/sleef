pipeline {
    agent { label 'jenkinsfile' }

    stages {
        stage('Preamble') {
            parallel {
                stage('x86_64 linux clang-19-lto') {
            	     agent { label 'x86_64 && ubuntu24 && avx512f' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "x86_64 clang-19 with LTO on" `hostname`
			 export CC=clang-19
			 export CXX=clang++-19
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEFDFT_ENABLE_STREAM=True -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SSE2=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENFORCE_TESTER=True -DSLEEF_ENABLE_LTO=True -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld-19"
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('x86_64 linux clang-19-asan') {
            	     agent { label 'x86_64 && ubuntu24 && avx512f' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "x86_64 clang-19 with ASAN on" `hostname`
			 export CC=clang-19
			 export CXX=clang++-19
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEFDFT_ENABLE_STREAM=True -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SSE2=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENFORCE_TESTER=True -DSLEEF_ENABLE_ASAN=True -DSLEEFDFT_ENABLE_PARALLELFOR=True
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('x86_64 linux clang-19 noexp') {
            	     agent { label 'x86_64 && ubuntu24 && avx512f' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "x86_64 clang-19 without experimental features on" `hostname`
			 export CC=clang-19
			 export CXX=clang++-19
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEFDFT_ENABLE_STREAM=True -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENFORCE_TESTER=True
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('x86_64 linux gcc-13') {
            	     agent { label 'x86_64 && ubuntu24 && cuda' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "x86_64 gcc-13 on" `hostname`
			 export CC=gcc-13
			 export CXX=g++-13
			 export CUDACXX=/opt/cuda-12.6/bin/nvcc
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_CUDA=True -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SSE2=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENFORCE_TESTER=True -DSLEEF_BUILD_SHARED_LIBS=ON
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('x86_64 windows vs2022') {
            	     agent { label 'windows11 && vs2022' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
		     	 bat """
			 call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"
			 if not %ERRORLEVEL% == 0 exit /b %ERRORLEVEL%
			 if exist ..\\install\\ rmdir /S /Q ..\\install
			 set PATH=%PATH%;%CD%\\..\\install\\bin
			 call "winbuild-msvc.bat" -DCMAKE_INSTALL_PREFIX=../../install -DCMAKE_BUILD_TYPE=Release -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_SSE2=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_BUILD_SHARED_LIBS=ON
			 if not %ERRORLEVEL% == 0 exit /b %ERRORLEVEL%
			 ctest -j 4 --output-on-failure
			 exit /b %ERRORLEVEL%
			 """
		     }
		}

                stage('x86_64 windows clang noexp') {
            	     agent { label 'windows11 && vs2022' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
		     	 bat """
			 call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"
			 if not %ERRORLEVEL% == 0 exit /b %ERRORLEVEL%
			 if exist ..\\install\\ rmdir /S /Q ..\\install
			 set PATH=%PATH%;%CD%\\..\\install\\bin
			 call "winbuild-clang.bat" -DCMAKE_INSTALL_PREFIX=../../install -DCMAKE_BUILD_TYPE=Release -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_SSL=False -DSLEEFDFT_ENABLE_PARALLELFOR=True
			 if not %ERRORLEVEL% == 0 exit /b %ERRORLEVEL%
			 ctest -j 4 --output-on-failure
			 exit /b %ERRORLEVEL%
			 """
		     }
		}

                stage('riscv linux gcc-14') {
            	     agent { label 'riscv && ubuntu24' }
                     options { skipDefaultCheckout() }
            	     steps {
		         script {
			     System.setProperty("org.jenkinsci.plugins.durabletask.BourneShellScript.HEARTBEAT_CHECK_INTERVAL", "86400");
			 }
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "riscv gcc-14 on" `hostname`
			 export CC=gcc-14
			 export CXX=g++-14
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_DFT=False -DSLEEF_ENFORCE_DFT=False -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_RVVM1=True -DSLEEF_ENFORCE_RVVM2=True
			 cmake -E time oomstaller --max-parallel `nproc` ninja -j `nproc`
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
		}

                stage('riscv linux clang-19') {
            	     agent { label 'riscv && ubuntu24' }
                     options { skipDefaultCheckout() }
            	     steps {
		         script {
			     System.setProperty("org.jenkinsci.plugins.durabletask.BourneShellScript.HEARTBEAT_CHECK_INTERVAL", "86400");
			 }
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "riscv clang-19 on" `hostname`
			 export CC=clang-19
			 export CXX=clang++-19
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_DFT=False -DSLEEF_ENFORCE_DFT=False -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_RVVM1=True -DSLEEF_ENFORCE_RVVM2=True
			 cmake -E time oomstaller --max-parallel `nproc` ninja -j `nproc`
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
		}

                stage('arm32 linux gcc-12') {
            	     agent { label 'armv7 && debian12' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "arm32 gcc-12 on" `hostname`
			 export CC=gcc-12
			 export CXX=g++-12
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_PURECFMA_SCALAR=False
			 cmake -E time oomstaller --max-parallel `nproc` ninja -j `nproc`
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('aarch64 linux clang-19-lto') {
            	     agent { label 'aarch64 && ubuntu24' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "aarch64 clang-19 with LTO on" `hostname`
			 export CC=clang-19
			 export CXX=clang++-19
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64-static -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_LTO=True -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld-19" -DSLEEFDFT_ENABLE_PARALLELFOR=True
			 cmake -E time oomstaller --max-parallel `nproc` ninja -j `nproc`
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('aarch64 linux gcc-14') {
            	     agent { label 'aarch64 && ubuntu24' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "aarch64 gcc-14 on" `hostname`
			 export CC=gcc-14
			 export CXX=g++-14
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64-static -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENFORCE_TESTER=False -DSLEEF_BUILD_SHARED_LIBS=ON -DSLEEFDFT_ENABLE_PARALLELFOR=True
			 cmake -E time oomstaller --max-parallel `nproc` ninja -j `nproc`
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('aarch64 linux gcc-14 noexp') {
            	     agent { label 'aarch64 && ubuntu24' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "aarch64 gcc-14 without experimental features on" `hostname`
			 export CC=gcc-14
			 export CXX=g++-14
			 export INSTALL_PREFIX=`pwd`/install
			 export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_BUILD_SHARED_LIBS=ON
			 cmake -E time oomstaller --max-parallel `nproc` ninja -j `nproc`
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

		stage('cross-ppc64el gcc') {
            	     agent { label 'x86_64 && ubuntu24 && cuda' }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "Cross ppc64el gcc on" `hostname`
			 export NATIVE_INSTALL_PREFIX=`pwd`/install-native
			 export LD_LIBRARY_PATH=$NATIVE_INSTALL_PREFIX/lib
			 export CROSS_INSTALL_PREFIX=`pwd`/install
			 export QEMU_LD_PREFIX=$CROSS_INSTALL_PREFIX/lib
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 export NATIVE_BUILD_DIR=`pwd`
			 cmake -GNinja .. -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DCMAKE_INSTALL_PREFIX=$NATIVE_INSTALL_PREFIX
			 cmake -E time ninja
			 cd ..
 			 mkdir build
			 cd build
			 cmake -GNinja .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/ppc64el-gcc.cmake -DNATIVE_BUILD_DIR=$NATIVE_BUILD_DIR -DCMAKE_INSTALL_PREFIX=$CROSS_INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_VSX=True -DSLEEF_ENFORCE_VSX3=True -DSLEEF_BUILD_SHARED_LIBS=ON
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
			 export LD_LIBRARY_PATH=/usr/powerpc64le-linux-gnu/lib
		         ctest -j `nproc`
			 ninja install
			 '''
            	     }
		 }

		stage('cross-s390x gcc') {
            	     agent { label 'x86_64 && ubuntu24 && cuda' }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "Cross s390x gcc on" `hostname`
			 export NATIVE_INSTALL_PREFIX=`pwd`/install-native
			 export LD_LIBRARY_PATH=$NATIVE_INSTALL_PREFIX/lib
			 export CROSS_INSTALL_PREFIX=`pwd`/install
			 export QEMU_LD_PREFIX=$CROSS_INSTALL_PREFIX/lib
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 export NATIVE_BUILD_DIR=`pwd`
			 cmake -GNinja .. -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DCMAKE_INSTALL_PREFIX=$NATIVE_INSTALL_PREFIX
			 cmake -E time ninja
			 cd ..
 			 mkdir build
			 cd build
			 cmake -GNinja .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/s390x-gcc.cmake -DNATIVE_BUILD_DIR=$NATIVE_BUILD_DIR -DCMAKE_INSTALL_PREFIX=$CROSS_INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_ENFORCE_VXE=True -DSLEEF_ENFORCE_VXE2=True
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 ninja install
			 '''
            	     }
		 }

		stage('cross-ppc64el gcc noexp') {
            	     agent { label 'x86_64 && ubuntu24 && cuda' }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "Cross ppc64el gcc without experimental features on" `hostname`
			 export NATIVE_INSTALL_PREFIX=`pwd`/install-native
			 export LD_LIBRARY_PATH=$NATIVE_INSTALL_PREFIX/lib
			 export CROSS_INSTALL_PREFIX=`pwd`/install
			 export QEMU_LD_PREFIX=$CROSS_INSTALL_PREFIX/lib
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 export NATIVE_BUILD_DIR=`pwd`
			 cmake -GNinja .. -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DCMAKE_INSTALL_PREFIX=$NATIVE_INSTALL_PREFIX
			 cmake -E time ninja
			 cd ..
 			 mkdir build
			 cd build
			 cmake -GNinja .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/ppc64el-gcc.cmake -DNATIVE_BUILD_DIR=$NATIVE_BUILD_DIR -DCMAKE_INSTALL_PREFIX=$CROSS_INSTALL_PREFIX -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_DFT=TRUE -DSLEEF_ENFORCE_TESTER4=True -DSLEEF_BUILD_SHARED_LIBS=ON
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
			 export LD_LIBRARY_PATH=/usr/powerpc64le-linux-gnu/lib
		         ctest -j `nproc`
			 ninja install
			 '''
            	     }
		 }
            }
        }
    }
}
