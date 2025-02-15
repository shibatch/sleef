pipeline {
    agent { label 'jenkinsfile' }

    stages {
        stage('Preamble') {
            parallel {
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
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=../../install -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENABLE_CUDA=True -DSLEEF_ENFORCE_CUDA=True -DSLEEF_BUILD_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SSE2=TRUE -DSLEEF_ENFORCE_SSE4=TRUE -DSLEEF_ENFORCE_AVX=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENABLE_TESTER4=True
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
			 call "winbuild-msvc.bat" -DCMAKE_BUILD_TYPE=Release -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENFORCE_SSE2=TRUE -DSLEEF_ENFORCE_SSE4=TRUE -DSLEEF_ENFORCE_AVX=TRUE -DSLEEF_ENFORCE_AVX2=TRUE -DSLEEF_ENFORCE_AVX512F=TRUE -DSLEEF_ENABLE_TESTER4=True -DSLEEF_DISABLE_SSL=True
			 if not %ERRORLEVEL% == 0 exit /b %ERRORLEVEL%
			 ctest -j 4 --output-on-failure
			 exit /b %ERRORLEVEL%
			 """
		     }
		}

                stage('riscv linux gcc-14') {
            	     agent { label 'riscv && ubuntu23' }
                     options { skipDefaultCheckout() }
            	     steps {
		         script {
			     System.setProperty("org.jenkinsci.plugins.durabletask.BourneShellScript.HEARTBEAT_CHECK_INTERVAL", "86400");
			 }
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "riscv gcc-14 on" `hostname`
			 export CC=gcc-14.2.0
			 export CXX=g++-14.2.0
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=../../install -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENABLE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_RVVM1=True -DSLEEF_ENFORCE_RVVM2=True
			 cmake -E time oomstaller ninja -j `nproc`
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
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=../../install -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_ENABLE_TESTER4=True -DSLEEF_ENABLE_TESTER=False
			 cmake -E time oomstaller ninja -j `nproc`
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('aarch64 linux gcc-14') {
            	     agent { label 'aarch64 && ubuntu24 && apple' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "aarch64 gcc-14 on" `hostname`
			 export CC=gcc-14
			 export CXX=g++-14
 			 mkdir build
			 cd build
			 cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=../../install -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_DFT=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_INLINE_HEADERS=TRUE -DSLEEF_ENFORCE_SVE=TRUE -DEMULATOR=qemu-aarch64-static -DSLEEF_ENABLE_TESTER4=True -DSLEEF_ENABLE_TESTER=False
			 cmake -E time ninja
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
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 cmake -GNinja .. -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_DFT=TRUE
			 cmake -E time ninja
			 cd ..
 			 mkdir build
			 cd build
			 cmake -GNinja .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/ppc64el-gcc.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_TESTER3=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_DFT=TRUE -DSLEEF_ENABLE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_VSX=True -DSLEEF_ENFORCE_VSX3=True
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
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
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 cmake -GNinja .. -DSLEEF_SHOW_CONFIG=1 -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_DFT=TRUE
			 cmake -E time ninja
			 cd ..
 			 mkdir build
			 cd build
			 cmake -GNinja .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/s390x-gcc.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DSLEEF_ENFORCE_TESTER3=TRUE -DSLEEF_BUILD_QUAD=TRUE -DSLEEF_BUILD_DFT=TRUE -DSLEEF_ENABLE_TESTER4=True -DSLEEF_ENABLE_TESTER=False -DSLEEF_ENFORCE_VXE=True -DSLEEF_ENFORCE_VXE2=True
			 cmake -E time ninja
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
