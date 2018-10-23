pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('AArch64 SVE') {
            	     agent { label 'aarch64' }
            	     steps {
	    	     	 sh '''
                	 echo "AArch64 SVE on" `hostname`
			 export PATH=$PATH:/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/bin
			 export LD_LIBRARY_PATH=/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/lib:/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
			 export CC=/opt/arm/arm-hpc-compiler-18.4_Generic-AArch64_Ubuntu-16.04_aarch64-linux/bin/armclang
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('AArch64 SVE + Advanced SIMD with AAVPCS + LIBSLEEFGNUABI') {
            	     agent { label 'aarch64' }
            	     steps {
	    	     	 sh '''
                	 echo "AArch64 SVE on" `hostname`
			 export PATH=$PATH:/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/bin
			 export LD_LIBRARY_PATH=/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/lib:/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
			 export CC=/opt/arm/arm-hpc-compiler-18.4_Generic-AArch64_Ubuntu-16.04_aarch64-linux/bin/armclang
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DFORCE_AAVPCS=On -DENABLE_GNUABI=On ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('Intel Compiler') {
                    agent { label 'icc' }
                    steps {
    	    	        sh '''
                        echo "Intel Compiler on" `hostname`
			export PATH=$PATH:/export/opt/sde-external-8.16.0-2018-01-30-lin:/export/opt/compilers_and_libraries_2018/linux/bin/intel64
                        export LD_LIBRARY_PATH=/export/opt/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin/
		        export CC=icc
		        rm -rf build
 		        mkdir build
		        cd build
		        cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
		        make -j 4 all
			export OMP_WAIT_POLICY=passive
		        export CTEST_OUTPUT_ON_FAILURE=TRUE
		        ctest -j 4
		        make install
		        '''
                    }
                }

                stage('FMA4') {
            	     agent { label 'fma4' }
            	     steps {
	    	     	 sh '''
                	 echo "FMA4 on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/bin:/opt/sde-external-8.16.0-2018-01-30-lin
			 export LD_LIBRARY_PATH=/opt/local/lib:/opt/lib
		         export CC=gcc-8.2.0
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('GCC-4.8') {
            	     agent { label 'x86' }
            	     steps {
	    	     	 sh '''
                	 echo "gcc-4 on" `hostname`
			 export PATH=$PATH:/opt/sde-external-8.16.0-2018-01-30-lin
		         export CC=gcc-4.8
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('Static libs on mac') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "On" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DENFORCE_TESTER3=TRUE ..
			 make -j 2 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 2
		         make install
			 '''
            	     }
                }

                stage('Windows') {
            	     agent { label 'win' }
            	     steps {
	    	     	 bat '''
			 set "PROJECT_DIR=%cd%"
			 set "ORG_PATH=%PATH%"
			 PATH C:/Cygwin64/bin;C:/Cygwin64/usr/bin;%PROJECT_DIR%/build-cygwin/bin;%PATH%
			 rmdir /S /Q build-cygwin
			 C:/Cygwin64/bin/bash -c 'mkdir build-cygwin;cd build-cygwin;cmake -g"Unix Makefiles" ..;make -j 4'
			 del /Q /F %PROJECT_DIR%/build-cygwin/bin/iut*
			 PATH %ORG_PATH%;C:/Cygwin64/bin;C:/Cygwin64/usr/bin;%PROJECT_DIR%/build-cygwin/bin;%PROJECT_DIR%/build/bin
			 cd %PROJECT_DIR%
			 rmdir /S /Q build
                         mkdir build
                         cd build
                         cmake -G"Visual Studio 15 2017 Win64" .. -DCMAKE_INSTALL_PREFIX=install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE -DENFORCE_TESTER3=TRUE
                         cmake --build . --target install --config Release
			 ctest --output-on-failure -j 4 -C Release
			 '''
            	     }
                }

                stage('i386') {
            	     agent { label 'i386' }
            	     steps {
	    	     	 sh '''
                	 echo "i386 on" `hostname`
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

		stage('PowerPC VSX') {
            	     agent { label 'x86 && xenial' }
            	     steps {
	    	     	 sh '''
                	 echo "PowerPC VSX on" `hostname`
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 cmake -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
			 cd ..
			 export PATH=$PATH:`pwd`/travis
			 export QEMU_CPU=POWER8
			 chmod +x travis/ppc64el-cc
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_TOOLCHAIN_FILE=../travis/toolchain-ppc64el.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DEMULATOR=qemu-ppc64le-static -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
		 }

                stage('AArch32') {
            	     agent { label 'aarch32' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch32 on" `hostname`
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 4 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
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
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE ..
			 make -j 3 all
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 3
		         make install
			 '''
            	     }
                }
            }
        }
    }
}
