pipeline {
    agent any

    stages {
        stage('Everything') {
            parallel {
                stage('AArch64 SVE') {
            	     agent { label 'aarch64' }
            	     steps {
	    	     	 sh '''
                	 echo "AArch64 SVE on" `hostname`
			 export PATH=$PATH:/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/bin
			 export LD_LIBRARY_PATH=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
			 export CC=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/bin/armclang
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
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
		        cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
		        make -j 4 all
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
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
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
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('GCC-8.1') {
            	     agent { label 'x86' }
            	     steps {
	    	     	 sh '''
                	 echo "gcc-8 on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/bin:/opt/sde-external-8.16.0-2018-01-30-lin
			 export LD_LIBRARY_PATH=/opt/local/lib:/opt/lib
		         export CC=gcc-8.1.0
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('clang-6.0') {
            	     agent { label 'x86' }
            	     steps {
	    	     	 sh '''
                	 echo "clang-6 on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/bin:/opt/sde-external-8.16.0-2018-01-30-lin
			 export LD_LIBRARY_PATH=/opt/local/lib:/opt/lib
		         export CC=clang-6.0
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }

                stage('Static lib on mac') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "On" `hostname`
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE ..
			 make -j 2 all
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 2
		         make install
			 '''
            	     }
                }
            }
        }
    }
}
