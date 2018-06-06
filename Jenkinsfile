pipeline {
    agent any

    stages {
        stage('Everything') {
            parallel {
                stage('Testing on AArch64') {
            	     agent { label 'aarch64' }
            	     steps {
	    	     	 sh '''
                	 echo "AArch64"
			 rm -rf build
 			 mkdir build
			 cd build
			 export PATH=$PATH:/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/bin
			 export CC=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/bin/armclang
			 export LD_LIBRARY_PATH=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
			 cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
			 make -j 4 all
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 4
		         make install
			 '''
            	     }
                }
                
                stage('Testing with Intel Compiler') {
                    agent { label 'icc' }
                    steps {
    	    	        sh '''
                        echo "Intel Compiler"
		        rm -rf build
 		        mkdir build
		        cd build
		        export CC=icc
		        cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
		        make -j 4 all
		        export CTEST_OUTPUT_ON_FAILURE=TRUE
		        ctest -j 4
		        make install
		        '''
                    }
                }
            }
        }
    }
}