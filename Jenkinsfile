pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
	    	sh '''
                echo "Building.."
		rm -rf build
 		mkdir build
		cd build
		#export CC=gcc
		export CC=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/bin/armclang
		cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
		make -j 4 all
		'''
            }
        }
        stage('Test') {
            steps {
	    	sh '''
                echo 'Testing..'
		export PATH=$PATH:/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/bin
		cd build
		export CTEST_OUTPUT_ON_FAILURE=TRUE
		ctest -j 4
		'''
            }
        }
        stage('Deploy') {
            steps {
	    	sh '''
                echo 'Deploying....'
		cd build
		make install
		'''
            }
        }
    }
}