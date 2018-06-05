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
		export PATH=$PATH:/opt/arm/arm-instruction-emulator-1.2.1_Generic-AArch64_Ubuntu-14.04_aarch64-linux/bin
		export CC=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/bin/armclang
		export LD_LIBRARY_PATH=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
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
		export LD_LIBRARY_PATH=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
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
		export LD_LIBRARY_PATH=/opt/arm/arm-hpc-compiler-18.1_Generic-AArch64_Ubuntu-16.04_aarch64-linux/lib
		cd build
		make install
		'''
            }
        }
    }
}