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
		export CC=gcc
		cmake -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 ..
		make -j 4 all
		'''
            }
        }
        stage('Test') {
            steps {
	    	sh '''
                echo 'Testing..'
		export CTEST_OUTPUT_ON_FAILURE=TRUE
		ctest --verbose -j 2
		'''
            }
        }
        stage('Deploy') {
            steps {
	    	sh '''
                echo 'Deploying....'
		make install
		'''
            }
        }
    }
}