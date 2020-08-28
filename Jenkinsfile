pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
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
			 cp sleef/doc/*.c .
			 rm -rf build
 			 mkdir build
			 cd build
			 /usr/local/bin/cmake -GNinja ..
			 ninja
			 '''
            	     }
                }
            }
        }
    }
}
