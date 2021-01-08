pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('iOS') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "Cross compiling for iOS on" `hostname`
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 cmake -GNinja -DBUILD_QUAD=TRUE ..
			 ninja
			 cd ..
			 rm -rf build-cross
			 mkdir build-cross
			 cd build-cross
			 cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=~/ios.toolchain.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DBUILD_QUAD=TRUE -DDISABLE_MPFR=TRUE -DDISABLE_SSL=TRUE ..
			 ninja
			 '''
            	     }
                }
            }
        }
    }
}
