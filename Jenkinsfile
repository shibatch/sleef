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
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 rm -rf build-native
 			 mkdir build-native
			 cd build-native
			 cmake -GNinja -DBUILD_QUAD=TRUE ..
			 ninja
			 cd ..
			 rm -rf build-cross
			 mkdir build-cross
			 cd build-cross
			 # You need to use ios.toolchain.cmake at https://github.com/leetal/ios-cmake
			 cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=~/ios.toolchain.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DBUILD_QUAD=TRUE -DDISABLE_MPFR=TRUE -DDISABLE_SSL=TRUE ..
			 ninja
			 '''
            	     }
                }
            }
        }
    }
}
