pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('Android') {
            	     agent { label 'android-ndk' }
            	     steps {
	    	     	 sh '''
                	 echo "Android on" `hostname`
			 rm -rf build-native
			 mkdir build-native
			 cd build-native
			 /usr/bin/cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE ..
			 cd ..
			 rm -rf build-cross
 			 mkdir build-cross
			 cd build-cross
			 /usr/bin/cmake -GNinja -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r21d/build/cmake/android.toolchain.cmake -DNATIVE_BUILD_DIR=`pwd`/../build-native -DANDROID_ABI=arm64-v8a -DBUILD_INLINE_HEADERS=TRUE -DBUILD_QUAD=TRUE ..
			 ninja
			 '''
            	     }
                }
            }
        }
    }
}
