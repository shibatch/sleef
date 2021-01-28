pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('ppc64le gcc') {
            	     agent { label 'ppc64' }
            	     steps {
	    	     	 sh '''
                	 echo "ppc64 gcc on" `hostname`
			 export CC=gcc-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VSX=TRUE -DENFORCE_VSX3=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('ppc64le clang') {
            	     agent { label 'ppc64' }
            	     steps {
	    	     	 sh '''
                	 echo "ppc64 clang on" `hostname`
			 export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VSX=TRUE -DENFORCE_VSX3=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('s390x gcc') {
            	     agent { label 's390x' }
            	     steps {
	    	     	 sh '''
                	 echo "s390x gcc on" `hostname`
			 export CC=gcc
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -G Ninja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VXE=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

                stage('s390x clang') {
            	     agent { label 's390x' }
            	     steps {
	    	     	 sh '''
                	 echo "s390x clang on" `hostname`
			 export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -G Ninja -DRUNNING_ON_TRAVIS=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_DFT=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_VXE=TRUE -DENFORCE_VXE2=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
			 '''
            	     }
                }

            }
        }
    }
}
