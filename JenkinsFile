pipeline {
    agent { label 'jenkinsfile' }

    stages {
        stage('Preamble') {
            parallel {
                stage('x86_64 linux gcc-13') {
            	     agent { label 'x86_64 && ubuntu24' }
                     options { skipDefaultCheckout() }
            	     steps {
                         cleanWs()
                         checkout scm
	    	     	 sh '''
                	 echo "x86_64 gcc-13 on" `hostname`
			 export CC=gcc-13
			 export CXX=g++-13
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_QUAD=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..
			 cmake -E time ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }
            }
        }
    }
}
