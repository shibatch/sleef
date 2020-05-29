pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('Armclang') {
            	     agent { label 'armclang' }
            	     steps {
	    	     	 sh '''
                	 echo "armclang+SVE on" `hostname`
			 export CC=armclang
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('Armclang AAVPCS') {
            	     agent { label 'armclang' }
            	     steps {
	    	     	 sh '''
                	 echo "armclang+SVE+AAVPCS on" `hostname`
			 export CC=armclang
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DFORCE_AAVPCS=On -DENABLE_GNUABI=On -DBUILD_QUAD=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('Intel Compiler') {
            	     agent { label 'icc' }
            	     steps {
	    	     	 sh '''
                	 echo "Intel Compiler on" `hostname`
		         export CC=icc
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE ..
			 ninja
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
