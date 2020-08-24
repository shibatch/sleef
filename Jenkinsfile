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

                stage('Static libs on mac') {
            	     agent { label 'mac' }
            	     steps {
	    	     	 sh '''
                	 echo "macOS on" `hostname`
			 export PATH=$PATH:/opt/local/bin:/opt/local/bin:/usr/local/bin:/usr/bin:/bin
			 export CC=gcc-9
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DBUILD_SHARED_LIBS=FALSE -DOPENSSL_ROOT_DIR=/usr/local/opt/openssl -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `sysctl -n hw.logicalcpu`
		         ninja install
			 '''
            	     }
                }

                stage('FreeBSD') {
            	     agent { label 'freebsd' }
            	     steps {
	    	     	 sh '''
                	 echo "FreeBSD on" `hostname`
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j 2
		         ninja install
			 '''
            	     }
                }

                stage('LTO with gcc') {
            	     agent { label 'x86 && gcc-10' }
            	     steps {
	    	     	 sh '''
                	 echo "LTO with gcc on" `hostname`
		         export CC=gcc-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_SHARED_LIBS=FALSE -DENABLE_LTO=TRUE ..
			 ninja
			 export OMP_WAIT_POLICY=passive
		         export CTEST_OUTPUT_ON_FAILURE=TRUE
		         ctest -j `nproc`
		         ninja install
			 '''
            	     }
                }

                stage('LTO with clang') {
            	     agent { label 'x86 && clang-10' }
            	     steps {
	    	     	 sh '''
                	 echo "LTO with clang on" `hostname`
		         export CC=clang-10
			 rm -rf build
 			 mkdir build
			 cd build
			 cmake -GNinja -DCMAKE_INSTALL_PREFIX=../install -DSLEEF_SHOW_CONFIG=1 -DENFORCE_TESTER3=TRUE -DBUILD_QUAD=TRUE -DBUILD_SHARED_LIBS=FALSE -DENABLE_LTO=TRUE ..
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
