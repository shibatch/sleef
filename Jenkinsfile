pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('i386') {
            	     agent { label 'docker-i386' }
            	     steps {
	    	     	 sh '''
                	 echo "i386 on" `hostname`
			 tar cfz /tmp/builddir.tgz .
			 docker start jenkins || true
			 docker exec jenkins apt-get update
			 docker exec jenkins apt-get -y dist-upgrade
			 docker exec jenkins rm -rf /build
			 docker exec jenkins mkdir /build
			 docker cp /tmp/builddir.tgz jenkins:/tmp/
			 docker exec jenkins tar xf /tmp/builddir.tgz -C /build
			 docker exec jenkins rm -f /tmp/builddir.tgz
			 rm -f /tmp/builddir.tgz
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;cmake -GNinja -DBUILD_QUAD=TRUE -DBUILD_INLINE_HEADERS=TRUE -DENFORCE_SSE2=TRUE -DENFORCE_SSE4=TRUE -DENFORCE_AVX=TRUE -DENFORCE_FMA4=TRUE -DENFORCE_AVX2=TRUE -DENFORCE_AVX512F=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }
            }
        }
    }
}
