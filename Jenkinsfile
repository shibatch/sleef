pipeline {
    agent any

    stages {
        stage('Preamble') {
            parallel {
                stage('aarch32 gcc') {
            	     agent { label 'docker-aarch32' }
            	     steps {
	    	     	 sh '''
                	 echo "aarch32 gcc on" `hostname`
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
			 docker exec jenkins bash -c "set -ev;export OMP_WAIT_POLICY=passive;cd /build;rm -rf build;mkdir build;cd build;export CC=gcc;export PATH=/opt/bin:$PATH;cmake -GNinja -DBUILD_QUAD=TRUE -DENFORCE_TESTER3=TRUE ..;ninja;ctest -j `nproc`"
			 docker stop jenkins
			 '''
            	     }
                }
            }
        }
    }
}
