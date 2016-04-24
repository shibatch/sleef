IUT.class : IUT.java org/naokishibata/sleef/FastMath.java
	javac IUT.java

doc : org/naokishibata/sleef/FastMath.java
	javadoc -d ./javadoc -subpackages org.naokishibata

../tester/tester :
	cd ../tester; make tester

../tester/testersp :
	cd ../tester; make testersp

test : IUT.class ../tester/tester ../tester/testersp
	../tester/tester java -ea IUT
	../tester/testersp java -ea IUT

clean :
	rm -f *~ IUT.class
	rm -rf javadoc
	find org -name "*.class" -exec rm {} ";"
	find org -name "*~" -exec rm {} ";"
