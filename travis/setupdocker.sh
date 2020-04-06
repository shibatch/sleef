#!/bin/bash
set -ev
docker pull ubuntu:bionic;
docker run -d --name bionic -dti ubuntu:bionic bash;
tar cfz /tmp/builddir.tgz .
docker cp /tmp/builddir.tgz bionic:/tmp/
docker exec bionic mkdir /build
docker exec bionic tar xfz /tmp/builddir.tgz -C /build
docker exec bionic rm -f /tmp/builddir.tgz
