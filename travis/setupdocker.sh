#!/bin/bash
set -ev
docker pull ubuntu:xenial;
docker run -d --name xenial -dti ubuntu:xenial bash;
tar cfz /tmp/builddir.tgz .
docker cp /tmp/builddir.tgz xenial:/tmp/
docker exec xenial mkdir /build
docker exec xenial tar xfz /tmp/builddir.tgz -C /build
docker exec xenial rm -f /tmp/builddir.tgz
