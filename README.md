[![Travis Build Status](https://travis-ci.org/shibatch/sleef.svg?branch=master)](https://travis-ci.org/shibatch/sleef)

Main Page   : https://sleef.org/
GitHub Repo : https://github.com/shibatch/sleef

SLEEF is a library that implements vectorized versions of C standard math functions. This library also includes DFT subroutines.

### How to build

1. Check out the source code from our GitHub repository : 
`git clone https://github.com/shibatch/sleef`

2. Make a separate directory to create an out-of-source build : 
`cd sleef && mkdir build && cd build`

3. Run cmake to configure the project : 
`cmake ..`

4. Run make to build the project : 
`make -j 1`

Refer to our web page for detailed instructions : https://sleef.org/compile.xhtml


### License

The software is distributed under the Boost Software License, Version
1.0.  See accompanying file LICENSE.txt or copy at
http://www.boost.org/LICENSE_1_0.txt.
Contributions to this project are accepted under the same license.

Copyright Naoki Shibata and contributors 2010 - 2021.
