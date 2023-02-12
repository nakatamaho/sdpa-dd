# sdpa-dd
# how to build
I verified build on Ubuntu 20.04
```
git clone https://github.com/nakatamaho/sdpa-dd.git
cd sdpa-dd
aclocal ; autoconf ; automake --add-missing
autoreconf --force --install
./configure --enable-openmp=yes
make -j4
```
