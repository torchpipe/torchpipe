precxx11

```
wget https://github.com/google/googletest/archive/release-1.12.1.tar.gz -O gtest_1_12_1.tar.gz
tar -zxvf gtest_1_12_1.tar.gz

cd googletest-release-1.12.1/
add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)
cmake -DBUILD_SHARED_LIBS=ON .
make

```

cxx11 abi:

```
apt-get install libgtest-dev
apt-get install libgmock-dev

```