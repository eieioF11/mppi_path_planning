# mppi_path_planning
MPPI経路計画

## Install
### 依存関係インストール
```bash
sudo apt install python3-dev python3-numpy python3-matplotlib libeigen3-dev
```
#### install matplotlibcpp17

https://soblin.github.io/matplotlibcpp17/
```bash
git clone https://github.com/soblin/matplotlibcpp17.git
cd matplotlibcpp17
mkdir build; cd build;
cmake .. -DADD_DEMO=0
make -j
sudo make install
```

### Clone
```bash
git clone https://github.com/eieioF11/mppi_path_planning.git
cd mppi_path_planning
git submodule update --init --recursive
```

## Build

```bash
g++ -g -Wall -std=c++17 -O3 -fopenmp $(pkg-config --cflags eigen3) src/main.cpp -I /usr/include/python3.10 -lpython3.10
```
or
```bash
. build.sh
```
