#!/bin/sh
g++ -Wall -fopenmp -std=c++17 $(pkg-config --cflags eigen3) src/main.cpp -I /usr/include/python3.10 -lpython3.10 -O3
