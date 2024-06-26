#!/bin/sh
g++ -g -Wall -std=c++17 -O3 -fopenmp $(pkg-config --cflags eigen3) src/main.cpp -I /usr/include/python3.10 -lpython3.10
