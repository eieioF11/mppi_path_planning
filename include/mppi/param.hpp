#pragma once
// std
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdio.h>
#include <unistd.h>
#include <vector>
// Eigen
#include <Eigen/Dense>

namespace MPPI {
  struct param_t{
    size_t T; // time horizon
    size_t K; // number of samples
    double dt; // control interval
    double lambda; // temperature parameter
    double alpha;
    double window_size;
    Eigen::Matrix<double, 3, 3> sigma; // disturbance of noise
    Eigen::Matrix<double, 6, 6> Q; // ステージコストの状態量重み
    Eigen::Matrix<double, 3, 3> R; // ステージコストの入力量重み
    Eigen::Matrix<double, 6, 6> Q_T; // 終端コスト
    double obstacle_cost;
    double robot_size; // [m]　半径
  };
} // namespace MPPI