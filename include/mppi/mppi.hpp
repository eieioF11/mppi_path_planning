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
#include <random>
#include <stdio.h>
#include <unistd.h>
#include <vector>
// OpenMP

#if defined(USE_OMP)
#include <omp.h>
#endif
// Eigen
#include <Eigen/Dense>
//
#include "math_util.hpp"
#include "param.hpp"

namespace MPPI {
  class MPPIPathPlanner {
  private:
    constexpr static int DIM_U = 3;
    constexpr static int DIM_X = 6;
    typedef Eigen::Matrix<double, DIM_U, 1> vec3_t;
    typedef Eigen::Matrix<double, DIM_X, 1> vec6_t;
    typedef vec3_t control_t;
    typedef vec6_t state_t;

    std::function<state_t(state_t, control_t, double)> f_;
    state_t x_t;
    param_t param_;
    double ganmma_;
    std::array<double, 3> VEL_MAX = {1.0, 1.0, 1.0};
    std::array<double, 3> VEL_MIN = {-1.0, -1.0, -1.0};
    std::vector<control_t> u_, u_pre_;
    std::vector<state_t> opt_path_;
    std::vector<std::vector<state_t>> sample_path_;

    std::vector<double> weight_;
    std::vector<double> stage_cost_;

    control_t clamp(const control_t& v) {
      control_t res;
      for (int i = 0; i < DIM_U; i++)
        res(i) = std::clamp(v(i), VEL_MIN[i], VEL_MAX[i]);
      return res;
    }

    // ノイズの生成
    std::vector<std::vector<vec3_t>> calc_epsilon(const Eigen::Matrix<double, 3, 3>& sigma, size_t K, size_t T) {
      std::vector<std::vector<vec3_t>> epsilon(K);
      for (size_t i = 0; i < K; i++) {
        epsilon[i].resize(T);
        for (size_t j = 0; j < T; j++) {
          std::mt19937 engine((std::random_device())());
          std::normal_distribution<> dist(0.0, 1.0);
          vec3_t v;
          for (size_t k = 0; k < 3; k++)
            v(k) = dist(engine);
          epsilon[i][j] = sigma * v;
        }
      }
      return epsilon;
    }
    // 　コスト関数
    double C(state_t x_t, control_t u_t, state_t x_goal) {
      double stage_cost = 0.0;
      stage_cost += (x_t - x_goal).transpose() * param_.Q * (x_t - x_goal);
      stage_cost += u_t.transpose() * param_.R * u_t;
      return stage_cost;
    }
    double phi(state_t x_t, state_t x_goal) {
      double terminal_cost = 0.0;
      terminal_cost += (x_t - x_goal).transpose() * param_.Q_T * (x_t - x_goal);
      return terminal_cost;
    }
    // 重み計算
    void calc_weight(const std::vector<double>& s) {
      auto min_element = std::min_element(s.begin(), s.end());
      double rho       = *min_element;
      // 正規化項計算
      double inv_lambda = 1.0 / param_.lambda;
      double eta        = 0;
      for (const auto& c : s)
        eta += std::exp(-inv_lambda * (c - rho));
      // 重み計算
      double inv_eta = 1.0 / eta;
      for (size_t i = 0; i < param_.K; i++)
        weight_[i] = inv_eta * std::exp(-inv_lambda * (s[i] - rho));
    }

    std::vector<control_t> moveing_average(const std::vector<control_t>& xx, size_t window_size) {
      size_t n = xx.size();
      std::vector<control_t> xx_mean(n, vec3_t::Zero(DIM_U, 1));
      std::vector<double> window(window_size, 1.0 / window_size);
      for (size_t d = 0; d < DIM_U; ++d) {
        std::vector<double> temp(n + window_size - 1, 0.0);
        // Padding the temp array with the first and last values
        for (size_t i = 0; i < n; ++i)
          temp[i + window_size / 2] = xx[i](d);
        for (size_t i = 0; i < n; ++i) {
          double sum = 0.0;
          for (size_t j = 0; j < window_size; ++j) {
            sum += temp[i + j] * window[j];
          }
          xx_mean[i](d) = sum;
        }
        size_t n_conv = ceil(window_size / 2.0);
        xx_mean[0](d) *= window_size / n_conv;
        for (size_t i = 1; i < n_conv; ++i) {
          xx_mean[i](d) *= window_size / (i + n_conv);
          xx_mean[n - i - 1](d) *= window_size / (i + n_conv - (window_size % 2));
        }
      }
      return xx_mean;
    }

  public:

    /**
     * @brief Construct a new MPPIPathPlanner object
     *
     * @param param MPPIのパラメータ
     * @param f 運動学モデル
     */
    MPPIPathPlanner(param_t param, std::function<state_t(state_t, control_t, double)> f) {
      f_      = f;
      param_  = param;
      ganmma_ = param_.lambda * (1.0 - param_.alpha);
      // メモリ確保
      sample_path_.resize(param_.K);
      for (size_t i = 0; i < param_.K; i++)
        sample_path_[i].resize(param_.T);
      stage_cost_.resize(param_.K);
      weight_.resize(param_.K);
      u_.resize(param_.T);
      for (auto& u : u_)
        u = vec3_t::Zero(DIM_U, 1);
      u_pre_ = u_;
      opt_path_.resize(param_.T);
    }
    /**
     * @brief 制御入力の制限設定
     *
     * @param min 最小値
     * @param max 最大値
     */
    void set_velocity_limit(std::array<double, 3> min, std::array<double, 3> max) {
      VEL_MAX = max;
      VEL_MIN = min;
    }
    /**
     * @brief MPPI計算
     *
     * @param x_t 現在の状態
     * @param x_goal ゴールの状態
     * @return std::vector<control_t> 制御入力
     */
    std::vector<control_t> path_planning(state_t x_t, state_t x_goal) {
      x_t(5)                                          = normalize_angle(x_t(5));
      u_                                              = u_pre_;
      const std::vector<std::vector<vec3_t>>& epsilon = calc_epsilon(param_.sigma, param_.K, param_.T);

      // サンプルとコストの計算
#pragma omp parallel for
      for (size_t i = 0; i < param_.K; i++) {
        stage_cost_[i] = 0.0;
        state_t x      = x_t;
        for (size_t j = 0; j < param_.T; j++) {
          control_t v        = u_[j] + epsilon[i][j];      // ノイズ付き制御入力
          sample_path_[i][j] = f_(x, clamp(v), param_.dt); // 状態計算
          x                  = sample_path_[i][j];
          x(5)               = normalize_angle(x(5));
          stage_cost_[i] += C(x, u_[j], x_goal) + ganmma_ * u_[j].transpose() * param_.sigma.inverse() * v; // ステージコスト
        }
        stage_cost_[i] += phi(x, x_goal); // ターミナルコスト
      }
      calc_weight(stage_cost_);
      std::vector<control_t> w_epsilon(param_.T);
      for (size_t i = 0; i < param_.T; i++) {
        w_epsilon[i] = vec3_t::Zero(DIM_U, 1);
        for (size_t j = 0; j < param_.K; j++)
          w_epsilon[i] += weight_[j] * epsilon[j][i];
      }
      w_epsilon = moveing_average(w_epsilon, 70);
      state_t x = x_t;
      for (size_t i = 0; i < param_.T; i++) {
        u_[i] += w_epsilon[i];
        x            = f_(x, clamp(u_[i]), param_.dt);
        opt_path_[i] = x;
      }
      // 前回の制御入力シーケンスを更新（左に1ステップシフト）
      size_t U_N = u_.size();
      std::copy(u_.begin() + 1, u_.end(), u_pre_.begin());
      u_pre_[U_N - 1] = u_[U_N - 1];
      return u_;
    }
    /**
     * @brief 最適経路の取得
     *
     * @return std::vector<state_t> 最適経路
     */
    std::vector<state_t> get_opt_path() { return opt_path_; }
    /**
     * @brief サンプル経路の取得
     *
     * @return std::vector<std::vector<state_t>> サンプル経路
     */
    std::vector<std::vector<state_t>> get_sample_path() { return sample_path_; }
  };
} // namespace MPPI
