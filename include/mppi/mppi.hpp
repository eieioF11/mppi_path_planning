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
#include <omp.h>
// Eigen
#include <Eigen/Dense>
//
#include "grid_map.hpp"
#include "math_utility.hpp"
#include "param.hpp"
#include "utility.hpp"

namespace MPPI {
  class MPPIPathPlanner {
  private:
    typedef vec3_t control_t;
    typedef vec6_t state_t;

    std::shared_ptr<GridMap> map_;

    std::function<state_t(state_t, control_t, double)> f_;
    state_t x_t;
    param_t param_;
    double ganmma_;
    double inv_lambda_;
    Eigen::Matrix<double, 3, 3> inv_sigma_;
    std::array<double, 3> VEL_MAX = {1.0, 1.0, 1.0};
    std::array<double, 3> VEL_MIN = {-1.0, -1.0, -1.0};
    std::vector<control_t> u_, u_pre_;
    std::vector<state_t> opt_path_;
    std::vector<std::vector<state_t>> sample_path_;
    std::vector<std::vector<vec3_t>> epsilon_;

    std::vector<double> weight_;
    std::vector<double> cost_;

    control_t clamp(const control_t& v) {
      control_t res;
      for (int i = 0; i < DIM_U; ++i)
        res(i) = std::clamp(v(i), VEL_MIN[i], VEL_MAX[i]);
      return res;
    }
    // ノイズ計算
    control_t noise() {
      std::mt19937 engine((std::random_device())());
      std::normal_distribution<> dist(0.0, 1.0);
      Eigen::LLT<Eigen::MatrixXd> llt(param_.sigma);
      Eigen::MatrixXd L = llt.matrixL();
      vec3_t n;
      for (size_t i = 0; i < DIM_U; ++i)
        n(i) = dist(engine);
      return L * n;
    }
    // 　コスト関数
    double C(const state_t& x_t, const control_t& u_t, const state_t& x_tar) {
      double stage_cost = 0.0;
      state_t diff_x    = x_t - x_tar;
      diff_x(5)         = normalize_angle(diff_x(5));
      stage_cost += diff_x.transpose() * param_.Q * diff_x;
      stage_cost += u_t.transpose() * param_.R * u_t;
      if (map_) {
        auto [vx, vy] = map_->get_grid_pos(x_t(3), x_t(4));
        // std::cout << "xt:"<< x_t(3) << " yt:"<< x_t(4) << std::endl;
        // std::cout << "vx:"<< vx << " vy:"<< vy << std::endl;
        if (map_->is_wall(vx, vy)) {
          stage_cost += param_.obstacle_cost;
          // std::cout << "obstacle" << std::endl;
        }
      }
      return stage_cost;
    }
    double phi(const state_t& x_t, const state_t& x_tar) {
      double terminal_cost = 0.0;
      state_t diff_x       = x_t - x_tar;
      diff_x(5)            = normalize_angle(diff_x(5));
      terminal_cost += diff_x.transpose() * param_.Q_T * diff_x;
      if (map_) {
        auto [vx, vy] = map_->get_grid_pos(x_t(3), x_t(4));
        if (map_->is_wall(vx, vy)) {
          terminal_cost += param_.obstacle_cost;
          // std::cout << "obstacle" << std::endl;
        }
      }
      return terminal_cost;
    }
    // 重み計算
    void calc_weight(const std::vector<double>& s) {
      auto min_element = std::min_element(s.begin(), s.end());
      double rho       = *min_element;
      // 正規化項計算
      double eta = 0;
#pragma omp parallel for reduction(+ : eta) schedule(dynamic)
      for (const auto& c : s)
        eta += std::exp(-inv_lambda_ * (c - rho));
      // 重み計算
      double inv_eta = 1.0 / eta;
#pragma omp parallel for schedule(dynamic)
      for (size_t i = 0; i < param_.K; ++i)
        weight_[i] = inv_eta * std::exp(-inv_lambda_ * (s[i] - rho));
    }

    std::vector<control_t> moveing_average(const std::vector<control_t>& xx, const size_t& window_size) {
      const size_t n = xx.size();
      std::vector<control_t> xx_mean(n, vec3_t::Zero(DIM_U, 1));
      std::vector<double> window(window_size, 1.0 / window_size);
#pragma omp parallel for schedule(dynamic)
      for (size_t d = 0; d < DIM_U; ++d) {
        std::vector<double> temp(n + window_size - 1, 0.0);
        // Padding the temp array with the first and last values
        for (size_t i = 0; i < n; ++i)
          temp[i + window_size * 0.5] = xx[i](d);
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
      epsilon_.resize(param_.K);
      sample_path_.resize(param_.K);
      for (size_t i = 0; i < param_.K; ++i) {
        epsilon_[i].resize(param_.T);
        sample_path_[i].resize(param_.T);
      }
      cost_.resize(param_.K);
      weight_.resize(param_.K);
      u_.resize(param_.T);
      for (auto& u : u_)
        u = vec3_t::Zero(DIM_U, 1);
      u_pre_ = u_;
      opt_path_.resize(param_.T);
      // inverse calculation
      inv_sigma_  = param_.sigma.inverse();
      inv_lambda_ = 1.0 / param_.lambda;
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
     * @brief グリッドマップの設定
     *
     * @param map グリッドマップ
     */
    void set_map(const GridMap& map) { map_ = std::make_shared<GridMap>(map); }
    /**
     * @brief MPPI計算
     *
     * @param x_t 現在の状態
     * @param x_tar ゴールの状態
     * @return std::vector<control_t> 制御入力
     */
    std::vector<control_t> path_planning(state_t x_t, state_t x_tar) {
      double ts, te;
      ts     = omp_get_wtime();
      x_t(5) = normalize_angle(x_t(5));
      u_     = u_pre_;
      // サンプルとコストの計算
#pragma omp parallel for schedule(dynamic)
      for (size_t k = 0; k < param_.K; ++k) {
        cost_[k]  = 0.0;
        state_t x = x_t;
        for (size_t t = 1; t < param_.T + 1; ++t) {
          // ノイズ生成
          epsilon_[k][t - 1] = noise();
          // ノイズ付き制御入力計算
          control_t v = u_[t - 1] + epsilon_[k][t - 1];
          // if (k < (1.0 - 0.6) * param_.K) v = epsilon_[k][t - 1];
          // 状態計算
          x = f_(x, clamp(v), param_.dt);
          // ステージコスト計算
          cost_[k] += C(x, v, x_tar) + ganmma_ * u_[t - 1].transpose() * inv_sigma_ * v; // ステージコスト
          // debug
          sample_path_[k][t - 1] = x;
        }
        // 終端コスト計算
        cost_[k] += phi(x, x_tar);
      }
      calc_weight(cost_);
      std::vector<control_t> w_epsilon(param_.T);
#pragma omp parallel for schedule(dynamic)
      for (size_t t = 0; t < param_.T; ++t) {
        w_epsilon[t] = vec3_t::Zero(DIM_U, 1);
        for (size_t k = 0; k < param_.K; ++k)
          w_epsilon[t] += weight_[k] * epsilon_[k][t];
      }
      w_epsilon = moveing_average(w_epsilon, param_.window_size);
      state_t x = x_t;
      for (size_t t = 0; t < param_.T; ++t) {
        u_[t] += w_epsilon[t];
        u_[t]        = clamp(u_[t]);
        x            = f_(x, u_[t], param_.dt);
        opt_path_[t] = x;
      }
      // 前回の制御入力シーケンスを更新（左に1ステップシフト）
      size_t U_N = u_.size();
      std::copy(u_.begin() + 1, u_.end(), u_pre_.begin());
      u_pre_[U_N - 1] = u_[U_N - 1];
      te              = omp_get_wtime();
      log_info("calc time:%f[s]", te - ts);
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
