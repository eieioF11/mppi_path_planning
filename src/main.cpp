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
// matplotlibcpp17
#include <matplotlibcpp17/pyplot.h>
// Eigen
#include <Eigen/Dense>
// OpenMP
#include <omp.h>
// MPPI
#include "../include/cpp_robot_sim/simulator.hpp"
#include "../include/mppi/mppi.hpp"

#define WINDOW_SIZE 0.3
#define ALL_WINDOW_LIM 0.1
#define ROBOT_SIZE 0.1
#define VEL_LIM 3.0 // -VEL_LIM~VEL_LIMで表示
// #define HOLONOMIC

#ifdef HOLONOMIC
cpp_robot_sim::state_t f(cpp_robot_sim::state_t x_t, cpp_robot_sim::control_t v_t, double dt)
{
  cpp_robot_sim::state_t x_next;
  Eigen::Matrix<double, 6, 6> A;
  A << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix<double, 6, 3> B;
  B << 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0;
  x_next = A * x_t + dt * B * v_t;
  return x_next;
}
#else
cpp_robot_sim::state_t f(cpp_robot_sim::state_t x_t, cpp_robot_sim::control_t v_t, double dt)
{
  cpp_robot_sim::state_t x_next;
  Eigen::Matrix<double, 6, 6> A;
  A << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix<double, 6, 3> B;
  B << 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      std::cos(x_t(5)), 0.0, 0.0,
      std::sin(x_t(5)), 0.0, 0.0,
      0.0, 0.0, 1.0;
  x_next = A * x_t + dt * B * v_t;
  return x_next;
}
#endif

void draw_vel(matplotlibcpp17::pyplot::PyPlot &plt, const std::vector<cpp_robot_sim::control_t> &u, double dt)
{
  std::vector<double> vx, vy, w, k;
  double i = 0;
  for (const auto &v : u)
  {
    vx.push_back(v(0));
    vy.push_back(v(1));
    w.push_back(v(2));
    k.push_back(i * dt);
    i++;
  }
  plt.grid();
  plt.xlim(Args(0, k.size() * dt));
  plt.ylim(Args(-VEL_LIM, VEL_LIM));
  plt.plot(Args(k, vx), Kwargs("color"_a = "red", "linestyle"_a = "--", "label"_a = "v_x"));
  plt.plot(Args(k, vy), Kwargs("color"_a = "green", "linestyle"_a = "--", "label"_a = "v_y"));
  plt.plot(Args(k, w), Kwargs("color"_a = "blue", "linestyle"_a = "--", "label"_a = "w"));
  plt.legend();
}

int main()
{
  int THREAD_NUM = omp_get_max_threads();
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  // auto fig = plt.figure();
  auto [fig, ax] = plt.subplots(1, 3, Kwargs("figsize"_a = py::make_tuple(18, 7), "subplot_kw"_a = py::dict("aspect"_a = "equal")));
  cpp_robot_sim::simulator sim(f, ROBOT_SIZE, ROBOT_SIZE);
  MPPI::param_t param;
  param.T = 30;
  param.K = 500;
  param.dt = 0.01;
  param.lambda = 1.0;
  param.alpha = 0.2;
#ifdef HOLONOMIC
  Eigen::Matrix<double, 3, 3> sigma;
  sigma << 0.5, 0.0, 0.0,
      0.0, 0.5, 0.0,
      0.0, 0.0, 1.0;
  Eigen::Matrix<double, 6, 6> Q;
  Q << 5.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 60.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 60.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 10.0;
  Eigen::Matrix<double, 6, 6> Q_T;
  Q_T << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 200.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 200.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 70.0;
  Eigen::Matrix<double, 3, 3> R;
  R << 5.0, 0.0, 0.0,
      0.0, 5.0, 0.0,
      0.0, 0.0, 2.0;
#else
  Eigen::Matrix<double, 3, 3> sigma;
  sigma << 0.5, 0.0, 0.0,
      0.0, 0.0001, 0.0,
      0.0, 0.0, 2.5;
  Eigen::Matrix<double, 6, 6> Q;
  Q << 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 800.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 800.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  Eigen::Matrix<double, 6, 6> Q_T;
  Q_T << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1200.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 1200.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix<double, 3, 3> R;
  R << 5.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 2.0;
#endif
  param.sigma = sigma;
  param.Q = Q;
  param.R = R;
  param.Q_T = Q_T;
  param.window_size = 50.0; // 70.0
  param.obstacle_cost = 1000.0;
  MPPI::MPPIPathPlanner mppi(param, f);
  MPPI::GridMap map;
  map.resize(3.0, 3.0);
  map.info.resolution = 0.1;
  map.info.height = 3.0;
  map.info.width = 3.0;
  // mppi.set_map();
#ifdef HOLONOMIC
  mppi.set_velocity_limit({-0.3, -0.3, -2.4}, {0.3, 0.3, 2.4});
#else
  mppi.set_velocity_limit({-0.5, 0.0, -2.4}, {0.5, 0.0, 2.4});
#endif
  cpp_robot_sim::state_t x_tar;
  x_tar << 0.0, 0.0, 0.0, 1.0, 2.0, MPPI::constants::HALF_PI;
  // x_tar << 0.0, 0.0, 0.0, 1.0, 2.0, std::atan2(2.0, 1.0);
  const double all_window_max = std::max(x_tar(3), x_tar(4));
  while (1)
  {
    std::cout << "MAX threads NUM:" << THREAD_NUM << std::endl;
    // mppi計算
    const std::vector<cpp_robot_sim::control_t> &u = mppi.path_planning(sim.x_t, x_tar);
    const std::vector<cpp_robot_sim::state_t> &opt_path = mppi.get_opt_path();
    const std::vector<std::vector<cpp_robot_sim::state_t>> sample_path = mppi.get_sample_path();
    cpp_robot_sim::control_t v_t;
    v_t = u[0];
    std::cout << "dt:" << sim.deltaT() << std::endl;
    sim.update(v_t, param.dt);
    // デバック用
    sim.x_t(0) = v_t(0);
    sim.x_t(1) = v_t(1);
    sim.x_t(2) = v_t(2);
    plt.subplot(132);
    // ax.set_aspect(Args("equal"));
    // plt.axes().set_aspect(Args("equal"));
    plt.cla();
    plt.grid();
    plt.xlim(Args(sim.x_t(3) - WINDOW_SIZE, sim.x_t(3) + WINDOW_SIZE));
    plt.ylim(Args(sim.x_t(4) - WINDOW_SIZE, sim.x_t(4) + WINDOW_SIZE));
    std::vector<double> sample_x(sample_path[0].size()), sample_y(sample_path[0].size());
    for (const auto &path : sample_path)
    {
      for (size_t i = 0; i < path.size(); i++)
      {
        sample_x[i] = path[i](3);
        sample_y[i] = path[i](4);
      }
      plt.plot(Args(sample_x, sample_y), Kwargs("color"_a = "grey", "linewidth"_a = 0.1));
    }
    std::vector<double> opt_x(opt_path.size()), opt_y(opt_path.size());
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < opt_path.size(); i++)
    {
      opt_x[i] = opt_path[i](3);
      opt_y[i] = opt_path[i](4);
    }
    plt.plot(Args(opt_x, opt_y), Kwargs("color"_a = "green", "linewidth"_a = 1.0));
    plt.plot(Args(x_tar(3), x_tar(4)), Kwargs("color"_a = "blue", "linewidth"_a = 1.0, "marker"_a = "o"));
    sim.draw(plt, true);
    plt.subplot(131);
    plt.cla();
    plt.grid();
    plt.xlim(Args(-ALL_WINDOW_LIM, all_window_max + ALL_WINDOW_LIM));
    plt.ylim(Args(-ALL_WINDOW_LIM, all_window_max + ALL_WINDOW_LIM));
    plt.plot(Args(opt_x, opt_y), Kwargs("color"_a = "green", "linewidth"_a = 1.0));
    plt.plot(Args(x_tar(3), x_tar(4)), Kwargs("color"_a = "blue", "linewidth"_a = 1.0, "marker"_a = "o"));
    sim.draw(plt, true);
    plt.subplot(133);
    plt.cla();
    draw_vel(plt, u, 0.1);
    plt.pause(Args(0.001));
  }
  return 0;
}