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
// MPPI
#define USE_OMP
#include "../include/cpp_robot_sim/simulator.hpp"
#include "../include/mppi/mppi.hpp"

#define WINDOW_SIZE 0.3
#define ALL_WINDOW_LIM 0.1
#define ROBOT_SIZE 0.1
#define HOLONOMIC

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

void draw_vel(matplotlibcpp17::pyplot::PyPlot &plt, const std::vector<cpp_robot_sim::control_t> &u)
{
  std::vector<double> vx, vy, w, k;
  double i = 0;
  for (const auto &v : u)
  {
    vx.push_back(v(0));
    vy.push_back(v(1));
    w.push_back(v(2));
    k.push_back(i);
    i++;
  }
  plt.grid();
  plt.xlim(Args(0, k.size()));
  plt.ylim(Args(-3., 3.));
  plt.plot(Args(k, vx), Kwargs("color"_a = "red", "linewidth"_a = 1.0, "marker"_a = "."));
  plt.plot(Args(k, vy), Kwargs("color"_a = "green", "linewidth"_a = 1.0, "marker"_a = "."));
  plt.plot(Args(k, w), Kwargs("color"_a = "blue", "linewidth"_a = 1.0, "marker"_a = "."));
}

int main()
{
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  // auto fig = plt.figure();
  auto [fig, ax] = plt.subplots(1, 2, Kwargs("figsize"_a = py::make_tuple(15, 15), "subplot_kw"_a = py::dict("aspect"_a = "equal")));
  cpp_robot_sim::simulator sim(plt, f, ROBOT_SIZE, ROBOT_SIZE);
  MPPI::param_t param;
  param.T = 40;
  param.K = 200;
  param.dt = 0.01;
  param.lambda = 1.0;
  param.alpha = 0.8;
  Eigen::Matrix<double, 3, 3> sigma;
  sigma << 1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0;
  Eigen::Matrix<double, 6, 6> Q;
  Q << 10.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 10.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 5.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 60.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 60.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 10.0;
  Eigen::Matrix<double, 6, 6> Q_T;
  Q_T << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 100.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 100.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 50.0;
  Eigen::Matrix<double, 3, 3> R;
  R << 5.0, 0.0, 0.0,
      0.0, 5.0, 0.0,
      0.0, 0.0, 2.0;
  param.sigma = sigma;
  param.Q = Q;
  param.R = R;
  param.Q_T = Q_T;
  MPPI::MPPIPathPlanner mppi(param, f);
  mppi.set_velocity_limit({-0.3, -0.3, -1.0}, {0.3, 0.3, 1.0});
  cpp_robot_sim::state_t x_goal;
  x_goal << 0.0, 0.0, 0.0, 1.0, 2.0, MPPI::constants::HALF_PI;
  const double all_window_max = std::max(x_goal(3), x_goal(4));
  while (1)
  {
    // mppi計算
    std::vector<cpp_robot_sim::control_t> u = mppi.path_planning(sim.x_t, x_goal);
    std::vector<cpp_robot_sim::state_t> opt_path = mppi.get_opt_path();
    std::vector<std::vector<cpp_robot_sim::state_t>> sample_path = mppi.get_sample_path();
    cpp_robot_sim::control_t v_t;
    v_t = u[0];
    sim.update(v_t, param.dt);
    // デバック用
    sim.x_t(0) = v_t(0);
    sim.x_t(1) = v_t(1);
    sim.x_t(2) = v_t(2);
    // ax.set_aspect(Args("equal"));
    plt.subplot(122);
    plt.cla();
    plt.grid();
    plt.xlim(Args(sim.x_t(3) - WINDOW_SIZE, sim.x_t(3) + WINDOW_SIZE));
    plt.ylim(Args(sim.x_t(4) - WINDOW_SIZE, sim.x_t(4) + WINDOW_SIZE));
    for (const auto &i : sample_path)
    {
      std::vector<double> sample_x, sample_y;
      for (const auto &j : i)
      {
        sample_x.push_back(j(3));
        sample_y.push_back(j(4));
      }
      plt.plot(Args(sample_x, sample_y), Kwargs("color"_a = "grey", "linewidth"_a = 0.1));
    }
    std::vector<double> opt_x, opt_y;
    for (const auto &i : opt_path)
    {
      opt_x.push_back(i(3));
      opt_y.push_back(i(4));
    }
    plt.plot(Args(opt_x, opt_y), Kwargs("color"_a = "green", "linewidth"_a = 1.0));
    plt.plot(Args(x_goal(3), x_goal(4)), Kwargs("color"_a = "blue", "linewidth"_a = 1.0, "marker"_a = "o"));
    sim.draw(true);
    plt.subplot(221);
    plt.cla();
    plt.grid();
    plt.xlim(Args(-ALL_WINDOW_LIM, all_window_max + ALL_WINDOW_LIM));
    plt.ylim(Args(-ALL_WINDOW_LIM, all_window_max + ALL_WINDOW_LIM));
    plt.plot(Args(opt_x, opt_y), Kwargs("color"_a = "green", "linewidth"_a = 1.0));
    plt.plot(Args(x_goal(3), x_goal(4)), Kwargs("color"_a = "blue", "linewidth"_a = 1.0, "marker"_a = "o"));
    sim.draw(true);
    plt.subplot(223);
    plt.cla();
    draw_vel(plt, u);
    plt.pause(Args(0.01));
    // sleep(1);
  }
  return 0;
}