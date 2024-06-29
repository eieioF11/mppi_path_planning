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
#include <iostream>
#include <sstream>
// OpenMP
#include <omp.h>
// Eigen
#include <Eigen/Dense>

#define LOG_LEVEL_ERROR 3
#define LOG_LEVEL_WARN 2
#define LOG_LEVEL_INFO 1

#define LOG_LEVEL LOG_LEVEL_INFO

namespace MPPI {
  constexpr int DIM_U = 3;
  constexpr int DIM_X = 6;
  typedef Eigen::Matrix<double, DIM_U, 1> vec3_t;
  typedef Eigen::Matrix<double, DIM_X, 1> vec6_t;

  /**
   * @brief log_info
   *
   * @param fmt
   * @param args
   * @return std::string
   */
  template <class... Args>
  inline void log_info(const char* fmt, Args... args) {
#if LOG_LEVEL <= LOG_LEVEL_INFO
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.resize(len);
    std::snprintf(buf.data(), len + 1, fmt, args...);
    std::cout << "[info] " << buf << std::endl;
#endif
  }

  /**
   * @brief log_warn
   *
   * @param fmt
   * @param args
   * @return std::string
   */
  template <class... Args>
  inline void log_warn(const char* fmt, Args... args) {
#if LOG_LEVEL <= LOG_LEVEL_WARN
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.resize(len);
    std::snprintf(buf.data(), len + 1, fmt, args...);
    std::cout << "[warn] " << buf << std::endl;
#endif
  }

  /**
   * @brief log_error
   *
   * @param fmt
   * @param args
   * @return std::string
   */
  template <class... Args>
  inline void log_error(const char* fmt, Args... args) {
#if LOG_LEVEL <= LOG_LEVEL_ERROR
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string buf;
    buf.resize(len);
    std::snprintf(buf.data(), len + 1, fmt, args...);
    std::cout << "[error] " << buf << std::endl;
#endif
  }

  template <typename T>
  std::string print_vector(const std::vector<T> &data)
  {
      std::stringstream ss;
      std::ostream_iterator<T> out_it(ss, ", ");
      ss << "[";
      std::copy(data.begin(), data.end() - 1, out_it);
      ss << data.back() << "]";
      return ss.str();
  }

} // namespace MPPI
