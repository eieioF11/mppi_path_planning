#pragma once
// std
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>

namespace MPPI {
  struct map_info_t {
    double resolution; // The map resolution [m/cell]
    uint32_t width;    // Map width [cells]
    uint32_t height;   // Map height [cells]

    // The origin of the map [m, m, rad]. This is the real-world pose of the
    // bottom left corner of cell (0,0) in the map.
    double origin_x;
    double origin_y;
  };
  /**
   * @brief GridMap(OccupancyGrid)クラス
   *
   */
  class GridMap {
  public:
    static constexpr int8_t WALL_VALUE = 100;
    map_info_t info;
    std::vector<int8_t> data; // mapの内容が100だと壁(占有数の確率は[0,100]の範囲) -1は不明
    /*
    参考:nav_msgs/OccupancyGrid
    https://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
    https://docs.ros2.org/foxy/api/nav_msgs/msg/OccupancyGrid.html
    */

    GridMap() = default;
    GridMap(const size_t& x, const size_t& y) { resize(x, y); }
    GridMap(const map_info_t& i, const std::vector<int8_t>& d) { set_map(i, d); }
    GridMap(const GridMap& map) : info(map.info), data(map.data) {}
    GridMap(GridMap&& map) {
      info = std::move(map.info);
      data = std::move(map.data);
    }
    GridMap& operator=(GridMap&& map) {
      info = std::move(map.info);
      data = std::move(map.data);
      return *this;
    }
    /**
     * @brief  値のセット
     *
     * @param map_info_t
     * @param std::vector<int8_t>
     */
    void set_map(const map_info_t& i, const std::vector<int8_t>& d) {
      info = i;
      data = d;
    }
    /**
     * @brief  壁判定
     *
     * @param val
     * @return bool
     */
    bool is_wall(const int8_t& val) const { return val > (WALL_VALUE - 1); }
    /**
     * @brief  cellの壁判定
     *
     * @param x
     * @param y
     * @return bool
     */
    bool is_wall(const int& x, const int& y) const { return is_wall(data[y * col() + x]); }
    /**
     * @brief  サイズ変更
     *
     * @param x
     * @param y
     */
    void resize(const size_t& x, const size_t& y) { data.resize(x * y); }
    /**
     * @brief  実際の位置からグリッドマップ上での位置に変換
     *
     * @param x 実際のx
     * @param y 実際のy
     * @return std::pair<double,double> グリッドマップ上の位置{vx,vy}
     */
    std::pair<double, double> get_grid_pos(const double& x, const double& y) const {
      double inv_resolution = 1.0 / info.resolution;
      double index_origin_x = std::round(-info.origin_x * inv_resolution);
      double index_origin_y = std::round(-info.origin_y * inv_resolution);
      double vx             = std::round((x)*inv_resolution + index_origin_x);
      double vy             = std::round((y)*inv_resolution + index_origin_y);
      return {vx, vy};
    }
    /**
     * @brief  グリッドマップ上の位置から実際の位置に変換
     *
     * @param x グリッドマップ上のx
     * @param y グリッドマップ上のy
     * @return std::pair<double,double> 実際の位置 {x,y}
     */
    std::pair<double, double> grid_to_pos(const double& x, const double& y) const {
      double index_origin_x = std::round(-info.origin_x / info.resolution);
      double index_origin_y = std::round(-info.origin_y / info.resolution);
      double px             = (x - index_origin_x) * info.resolution;
      double py             = (y - index_origin_y) * info.resolution;
      return {px, py};
    }
    /**
     * @brief  マップ上に存在するか
     *
     * @param Vector2d グリッドマップ上の位置
     * @return bool
     */
    bool is_contain(const double& x, const double& y) const {
      if (0 <= x && x < row() && 0 <= y && y < col())
        return true;
      else
        return false;
    }

    /**
     * @brief  行の最大取得
     *
     * @return size_t
     */
    size_t row() const { return info.height; }
    /**
     * @brief  列の最大取得
     *
     * @return size_t
     */
    size_t col() const { return info.width; }
    /**
     * @brief  cellに値を代入
     *
     * @param Vector2d v
     * @param int8_t value
     */
    void set(const double& x, const double& y, int8_t value) { data[y * col() + x] = value; }
    int8_t at(const int x, const int y) const { return data[y * col() + x]; }
    int8_t operator()(const int x, const int y) const { return at(x, y); }
    int8_t& operator()(const int x, const int y) { return data[y * col() + x]; }
  };
} // namespace MPPI