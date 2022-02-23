#pragma once
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class SemanticsColorMap {
public:
  static constexpr int NUM_CLASSES_INDOOR = 32;
  static constexpr int NUM_CLASSES_COLOR_ONLY = 1;
  Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor> colormap;
  explicit SemanticsColorMap(int num_classes);
};

class PlasmaColorMap {
public:
    cv::Mat _lut;
    explicit PlasmaColorMap();
    void init( const int & n);
    void operator()( const cv::InputArray & _src, cv::OutputArray & _dst) const;
};
