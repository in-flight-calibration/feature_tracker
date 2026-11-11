#pragma once

#include <ifc_msgs/msg/features.hpp>

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace feature_tracker
{

constexpr size_t MAX_FEATURES = ifc_msgs::msg::Features::MAX_MATCHES;

using FeatureId = std::array<uint8_t, 32>;

struct Match
{
    FeatureId id;
    cv::Point2f previous;
    cv::Point2f current;
    float quality;
    bool inlier = false;
};

struct Features
{
    std::array<int64_t, 2> timestamp{0, 0};
    std::vector<Match> matches;
    cv::Matx33d fundamental = cv::Matx33d::all(std::numeric_limits<double>::quiet_NaN());
};

}
