#pragma once

#include <features.hpp>
#include <ifc_tools/debug_viewer.hpp>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include <optional>
#include <string>

namespace feature_tracker
{

class Tracker : public rclcpp::Node
{
public:
    virtual ~Tracker() = default;

    virtual std::optional<Features> track(const rclcpp::Time& image_timestamp, const cv::Mat& image) = 0;

protected:
    Tracker(const std::string& node_name, std::optional<ifc_tools::DebugViewer>& viewer, const rclcpp::NodeOptions& options)
        : rclcpp::Node(node_name, options),
          _viewer(viewer)
    {
    }

    std::optional<ifc_tools::DebugViewer>& _viewer;
};

}
