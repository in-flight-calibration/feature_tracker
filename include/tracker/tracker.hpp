#pragma once

#include <features.hpp>
#include <ifc_tools/debug_viewer.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

namespace feature_tracker
{

class Tracker : public rclcpp::Node
{
public:
    virtual ~Tracker() = default;

    std::optional<Features> process(const rclcpp::Time& image_timestamp, const cv::Mat& image)
    {
        const int64_t timestamp = image_timestamp.nanoseconds();
        std::optional<Features> features = _process(image_timestamp, image);

        if (features)
        {
            features->timestamp = {_previous_timestamp, timestamp};

            if (_viewer)
            {
                draw(image, *features);
            }
        }

        _previous_timestamp = timestamp;
        return features;
    }

protected:
    Tracker(const std::string& node_name, std::optional<ifc_tools::DebugViewer>& viewer, const rclcpp::NodeOptions& options)
        : rclcpp::Node(node_name, options),
          _viewer(viewer)
    {
    }

    virtual std::optional<Features> _process(const rclcpp::Time& image_timestamp, const cv::Mat& image) = 0;

    std::optional<ifc_tools::DebugViewer>& _viewer;

private:
    int64_t _previous_timestamp = 0;

    void draw(const cv::Mat& image, const Features& features)
    {
        const bool valid = std::isfinite(features.fundamental(0, 0));
        cv::Mat annotated = image.clone();

        for (const Match& match : features.matches)
        {
            cv::circle(annotated, match.current, 3,
                !valid || match.inlier ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), cv::FILLED);
        }

        _viewer->update(0, 0, annotated, "FEATURES");
        _viewer->render(static_cast<uint64_t>(features.timestamp[1] / 1000));
    }
};

}
