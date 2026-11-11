#pragma once

#include <affine.hpp>
#include <ifc_tools/horizon_utils.hpp>
#include <tracker/tracker.hpp>
#include <tracker/tracker_factory.hpp>
#include <ifc_msgs/msg/features.hpp>
#include <ifc_msgs/msg/horizon.hpp>
#include <ifc_msgs/msg/match.hpp>
#include <ifc_tools/debug_viewer.hpp>
#include <ifc_tools/image_receiver.hpp>
#include <ifc_tools/performance_monitor.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace feature_tracker
{

class FeatureTracker final : public rclcpp::Node
{
public:
    explicit FeatureTracker(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node("feature_tracker", options),
          _performance_monitor(std::make_shared<ifc_tools::PerformanceMonitor>("feature_tracker"))
    {
        const std::string image_topic = declare_parameter<std::string>("image.topic", "/flightgear/image");
        const std::string image_transport = declare_parameter<std::string>("image.transport", "compressed");
        _receiver = std::make_shared<ifc_tools::ImageReceiver>("feature_tracker_receiver", image_topic, image_transport);
        _receiver->setCallback(std::bind(&FeatureTracker::onImage, this, std::placeholders::_1, std::placeholders::_2));

        const std::string horizon_topic = declare_parameter<std::string>("horizon.topic", "/horizon_detection/horizon");
        _horizon_subscription = create_subscription<ifc_msgs::msg::Horizon>(
            horizon_topic, rclcpp::ServicesQoS(),
            std::bind(&FeatureTracker::onHorizon, this, std::placeholders::_1));

        const std::string tracker_name = declare_parameter<std::string>("tracker.type", "orb");
        if (declare_parameter<bool>("tracker.debug_view", false))
        {
            _viewer.emplace(cv::Size(960, 540), 1, 1);
        }

        _tracker = TrackerFactory::create(tracker_name, _viewer);

        const std::string features_topic = declare_parameter<std::string>("tracker.topic", "/feature_tracker/features");
        _publisher = create_publisher<ifc_msgs::msg::Features>(features_topic, rclcpp::ServicesQoS());

        const std::string performance_topic = declare_parameter<std::string>("tracker.performance_topic", "/performance/feature_tracker");

        if (!performance_topic.empty())
        {
            _performance_monitor->setTopic(performance_topic);
        }

        _margin_px = static_cast<float>(declare_parameter<double>("tracker.horizon_margin_px", 10.0));
        _affine.min_matches = declare_parameter<int>("tracker.affine.min_matches", 12);
        _affine.ransac_threshold = declare_parameter<double>("tracker.affine.ransac_threshold", 0.005);
        _affine.min_spread = declare_parameter<double>("tracker.affine.min_spread", 0.01);
    }

    const std::shared_ptr<ifc_tools::ImageReceiver>& receiver() const
    {
        return _receiver;
    }

    const std::shared_ptr<ifc_tools::PerformanceMonitor>& performanceMonitor() const
    {
        return _performance_monitor;
    }

private:
    std::shared_ptr<ifc_tools::ImageReceiver> _receiver;
    std::optional<ifc_tools::DebugViewer> _viewer;
    std::unique_ptr<Tracker> _tracker;

    rclcpp::Publisher<ifc_msgs::msg::Features>::SharedPtr _publisher;
    rclcpp::Subscription<ifc_msgs::msg::Horizon>::SharedPtr _horizon_subscription;

    float _margin_px;
    AffineConfig _affine;

    std::mutex _mutex;
    cv::Mat _image;
    std::optional<ifc_msgs::msg::Horizon> _horizon;
    std::optional<rclcpp::Time> _image_timestamp;
    
    std::shared_ptr<ifc_tools::PerformanceMonitor> _performance_monitor;

    void onImage(const rclcpp::Time& image_timestamp, const cv::Mat& image)
    {
        std::scoped_lock<std::mutex> lock(_mutex);

        if (_image_timestamp)
        {
            process();
        }

        _image_timestamp = image_timestamp;
        _image = image.clone();
    }

    void onHorizon(const ifc_msgs::msg::Horizon::ConstSharedPtr& message)
    {
        std::scoped_lock<std::mutex> lock(_mutex);

        _horizon = *message;

        if (_image_timestamp && rclcpp::Time(message->image_timestamp, _image_timestamp->get_clock_type()) == *_image_timestamp)
        {
            process();
        }
    }

    cv::Mat maskedImage() const
    {
        if (!_horizon || _horizon->confidence <= 0.0f || !std::isfinite(_horizon->theta) || !std::isfinite(_horizon->rho))
        {
            return _image;
        }

        cv::Mat masked;
        _image.copyTo(masked, ifc_tools::groundMask(_image.size(), _horizon->theta, _horizon->rho, _margin_px));
        return masked;
    }

    void process()
    {
        const rclcpp::Time timestamp = *_image_timestamp;
        _image_timestamp.reset();

        _performance_monitor->start();
        const auto features = _tracker->process(timestamp, maskedImage());
        _performance_monitor->stop();

        if (features)
        {
            publish(*features);
        }
    }

    void publish(const Features& features) const
    {
        const cv::Point2f centre(0.5f * _image.cols, 0.5f * _image.rows);
        const cv::Matx33d A(centre.x, 0.0, centre.x, 0.0, centre.y, centre.y, 0.0, 0.0, 1.0);
        const cv::Matx33d fundamental = A.t() * features.fundamental * A;

        ifc_msgs::msg::Features message;
        message.header.stamp = static_cast<builtin_interfaces::msg::Time>(get_clock()->now());
        message.image_timestamp[0] = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(features.timestamp[0]));
        message.image_timestamp[1] = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(features.timestamp[1]));

        for (int i = 0; i < 9; ++i)
        {
            message.f[i] = static_cast<float>(fundamental.val[i]);
        }

        std::vector<cv::Point2f> p1;
        std::vector<cv::Point2f> p2;
        const bool has_fundamental = std::isfinite(features.fundamental(0, 0));
        message.affine.fill(std::numeric_limits<float>::quiet_NaN());

        for (const Match& match : features.matches)
        {
            ifc_msgs::msg::Match entry;
            entry.id = match.id;
            entry.x1 = (match.previous.x - centre.x) / centre.x;
            entry.y1 = (match.previous.y - centre.y) / centre.y;
            entry.x2 = (match.current.x - centre.x) / centre.x;
            entry.y2 = (match.current.y - centre.y) / centre.y;
            entry.quality = match.quality;
            entry.inlier = match.inlier;
            message.matches.push_back(entry);

            if (!has_fundamental || match.inlier)
            {
                p1.emplace_back(entry.x1, entry.y1);
                p2.emplace_back(entry.x2, entry.y2);
            }
        }

        if (const auto affine = estimateAffine(p1, p2, _affine))
        {
            for (int i = 0; i < 6; ++i)
            {
                message.affine[i] = static_cast<float>(affine->val[i]);
            }
        }

        _publisher->publish(message);
    }
};

}
