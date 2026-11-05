#pragma once

#include <feature_tracker/horizon_mask.hpp>
#include <feature_tracker/tracker.hpp>
#include <ifc_msgs/msg/feature_match.hpp>
#include <ifc_msgs/msg/feature_matches.hpp>
#include <ifc_msgs/msg/horizon.hpp>
#include <ifc_tools/debug_viewer.hpp>
#include <ifc_tools/image_receiver.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace feature_tracker
{

class FeatureTracker final : public rclcpp::Node
{
public:
    explicit FeatureTracker(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : rclcpp::Node("feature_tracker", options),
          _tracker(declareTrackerParameters())
    {
        const std::string image_topic = declare_parameter<std::string>("image.topic", "/flightgear/image");
        const std::string image_transport = declare_parameter<std::string>("image.transport", "compressed");
        const std::string horizon_topic = declare_parameter<std::string>("horizon.topic", "/horizon_detection/horizon");
        const std::string matches_topic = declare_parameter<std::string>("tracker.topic", "/feature_tracker/matches");
        _horizon_margin_px = static_cast<float>(declare_parameter<double>("tracker.horizon_margin_px", 10.0));
        _horizon_max_age = declare_parameter<double>("tracker.horizon_max_age_s", 0.2);
        _min_horizon_confidence = static_cast<float>(declare_parameter<double>("tracker.min_horizon_confidence", 0.3));

        if (declare_parameter<bool>("tracker.debug_view", false))
        {
            _viewer.emplace(cv::Size(1920, 540), 1, 2);
        }

        _receiver = std::make_shared<ifc_tools::ImageReceiver>("feature_tracker_receiver", image_topic, image_transport);
        _publisher = create_publisher<ifc_msgs::msg::FeatureMatches>(matches_topic, rclcpp::ServicesQoS());
        _horizon_subscription = create_subscription<ifc_msgs::msg::Horizon>(
            horizon_topic, rclcpp::ServicesQoS(),
            std::bind(&FeatureTracker::onHorizon, this, std::placeholders::_1));
        _receiver->setCallback(
            std::bind(&FeatureTracker::onImage, this, std::placeholders::_1, std::placeholders::_2));
    }

    const std::shared_ptr<ifc_tools::ImageReceiver>& receiver() const
    {
        return _receiver;
    }

private:
    TrackerParameters declareTrackerParameters()
    {
        TrackerParameters p;
        p.orb.max_features = static_cast<int>(declare_parameter<int64_t>("tracker.orb.max_features", p.orb.max_features));
        p.orb.scale_factor = static_cast<float>(declare_parameter<double>("tracker.orb.scale_factor", p.orb.scale_factor));
        p.orb.levels = static_cast<int>(declare_parameter<int64_t>("tracker.orb.levels", p.orb.levels));
        p.orb.fast_threshold = static_cast<int>(declare_parameter<int64_t>("tracker.orb.fast_threshold", p.orb.fast_threshold));
        p.ratio_test = static_cast<float>(declare_parameter<double>("tracker.matching.ratio_test", p.ratio_test));
        p.max_hamming_distance = static_cast<int>(
            declare_parameter<int64_t>("tracker.matching.max_hamming_distance", p.max_hamming_distance));
        p.search_radius_ratio = static_cast<float>(
            declare_parameter<double>("tracker.matching.search_radius_ratio", p.search_radius_ratio));
        p.ransac_threshold_px = declare_parameter<double>("tracker.ransac.threshold_px", p.ransac_threshold_px);
        p.ransac_confidence = declare_parameter<double>("tracker.ransac.confidence", p.ransac_confidence);
        p.min_inliers = static_cast<int>(declare_parameter<int64_t>("tracker.ransac.min_inliers", p.min_inliers));
        p.min_inlier_ratio = declare_parameter<double>("tracker.ransac.min_inlier_ratio", p.min_inlier_ratio);
        return p;
    }

    void onHorizon(const ifc_msgs::msg::Horizon::ConstSharedPtr& message)
    {
        if (message->confidence <= 0.0f || !std::isfinite(message->theta) || !std::isfinite(message->rho))
        {
            return;
        }

        _horizon = *message;
    }

    std::optional<cv::Mat> currentMask(const rclcpp::Time& image_timestamp, cv::Size size) const
    {
        if (!_horizon || _horizon->confidence < _min_horizon_confidence)
        {
            return std::nullopt;
        }

        const double age = std::abs((image_timestamp - rclcpp::Time(_horizon->image_timestamp, image_timestamp.get_clock_type())).seconds());

        if (age > _horizon_max_age)
        {
            return std::nullopt;
        }

        return groundMask(size, _horizon->theta, _horizon->rho, _horizon_margin_px);
    }

    void onImage(const rclcpp::Time& image_timestamp, const cv::Mat& image)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        const std::optional<cv::Mat> mask = currentMask(image_timestamp, image.size());

        if (!mask)
        {
            _tracker.reset();
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "No current horizon; frames are not tracked");
            return;
        }

        const std::optional<TrackingResult> result = _tracker.process(gray, *mask);
        const std::optional<rclcpp::Time> previous_timestamp = _previous_timestamp;
        _previous_timestamp = image_timestamp;

        if (!result || !previous_timestamp)
        {
            return;
        }

        publish(*result, *previous_timestamp, image_timestamp, image.size());

        if (_viewer)
        {
            draw(image, *mask, *result, image_timestamp);
        }
    }

    void publish(
        const TrackingResult& result, const rclcpp::Time& previous_timestamp,
        const rclcpp::Time& image_timestamp, cv::Size size) const
    {
        ifc_msgs::msg::FeatureMatches message;
        message.header.stamp = static_cast<builtin_interfaces::msg::Time>(get_clock()->now());
        message.prev_image_timestamp = static_cast<builtin_interfaces::msg::Time>(previous_timestamp);
        message.image_timestamp = static_cast<builtin_interfaces::msg::Time>(image_timestamp);
        message.image_width = static_cast<uint32_t>(size.width);
        message.image_height = static_cast<uint32_t>(size.height);
        message.f_status = result.fundamental_valid
            ? ifc_msgs::msg::FeatureMatches::F_VALID
            : ifc_msgs::msg::FeatureMatches::F_INVALID;
        message.inlier_count = static_cast<uint32_t>(result.inlier_count);

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                message.f[r * 3 + c] = result.fundamental(r, c);
            }
        }

        message.matches.reserve(result.matches.size());

        for (const Match& match : result.matches)
        {
            ifc_msgs::msg::FeatureMatch entry;
            entry.id = match.id;
            entry.prev_x = match.previous.x;
            entry.prev_y = match.previous.y;
            entry.curr_x = match.current.x;
            entry.curr_y = match.current.y;
            entry.quality = match.quality;
            entry.inlier = match.inlier;
            message.matches.push_back(entry);
        }

        _publisher->publish(message);
    }

    void draw(
        const cv::Mat& image, const cv::Mat& mask, const TrackingResult& result,
        const rclcpp::Time& image_timestamp)
    {
        cv::Mat annotated = image.clone();

        for (const cv::KeyPoint& keypoint : result.keypoints)
        {
            cv::circle(annotated, keypoint.pt, 2, cv::Scalar(255, 160, 0), cv::FILLED);
        }

        for (const Match& match : result.matches)
        {
            const cv::Scalar color = match.inlier ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            cv::line(annotated, match.previous, match.current, color, 1, cv::LINE_AA);
            cv::circle(annotated, match.current, 3, color, cv::FILLED);
        }

        char label[96];
        std::snprintf(
            label, sizeof(label), "matches %zu inliers %d F %s",
            result.matches.size(), result.inlier_count, result.fundamental_valid ? "valid" : "invalid");

        cv::Mat ground;
        image.copyTo(ground, mask);

        _viewer->update(0, 0, annotated, label);
        _viewer->update(0, 1, ground, "ground mask");
        _viewer->render(static_cast<uint64_t>(image_timestamp.nanoseconds() / 1000));
    }

    std::shared_ptr<ifc_tools::ImageReceiver> _receiver;
    rclcpp::Publisher<ifc_msgs::msg::FeatureMatches>::SharedPtr _publisher;
    rclcpp::Subscription<ifc_msgs::msg::Horizon>::SharedPtr _horizon_subscription;
    Tracker _tracker;
    std::optional<ifc_msgs::msg::Horizon> _horizon;
    std::optional<rclcpp::Time> _previous_timestamp;
    std::optional<ifc_tools::DebugViewer> _viewer;
    float _horizon_margin_px = 10.0f;
    double _horizon_max_age = 0.2;
    float _min_horizon_confidence = 0.3f;
};

}
