#pragma once

#include <tracker/tracker.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace feature_tracker
{

class TrackerOrb final : public Tracker
{
public:
    explicit TrackerOrb(
        std::optional<ifc_tools::DebugViewer>& viewer,
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Tracker("feature_tracker_orb", viewer, options),
          _orb(cv::ORB::create(static_cast<int>(MAX_FEATURES))),
          _matcher(cv::NORM_HAMMING),
          _ratio_test(declare_parameter<double>("feature_tracker.orb.ratio_test", 0.8)),
          _ransac_threshold(declare_parameter<double>("feature_tracker.orb.ransac_threshold_px", 1.0)),
          _ransac_confidence(declare_parameter<double>("feature_tracker.orb.ransac_confidence", 0.999))
    {
    }

protected:
    std::optional<Features> _process(const rclcpp::Time&, const cv::Mat& image) override
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detect(image, keypoints, descriptors);

        std::vector<FeatureId> ids(keypoints.size());
        std::vector<bool> tracked(keypoints.size(), false);

        if (_descriptors.empty())
        {
            assignIds(descriptors, tracked, ids);
            save(keypoints, descriptors, ids);
            return std::nullopt;
        }

        const std::vector<Match> matches = match(keypoints, descriptors, ids, tracked);
        assignIds(descriptors, tracked, ids);

        std::vector<uchar> inliers;
        const std::optional<cv::Matx33d> fundamental = estimateFundamental(matches, inliers);

        Features features;
        features.matches = matches;

        if (fundamental)
        {
            features.fundamental = *fundamental;

            for (size_t i = 0; i < matches.size(); ++i)
            {
                features.matches[i].inlier = inliers[i] != 0;
            }
        }

        save(keypoints, descriptors, ids);
        return features;
    }

private:
    cv::Ptr<cv::ORB> _orb;
    cv::BFMatcher _matcher;
    double _ratio_test;
    double _ransac_threshold;
    double _ransac_confidence;
    std::vector<cv::KeyPoint> _keypoints;
    cv::Mat _descriptors;
    std::vector<FeatureId> _ids;
    
    void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        _orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    }

    std::vector<Match> match(
        const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
        std::vector<FeatureId>& ids, std::vector<bool>& tracked) const
    {
        std::vector<Match> matches;

        if (descriptors.empty())
        {
            return matches;
        }

        std::vector<std::vector<cv::DMatch>> candidates;
        _matcher.knnMatch(_descriptors, descriptors, candidates, 2);

        for (const auto& pair : candidates)
        {
            if (pair.size() < 2 || pair[0].distance >= _ratio_test * pair[1].distance || tracked[pair[0].trainIdx])
            {
                continue;
            }

            const cv::DMatch& match = pair[0];
            ids[match.trainIdx] = _ids[match.queryIdx];
            tracked[match.trainIdx] = true;
            matches.push_back({
                ids[match.trainIdx],
                _keypoints[match.queryIdx].pt,
                keypoints[match.trainIdx].pt,
                1.0f - match.distance / 256.0f,
                false});
        }

        return matches;
    }

    static void assignIds(const cv::Mat& descriptors, const std::vector<bool>& tracked, std::vector<FeatureId>& ids)
    {
        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (!tracked[i])
            {
                std::copy_n(descriptors.ptr<uint8_t>(static_cast<int>(i)), ids[i].size(), ids[i].begin());
            }
        }
    }

    std::optional<cv::Matx33d> estimateFundamental(const std::vector<Match>& matches, std::vector<uchar>& inliers) const
    {
        if (matches.size() < 8)
        {
            return std::nullopt;
        }

        std::vector<cv::Point2f> previous;
        std::vector<cv::Point2f> current;

        for (const Match& match : matches)
        {
            previous.push_back(match.previous);
            current.push_back(match.current);
        }

        const cv::Mat fundamental = cv::findFundamentalMat(
            previous, current, cv::FM_RANSAC, _ransac_threshold, _ransac_confidence, inliers);

        if (fundamental.rows != 3 || fundamental.cols != 3 || inliers.size() != matches.size())
        {
            return std::nullopt;
        }

        return cv::Matx33d(fundamental);
    }

    void save(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::vector<FeatureId>& ids)
    {
        _keypoints = std::move(keypoints);
        _descriptors = std::move(descriptors);
        _ids = std::move(ids);
    }
};

}
