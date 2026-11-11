#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace feature_tracker
{

struct AffineConfig
{
    int min_matches = 12;
    double ransac_threshold = 0.005;
    double min_spread = 0.01;
};

inline std::optional<cv::Matx23d> estimateAffine(const std::vector<cv::Point2f>& p1, const std::vector<cv::Point2f>& p2, const AffineConfig& config)
{
    if (p1.size() != p2.size() || static_cast<int>(p1.size()) < config.min_matches)
    {
        return std::nullopt;
    }

    cv::Point2d mean(0.0, 0.0);

    for (const cv::Point2f& p : p1)
    {
        mean += cv::Point2d(p);
    }

    mean *= 1.0 / static_cast<double>(p1.size());
    double xx = 0.0;
    double xy = 0.0;
    double yy = 0.0;

    for (const cv::Point2f& p : p1)
    {
        const cv::Point2d d = cv::Point2d(p) - mean;
        xx += d.x * d.x;
        xy += d.x * d.y;
        yy += d.y * d.y;
    }

    const double count = static_cast<double>(p1.size());
    xx /= count;
    xy /= count;
    yy /= count;

    if (0.5 * (xx + yy - std::hypot(xx - yy, 2.0 * xy)) < config.min_spread)
    {
        return std::nullopt;
    }

    std::vector<uchar> inliers;
    const cv::Mat affine = cv::estimateAffine2D(p1, p2, inliers, cv::RANSAC, config.ransac_threshold);

    if (affine.empty() || cv::countNonZero(inliers) < config.min_matches)
    {
        return std::nullopt;
    }

    return cv::Matx23d(
        affine.at<double>(0, 0), affine.at<double>(0, 1), affine.at<double>(0, 2),
        affine.at<double>(1, 0), affine.at<double>(1, 1), affine.at<double>(1, 2));
}

}
