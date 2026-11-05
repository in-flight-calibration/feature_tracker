#pragma once

#include <feature_tracker/orb_extractor.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace feature_tracker
{

struct TrackerParameters
{
    OrbParameters orb;
    float ratio_test = 0.8f;
    int max_hamming_distance = 80;
    float search_radius_ratio = 0.25f;
    double ransac_threshold_px = 1.0;
    double ransac_confidence = 0.999;
    int min_inliers = 15;
    double min_inlier_ratio = 0.3;
};

struct Match
{
    uint32_t id = 0;
    cv::Point2f previous;
    cv::Point2f current;
    float quality = 0.0f;
    bool inlier = false;
};

struct TrackingResult
{
    std::vector<Match> matches;
    cv::Matx33d fundamental = cv::Matx33d::zeros();
    bool fundamental_valid = false;
    int inlier_count = 0;
    std::vector<cv::KeyPoint> keypoints;
};

class Tracker
{
public:
    explicit Tracker(const TrackerParameters& parameters)
        : _parameters(parameters),
          _extractor(parameters.orb),
          _matcher(cv::NORM_HAMMING)
    {
    }

    // Returns nothing for the first frame, which only initializes the previous-frame state.
    std::optional<TrackingResult> process(const cv::Mat& gray, const cv::Mat& mask)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        _extractor.extract(gray, mask, keypoints, descriptors);

        TrackingResult result;
        result.keypoints = keypoints;
        std::vector<uint32_t> ids(keypoints.size(), 0);
        std::vector<bool> tracked(keypoints.size(), false);

        const bool has_previous = !_previous_keypoints.empty() && !_previous_descriptors.empty();

        if (has_previous && !keypoints.empty())
        {
            const double diagonal = std::hypot(gray.cols, gray.rows);
            match(keypoints, descriptors, diagonal * _parameters.search_radius_ratio, ids, tracked, result);
            estimateGeometry(result);
        }

        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (!tracked[i])
            {
                ids[i] = _next_id++;
            }
        }

        const bool first_frame = !has_previous && !_initialized;
        _initialized = true;
        _previous_keypoints = std::move(keypoints);
        _previous_descriptors = std::move(descriptors);
        _previous_ids = std::move(ids);

        if (first_frame)
        {
            return std::nullopt;
        }

        return result;
    }

    void reset()
    {
        _previous_keypoints.clear();
        _previous_descriptors.release();
        _previous_ids.clear();
        _initialized = false;
    }

private:
    void match(
        const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, double radius,
        std::vector<uint32_t>& ids, std::vector<bool>& tracked, TrackingResult& result) const
    {
        cv::Mat allowed(
            static_cast<int>(_previous_keypoints.size()), static_cast<int>(keypoints.size()), CV_8UC1);

        for (int i = 0; i < allowed.rows; ++i)
        {
            uchar* row = allowed.ptr<uchar>(i);

            for (int j = 0; j < allowed.cols; ++j)
            {
                const cv::Point2f delta = keypoints[j].pt - _previous_keypoints[i].pt;
                row[j] = std::hypot(delta.x, delta.y) <= radius ? 255 : 0;
            }
        }

        std::vector<std::vector<cv::DMatch>> candidates;
        _matcher.knnMatch(_previous_descriptors, descriptors, candidates, 2, allowed);

        std::vector<int> best(keypoints.size(), -1);
        std::vector<cv::DMatch> accepted;

        for (const auto& pair : candidates)
        {
            if (pair.empty() || pair[0].distance > _parameters.max_hamming_distance)
            {
                continue;
            }

            if (pair.size() > 1 && pair[0].distance >= _parameters.ratio_test * pair[1].distance)
            {
                continue;
            }

            const cv::DMatch& match = pair[0];
            int& slot = best[match.trainIdx];

            if (slot < 0)
            {
                slot = static_cast<int>(accepted.size());
                accepted.push_back(match);
            }
            else if (match.distance < accepted[slot].distance)
            {
                accepted[slot] = match;
            }
        }

        for (const cv::DMatch& match : accepted)
        {
            ids[match.trainIdx] = _previous_ids[match.queryIdx];
            tracked[match.trainIdx] = true;

            Match entry;
            entry.id = ids[match.trainIdx];
            entry.previous = _previous_keypoints[match.queryIdx].pt;
            entry.current = keypoints[match.trainIdx].pt;
            entry.quality = std::clamp(
                1.0f - match.distance / static_cast<float>(OrbExtractor::descriptor_bits), 0.0f, 1.0f);
            result.matches.push_back(entry);
        }
    }

    void estimateGeometry(TrackingResult& result) const
    {
        if (static_cast<int>(result.matches.size()) < std::max(8, _parameters.min_inliers))
        {
            return;
        }

        std::vector<cv::Point2f> previous;
        std::vector<cv::Point2f> current;

        for (const Match& match : result.matches)
        {
            previous.push_back(match.previous);
            current.push_back(match.current);
        }

        std::vector<uchar> inliers;
        const cv::Mat fundamental = cv::findFundamentalMat(
            previous, current, cv::FM_RANSAC,
            _parameters.ransac_threshold_px, _parameters.ransac_confidence, inliers);

        if (fundamental.empty() || fundamental.rows != 3 || fundamental.cols != 3 ||
            inliers.size() != result.matches.size())
        {
            return;
        }

        for (size_t i = 0; i < inliers.size(); ++i)
        {
            result.matches[i].inlier = inliers[i] != 0;
            result.inlier_count += inliers[i] != 0 ? 1 : 0;
        }

        const double ratio = static_cast<double>(result.inlier_count) / result.matches.size();

        if (result.inlier_count < _parameters.min_inliers || ratio < _parameters.min_inlier_ratio)
        {
            for (Match& match : result.matches)
            {
                match.inlier = false;
            }

            result.inlier_count = 0;
            return;
        }

        for (int r = 0; r < 3; ++r)
        {
            for (int c = 0; c < 3; ++c)
            {
                result.fundamental(r, c) = fundamental.at<double>(r, c);
            }
        }

        result.fundamental_valid = true;
    }

    TrackerParameters _parameters;
    OrbExtractor _extractor;
    cv::BFMatcher _matcher;
    std::vector<cv::KeyPoint> _previous_keypoints;
    cv::Mat _previous_descriptors;
    std::vector<uint32_t> _previous_ids;
    uint32_t _next_id = 0;
    bool _initialized = false;
};

}
