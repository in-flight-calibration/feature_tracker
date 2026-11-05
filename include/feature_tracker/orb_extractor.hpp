#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

namespace feature_tracker
{

struct OrbParameters
{
    int max_features = 1000;
    float scale_factor = 1.2f;
    int levels = 8;
    int fast_threshold = 20;
};

class OrbExtractor
{
public:
    explicit OrbExtractor(const OrbParameters& parameters)
        : _orb(cv::ORB::create(
              parameters.max_features, parameters.scale_factor, parameters.levels,
              31, 0, 2, cv::ORB::HARRIS_SCORE, 31, parameters.fast_threshold))
    {
    }

    void extract(
        const cv::Mat& gray, const cv::Mat& mask,
        std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const
    {
        _orb->detectAndCompute(gray, mask, keypoints, descriptors);
    }

    static constexpr int descriptor_bits = 256;

private:
    cv::Ptr<cv::ORB> _orb;
};

}
