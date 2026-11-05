#include <feature_tracker/horizon_mask.hpp>
#include <feature_tracker/tracker.hpp>

#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <cstdlib>

namespace
{

bool check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAILED: %s\n", message);
    }

    return condition;
}

cv::Mat texture(cv::Size size)
{
    cv::Mat image(size, CV_8UC1);
    cv::RNG rng(7);
    rng.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.5);
    return image;
}

}

int main()
{
    const cv::Size size(640, 480);
    bool ok = true;

    const cv::Mat mask = feature_tracker::groundMask(size, 0.0f, 0.0f, 0.0f);
    ok &= check(mask.at<uchar>(10, 100) == 0, "sky pixel is masked out");
    ok &= check(mask.at<uchar>(470, 100) == 255, "ground pixel is kept");

    const cv::Mat tilted = feature_tracker::groundMask(size, 0.3f, 0.0f, 0.0f);
    ok &= check(tilted.at<uchar>(470, 320) == 255 && tilted.at<uchar>(10, 320) == 0, "tilted horizon splits the image");

    const cv::Mat base = texture(cv::Size(800, 600));
    const cv::Mat full = cv::Mat(size, CV_8UC1, cv::Scalar(255));
    feature_tracker::Tracker tracker{feature_tracker::TrackerParameters{}};

    ok &= check(!tracker.process(base(cv::Rect(0, 0, 640, 480)).clone(), full).has_value(), "first frame yields no matches");

    const auto result = tracker.process(base(cv::Rect(12, 8, 640, 480)).clone(), full);
    ok &= check(result.has_value(), "second frame yields a result");

    if (result)
    {
        ok &= check(result->fundamental_valid, "fundamental matrix is valid");
        ok &= check(result->inlier_count >= 50, "enough inliers");

        int consistent = 0;

        for (const auto& match : result->matches)
        {
            const cv::Point2f shift = match.previous - match.current;

            if (match.inlier && std::abs(shift.x - 12.0f) < 2.0f && std::abs(shift.y - 8.0f) < 2.0f)
            {
                ++consistent;
            }
        }

        ok &= check(consistent >= 50, "inliers follow the known shift");

        const auto next = tracker.process(base(cv::Rect(24, 16, 640, 480)).clone(), full);
        bool persistent = false;

        if (next)
        {
            for (const auto& a : result->matches)
            {
                for (const auto& b : next->matches)
                {
                    persistent |= a.id == b.id;
                }
            }
        }

        ok &= check(persistent, "ids persist across frames");
    }

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
