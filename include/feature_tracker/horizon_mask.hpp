#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace feature_tracker
{

// Marks pixels below the horizon (-u*sin(theta) + v*cos(theta) < rho) in the normalized
// image-centred frame, shrunk by margin_px towards the ground.
inline cv::Mat groundMask(cv::Size size, float theta, float rho, float margin_px)
{
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    const double half_diagonal = 0.5 * std::hypot(size.width, size.height);
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    const double offset = rho - margin_px / half_diagonal;

    for (int y = 0; y < size.height; ++y)
    {
        const double v = (0.5 * size.height - y) / half_diagonal;
        uchar* row = mask.ptr<uchar>(y);

        if (std::abs(sin_theta) < 1e-9)
        {
            if (v * cos_theta < offset)
            {
                std::fill(row, row + size.width, uchar(255));
            }

            continue;
        }

        const double u_limit = (v * cos_theta - offset) / sin_theta;
        const double x_limit = u_limit * half_diagonal + 0.5 * size.width;

        if (sin_theta > 0.0)
        {
            const int first = std::clamp(static_cast<int>(std::floor(x_limit)) + 1, 0, size.width);
            std::fill(row + first, row + size.width, uchar(255));
        }
        else
        {
            const int last = std::clamp(static_cast<int>(std::ceil(x_limit)), 0, size.width);
            std::fill(row, row + last, uchar(255));
        }
    }

    return mask;
}

}
