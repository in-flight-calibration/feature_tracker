#pragma once

#include <tracker/tracker.hpp>
#include <tracker/orb/tracker_orb.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace feature_tracker
{

class TrackerFactory
{
public:
    static std::unique_ptr<Tracker> create(const std::string& name, std::optional<ifc_tools::DebugViewer>& viewer)
    {
        if (name == "orb")
        {
            return std::make_unique<TrackerOrb>(viewer);
        }

        throw std::invalid_argument("Unknown feature tracker: " + name);
    }
};

}
