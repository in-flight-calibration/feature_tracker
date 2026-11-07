#include <feature_tracker.hpp>

#include <memory>

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<feature_tracker::FeatureTracker>();

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.add_node(node->receiver());
    executor.add_node(node->performanceMonitor());
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
