#pragma once

#include <string>
#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include "franka_semantic_components/franka_robot_model.hpp"
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include "std_msgs/msg/float64_multi_array.hpp"
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter.hpp>


using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_example_controllers {
  using Eigen::Matrix3d;
  using Matrix4d = Eigen::Matrix<double, 4, 4>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Matrix7d = Eigen::Matrix<double, 7, 7>;

  using Vector3d = Eigen::Matrix<double, 3, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  
  using Eigen::Quaterniond;

/**
 * The cartesian impedance example controller implements the Hogan formulation.
 */
class CartesianJointImpedanceForceController : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string arm_id_;
  const int num_joints = 7;
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  rclcpp::Time start_time_;
  Quaterniond desired_orientation;
  Vector3d desired_position;
  Vector7d desired_qn;
  Matrix4d desired;

  Quaterniond target_desired_orientation;
  Vector3d target_desired_position;
  Vector7d target_desired_qn;

  Quaterniond delta_orientation;
  double delta_angle;
  Vector3d delta_axis;

  double limit_orientation;
  Vector3d limit_position;
  Vector7d limit_qn;
  
  Vector7d home_q_;

  Matrix6d stiffness;
  Matrix6d damping;
  double n_stiffness;

  double pos_stiff_x_;
  double pos_stiff_y_;
  double pos_stiff_z_;
  double rot_stiff_x_;
  double rot_stiff_y_;
  double rot_stiff_z_;

  rclcpp_lifecycle::LifecycleNode::OnSetParametersCallbackHandle::SharedPtr desired_impedance_handle_;
  rcl_interfaces::msg::SetParametersResult impedanceCallback(const std::vector<rclcpp::Parameter> & parameters);


  void desiredCartesianCallback(const std_msgs::msg::Float64MultiArray& msg);
  void desiredJointCallback(const std_msgs::msg::Float64MultiArray& msg);

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_desired_cartesian_; 
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_desired_joint_; 

};
}  // namespace franka_example_controllers
