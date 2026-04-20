#include <franka_example_controllers/subscriber/cartesian_joint_impedence_force_controller.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <algorithm>

#include <franka/model.h>
#include <Eigen/Geometry>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter.hpp>

using std::max;
using std::min;

// You can keep this pseudoInverse helper if you use it elsewhere.
// It is no longer used for the nullspace.
inline void pseudoInverse(const Eigen::MatrixXd& M_,
                          Eigen::MatrixXd& M_pinv_,
                          bool damped = true) {
  double lambda_ = damped ? 0.2 : 0.0;

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ =
      svd.singularValues();
  Eigen::MatrixXd S_ = M_;  // copy dims only
  S_.setZero();

  for (int i = 0; i < sing_vals_.size(); i++) {
    S_(i, i) =
        (sing_vals_(i)) /
        (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);
  }

  M_pinv_ =
      Eigen::MatrixXd(svd.matrixV() * S_.transpose() *
                      svd.matrixU().transpose());
}

namespace franka_example_controllers {

controller_interface::InterfaceConfiguration
CartesianJointImpedanceForceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CartesianJointImpedanceForceController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& franka_robot_model_name :
       franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }
  return config;
}

/******************************************************************
 * UPDATED update(): dynamic-consistent nullspace + rate limits
 ******************************************************************/
controller_interface::return_type
CartesianJointImpedanceForceController::update(const rclcpp::Time& /*time*/,
                                               const rclcpp::Duration& period) {
  // --- Read robot state ---
  Eigen::Map<const Matrix4d> current(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  Eigen::Vector3d    current_position(current.block<3,1>(0,3));
  Eigen::Quaterniond current_orientation(current.block<3,3>(0,0));

  Eigen::Map<const Matrix7d> inertia(
      franka_robot_model_->getMassMatrix().data());
  Eigen::Map<const Vector7d> coriolis(
      franka_robot_model_->getCoriolisForceVector().data());
  Eigen::Matrix<double, 6, 7> jacobian(
      franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector).data());

  Eigen::Map<const Vector7d> qD(
      franka_robot_model_->getRobotState()->dq.data());
  Eigen::Map<const Vector7d> q(
      franka_robot_model_->getRobotState()->q.data());

  // --- Time step and safety limits ---
  const double dt = period.seconds();  // ~0.001 at 1 kHz

  const double max_lin_speed   = 0.40;  // [m/s]
  const double max_rot_speed   = 0.30;  // [rad/s]
  const double max_joint_speed = 0.30;  // [rad/s]

  Eigen::Vector3d max_pos_step   =
      Eigen::Vector3d::Constant(max_lin_speed * dt);
  Vector7d        max_q_step     =
      Vector7d::Constant(max_joint_speed * dt);
  const double    max_angle_step = max_rot_speed * dt;

  // === ORIENTATION: smoothly move desired_orientation towards target_desired_orientation ===
  Eigen::Quaterniond delta_q =
      target_desired_orientation * desired_orientation.inverse();
  delta_q.normalize();

  // Use shortest rotation
  if (delta_q.w() < 0.0) {
    delta_q.coeffs() *= -1.0;
  }

  Eigen::AngleAxisd aa(delta_q);
  double           delta_angle = aa.angle();
  Eigen::Vector3d  delta_axis  = aa.axis();

  double step_angle = std::min(delta_angle, max_angle_step);
  if (step_angle > 1e-6) {
    Eigen::AngleAxisd step_aa(step_angle, delta_axis);
    Eigen::Quaterniond step_q(step_aa);
    desired_orientation = step_q * desired_orientation;
    desired_orientation.normalize();
  }
  // keep an interpretation for limit_orientation if used elsewhere
  limit_orientation = max_rot_speed * dt;

  // === POSITION: smoothly move desired_position towards target_desired_position ===
  Eigen::Vector3d pos_err  = target_desired_position - desired_position;
  Eigen::Vector3d pos_step = pos_err;
  pos_step = pos_step.cwiseMax(-max_pos_step).cwiseMin(max_pos_step);
  desired_position += pos_step;

  // === NULLSPACE (posture) joints: smoothly move desired_qn towards target_desired_qn ===
  Vector7d qn_err  = target_desired_qn - desired_qn;
  Vector7d qn_step = qn_err;
  qn_step = qn_step.cwiseMax(-max_q_step).cwiseMin(max_q_step);
  desired_qn += qn_step;

  // --- Task-space error based on "desired_*" (already rate-limited) ---
  Vector6d error;
  error.setZero();

  error.head<3>() = current_position - desired_position;

  // quaternion sign consistency
  if (desired_orientation.coeffs().dot(current_orientation.coeffs()) < 0.0) {
    current_orientation.coeffs() << -current_orientation.coeffs();
  }
  Eigen::Quaterniond rot_error(current_orientation *
                               desired_orientation.inverse());
  Eigen::AngleAxisd rot_error_aa(rot_error);
  error.tail<3>() = rot_error_aa.axis() * rot_error_aa.angle();

  // --- Task torque ---
  Vector7d tau_task, tau_nullspace, tau_d;
  tau_task.setZero();
  tau_nullspace.setZero();
  tau_d.setZero();

  tau_task = jacobian.transpose() *
             (-stiffness * error - damping * (jacobian * qD));

  // === Dynamically consistent nullspace projector ===
  //
  // Λ = (J M^{-1} Jᵀ)⁻¹
  // J# = M^{-1} Jᵀ Λ  (dynamically consistent generalized inverse)
  // N  = I - J# J     (M-weighted projector onto nullspace of J)
  //
  Eigen::Matrix<double, 6, 6> lambda;
  {
    Eigen::Matrix<double, 6, 7> JM_inv = jacobian * inertia.inverse();
    Eigen::Matrix<double, 6, 6> JM_inv_JT = JM_inv * jacobian.transpose();
    lambda = JM_inv_JT.inverse();  // for 6x6 this is usually OK; you can switch to .ldlt() if needed
  }

  Eigen::Matrix<double, 7, 6> Jsharp =
      inertia.inverse() * jacobian.transpose() * lambda;  // 7x6

  Matrix7d N = Matrix7d::Identity() - Jsharp * jacobian;  // 7x7

  // Nullspace PD in joint space
  Vector7d tau_ns_des =
      n_stiffness * (desired_qn - q) -
      (2.0 * std::sqrt(n_stiffness)) * qD;

  tau_nullspace = N * tau_ns_des;

  // --- Total commanded torque: task + coriolis + nullspace ---
  tau_d = tau_task + coriolis + tau_nullspace;

  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }

  return controller_interface::return_type::OK;
}
/******************************************************************
 * end of updated update()
 ******************************************************************/

controller_interface::CallbackReturn
CartesianJointImpedanceForceController::on_init(){\
  auto node = this->get_node();

  desired_impedance_handle_ = node->add_on_set_parameters_callback(
      std::bind(&CartesianJointImpedanceForceController::impedanceCallback,
                this,
                std::placeholders::_1));
  try {
    auto_declare<std::string>("arm_id", "panda");

    auto_declare<double>("pos_stiff_x", 100.0);
    auto_declare<double>("pos_stiff_y", 100.0);
    auto_declare<double>("pos_stiff_z", 100.0);

    auto_declare<double>("rot_stiff_x", 10.0);
    auto_declare<double>("rot_stiff_y", 10.0);
    auto_declare<double>("rot_stiff_z", 10.0);

    // initial per-step limits (will be overwritten by dt-based logic)
    limit_orientation = 0.005;
    limit_position    << 0.0025, 0.0025, 0.0025;
    limit_qn          << 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005;

    sub_desired_cartesian_ =
        get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/cartesian_impedance/pose_desired", 1,
            std::bind(
                &CartesianJointImpedanceForceController::desiredCartesianCallback,
                this,
                std::placeholders::_1));

    sub_desired_joint_ =
        get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/cartesian_impedance/joint_desired", 1,
            std::bind(
                &CartesianJointImpedanceForceController::desiredJointCallback,
                this,
                std::placeholders::_1));

  } catch (const std::exception& e) {
    fprintf(stderr,
            "Exception thrown during init stage with message: %s \n",
            e.what());
    return CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

CallbackReturn CartesianJointImpedanceForceController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_      = get_node()->get_parameter("arm_id").as_string();
  pos_stiff_x_ = get_node()->get_parameter("pos_stiff_x").as_double();
  pos_stiff_y_ = get_node()->get_parameter("pos_stiff_y").as_double();
  pos_stiff_z_ = get_node()->get_parameter("pos_stiff_z").as_double();

  rot_stiff_x_ = get_node()->get_parameter("rot_stiff_x").as_double();
  rot_stiff_y_ = get_node()->get_parameter("rot_stiff_y").as_double();
  rot_stiff_z_ = get_node()->get_parameter("rot_stiff_z").as_double();

  franka_robot_model_ =
      std::make_unique<franka_semantic_components::FrankaRobotModel>(
          franka_semantic_components::FrankaRobotModel(
              arm_id_ + "/robot_model", arm_id_));
  auto parameters = get_node()->list_parameters({}, 10);
  (void)parameters;

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianJointImpedanceForceController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  desired = Matrix4d(
      franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
  desired_position    = Vector3d(desired.block<3,1>(0,3));
  desired_orientation = Quaterniond(desired.block<3,3>(0,0));
  desired_qn          = Vector7d(franka_robot_model_->getRobotState()->q.data());

  target_desired_orientation = desired_orientation;
  target_desired_position    = desired_position;
  target_desired_qn          = desired_qn;

  n_stiffness = 10.0;

  stiffness.setIdentity();
  stiffness.topLeftCorner(3, 3) =
      Vector3d(pos_stiff_x_, pos_stiff_y_, pos_stiff_z_).asDiagonal();
  stiffness.bottomRightCorner(3, 3) =
      Vector3d(rot_stiff_x_, rot_stiff_y_, rot_stiff_z_).asDiagonal();

  // Simple critical damping in task space
  damping.setIdentity();
  damping.topLeftCorner(3, 3) =
      (2.0 * Vector3d(std::sqrt(pos_stiff_x_),
                      std::sqrt(pos_stiff_y_),
                      std::sqrt(pos_stiff_z_)))
          .asDiagonal();
  damping.bottomRightCorner(3, 3) =
      (0.8 * 2.0 *
       Vector3d(std::sqrt(rot_stiff_x_),
                std::sqrt(rot_stiff_y_),
                std::sqrt(rot_stiff_z_)))
          .asDiagonal();

  n_stiffness = 10.0;

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianJointImpedanceForceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

void CartesianJointImpedanceForceController::desiredCartesianCallback(
    const std_msgs::msg::Float64MultiArray& msg) {
  if (msg.data.empty()) {
    return;
  }

  if (msg.data[0]) {
    // position [0..2]
    for (auto i = 0; i < 3; ++i) {
      target_desired_position[i] = msg.data[i];
    }
    // orientation as 3x3 rotation matrix, if present
    if (msg.data.size() >= 12 && msg.data[11]) {
      Matrix3d desired_orientation_mat;
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          desired_orientation_mat(i, j) = msg.data[3 + 3 * i + j];
        }
      }
      target_desired_orientation =
          Eigen::Quaterniond(desired_orientation_mat);
    }
  }
}

void CartesianJointImpedanceForceController::desiredJointCallback(
    const std_msgs::msg::Float64MultiArray& msg) {
  if (msg.data.empty()) {
    return;
  }
  if (msg.data[0]) {
    for (auto i = 0; i < 7; ++i) {
      target_desired_qn[i] = msg.data[i];
    }
  }
}

rcl_interfaces::msg::SetParametersResult
CartesianJointImpedanceForceController::impedanceCallback(
    const std::vector<rclcpp::Parameter>& parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason     = "";

  for (const auto& p : parameters) {
    if (p.get_name() == "pos_stiff_x") pos_stiff_x_ = p.as_double();
    if (p.get_name() == "pos_stiff_y") pos_stiff_y_ = p.as_double();
    if (p.get_name() == "pos_stiff_z") pos_stiff_z_ = p.as_double();

    if (p.get_name() == "rot_stiff_x") rot_stiff_x_ = p.as_double();
    if (p.get_name() == "rot_stiff_y") rot_stiff_y_ = p.as_double();
    if (p.get_name() == "rot_stiff_z") rot_stiff_z_ = p.as_double();
  }

  // Update stiffness matrix when parameters change
  stiffness.topLeftCorner(3, 3) =
      Vector3d(pos_stiff_x_, pos_stiff_y_, pos_stiff_z_).asDiagonal();
  stiffness.bottomRightCorner(3, 3) =
      Vector3d(rot_stiff_x_, rot_stiff_y_, rot_stiff_z_).asDiagonal();

  return result;
}

}  // namespace franka_example_controllers

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianJointImpedanceForceController,
                       controller_interface::ControllerInterface)

