#include <franka_example_controllers/subscriber/cartesian_joint_impedence_force_controller.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>
#include <franka/model.h>
#include <algorithm>
#include <Eigen/Geometry>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/parameter.hpp>


using std::max;
using std::min;

inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
    double lambda_ = damped ? 0.2 : 0.0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
    Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
    S_.setZero();

    for (int i = 0; i < sing_vals_.size(); i++)
        S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

    M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
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
      for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {  
       config.names.push_back(franka_robot_model_name);
    }
  
    return config;
  }

  controller_interface::return_type CartesianJointImpedanceForceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
    Eigen::Map<const Matrix4d> current(franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
    Eigen::Vector3d current_position(current.block<3,1>(0,3));
    Eigen::Quaterniond current_orientation(current.block<3,3>(0,0));
    Eigen::Map<const Matrix7d> inertia(franka_robot_model_->getMassMatrix().data());
    Eigen::Map<const Vector7d> coriolis(franka_robot_model_->getCoriolisForceVector().data());
    Eigen::Matrix<double, 6, 7> jacobian(
        franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector).data());
    Eigen::Map<const Vector7d> qD(franka_robot_model_->getRobotState()->dq.data());
    Eigen::Map<const Vector7d> q(franka_robot_model_->getRobotState()->q.data());
    Vector6d error;
    //clipping of delta pose
    delta_orientation = target_desired_orientation * desired_orientation.inverse();
    delta_orientation.normalize();

    Eigen::AngleAxisd aa(delta_orientation);
    Eigen::Vector3d delta_axis  = aa.axis();
    delta_angle = aa.angle();
    limit_orientation = 0.005;
    double limited_angle = std::min(delta_angle, limit_orientation);
    delta_orientation = Eigen::AngleAxisd(limited_angle, delta_axis);

    desired_orientation = delta_orientation * desired_orientation;
    desired_orientation.normalize();

    desired_position += (target_desired_position - desired_position).cwiseMin(limit_position).cwiseMax(-limit_position);
    desired_qn += (target_desired_qn - desired_qn).cwiseMin(limit_qn).cwiseMax(-limit_qn);

    auto desired_position_cur = desired_position;
    error.head(3) << current_position - desired_position_cur;
    if (desired_orientation.coeffs().dot(current_orientation.coeffs()) < 0.0) {
      current_orientation.coeffs() << -current_orientation.coeffs();
    }
    Eigen::Quaterniond rot_error(
        current_orientation * desired_orientation.inverse());
    Eigen::AngleAxisd rot_error_aa(rot_error);
    error.tail(3) << rot_error_aa.axis() * rot_error_aa.angle();
    Vector7d tau_task, tau_nullspace, tau_d;
    tau_task.setZero();
    tau_nullspace.setZero();
    tau_d.setZero();
    tau_task << jacobian.transpose() * (-stiffness*error - damping*(jacobian*qD));

    Eigen::MatrixXd jacobian_transpose_pinv;
      pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                        jacobian.transpose() * jacobian_transpose_pinv) *
                          (n_stiffness * (desired_qn - q) -
                            (2.0 * sqrt(n_stiffness)) * qD);

    tau_d <<  tau_task + coriolis + tau_nullspace;
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(tau_d(i));
    }
    return controller_interface::return_type::OK;
  }

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

      //check values
      limit_orientation=0.005;
      limit_position<<0.0025,0.0025,0.0025;
      limit_qn<<0.005,0.005,0.005,0.005,0.005,0.005,0.005;


      sub_desired_cartesian_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/cartesian_impedance/pose_desired", 1,
        std::bind(&CartesianJointImpedanceForceController::desiredCartesianCallback, this, std::placeholders::_1)
      );
      sub_desired_joint_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/cartesian_impedance/joint_desired", 1,
        std::bind(&CartesianJointImpedanceForceController::desiredJointCallback, this, std::placeholders::_1)
      );


    } catch (const std::exception& e) {
      fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
      return CallbackReturn::ERROR;
    }
    return controller_interface::CallbackReturn::SUCCESS;
  }


  CallbackReturn CartesianJointImpedanceForceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    pos_stiff_x_ = get_node()->get_parameter("pos_stiff_x").as_double();
    pos_stiff_y_ = get_node()->get_parameter("pos_stiff_y").as_double();
    pos_stiff_z_ = get_node()->get_parameter("pos_stiff_z").as_double();

    rot_stiff_x_ = get_node()->get_parameter("rot_stiff_x").as_double();
    rot_stiff_y_ = get_node()->get_parameter("rot_stiff_y").as_double();
    rot_stiff_z_ = get_node()->get_parameter("rot_stiff_z").as_double();
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        franka_semantic_components::FrankaRobotModel(arm_id_ + "/robot_model",
                                                    arm_id_));
    auto parameters = get_node()->list_parameters({}, 10);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn CartesianJointImpedanceForceController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
    desired = Matrix4d(franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector).data());
    desired_position = Vector3d(desired.block<3,1>(0,3));
    desired_orientation = Quaterniond(desired.block<3,3>(0,0));
    desired_qn = Vector7d(franka_robot_model_->getRobotState()->q.data());

    target_desired_orientation=desired_orientation;
    target_desired_position=desired_position;
    target_desired_qn=desired_qn;


    stiffness.setIdentity();
    stiffness.topLeftCorner(3,3) =
        Vector3d(pos_stiff_x_, pos_stiff_y_, pos_stiff_z_).asDiagonal();
    stiffness.bottomRightCorner(3,3) =
        Vector3d(rot_stiff_x_, rot_stiff_y_, rot_stiff_z_).asDiagonal();
    // Simple critical damping
    damping.setIdentity();
    damping.topLeftCorner(3,3) =
      (2.0 * Vector3d(
          sqrt(pos_stiff_x_),
          sqrt(pos_stiff_y_),
          sqrt(pos_stiff_z_)
      )).asDiagonal();
    damping.bottomRightCorner(3,3) =
      (0.8 * 2.0 * Vector3d(
          sqrt(rot_stiff_x_),
          sqrt(rot_stiff_y_),
          sqrt(rot_stiff_z_)
      )).asDiagonal();

    n_stiffness = 10.0;

    return CallbackReturn::SUCCESS;

  }

  CallbackReturn CartesianJointImpedanceForceController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/){
      franka_robot_model_->release_interfaces();
      return CallbackReturn::SUCCESS;
    }

  void CartesianJointImpedanceForceController::desiredCartesianCallback(const std_msgs::msg::Float64MultiArray& msg) {
    if (msg.data[0]){
      for (auto i = 0; i < 3; ++i) {
        target_desired_position[i] = msg.data[i];
      }
      if (msg.data[11]){ // for orientation matrix
        Matrix3d desired_orientation_mat;
        for (auto i = 0; i < 3; ++i) {
          for (auto j = 0; j < 3; ++j) {
            desired_orientation_mat(i, j) = msg.data[3+3*i+j];
          }
        }
        target_desired_orientation = Eigen::Quaterniond(desired_orientation_mat);
      }

    }
  }

  void CartesianJointImpedanceForceController::desiredJointCallback(
    const std_msgs::msg::Float64MultiArray& msg) {
    if (msg.data[0]){
      for (auto i = 0; i < 7; ++i) {
        target_desired_qn[i] = msg.data[i];
      }
    }
  }


  rcl_interfaces::msg::SetParametersResult 
  CartesianJointImpedanceForceController::impedanceCallback(const std::vector<rclcpp::Parameter>& parameters){
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "";
    for (const auto& p : parameters) {
        if (p.get_name() == "pos_stiff_x") pos_stiff_x_ = p.as_double();
        if (p.get_name() == "pos_stiff_y") pos_stiff_y_ = p.as_double();
        if (p.get_name() == "pos_stiff_z") pos_stiff_z_ = p.as_double();

        if (p.get_name() == "rot_stiff_x") rot_stiff_x_ = p.as_double();
        if (p.get_name() == "rot_stiff_y") rot_stiff_y_ = p.as_double();
        if (p.get_name() == "rot_stiff_z") rot_stiff_z_ = p.as_double();
    }

    // Update stiffness matrix
    stiffness.topLeftCorner(3,3) =
        Vector3d(pos_stiff_x_, pos_stiff_y_, pos_stiff_z_).asDiagonal();

    stiffness.bottomRightCorner(3,3) =
        Vector3d(rot_stiff_x_, rot_stiff_y_, rot_stiff_z_).asDiagonal();


    return result;

  }


}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianJointImpedanceForceController,
                       controller_interface::ControllerInterface)
