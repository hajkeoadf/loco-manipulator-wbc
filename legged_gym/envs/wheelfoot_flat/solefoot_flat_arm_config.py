# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.base_config import BaseConfig
from .solefoot_flat_config import BipedCfgSF

class BipedCfgSFArm(BipedCfgSF):
    """机械臂子类的配置文件，继承自BipedCfgSF"""
    
    class env:
        num_envs = 4096
        num_observations = 54 + 6 + 7 + 6 + 6  # 基础观测 + 机械臂观测 (关节状态 + 末端状态 + 目标状态)
        num_critic_observations = 3 + num_observations
        num_height_samples = 187
        num_privileged_obs = (
            num_observations + 3 + 12 + num_height_samples + 6 + 20 + 6
        )
        num_actions = 14  # 8腿 + 6臂
        ee_idx = 10
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 5
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

        # privileged obs flags
        priv_observe_friction = True
        priv_observe_friction_indep = True
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_restitution = True
        priv_observe_base_mass = True
        priv_observe_com_displacement = True
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_joint_friction = True
        priv_observe_Kp_factor = True
        priv_observe_Kd_factor = True
        priv_observe_contact_forces = False
        priv_observe_contact_states = False
        priv_observe_body_velocity = False
        priv_observe_foot_height = False
        priv_observe_body_height = False
        priv_observe_gravity = False
        priv_observe_terrain_type = False
        priv_observe_clock_inputs = False
        priv_observe_doubletime_clock_inputs = False
        priv_observe_halftime_clock_inputs = False
        priv_observe_desired_contact_states = False
        priv_observe_dummy_variable = False

    class terrain:
        mesh_type = "plane"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        critic_measure_heights = True
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        ]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5 + 4
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        slope_treshold = 0.75
        simplify_grid = False
        edge_width_thresh = 0.01
        high_horizontal_scale = 0.01
        edge_width_thresh_up = 0.18
        edge_width_thresh_down = 0.05

    class commands:
        curriculum = False
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 5
        resampling_time = 10.0
        heading_command = False
        min_norm = 0.1
        zero_command_prob = 0.8

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 1.0]
            heading = [-3.14159, 3.14159]
            base_height = [0.68, 0.78]
            stand_still = [0, 1]

    class gait:
        num_gait_params = 4
        resampling_time = 5
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5]
            offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.10, 0.20]

    class init_state:
        pos = [0.0, 0.0, 0.8]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            # 双足
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0,
            # 机械臂 - 设置为更稳定的初始姿态
            "J1": 0.0,
            "J2": -1.57,  # 机械臂向下
            "J3": 0.0,
            "J4": 0.0,
            "J5": 0.0,
            "J6": 0.0,
        }

        init_stand_joint_angles = {
            # 双足
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0,
            # 机械臂
            "J1": 0.0,
            "J2": -1.57,
            "J3": 0.0,
            "J4": 0.0,
            "J5": 0.0,
            "J6": 0.0,
        }

    class control:
        action_scale = 0.25
        control_type = "P"
        stiffness = {
            # 双足关节
            "abad_L_Joint": 10,
            "hip_L_Joint": 10,
            "knee_L_Joint": 10,
            "ankle_L_Joint": 10,
            "abad_R_Joint": 10,
            "hip_R_Joint": 10,
            "knee_R_Joint": 10,
            "ankle_R_Joint": 10,
            # 机械臂关节 - 更高的刚度
            "J1": 100,
            "J2": 100,
            "J3": 100,
            "J4": 50,
            "J5": 50,
            "J6": 50,
        }
        damping = {
            # 双足关节
            "abad_L_Joint": 0.2,
            "hip_L_Joint": 0.2,
            "knee_L_Joint": 0.2,
            "ankle_L_Joint": 10,
            "abad_R_Joint": 0.2,
            "hip_R_Joint": 0.2,
            "knee_R_Joint": 0.2,
            "ankle_R_Joint": 10,
            # 机械臂关节 - 更高的阻尼
            "J1": 10,
            "J2": 10,
            "J3": 10,
            "J4": 5,
            "J5": 5,
            "J6": 5,
        }
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0

        pull_off_robots = False
        pull_interval_s = 6
        max_pull_vel_z = 0.25
        force_duration_s = 3.0

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf"
        name = "pointfoot_flat_arm"
        foot_name = "ankle"
        foot_radius = 0.00
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01
        disable_gravity = False

    class domain_rand:
        randomize_friction = False
        friction_range = [0.2, 1.5]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_base_mass = False
        added_mass_range = [-4., 4.]
        randomize_base_com = False
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = False
        randomize_inertia_range = [0.8, 1.2]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.0
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = False
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = False
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = False
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = False
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = False
        randomize_imu_offset = False
        delay_ms_range = [0, 20]

    class rewards:
        class scales:
            # 基础奖励
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.5
            torques = -0.0002
            dof_vel = -0.0
            dof_acc = -2.5e-8
            base_height = -0.2
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0
            dof_pos_limits = -10.0
            
            # 机械臂奖励
            arm_tracking = 1.0
            arm_energy = -0.01
            arm_torques = -0.001
            arm_stability = -0.01

        only_positive_rewards = True
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.75
        feet_height_target = 0.10
        min_feet_distance = 0.20
        max_contact_force = 100.0
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        about_landing_threshold = 0.05

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05
            base_z = 1./0.6565

        clip_observations = 100.0
        clip_actions = 100.0

        class friction_range:
            min = 0.0
            max = 1.0

        class restitution_range:
            min = 0.0
            max = 1.0

        class added_mass_range:
            min = -4.0
            max = 4.0

        class com_displacement_range:
            min = -0.1
            max = 0.1

        class motor_strength_range:
            min = 0.8
            max = 1.2

        class motor_offset_range:
            min = -0.05
            max = 0.05

        class body_height_range:
            min = 0.6
            max = 0.8

        class gravity_range:
            min = 9.0
            max = 10.0

        class ground_friction_range:
            min = 0.2
            max = 1.5

    class noise:
        add_noise = True
        noise_level = 1.5

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [5, -5, 3]
        lookat = [0, 0, 0]
        realtime_plot = True

    class sim:
        dt = 0.0025
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 0
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class goal_ee:
        class ranges:
            init_pos_l = [0.3, 0.5]
            init_pos_p = [-0.2, 0.2]
            init_pos_y = [-0.2, 0.2]
            final_pos_l = [0.3, 0.5]
            final_pos_p = [-0.2, 0.2]
            final_pos_y = [-0.2, 0.2]
            final_delta_orn = [-0.1, 0.1]

        l_schedule = [0.4, 2000, 4000]
        p_schedule = [0.0, 2000, 4000]
        y_schedule = [0.0, 2000, 4000]
        tracking_ee_reward_schedule = [0.1, 1000, 1000]
        final_tracking_ee_reward = 1.0


class BipedCfgPPOSFArm(BaseConfig):
    seed = 1
    runner_class_name = "RSLOnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = (BipedCfgSFArm.env.num_observations + BipedCfgSFArm.env.num_height_samples) * BipedCfgSFArm.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False
        encoder_des = "Base linear velocity"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        orthogonal_init = False
        fix_std_noise_value = None

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "RSLPPO"
        num_steps_per_env = 24
        max_iterations = 10000

        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_SF_Arm"
        save_interval = 500
        experiment_name = "SF_TRON1A_Arm"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None 