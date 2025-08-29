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

from legged_gym.envs.solefoot_flat.solefoot_flat_config import BipedCfgSF, BipedCfgPPOSF
from legged_gym.envs.base.base_config import BaseConfig
import numpy as np
RESUME = True

class BipedCfgSFWithArm(BipedCfgSF):
    class env:
        num_envs = 4096
        num_observations = 3 + 3 + 14 + 14 + 14 + 1 + 1 + 4 + 5 + 6  
        num_critic_observations = 5 + num_observations
        num_height_samples = 187
        num_privileged_obs = 5
        num_actions = 14  # 8 (legs) + 6 (arm)
        ee_idx = 10 # !!!!要确认
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        obs_history_length = 10
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5
        action_delay = 0  # 动作延迟
        observe_priv = True  # 是否使用privileged observations

        class dog:
            dog_num_observations = 3 + 3 + 8 + 8 + 8 + 1 + 1 + 4 + 5 
            dog_num_privileged_obs = 5
            dog_num_obs_history = 10
            dog_num_actions = 8

        class arm:
            arm_num_observations = 6 + 6 + 6 + 6
            arm_num_privileged_obs = 5
            arm_num_obs_history = 10
            arm_num_actions = 6


    class goal_ee:
        command_mode = 'sphere'  # 'cart' or 'sphere' - 内部仍使用球坐标系
        traj_time = [2.0, 4.0]  # 轨迹时间范围
        hold_time = [1.0, 2.0]  # 保持时间范围
        collision_lower_limits = [-0.6, -0.6, 0.1]  # 碰撞检测下限 - 提高z下限避免目标过低
        collision_upper_limits = [0.6, 0.6, 1.0]  # 碰撞检测上限 - 增加z上限给更多空间
        underground_limit = 0.05  # 地下限制 - 提高地面限制
        num_collision_check_samples = 10  # 碰撞检测采样数
        sphere_error_scale = [1.0, 1.0, 1.0]  # 球坐标系误差缩放
        orn_error_scale = [1.0, 1.0, 1.0]  # 姿态误差缩放
        
        # 课程学习调度参数
        l_schedule = [0, 1]  # 长度课程学习
        p_schedule = [0, 1]  # 俯仰课程学习
        y_schedule = [0, 1]  # 偏航课程学习
        tracking_ee_reward_schedule = [0, 1]  # 跟踪奖励课程学习
        
        class ranges:
            # 笛卡尔坐标范围 (相对于机器人基座) - 便于调试
            init_pos_x = [0.6, 0.8]  # 初始X范围 - 前方距离
            init_pos_y = [0, 0]  # 初始Y范围 - 横向范围，±5cm
            init_pos_z = [0.35, 0.35]  # 初始Z范围 - 高度范围
            init_pos_l = [0.4, 0.8]  # 初始长度范围
            init_pos_p = [np.pi/6, np.pi/3]  # 初始俯仰范围
            init_pos_y = [-np.pi/36, np.pi/36]  # 初始偏航范围
            
            # 最终位置范围 (相对于机器人基座)
            final_pos_x = [0.8, 1.8]  # 最终X范围 - 前方距离
            final_pos_y = [-0.08, 0.08]  # 最终Y范围 - 横向范围，±8cm
            final_pos_z = [-1.0, 1.0]  # 最终Z范围 - 高度范围
            final_pos_l = [0.5, 1.0]  # 最终长度范围
            final_pos_p = [-2*np.pi/5, np.pi/5]  # 最终俯仰范围
            final_pos_y = [-8*np.pi/180, 8*np.pi/180]  # 最终偏航范围
            
            final_delta_orn = [[-0, 0], [-0, 0], [-0, 0]]  # 最终姿态变化范围
            final_tracking_ee_reward = 1.0  # 最终跟踪奖励

    class commands:
        curriculum = True
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 1.0
        non_smooth_max_lin_vel_x = 1.0
        non_smooth_max_lin_vel_y = 1.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 5  # 命令数量：lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_still
        resampling_time = 10.0  # 增加重采样时间
        heading_command = False  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1
        zero_command_prob = 0.8

        class ranges:
            lin_vel_x = [-1.0, 1.5]  # x方向线速度（前进/后退）
            lin_vel_y = [-1.0, 1.0]  # y方向线速度（横移）
            ang_vel_yaw = [-1.0, 1.0]  # z轴角速度（原地转圈）
            heading = [-3.14159, 3.14159] # 期望的朝向角
            base_height = [0.68, 0.78] # 期望的base高度
            stand_still = [0, 1] # 是否站立

    class arm:
        osc_kp = [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]  # 操作空间控制Kp
        osc_kd = [10.0, 10.0, 10.0, 5.0, 5.0, 5.0]  # 操作空间控制Kd
        grasp_offset = 0.05  # 抓取偏移
        init_target_ee_base = [0.2, 0.0, 0.3]  # 初始目标末端执行器位置

    class init_state:
        pos = [0.0, 0.0, 0.8]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            # 双足关节
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0,
            # 机械臂关节
            "J1": 0.0,    # 肩部旋转
            "J2": 0.0,    # 肩部俯仰
            "J3": 0.0,    # 肘部
            "J4": 0.0,    # 腕部俯仰
            "J5": 0.0,    # 腕部偏航
            "J6": 0.0,    # 腕部旋转
        }

    class control:
        action_scale = 0.5  # 增加动作缩放以提高机械臂响应性
        control_type = "P"
        adaptive_arm_gains = False  # 是否使用自适应机械臂增益
        adaptive_arm_gains_scale = 10.0  # 自适应机械臂增益缩放
        torque_supervision = False  # 是否使用力矩监督 - 暂时禁用操作空间控制
        
        stiffness = {
            # 双足关节
            "abad_L_Joint": 45,
            "hip_L_Joint": 45,
            "knee_L_Joint": 45,
            "ankle_L_Joint": 45,
            "abad_R_Joint": 45,
            "hip_R_Joint": 45,
            "knee_R_Joint": 45,
            "ankle_R_Joint": 45,
            # 机械臂关节 - 增加刚度以提高响应性
            "J1": 50,
            "J2": 50,
            "J3": 50,
            "J4": 50,
            "J5": 50,
            "J6": 50,
        }
        
        damping = {
            # 双足关节
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "ankle_L_Joint": 0.8,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,
            "ankle_R_Joint": 0.8,
            # 机械臂关节 - 增加阻尼以提高稳定性
            "J1": 2.0,
            "J2": 2.0,
            "J3": 2.0,
            "J4": 2.0,
            "J5": 2.0,
            "J6": 2.0,
        }
        
        decimation = 8
        user_torque_limit = 80.0
        max_power = 1000.0

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf"
        name = "solefoot_flat_with_arm"
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

    class domain_rand:
        randomize_friction = True
        friction_range = [0.0, 1.6]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 5]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 1.0
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = False
        delay_ms_range = [0, 20]

    class rewards:
        class scales:
            # 腿部奖励
            keep_balance = 3.0
            tracking_lin_vel_x = 1.5
            tracking_lin_vel_y = 1.5
            tracking_ang_vel = 1
            base_height = -10.0  # 从 -10 减少到 -2.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00008
            dof_acc = -2.5e-7
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -100.0  
            action_smooth = -0.01
            orientation = -0.5  # 从 -1.0 减少到 -0.5
            feet_distance = -100.0  
            feet_regulation = -0.05
            tracking_contacts_shaped_force = -2.0
            tracking_contacts_shaped_vel = -2.0
            tracking_contacts_shaped_height = -2.0
            feet_contact_forces = -0.002
            ankle_torque_limits = -0.1
            power = -2e-4
            relative_feet_height_tracking = 1.0
            zero_command_nominal_state = -10.0
            keep_ankle_pitch_zero_in_air = 1.0
            foot_landing_vel = -10.0
            
            # 机械臂奖励 - 大幅增加权重以鼓励运动
            tracking_ee_cart = 0.5  # 大幅增加跟踪奖励
            tracking_ee_sphere = 0.5  # 大幅增加跟踪奖励
            tracking_ee_orn = 0.3  # 增加姿态跟踪奖励
            arm_energy_abs_sum = -0.00005  # 减少能量惩罚，鼓励更多运动

        only_positive_rewards = False
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2
        ang_tracking_sigma = 0.25
        height_tracking_sigma = 0.01
        tracking_ee_sigma = 0.5  # 末端执行器跟踪sigma - 增大以提供更宽容的奖励范围
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

    class termination:
        z_threshold = 0.3  # 终止高度阈值

class BipedCfgPPOSFWithArm(BipedCfgPPOSF):
    seed = 1
    runner_class_name = "OnPolicyRunner"
    
    class MLP_Encoder:
        output_detach = True
        num_input_dim = (BipedCfgSFWithArm.env.num_observations + BipedCfgSFWithArm.env.num_height_samples) * BipedCfgSFWithArm.env.obs_history_length
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
        entropy_coef = 0.05  # 从 0.01 增加到 0.05，鼓励探索
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 5.0e-4  # 从 1.0e-3 减少到 5.0e-4
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        est_learning_rate = 5.0e-4  # 从 1.0e-3 减少到 5.0e-4
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

        mixing_schedule=[0.3, 0, 3000] if not RESUME else [0.3, 0, 1]  # 从 1.0 减少到 0.3

        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.05, 3000, 7000] if not RESUME else [0, 0.05, 1000, 1000]  # 从 0.1 减少到 0.05

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "RSLPPO"
        num_steps_per_env = 40
        max_iterations = 10000
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_SF_with_arm"
        save_interval = 500
        experiment_name = "SF_TRON1A_with_arm"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None 