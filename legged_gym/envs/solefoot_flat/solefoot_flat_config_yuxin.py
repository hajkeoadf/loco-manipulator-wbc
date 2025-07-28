
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

class BipedCfgSF(BaseConfig):
    class arm:
        num_actions_arm = 6
        arm_num_privileged_obs = 9
        arm_num_observation_history = 30
        arm_num_observations = 26 - 6
        arm_num_obs_history = arm_num_observations * arm_num_observation_history
        arm_num_commands = 6
        num_actions_arm_cd = 8

        class commands:
            angle75 = torch.deg2rad(torch.tensor(75))
            angle60 = torch.deg2rad(torch.tensor(60))
            l = [0.3, 0.77]
            p = [-torch.pi*0.45 , torch.pi*0.45]  # 75 
            y = [-torch.pi/2 , torch.pi/2]
            roll_ee = [-torch.pi * 0.45, torch.pi * 0.45]
            pitch_ee = [-angle60 , angle60]
            yaw_ee = [-angle75 , angle75]
            
            T_traj = [2, 3.]
            T_force_range = [1, 4.]
            add_force_thres = 0.3

        class obs_scales:
            l = 1.
            p = 1.
            y = 1.
            wx = 1.
            wy = 1.
            wz = 1.

    class legged:
        num_actions_loco = 8
        legged_num_privileged_obs = 2
        legged_num_observation_history = 30
        legged_num_observations = 56
        legged_num_obs_history = legged_num_observations * legged_num_observation_history
        legged_num_commands = 5
        legged_actions = 12

    class env:
        num_envs = 4096  # 减少环境数量以提高稳定性
        # num_privileged_group = 0 # 4096
        # num_proprio_group = num_envs - num_privileged_group
        num_observations = 54  # 修正观测空间维度：3(ang_vel) + 3(gravity) + 14(dof_pos) + 14(dof_vel) + 14(actions) + 1(clock_sin) + 1(clock_cos) + 4(gaits) = 54
        num_critic_observations = 3 + num_observations # add lin_vel to the front
        num_height_samples = 187  # 增加高度采样点
        num_privileged_obs = (
            num_observations + 3 + 12 + num_height_samples + 6 + 20 + 6
        )  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 14  # 修正动作数量，与实际关节数量匹配
        ee_idx = 10
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 5  # number of observations stacked together
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
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0  # 增加摩擦力
        dynamic_friction = 1.0  # 增加摩擦力
        restitution = 0.0  # 减少弹性
        # rough terrain only:
        measure_heights = True  # 启用高度测量
        critic_measure_heights = True
        measured_points_x = [
            -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
        ]  # 参考airbot的测量点配置
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5 + 4  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )
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
        num_commands = 5  # 命令数量：lin_vel_x, lin_vel_y, ang_vel_yaw, heading, stand_still
        resampling_time = 10.0  # 增加重采样时间
        heading_command = False  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1
        zero_command_prob = 0.8

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5] # [1.0, 2.5]
            offsets = [0.5, 0.5]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.10, 0.20] # [0.0, 0.1]

    class init_state:
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
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
            "J1": 0.0,    # 肩部旋转
            "J2": 0.0,  # 肩部俯仰
            "J3": 0.0,    # 肘部
            "J4": 0.0,    # 腕部俯仰
            "J5": 0.0,    # 腕部偏航
            "J6": 0.0,    # 腕部旋转
        }

        init_stand_joint_angles = {
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.58,
            "knee_L_Joint": 1.35,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.58,
            "knee_R_Joint": -1.35,

            "ankle_L_Joint": -0.8,
            "ankle_R_Joint": -0.8,
            
            # 机械臂站立时的关节角度
            "J1": 0.0,    # 肩部旋转
            "J2": 0.0,  # 肩部俯仰
            "J3": 0.0,    # 肘部
            "J4": 0.0,    # 腕部俯仰
            "J5": 0.0,    # 腕部偏航
            "J6": 0.0,    # 腕部旋转
        }

    class control:
        action_scale = 0.25

        control_type = "P"
        stiffness = {
            # 双足关节 - 参考airbot的刚度
            "abad_L_Joint": 10,
            "hip_L_Joint": 10,
            "knee_L_Joint": 10,
            "ankle_L_Joint": 10,
            "abad_R_Joint": 10,
            "hip_R_Joint": 10,
            "knee_R_Joint": 10,
            "ankle_R_Joint": 10,
            # 机械臂关节 - 参考airbot的刚度
            "J1": 20,
            "J2": 20,
            "J3": 20,
            "J4": 20,
            "J5": 20,
            "J6": 20,
        }  # [N*m/rad]
        damping = {
            # 双足关节 - 参考airbot的阻尼
            "abad_L_Joint": 0.2,
            "hip_L_Joint": 0.2,
            "knee_L_Joint": 0.2,
            "ankle_L_Joint": 10,
            "abad_R_Joint": 0.2,
            "hip_R_Joint": 0.2,
            "knee_R_Joint": 0.2,
            "ankle_R_Joint": 10,
            # 机械臂关节 - 参考airbot的阻尼
            "J1": 0.5,
            "J2": 0.5,
            "J3": 0.5,
            "J4": 0.5,
            "J5": 0.5,
            "J6": 0.5,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 减少decimation提高控制频率
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]

        pull_off_robots = False
        pull_interval_s = 6
        max_pull_vel_z = 0.25
        force_duration_s = 3.0  # 施加外力的持续时间

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/SF_TRON1A/urdf/robot.urdf"
        name = "pointfoot_flat"
        foot_name = "ankle"
        foot_radius = 0.00
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        # 暂时关闭大部分随机化以提高稳定性
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
            # 参考airbot的成功奖励权重
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
            # 移除可能导致不稳定的复杂奖励
            # keep_balance = 1.0
            # tracking_lin_vel_x = 1.5
            # tracking_lin_vel_y = 1.5
            # feet_distance = -100
            # feet_regulation = -0.05
            # tracking_contacts_shaped_force = -2.0
            # tracking_contacts_shaped_vel = -2.0
            # tracking_contacts_shaped_height = -2.0
            # feet_contact_forces = -0.002
            # ankle_torque_limits = -0.1
            # power = -2e-4
            # relative_feet_height_tracking = 1.0
            # zero_command_nominal_state = -10.0
            # keep_ankle_pitch_zero_in_air = 1.0
            # foot_landing_vel = -10.0
            # arm_stability = -0.5

        only_positive_rewards = True  # 启用只正奖励以提高稳定性
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.75 # 0.56 # lower than previous height
        feet_height_target = 0.10
        min_feet_distance = 0.20
        max_contact_force = 100.0  # forces above this value are penalized
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
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [5, -5, 3]  # [m]
        # lookat = [11.0, 5, 3.0]  # [m]
        lookat = [0, 0, 0]  # [m]
        realtime_plot = True

    class sim:
        dt = 0.0025
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 0
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class BipedCfgPPOSF(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = (BipedCfgSF.env.num_observations + BipedCfgSF.env.num_height_samples) * BipedCfgSF.env.obs_history_length  # (48 + 187) * 5 = 1175
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False
        encoder_des = "Base linear velocity"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False
        fix_std_noise_value = None

    class algorithm:
        # PPO training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_SF"
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "SF_TRON1A"
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
