import os
from isaacgym import gymapi, gymutil, gymtorch
import torch

def load_urdf_and_test_functions(urdf_path):
    # 初始化 Isaac Gym
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # 创建默认地面平面
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # 加载 URDF 资产
    asset_root = os.path.dirname(urdf_path)
    asset_file = os.path.basename(urdf_path)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False  # 允许基座运动（测试接触力）
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # 打印关节和刚体信息
    dof_names = gym.get_asset_dof_names(robot_asset)
    print("=== DOF Names (Non-fixed Joints) ===")
    for name in dof_names:
        print(f" - {name}")

    link_names = gym.get_asset_rigid_body_names(robot_asset)
    print("\n=== Link Names ===")
    for name in link_names:
        print(f" - {name}")

    # 创建机器人实例
    spacing = 2.0
    env = gym.create_env(sim, gymapi.Vec3(-spacing, 0.0, -spacing), 
                        gymapi.Vec3(spacing, spacing, spacing), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # 悬空放置以测试自由落体接触力
    actor_handle = gym.create_actor(env, robot_asset, pose, "robot", 0, 0)

    # 初始化 GPU 张量访问
    gym.prepare_sim(sim)

    # 获取张量接口
    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    net_contact_forces = gym.acquire_net_contact_force_tensor(sim)

    # 包装为 PyTorch 张量
    root_states = gymtorch.wrap_tensor(root_tensor)
    dof_states = gymtorch.wrap_tensor(dof_state_tensor)
    contact_forces = gymtorch.wrap_tensor(net_contact_forces)

    # 模拟几步以生成数据
    for _ in range(10):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    # 刷新张量数据
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    # 打印测试结果
    print("\n=== Actor Root State ===")
    print(f"Position (X,Y,Z): {root_states[actor_handle, :3]}")
    print(f"Rotation (Quat W,X,Y,Z): {root_states[actor_handle, 3:7]}")
    print(f"Linear Velocity (X,Y,Z): {root_states[actor_handle, 7:10]}")
    print(f"Angular Velocity (X,Y,Z): {root_states[actor_handle, 10:13]}")

    print("\n=== DOF State ===")
    for i, name in enumerate(dof_names):
        pos = dof_states.view(-1, 2)[i, 0]  # 位置在每行的第0列
        vel = dof_states.view(-1, 2)[i, 1]  # 速度在每行的第1列
        print(f"{name}: Pos={pos:.3f}, Vel={vel:.3f}")

    print("\n=== Net Contact Forces ===")
    for i, name in enumerate(link_names):
        force = contact_forces[i]
        print(f"{name}: Force (X,Y,Z) = {force}")

    gym.destroy_sim(sim)

if __name__ == "__main__":
    # 替换为你的 URDF 文件路径
    urdf_path = "/home/ril/limx_rl/pointfoot-legged-gym/resources/robots/SF_TRON1A/urdf/robot.urdf"
    load_urdf_and_test_functions(urdf_path)