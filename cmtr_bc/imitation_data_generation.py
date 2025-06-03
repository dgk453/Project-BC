"""Extract expert states and actions from Waymo Open Dataset."""
import torch
import numpy as np
import imageio
import logging
import argparse

from gpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.roadgraph import LocalRoadGraphPoints
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
)

logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Select the dynamics model that you use")
    parser.add_argument(
        "--dynamics-model",
        "-d",
        type=str,
        default="delta_local",
        choices=["delta_local", "bicycle", "classic"],
    )
    args = parser.parse_args()
    return args


def map_to_closest_discrete_value(grid, cont_actions):
    """
    Find the nearest value in the action grid for a given expert action.
    """
    # Calculate the absolute differences and find the indices of the minimum values
    abs_diff = torch.abs(grid.unsqueeze(0) - cont_actions.unsqueeze(-1))
    indx = torch.argmin(abs_diff, dim=-1)

    # Gather the closest values based on the indices
    closest_values = grid[indx]

    return closest_values, indx


def create_infos(trajectories, local_polylines, local_goals, env, waymo_dataset, prior_frame=5, future_frame=1, include_goals=False): 
    processed_obs = []
    processed_actions = []

    control_agents = env.get_controlled_agents_mask()

    object_type_dict = {
        0: 'TYPE_UNSET',
        7: 'TYPE_VEHICLE',
        8: 'TYPE_PEDESTRIAN',
        9: 'TYPE_CYCLIST',
        4: 'TYPE_OTHER'
    }
    
    for i in range(len(trajectories)): 
        control_mask = control_agents[i].bool()
        full_trajectories = torch.concat(trajectories[i], dim=-2)
        if (include_goals): 
            full_goals = torch.concat(local_goals[i], dim=-2)
        init_mask = full_trajectories[:, 0, -3] == 1 # make mask for valid trajectories
        full_trajectories = full_trajectories[init_mask]
        if (include_goals): 
            full_goals = full_goals[init_mask]
        episode_len = full_trajectories.shape[1]
        control_mask = control_mask[init_mask]

        # TODO: Maybe clip full trajectories so that there are no crazy jumps? 

        for time_index in range(prior_frame - 1, episode_len - future_frame): 
            left_cut_off = max(0, time_index - prior_frame + 1)
            right_cut_off = min(episode_len, time_index + future_frame + 1)

            current_time_index = time_index - left_cut_off

            traj = full_trajectories[:, left_cut_off:right_cut_off]
            if (include_goals): 
                goals = full_goals[:, left_cut_off:right_cut_off]

            pred_mask = traj[:, current_time_index, -3] == 1
            control_mask = control_mask & pred_mask

            localized_polylines = local_polylines[time_index][i][init_mask][control_mask]
            localized_polylines_mask = localized_polylines[..., 0] != 0 # Undefined polyline types
            # print(localized_polylines.shape, localized_polylines_mask.shape)
            localized_polylines_mask = torch.ones_like(localized_polylines_mask)
            localized_polylines_center = localized_polylines[...,0, :3]

            # actions = expert_actions[i][init_mask][control_mask][:, time_index]

            if ((control_mask == False).all() or (localized_polylines == 0).all()):
                break

            obj_types_pre = traj[:, 0, -1].cpu().numpy()
            obj_types = np.array([object_type_dict[i] for i in obj_types_pre])
            object_trajectories = None
            if (include_goals): 
                object_trajectories = torch.cat([traj[..., :-3], goals, traj[..., -3].unsqueeze(-1)], dim=-1) # attributes, goals, valid
            else: 
                object_trajectories = traj[..., :-2]

            if (control_mask == False).all(): 
                break 

            # print(f"Object trajectory shape: {object_trajectories.shape}")

            infos = {
                "scenario_id": "0",
                "timestamps_seconds": torch.arange(traj.shape[1]).cpu().numpy()/10, 
                "current_time_index": current_time_index,
                "sdc_track_index": 0,
                "objects_of_interest": [],
                "tracks_to_predict": {
                    "track_index": control_mask.nonzero().flatten().cpu().numpy(),
                    "difficulty": None,
                    "object_type": obj_types[control_mask.cpu().numpy()], #types in waymo_types of CMTR/mtr/datasets/waymo/waymo_types.py
                },
                "track_infos": {
                    "object_id": traj[:, 0, -2].cpu().numpy(),
                    "object_type": obj_types, #types in waymo_types of CMTR/mtr/datasets/waymo/waymo_types.py
                    "trajs": object_trajectories.cpu().numpy() 
                },
                "dynamic_map_infos": {
                    "lane_id": None,
                    "state": None,
                    "stop_point": None
                },
                "map_infos": {
                    "lane": None,
                    "road_line": None,
                    "road_edge": None,
                    "stop_sign": None,
                    "crosswalk": None,
                    "speed_bump": None,
                    "localized_polylines": localized_polylines.cpu().numpy(), 
                    "localized_polylines_mask": localized_polylines_mask.cpu().numpy(),
                    "localized_polylines_center": localized_polylines_center.cpu().numpy()
                }
            }

            ret = waymo_dataset.create_scene_level_data(None, infos)
            processed_obs.append(ret)
            # processed_actions.append(actions)

    return processed_obs


def create_infos_simple(trajectories, local_polylines, local_goals, env, waymo_dataset, expert_actions, prior_frame=5, future_frame=1, include_goals=False): 
    processed_obs = []
    processed_actions = []

    control_agents = env.get_controlled_agents_mask()

    object_type_dict = {
        0: 'TYPE_UNSET',
        7: 'TYPE_VEHICLE',
        8: 'TYPE_PEDESTRIAN',
        9: 'TYPE_CYCLIST',
        4: 'TYPE_OTHER'
    }
    
    for i in range(len(trajectories)): 
        control_mask = control_agents[i]
        full_trajectories = torch.concat(trajectories[i], dim=-2).cpu()
        if (include_goals): 
            full_goals = torch.concat(local_goals[i], dim=-2).cpu()
        init_mask = full_trajectories[:, 0, -3] == 1 # make mask for valid trajectories
        full_trajectories = full_trajectories[init_mask]
        episode_len = full_trajectories.shape[1]
        current_time_index = 11
        control_mask = control_mask[init_mask].cpu()
        control_mask = control_mask & (full_trajectories[:, current_time_index, -3] == 1)

        localized_polylines = local_polylines[current_time_index][i][init_mask][control_mask]
        localized_polylines_mask = localized_polylines[..., 0] != 0 # Undefined polyline types
        localized_polylines_mask = torch.ones_like(localized_polylines_mask)
        localized_polylines_center = localized_polylines[...,0, :3]

        actions = expert_actions[i][init_mask][control_mask][:, current_time_index]
        # if (predict_mask == False).all():
        #     break

        # print(f"Prd mask: {predict_mask}")

        obj_types_pre = full_trajectories[:, 0, -1].cpu().numpy()
        obj_types = np.array([object_type_dict[i] for i in obj_types_pre])
        if (include_goals):
            trajs = torch.cat([full_trajectories[..., :-3], full_goals[init_mask], full_trajectories[..., -3].unsqueeze(-1)], dim=-1) # attributes, goals, valid
        else: 
            trajs = full_trajectories[..., :-2]
        infos = {
            "scenario_id": "0",
            "timestamps_seconds": torch.arange(episode_len).cpu().numpy()/10, 
            "current_time_index": current_time_index,
            "sdc_track_index": 0,
            "objects_of_interest": [],
            "tracks_to_predict": {
                "track_index": control_mask.nonzero().flatten().cpu().numpy(),
                "difficulty": None,
                "object_type": obj_types[control_mask.cpu()], #types in waymo_types of CMTR/mtr/datasets/waymo/waymo_types.py
            },
            "track_infos": {
                "object_id": full_trajectories[:, 0, -2].cpu().numpy(),
                "object_type": obj_types, #types in waymo_types of CMTR/mtr/datasets/waymo/waymo_types.py
                "trajs": trajs.cpu().numpy(),
            },
            "dynamic_map_infos": {
                "lane_id": None,
                "state": None,
                "stop_point": None
            },
            "map_infos": {
                "lane": None,
                "road_line": None,
                "road_edge": None,
                "stop_sign": None,
                "crosswalk": None,
                "speed_bump": None,
                "localized_polylines": localized_polylines.cpu().numpy(), 
                "localized_polylines_mask": localized_polylines_mask.cpu().numpy(),
                "localized_polylines_center": localized_polylines_center.cpu().numpy()
            }
        }

        ret = waymo_dataset.create_scene_level_data(None, infos)
        processed_obs.append(ret)
        processed_actions.append(actions)

    return processed_obs, processed_actions

def generate_state_action_pairs(
    env,
    device,
    action_space_type="discrete",
    use_action_indices=False,
    make_video=False,
    render_index=[0],
    save_path="output_video.mp4",
    CMTR = False,
    waymo_dataset = None,
    prior_frame=None, 
    future_frame=None, 
    include_goals = False,
    simple = False,
):
    """Generate pairs of states and actions from the Waymo Open Dataset.

    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
        device (str): Where to run the simulation (cpu or cuda).
        action_space_type (str): discrete, multi-discrete, continuous
        use_action_indices (bool): Whether to return action indices instead of action values.
        make_video (bool): Whether to save a video of the expert trajectory.
        render_index (int): Index of the world to render (must be <= num_worlds).

    Returns:
        expert_actions: Expert actions for the controlled agents. An action is a
            tuple with (acceleration, steering, heading).
        obs_tensor: Expert observations for the controlled agents.
    """
    frames = [[] for _ in range(render_index[1] - render_index[0])]

    # logging.info(
    #     f"Generating expert actions and observations for {env.num_worlds} worlds \n"
    # )

    # Reset the environment
    obs = env.reset()

    # Get expert actions for full trajectory in all worlds
    expert_actions, expert_speeds, expert_positions, expert_yaws = env.get_expert_actions()
    if action_space_type == "discrete":
        # Discretize the expert actions: map every value to the closest
        # value in the action grid.
        disc_expert_actions = expert_actions.clone()
        if env.config.dynamics_model == "delta_local":
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.dx, cont_actions=expert_actions[:, :, :, 0]
            )
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.dy, cont_actions=expert_actions[:, :, :, 1]
            )
            disc_expert_actions[:, :, :, 2], _ = map_to_closest_discrete_value(
                grid=env.dyaw, cont_actions=expert_actions[:, :, :, 2]
            )
        else:
            # Acceleration
            disc_expert_actions[:, :, :, 0], _ = map_to_closest_discrete_value(
                grid=env.accel_actions, cont_actions=expert_actions[:, :, :, 0]
            )
            # Steering
            disc_expert_actions[:, :, :, 1], _ = map_to_closest_discrete_value(
                grid=env.steer_actions, cont_actions=expert_actions[:, :, :, 1]
            )

        if use_action_indices:  # Map action values to joint action index
            logging.info("Mapping expert actions to joint action index... \n")
            expert_action_indices = torch.zeros(
                expert_actions.shape[0],
                expert_actions.shape[1],
                expert_actions.shape[2],
                1,
                dtype=torch.int32,
            ).to(device)
            for world_idx in range(disc_expert_actions.shape[0]):
                for agent_idx in range(disc_expert_actions.shape[1]):
                    for time_idx in range(disc_expert_actions.shape[2]):
                        action_val_tuple = tuple(
                            round(x, 3)
                            for x in disc_expert_actions[
                                world_idx, agent_idx, time_idx, :
                            ].tolist()
                        )
                        if not env.config.dynamics_model == "delta_local":
                            action_val_tuple = (
                                action_val_tuple[0],
                                action_val_tuple[1],
                                0.0,
                            )

                        action_idx = env.values_to_action_key.get(
                            action_val_tuple
                        )
                        expert_action_indices[
                            world_idx, agent_idx, time_idx
                        ] = action_idx

            expert_actions = expert_action_indices
        else:
            # Map action values to joint action index
            expert_actions = disc_expert_actions
    elif action_space_type == "multi_discrete":
        """will be update"""
        pass
    else:
        pass
        # logging.info("Using continuous expert actions... \n")

    # Storage
    expert_observations_lst = []
    expert_actions_lst = []
    expert_next_obs_lst = []
    expert_dones_lst = []

    # Initialize dead agent mask

    dead_agent_mask = ~env.cont_agent_mask.clone()

    trajectories = [[] for i in range(env.num_worlds)]
    localized_goals = [[] for i in range(env.num_worlds)]
    local_polylines = []

    def collate_trajectories(): 
        global_ego = GlobalEgoState.from_tensor(
            abs_self_obs_tensor=env.sim.absolute_self_observation_tensor(),
            backend=env.backend,
            device=device
        )

        logal_ego = LocalEgoState.from_tensor(
            self_obs_tensor=env.sim.self_observation_tensor(),
            backend=env.backend,
            device=device,
        )

        roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=env.sim.agent_roadmap_tensor(),
            backend=env.backend,
            device=env.device,
        )

        type = env.sim.info_tensor().to_torch()[..., -1]

        vel_x = logal_ego.speed * torch.cos(global_ego.rotation_angle)
        vel_y = logal_ego.speed * torch.sin(global_ego.rotation_angle)

        # print(f"dones {~(dones[0].bool().unsqueeze(-1))}")

        roadgraph = LocalRoadGraphPoints.from_tensor(
            local_roadgraph_tensor=env.sim.agent_roadmap_tensor(),
            backend=env.backend,
            device=env.device,
        )

        roadgraph.one_hot_encode_road_point_types()

        for i in range(env.num_worlds): 
            traj_lst = [
                global_ego.pos_x[i].to(device).unsqueeze(-1), # center_x
                global_ego.pos_y[i].to(device).unsqueeze(-1), # center_y
                torch.zeros_like(global_ego.pos_y[i].to(device).unsqueeze(-1)), # center_z
                global_ego.vehicle_length[i].to(device).unsqueeze(-1), # length
                global_ego.vehicle_width[i].to(device).unsqueeze(-1), # width
                global_ego.vehicle_height[i].to(device).unsqueeze(-1), # height
                global_ego.rotation_angle[i].to(device).unsqueeze(-1), # heading
                vel_x[i].to(device).unsqueeze(-1), # vel_x
                vel_y[i].to(device).unsqueeze(-1), # vel_y
                ((type[i] != 0).bool() & (abs(global_ego.pos_x[i]) < 1_000)).int().unsqueeze(-1),
                logal_ego.id[i].int().to(device).unsqueeze(-1), # id
                type.to(device)[i].unsqueeze(-1) # type
            ]

            goal_dx = global_ego.goal_x[i] - global_ego.pos_x[i]
            goal_dy = global_ego.goal_y[i] - global_ego.pos_y[i]
            goal_length = torch.sqrt(goal_dx**2 + goal_dy**2) 
            goal_dx /= goal_length
            goal_dy /= goal_length

            goal = torch.concat((goal_dx.unsqueeze(-1), 
                                  goal_dy.unsqueeze(-1)), 
                                  dim=-1)
            
            traj = torch.concat(traj_lst, dim=-1)
            trajectories[i].append(traj.unsqueeze(1))
            localized_goals[i].append(goal.unsqueeze(1))

        polylines = torch.concat([
                        roadgraph.x.unsqueeze(-1), # Center x
                        roadgraph.y.unsqueeze(-1), # Center y
                        torch.zeros_like(roadgraph.x.unsqueeze(-1)), # Zerod z center
                        roadgraph.segment_length.unsqueeze(-1),
                        roadgraph.segment_width.unsqueeze(-1), 
                        roadgraph.orientation.unsqueeze(-1),
                        roadgraph.type
                    ], dim=-1).unsqueeze(-2)
            
        local_polylines.append(polylines)

    for time_step in range(env.episode_len):

        if (CMTR):
            collate_trajectories()
            env.step_dynamics(expert_actions[:, :, time_step, :])
            dones = env.get_dones()
            dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        else: 
            env.step_dynamics(expert_actions[:, :, time_step, :])
            next_obs = env.get_obs()

            dones = env.get_dones()
            # infos = env.get_()

            # print("obs shape: ") 
            # print(next_obs.shape)

            # Unpack and store (obs, action, next_obs, dones) pairs for controlled agents
            expert_observations_lst.append(obs[~dead_agent_mask, :])
            expert_actions_lst.append(
                expert_actions[~dead_agent_mask][:, time_step, :]
            )

            expert_next_obs_lst.append(next_obs[~dead_agent_mask, :])
            expert_dones_lst.append(dones[~dead_agent_mask])

            # Update
            obs = next_obs
            dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        # Render
        if make_video:
            for render in range(render_index[0], render_index[1]):
                frame = env.render(world_render_idx=render)
                frames[render].append(frame)
        if (dead_agent_mask == True).all():
            break

    # is_collision = infos[:, :, :3].sum(dim=-1)
    # is_goal = infos[:, :, 3]
    # collision_mask = is_collision != 0
    # goal_mask = is_goal != 0
    # valid_collision_mask = collision_mask & alive_agent_mask
    # valid_goal_mask = goal_mask & alive_agent_mask
    # collision_rate = (
    #     valid_collision_mask.sum().float() / alive_agent_mask.sum().float()
    # )
    # goal_rate = valid_goal_mask.sum().float() / alive_agent_mask.sum().float()

    # print(f"Collision {collision_rate} Goal {goal_rate}")

    if make_video:
        for render in range(render_index[0], render_index[1]):
            imageio.mimwrite(
                f"{save_path}_world_{render}.mp4",
                np.array(frames[render]),
                fps=30,
            )

    if (not CMTR): 
        flat_expert_obs = torch.cat(expert_observations_lst, dim=0)
        flat_expert_actions = torch.cat(expert_actions_lst, dim=0)
        flat_next_expert_obs = torch.cat(expert_next_obs_lst, dim=0)
        flat_expert_dones = torch.cat(expert_dones_lst, dim=0)
        return (
            flat_expert_obs,
            flat_expert_actions,
            flat_next_expert_obs,
            flat_expert_dones,
            # goal_rate,
            # collision_rate,
        )
    else: 
        # create info
        assert not (waymo_dataset is None)
        # observations, actions = create_infos_simple(trajectories, env, waymo_dataset, expert_actions, prior_frame=prior_frame, future_frame=future_frame)
        if simple: 
            return create_infos_simple(trajectories, local_polylines, localized_goals, 
                                       env, waymo_dataset, expert_actions, prior_frame=prior_frame, 
                                       future_frame=future_frame, include_goals=include_goals)
        else: 
            return create_infos(trajectories, local_polylines, localized_goals, 
                                env, waymo_dataset, prior_frame=prior_frame, future_frame=future_frame, 
                                include_goals=include_goals)




if __name__ == "__main__":
    import argparse

    args = parse_args()
    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = 10
    MAX_NUM_OBJECTS = 128

    # Initialize lists to store results
    num_actions = []
    goal_rates = []
    collision_rates = []

    # Set the environment and render configurations
    # Action space (joint discrete)

    render_config = RenderConfig(draw_obj_idx=True)
    scene_config = SceneConfig(
        "/data/formatted_json_v2_no_tl_train/", NUM_WORLDS
    )
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(torch.linspace(-0.3, 0.3, 7), decimals=3),
        accel_actions=torch.round(torch.linspace(-6.0, 6.0, 7), decimals=3),
        dx=torch.round(torch.linspace(-3.0, 3.0, 100), decimals=3),
        dy=torch.round(torch.linspace(-3.0, 3.0, 100), decimals=3),
        dyaw=torch.round(torch.linspace(-1.0, 1.0, 300), decimals=3),
    )

    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,  # Number of agents to control
        device="cpu",
        render_config=render_config,
        action_type="continuous",
    )
    # Generate expert actions and observations
    (
        expert_obs,
        expert_actions,
        next_expert_obs,
        expert_dones,
        goal_rate,
        collision_rate,
    ) = generate_state_action_pairs(
        env=env,
        device="cpu",
        action_space_type="continuous",  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=True,  # Record the trajectories as sanity check
        render_index=[0, 1],  # start_idx, end_idx
        save_path="use_discr_actions_fix",
    )
    env.close()
    del env
    del env_config

    # Uncommment to save the expert actions and observations
    # torch.save(expert_actions, "expert_actions.pt")
    # torch.save(expert_obs, "expert_obs.pt")
