import torch
from gpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.datatypes.roadgraph import LocalRoadGraphPoints
from gpudrive.datatypes.observation import (
    LocalEgoState,
    GlobalEgoState,
)
import numpy as np


OBJECT_TYPE_DICT = {
    0: 'TYPE_UNSET',
    7: 'TYPE_VEHICLE',
    8: 'TYPE_PEDESTRIAN',
    9: 'TYPE_CYCLIST',
    4: 'TYPE_OTHER'
}

def get_action_from_batch_dict(batch_dict, control_mask_idx, num_env, num_obj): 
    pred_trajs = batch_dict['pred_trajs']
    num_ego_agents, num_modes, num_time_frames, num_features = pred_trajs.shape
    # Choosing the first mode for now
    trajs = pred_trajs[:, 0, :, :]
    padding = torch.zeros(num_ego_agents, 1, num_features)
    trajs = torch.concat([padding, trajs], dim=1)
    delta_pos = torch.diff(pred_trajs, dim=1)[..., :2]
    actions = torch.concat([delta_pos, torch.atan2(actions[..., 1], actions[..., 0])], dim=-1)    
    ret = torch.zeros(num_env, num_obj, 3)
    ret[0, control_mask_idx] = actions
    return ret

def get_action(model, trajectories, local_polylines, local_goals, env, waymo_dataset, prior_frame, future_frame, include_goals=False): 
    # Takes in a single env
    control_mask = env.get_controlled_agents_mask()[0]
    full_trajectories = torch.concat(trajectories, dim=-2)
    if (include_goals): 
        full_goals = torch.concat(local_goals, dim=-2)
    init_mask = full_trajectories[:, 0, -3] == 1 # make mask for valid trajectories
    full_trajectories = full_trajectories[init_mask]
    if (include_goals): 
        goals = full_goals[init_mask]

    control_mask_idx = control_mask.nonzero().flatten()
    control_mask_idx = control_mask_idx[init_mask]
    control_mask = control_mask[init_mask]

    traj = full_trajectories[:, :-prior_frame]
    pred_mask = traj[:, -1, -3] == 1
    control_mask = control_mask & pred_mask
    control_mask_idx = control_mask_idx[control_mask]
    localized_polylines = local_polylines[-1][init_mask][control_mask]
    localized_polylines_mask = localized_polylines[..., 0] != 0 # Undefined polyline types
    # print(localized_polylines.shape, localized_polylines_mask.shape)
    localized_polylines_mask = torch.ones_like(localized_polylines_mask)
    localized_polylines_center = localized_polylines[...,0, :3]

    if ((control_mask == False).all() or (localized_polylines == 0).all()):
        return None

    obj_types_pre = traj[:, 0, -1].cpu().numpy()
    obj_types = np.array([OBJECT_TYPE_DICT[i] for i in obj_types_pre])
    object_trajectories = None

    # Add padding for future observations
    future_traj = torch.zeros(traj.shape[0], future_frame, traj.shape[-1])
    traj = torch.concat([traj, future_traj], dim=1)

    if (include_goals): 
        object_trajectories = torch.cat([traj[..., :-3], goals, traj[..., -3].unsqueeze(-1)], dim=-1) # attributes, goals, valid
    else: 
        object_trajectories = traj[..., :-2]

        # print(f"Object trajectory shape: {object_trajectories.shape}")

    infos = {
        "scenario_id": "0",
        "timestamps_seconds": torch.arange(traj.shape[1]).cpu().numpy()/10, 
        "current_time_index": prior_frame - 1,
        "sdc_track_index": 0,
        "objects_of_interest": [],
        "tracks_to_predict": {
            "track_index": control_mask.nonzero().flatten().cpu().numpy(),
            "difficulty": None,
            "object_type": obj_types[control_mask.cpu()], #types in waymo_types of CMTR/mtr/datasets/waymo/waymo_types.py
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
    waymo_dataset.collate_batch([ret])
    model.eval()
    with torch.no_grad(): 
        loss, tb_dict, disp_dict, batch_dict = model(batch_dict)

    num_envs, num_obj = env.get_controlled_agents_mask().shape
    return get_action_from_batch_dict(batch_dict, control_mask_idx, num_envs, num_obj)




def rollout(env, model, waymo_dataset, prior_frame, future_frame, wrld_idx=0, device="cuda:1", include_goals=False): 
    env.reset()
    trajectories = []
    localized_goals = []
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

    expert_actions, expert_speeds, expert_positions, expert_yaws = env.get_expert_actions()
    for time_step in range(env.episode_len): 
        collate_trajectories()
        if (time_step < prior_frame): 
            env.step_dynamics(expert_actions[:, :, time_step, :])
        else: 
            action = get_action(trajectories, local_polylines, localized_goals, env, waymo_dataset, prior_frame, future_frame, include_goals)
            env.step_dynamics(action)