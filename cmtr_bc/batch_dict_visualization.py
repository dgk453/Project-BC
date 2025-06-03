import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize, colorConverter
from matplotlib.lines import Line2D

from mtr.utils.common_utils import to_numpy


road_color = 'lightgrey'
ego_agent_line_color = 'black'
ego_agent_face_color = 'royalblue'
other_agent_line_color = 'black'
other_agent_face_color = 'white'
history_color = 'red'
gt_future_color = 'green'
pred_future_color = 'orange'
attn_color = 'red'

road_z_order = 1
obj_z_order = 2
traj_z_order = 3

road_attn_cmap = LinearSegmentedColormap.from_list("", [(0, road_color), (1, attn_color)])
obj_attn_cmap = LinearSegmentedColormap.from_list("", [(0, other_agent_face_color), (1, attn_color)])
transparent = colorConverter.to_rgba('white', alpha=0)
gmm_prob_cmap = LinearSegmentedColormap.from_list("", [(0, transparent), (1, pred_future_color)])


def select_or_aggregate_attn(attn, idx=None, weight=None):
    if idx is not None:
        return attn[idx]
    if weight is not None:
        return np.sum(attn * weight[:, None], axis=0)
    assert attn.ndim == 1
    return attn


def plot_scenario(
        input_dict,
        forward_ret_dict,
        num_samples=1,
        plot_object_history=True,
        plot_object_gt_future=True,
        plot_object_dense_pred_future=False,
        plot_ego_object_pred_future=False,
        plot_gmm_mode="gmm_prob",                       # "gmm_prob", "min_ade", "min_fde"
        encoder_obj_to_plot=None,
        decoder_layer_to_plot=-1,
        attn_to_obj_and_map=None,
        attn_to_obj=None,
        attn_to_map=None,
):
    """
    Args:
    input_dict:
        scenario_id (num_center_objects):
        track_index_to_predict (num_center_objects):

        obj_trajs (num_center_objects, num_objects, num_history_timestamps, num_attrs):
            num_attrs 29:
                0:3: [x, y, z]
                3:6: [length, width, height]
                6:11: object type one-hot, [car, pedestrian, cyclist, center_obj, autonomous_vehicle]
                11:23: timestamp one-hot (11) + timestamp (1)
                23:25: [heading_sin, heading_cos]
                25:29: [vel_x, vel_y, acc_x, acc_y]
        obj_trajs_mask (num_center_objects, num_objects, num_history_timestamps):
        map_polylines (num_center_objects, num_polylines, num_points_each_polyline, 9)
            9: [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_polylines, num_points_each_polyline)

        obj_trajs_pos: (num_center_objects, num_objects, num_history_timestamps, 3)
        obj_trajs_last_pos: (num_center_objects, num_objects, 3)
        obj_types: (num_objects)
        obj_ids: (num_objects)

        center_objects_world: (num_center_objects, 10)  [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        center_objects_type: (num_center_objects)
        center_objects_id: (num_center_objects)

        obj_trajs_future_state (num_center_objects, num_objects, num_future_timestamps, 4): [x, y, vx, vy]
        obj_trajs_future_mask (num_center_objects, num_objects, num_future_timestamps):
        center_gt_trajs (num_center_objects, num_future_timestamps, 4): [x, y, vx, vy]
        center_gt_trajs_mask (num_center_objects, num_future_timestamps):
        center_gt_final_valid_idx (num_center_objects): the final valid timestamp in num_future_timestamps
    forward_ret_dict:
        intention_points: (num_center_objects, num_intentions, 2): [x, y]
        pred_list: list of gmm tuple, each tuple contains prediction of each decoder layer as:
            gmm probability: (num_center_objects, num_gmm)
            gmm traj: (num_center_objects, num_gmm, num_future_timestamps, 7)
                7: [x, y, log_std_x, log_std_y, rho, vx, vy]
        pred_dense_trajs: (num_center_objects, num_objects, num_future_timestamps, 7)
            7: [x, y, log_std_x, log_std_y, rho, vx, vy]

        center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx: same as input_dict
        obj_trajs_future_state, obj_trajs_future_mask: same as input_dict
    encoder_obj_to_plot: int (default None), the index of the object to plot its sparse mask
        if None, plot ego-agent's sparse mask
    """
    assert (attn_to_obj_and_map is not None) + (attn_to_obj is not None or attn_to_map is not None) < 2

    plot_obj_attn = plot_map_attn = overwrite_ego_idx = aggregate_attn = False
    if attn_to_obj is not None:
        plot_obj_attn = plot_ego_object_pred_future = True
        aggregate_attn = True

    if attn_to_map is not None:
        plot_map_attn = plot_ego_object_pred_future = True
        aggregate_attn = True

    if attn_to_obj_and_map is not None:
        plot_obj_attn = plot_map_attn = True
        plot_ego_object_pred_future = False
        overwrite_ego_idx = True
        num_pad_obj = input_dict['obj_trajs'].shape[1]
        attn_to_obj = attn_to_obj_and_map[..., :num_pad_obj]
        attn_to_map = attn_to_obj_and_map[..., num_pad_obj:]

    num_samples_in_data = len(input_dict['scenario_id'])

    figs = []
    for i in range(min(num_samples, num_samples_in_data)):
        fig = plt.figure(figsize=(15, 15))
        ax = plt.gca()
        xy_lim_half = 80

        input_dict = to_numpy(input_dict)

        ego_agent_idx = input_dict['track_index_to_predict'][i]
        obj_history_mask = input_dict['obj_trajs_mask'][i].astype(bool)     # (num_objects, num_history_timestamps)

        # extract attention from encoder or decoder
        if overwrite_ego_idx and encoder_obj_to_plot is not None:
            ego_agent_idx = encoder_obj_to_plot
            num_valid_obj = obj_history_mask.any(axis=-1)                                   # (num_objects, )
            assert ego_agent_idx < num_valid_obj.sum()

        attn_idx = gmm_prob = None
        if aggregate_attn:
            if plot_gmm_mode == "gmm_prob":
                gmm_score = forward_ret_dict['pred_list'][decoder_layer_to_plot][0][i]      # (num_gmm, )
                gmm_prob = scipy.special.softmax(gmm_score)                                 # (num_gmm, )
        else:
            attn_idx = ego_agent_idx

        # ---------------------------------- plot map ---------------------------------- #
        map_polylines = input_dict['map_polylines'][i]
        map_polylines_mask = input_dict['map_polylines_mask'][i]

        # adjust color based on attn score
        if plot_map_attn:
            ego_attn_to_map = select_or_aggregate_attn(attn_to_map[i], idx=attn_idx, weight=gmm_prob)
            attn_to_map_norm = Normalize(vmin=0, vmax=ego_attn_to_map.max())
            map_colors = road_attn_cmap(attn_to_map_norm(ego_attn_to_map))
        else:
            map_colors = [road_color] * len(map_polylines)

        polyline_center = map_polylines[:, 0, :2]
        polyline_heading = map_polylines[:, 0, 5, None]
        polyline_length = map_polylines[:, 0, 3, None]
        polyline_width = map_polylines[:, 0, 4]
        polyline_offset_x = polyline_length * np.cos(polyline_heading)
        polyline_offset_y = polyline_length * np.sin(polyline_heading)
        polyline_offset = np.concat([polyline_offset_x, polyline_offset_y], axis=-1)
        polyline_p1 = polyline_center - polyline_offset
        polyline_p2 = polyline_center + polyline_offset
        c_scale = 0.7
        for p1, p2, width in zip(polyline_p1, polyline_p2, polyline_width): 
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=[c_scale]*3, linewidth=(width+0.05)*20)
        # ---------------------------------- plot object current location as box ---------------------------------- #
        obj_current_pos = input_dict['obj_trajs_last_pos'][i]
        obj_size = input_dict['obj_trajs'][i][:, -1, 3:5]
        obj_head_sin = input_dict['obj_trajs'][i][:, -1, -6]
        obj_head_cos = input_dict['obj_trajs'][i][:, -1, -5]

        obj_mask = obj_history_mask[:, -1]

        # compute box location and angle
        obj_heading = np.arctan2(obj_head_sin, obj_head_cos)

        half_length = obj_size[:, 0] / 2
        half_width = obj_size[:, 1] / 2
        rectangle_x = obj_current_pos[:, 0] - half_length * obj_head_cos + half_width * obj_head_sin
        rectangle_y = obj_current_pos[:, 1] - half_length * obj_head_sin - half_width * obj_head_cos

        # adjust color based on attn score
        if plot_obj_attn:
            ego_attn_to_obj = select_or_aggregate_attn(attn_to_obj[i], idx=attn_idx, weight=gmm_prob)
            vmax = ego_attn_to_obj.max()
            if vmax > 1:
                flat = np.copy(ego_attn_to_obj)
                flat.sort()
                vmax = flat[-2] if len(flat) > 1 else flat[-1]
            attn_to_obj_norm = Normalize(vmin=0, vmax=vmax)
            obj_colors = obj_attn_cmap(attn_to_obj_norm(ego_attn_to_obj))
        else:
            obj_colors = [other_agent_face_color] * len(obj_mask)
            obj_colors[ego_agent_idx] = ego_agent_face_color

        for j, (obj_x_j, obj_y_j, obj_mask_j, obj_size_j, obj_heading_j, obj_face_color_j) in \
                enumerate(zip(rectangle_x, rectangle_y, obj_mask, obj_size, obj_heading, obj_colors)):
            if not obj_mask_j:
                continue

            if j == ego_agent_idx:
                edge_color = ego_agent_line_color
                obj_face_color_j = ego_agent_face_color
                linewidth = 1
            else:
                edge_color = other_agent_line_color
                linewidth = 1
            ax.add_patch(Rectangle((obj_x_j, obj_y_j),
                                   obj_size_j[0], obj_size_j[1],
                                   angle=np.rad2deg(obj_heading_j),
                                   edgecolor=edge_color,
                                   linewidth=linewidth,
                                   facecolor=obj_face_color_j,
                                   zorder=obj_z_order))

        # plot object history
        if plot_object_history:
            obj_history_pos = input_dict['obj_trajs_pos'][i, :, :, :2]
            for j, (obj_history_pos_j, obj_history_mask_j) in enumerate(zip(obj_history_pos, obj_history_mask)):
                obj_history_pos_j = obj_history_pos_j[obj_history_mask_j]

                if j == ego_agent_idx:
                    linewidth = 2
                else:
                    linewidth = 1
                
                ax.plot(obj_history_pos_j[:, 0], obj_history_pos_j[:, 1],
                        color=history_color,
                        linestyle='--',
                        linewidth=linewidth,
                        zorder=traj_z_order,
                        label='agent history' if j == 0 else '')

        # plot object ground truth future
        obj_future_mask = input_dict['obj_trajs_future_mask'][i].astype(bool)
        if plot_object_gt_future:
            obj_future_pos = input_dict['obj_trajs_future_state'][i, :, :, :2]
            xy_lim_half = max(xy_lim_half, np.abs(obj_future_pos[ego_agent_idx, :, :2]).max())
            for j, (obj_future_pos_j, obj_future_mask_j) in enumerate(zip(obj_future_pos, obj_future_mask)):
                obj_future_pos_j = obj_future_pos_j[obj_future_mask_j]
                ax.plot(obj_future_pos_j[:, 0], obj_future_pos_j[:, 1],
                        color=gt_future_color,
                        linestyle='--',
                        zorder=traj_z_order,
                        label='agent ground truth' if j == 0 else '')

        # plot object dense prediction from encoder
        if plot_object_dense_pred_future and 'pred_dense_trajs' in forward_ret_dict:
            obj_future_pos = forward_ret_dict['pred_dense_trajs'][i, :, :, :2]
            xy_lim_half = max(xy_lim_half, np.abs(obj_future_pos[ego_agent_idx, :, :2]).max())
            for j, (obj_future_pos_j, obj_future_mask_j) in enumerate(zip(obj_future_pos, obj_future_mask)):
                if plot_ego_object_pred_future and j == ego_agent_idx:
                    continue
                obj_future_pos_j = obj_future_pos_j[obj_future_mask_j]
                ax.plot(obj_future_pos_j[:, 0], obj_future_pos_j[:, 1],
                        color=pred_future_color,
                        linestyle='--',
                        zorder=traj_z_order)

        # plot object gmm prediction from decoder
        if plot_ego_object_pred_future:
            gmm_score, gmm_traj = forward_ret_dict['pred_scores'], forward_ret_dict['pred_trajs']
            # (num_gmm, ), (num_gmm, num_future_timestamps, 2)
            gmm_score, gmm_traj = gmm_score[i], gmm_traj[i, ..., :2]
            ego_gt_future_traj = input_dict['center_gt_trajs'][i, ..., :2]          # (num_future_timestamps, 2)
            ego_future_mask = input_dict['center_gt_trajs_mask'][i].astype(bool)    # (num_future_timestamps, )

            if plot_gmm_mode == "gmm_prob":
                gmm_prob = scipy.special.softmax(gmm_score)                         # (num_gmm, )
                gmm_prob_norm = Normalize(vmin=0, vmax=gmm_prob.max())
                gmm_prob = gmm_prob_norm(gmm_prob)

                xy_lim_half = max(xy_lim_half, np.abs(gmm_traj).max())
                for gmm_prob_j, gmm_traj_j in zip(gmm_prob, gmm_traj):
                    gmm_traj_j = gmm_traj_j[ego_future_mask]
                    ax.plot(gmm_traj_j[:, 0], gmm_traj_j[:, 1],
                            color=pred_future_color,
                            linestyle='-',
                            alpha=gmm_prob_j,
                            zorder=traj_z_order)
            elif plot_gmm_mode == "min_ade":
                # (num_gmm, )
                ade = np.linalg.norm(gmm_traj - ego_gt_future_traj, axis=-1)[:, ego_future_mask].mean(axis=-1)
                min_ade_idx = np.argmin(ade)
                gmm_traj_j = gmm_traj[min_ade_idx, ego_future_mask]
                xy_lim_half = max(xy_lim_half, np.abs(gmm_traj_j).max())
                ax.plot(gmm_traj_j[:, 0], gmm_traj_j[:, 1],
                        color=pred_future_color,
                        linestyle='-',
                        zorder=traj_z_order,
                        label='prediction')
                ego_gt_final_valid_idx = int(input_dict['center_gt_final_valid_idx'][i])
                ax.scatter(ego_gt_future_traj[ego_gt_final_valid_idx, 0], ego_gt_future_traj[ego_gt_final_valid_idx, 1],
                           color=gt_future_color,
                           marker='*',
                           zorder=traj_z_order)
                ax.scatter(gmm_traj_j[-1, 0], gmm_traj_j[-1, 1],
                           color=pred_future_color,
                           marker='*',
                           zorder=traj_z_order)
            elif plot_gmm_mode == "min_fde":
                ego_gt_final_valid_idx = int(input_dict['center_gt_final_valid_idx'][i])
                # (num_gmm, )
                fde = np.linalg.norm((gmm_traj - ego_gt_future_traj)[:, ego_gt_final_valid_idx], axis=-1)
                min_fde_idx = np.argmin(fde)
                gmm_traj_j = gmm_traj[min_fde_idx, ego_future_mask]
                xy_lim_half = max(xy_lim_half, np.abs(gmm_traj_j).max())
                ax.plot(gmm_traj_j[:, 0], gmm_traj_j[:, 1],
                        color=pred_future_color,
                        linestyle='-',
                        linewidth=2,
                        zorder=traj_z_order)
                ax.scatter(ego_gt_future_traj[ego_gt_final_valid_idx, 0], ego_gt_future_traj[ego_gt_final_valid_idx, 1],
                           color=gt_future_color,
                           marker='*',
                           zorder=traj_z_order)
                ax.scatter(gmm_traj_j[-1, 0], gmm_traj_j[-1, 1],
                           color=pred_future_color,
                           marker='*',
                           zorder=traj_z_order)
            else:
                raise ValueError(f"plot_gmm_mode {plot_gmm_mode} is not supported")

        # give some margin
        # xy_lim_half = min(xy_lim_half * 1.1, 60)
        xy_lim_half = xy_lim_half * 1.1
        # ego agent is at (0, 0)
        plt.axis([-xy_lim_half, xy_lim_half, -xy_lim_half, xy_lim_half])
        ax.set_aspect('equal', adjustable='box')

        # if plot_map_attn:
        #     plt.colorbar(plt.cm.ScalarMappable(cmap=road_attn_cmap, norm=attn_to_map_norm),
        #                  label='attention to map', ax=ax)
        #
        # if plot_obj_attn:
        #     plt.colorbar(plt.cm.ScalarMappable(cmap=obj_attn_cmap, norm=attn_to_obj_norm),
        #                  label='attention to agents', ax=ax)
        #
        # if plot_ego_object_pred_future and plot_gmm_mode == "gmm_prob":
        #     plt.colorbar(plt.cm.ScalarMappable(cmap=gmm_prob_cmap, norm=gmm_prob_norm), label='gmm prob', ax=ax)

        # plt.title(f'min ADE: {ade.min():.2f}', fontsize=25)
        plt.legend(loc='lower right', fontsize=25)
        plt.tick_params(axis='both', which='both',
                        bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)

        plt.tight_layout()
        figs.append(fig)
    return figs if len(figs) > 1 else figs[0]


def plot_encoder_attention(
        input_dict,
        forward_ret_dict,
        attn_to_obj_and_map,
        encoder_obj_to_plot=None,
        num_samples=1,
):
    return plot_scenario(input_dict, forward_ret_dict, num_samples=num_samples,
                         attn_to_obj_and_map=attn_to_obj_and_map,
                         encoder_obj_to_plot=encoder_obj_to_plot,
                         plot_object_dense_pred_future=True,
                         plot_ego_object_pred_future=False)


def plot_decoder_attention(
        input_dict,
        forward_ret_dict,
        attn_to_obj=None,
        attn_to_map=None,
        decoder_layer_to_plot=-1,
        plot_gmm_mode="gmm_prob",                       # "gmm_prob", "min_ade", "min_fde"
        num_samples=1,
):
    assert (attn_to_obj is not None) or (attn_to_map is not None)
    return plot_scenario(input_dict, forward_ret_dict, num_samples=num_samples,
                         attn_to_obj=attn_to_obj, attn_to_map=attn_to_map,
                         decoder_layer_to_plot=decoder_layer_to_plot,
                         plot_object_dense_pred_future=False,
                         plot_ego_object_pred_future=True,
                         plot_gmm_mode=plot_gmm_mode)


def plot_attn(
        input_dict,
        forward_ret_dict,
        input_grad_dict,
        num_samples=1,
):
    input_dict, input_grad_dict = to_numpy([input_dict, input_grad_dict])

    fig_dict = {}
    fig_dict['decoder_grad_wrt_inputs'] = plot_decoder_attention(input_dict, forward_ret_dict,
                                                                 attn_to_obj=input_grad_dict['obj'],
                                                                 # attn_to_map=input_grad_dict['map'],
                                                                 decoder_layer_to_plot=-1,
                                                                 plot_gmm_mode="min_ade",
                                                                 num_samples=num_samples)
    return fig_dict
