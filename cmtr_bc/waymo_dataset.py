# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

from numba import njit
import numpy as np
import pickle
import torch

from mtr.datasets.dataset import DatasetTemplate
from mtr.utils import common_utils
from mtr.config import cfg


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, test_mode=None, logger=None, include_goals=False):
        super().__init__(dataset_cfg=dataset_cfg, training=training, logger=logger)
        self.training = training
        self.dataset_cfg = dataset_cfg
        if test_mode is None:
            self.test_mode = self.dataset_cfg.TEST_MODE
        else:
            # overwrite the test_mode in some cases
            self.test_mode = test_mode
        self.include_goals = include_goals
        self.CMTR = True

    @property
    def mode(self):
        if self.training:
            return 'train'
        else:
            return f'test_{self.test_mode}'

    def get_all_infos(self, info_path):
        self.logger.info(f'Start to load infos from {info_path}')
        with open(info_path, 'rb') as f:
            src_infos = pickle.load(f)

        src_infos = list(sorted(src_infos, key=lambda info: info['scenario_id']))
        infos = src_infos[:len(src_infos) // self.dataset_cfg.SAMPLE_INTERVAL['train' if self.training else 'test']]

        # print(type(infos))
        # # print(infos)
        
        # print(type(infos['objects_of_interest']))
        # print(type(infos['tracks_to_predict']))


        # with open('/scratch/cluster/zzwang_new/CMTR/tools/eval_rslts/knn_scenarios.p', 'rb') as f:
        #     scenario_ids = pickle.load(f)
        #     src_infos = {info['scenario_id']: info for info in src_infos}
        #     infos = [src_infos[scenario_id] for scenario_id in scenario_ids[:30]]

        self.logger.info(f'Total scenes before filters: {len(infos)}')

        for func_name, val in self.dataset_cfg.INFO_FILTER_DICT.items():
            infos = getattr(self, func_name)(infos, val)

        return infos

    def filter_info_by_object_type(self, infos, valid_object_types=None):
        ret_infos = []
        for cur_info in infos:
            num_interested_agents = cur_info['tracks_to_predict']['track_index'].__len__()
            if num_interested_agents == 0:
                continue

            valid_mask = []
            for idx, cur_track_index in enumerate(cur_info['tracks_to_predict']['track_index']):
                valid_mask.append(cur_info['tracks_to_predict']['object_type'][idx] in valid_object_types)

            valid_mask = np.array(valid_mask) > 0
            if valid_mask.sum() == 0:
                continue

            assert len(cur_info['tracks_to_predict'].keys()) == 3, f"{cur_info['tracks_to_predict'].keys()}"
            cur_info['tracks_to_predict']['track_index'] = list(np.array(cur_info['tracks_to_predict']['track_index'])[valid_mask])
            cur_info['tracks_to_predict']['object_type'] = list(np.array(cur_info['tracks_to_predict']['object_type'])[valid_mask])
            cur_info['tracks_to_predict']['difficulty'] = list(np.array(cur_info['tracks_to_predict']['difficulty'])[valid_mask])

            ret_infos.append(cur_info)
        self.logger.info(f'Total scenes after filter_info_by_object_type: {len(ret_infos)}')
        return ret_infos

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        ret_infos = self.create_scene_level_data(index)

        return ret_infos
    

    def get_observation(self, observation): 
        #TODO this should make a info dictionary
        #observation --> gpudrive's observation
        info = dict()
        index = 0
        return self.create_scene_level_data(index, info)


    def create_scene_level_data(self, index, info=None):
        """
        Args:
            index (index): corresponds to the time stamp in the scene? TODO
        Returns:

        """
        self.CMTR = not (info is None)
        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)

        track_infos = info['track_infos']

        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_ids = np.array(track_infos['object_id'])
        obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]
        
        scene_id = info['scenario_id']

        center_objects, track_index_to_predict = self.get_interested_agents(
            track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full,
            current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        ) # just slices the agents we want to predict

        if self.mode == "test_perturb":
            # perturb agents that are both non-causal and non-overlapping with ego
            assert center_objects.shape[0] == 1
            assert len(track_index_to_predict) == 1
            is_causal_agent = track_infos['is_causal_agent']            # (num_objects)

            # check if trajectories overlap
            obj_loc = obj_trajs_full[..., 0:2]
            obj_size = obj_trajs_full[..., 3:5]
            obj_heading = obj_trajs_full[..., 6]
            obj_mask = obj_trajs_full[..., -1]
            overlap_ego = check_overlap_with_causal_agents(is_causal_agent, obj_loc, obj_size, obj_heading, obj_mask)

            # add time shift to the non-causal agents
            perturb_mask = ~is_causal_agent & ~overlap_ego
            obj_trajs_to_perturb = obj_trajs_full[perturb_mask]
            num_timestamp = obj_trajs_to_perturb.shape[1]

            # time_shift range: [current_time_index, num_timestamp - current_time_index)
            # at least shift by current_time_index
            # at most shift to the last timestamp
            time_shift_min, time_shift_max = current_time_index, num_timestamp - current_time_index
            time_shift = index % (time_shift_max - time_shift_min) + time_shift_min #TODO index is referenced here, fix this - ask Zizhao

            obj_trajs_to_perturb[:, :num_timestamp - time_shift] = obj_trajs_to_perturb[:, time_shift:]
            # set all attributes (including mask) to 0
            obj_trajs_to_perturb[:, num_timestamp - time_shift:] = 0

            obj_trajs_full[perturb_mask] = obj_trajs_to_perturb

            obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
            obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs,
            center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids) = self.create_agent_data_for_center_objects(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past, obj_trajs_future=obj_trajs_future,
            track_index_to_predict=track_index_to_predict, sdc_track_index=sdc_track_index,
            timestamps=timestamps, obj_types=obj_types, obj_ids=obj_ids
        )

        # print(f"Object past traj shape: {obj_trajs_past.shape}, Obj trajs data: {obj_trajs_data.shape}")
        obj_trajs_future_lw = obj_trajs_future_state[..., 4:6]
        obj_trajs_future_heading = obj_trajs_future_state[..., 6]
        center_gt_trajs = center_gt_trajs[..., :4]                      # [x, y, vx, vy]
        obj_trajs_future_state = obj_trajs_future_state[..., :4]        # [x, y, vx, vy]

        ret_dict = {
            'index': index, #TODO index is involved in this as well 
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,  # used to select center-features
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,
            'obj_types': obj_types,
            'obj_ids': obj_ids,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if self.dataset_cfg.get("WITH_FJMP_LABELS", False):
            # first dimension is the index of the center object
            # since interaction labels do not depend on the center object, 
            # the labels computed from the first element apply to all center objects
            # fjmp_label: (num_objects, num_objects)
            fjmp_label = get_interaction_labels_sparse(obj_trajs_future_state[0, ..., :2],
                                                       obj_trajs_future_lw[0],
                                                       np.linalg.norm(obj_trajs_future_state[0, ..., 2:], axis=-1),
                                                       obj_trajs_future_heading[0],
                                                       obj_trajs_future_mask[0])
            # (num_center_objects, num_objects, num_objects)
            ret_dict['obj_fjmp_label'] = fjmp_label[None].repeat(len(track_index_to_predict_new), axis=0)
            # (num_center_objects, num_objects)
            ret_dict['center_object_fjmp_label'] = fjmp_label[track_index_to_predict_new]

        if not self.dataset_cfg.get('WITHOUT_HDMAP', False):
            ret_dict['map_polylines'] = info['map_infos']['localized_polylines']
            ret_dict['map_polylines_mask'] = info['map_infos']['localized_polylines_mask']
            ret_dict['map_polylines_center'] = info['map_infos']['localized_polylines_center']
        return ret_dict

    def create_agent_data_for_center_objects(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types, obj_ids
        ):
        obj_trajs_data, obj_trajs_mask, obj_trajs_future_state, obj_trajs_future_mask = self.generate_centered_trajs_for_agents(
            center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_types=obj_types, center_indices=track_index_to_predict,
            sdc_index=sdc_track_index, timestamps=timestamps, obj_trajs_future=obj_trajs_future
        )

        # print(f"obj_trajs_data: {obj_trajs_data.shape}, obj_trajs_past: {obj_trajs_past.shape}")

        # generate the labels of track_objects for training
        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps, 4)
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]  # (num_center_objects, num_future_timestamps)
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        # filter invalid past trajs
        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)  # (num_objects (original))

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps)
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps, C)
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]  # (num_center_objects, num_objects (filtered), num_timestamps_future, 4):  [x, y, vx, vy]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]  # (num_center_objects, num_objects, num_timestamps_future):
        obj_types = obj_types[valid_past_mask]
        obj_ids = obj_ids[valid_past_mask]

        valid_index_cnt = valid_past_mask.cumsum(axis=0)
        track_index_to_predict_new = valid_index_cnt[track_index_to_predict] - 1
        sdc_track_index_new = valid_index_cnt[sdc_track_index] - 1  # TODO: CHECK THIS

        assert obj_trajs_future_state.shape[1] == obj_trajs_data.shape[1]
        assert len(obj_types) == obj_trajs_future_mask.shape[1]
        assert len(obj_ids) == obj_trajs_future_mask.shape[1]

        # generate the final valid position of each object
        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0  # (num_center_objects, num_objects)
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0  # (num_center_objects)
            center_gt_final_valid_idx[cur_valid_mask] = k

        ret = [obj_trajs_data, obj_trajs_mask > 0, obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
            track_index_to_predict_new, sdc_track_index_new, obj_types, obj_ids]

        return ret

    def get_interested_agents(self, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
        center_objects_list = []
        track_index_to_predict_selected = []

        for k in range(len(track_index_to_predict)):
            obj_idx = track_index_to_predict[k]

            assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

            center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
            track_index_to_predict_selected.append(obj_idx)

        center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
        track_index_to_predict = np.array(track_index_to_predict_selected)
        return center_objects, track_index_to_predict

    @staticmethod
    def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index, rot_vel_index=None):
        """
        Args:
            obj_trajs (num_objects, num_timestamps, num_attrs):
                first three values of num_attrs are [x, y, z] or [x, y]
            center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
            center_heading (num_center_objects):
            heading_index: the index of heading angle in the num_attr-axis of obj_trajs
        """
        num_objects, num_timestamps, num_attrs = obj_trajs.shape
        num_center_objects = center_xyz.shape[0]
        assert center_xyz.shape[0] == center_heading.shape[0]
        assert center_xyz.shape[1] in [3, 2]

        obj_trajs = obj_trajs.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
        obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
        obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, 0:2].view(num_center_objects, -1, 2),
            angle=-center_heading
        ).view(num_center_objects, num_objects, num_timestamps, 2)

        obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

        # rotate direction of velocity
        if rot_vel_index is not None:
            assert len(rot_vel_index) == 2
            obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
                points=obj_trajs[:, :, :, rot_vel_index].view(num_center_objects, -1, 2),
                angle=-center_heading
            ).view(num_center_objects, num_objects, num_timestamps, 2)

        return obj_trajs

    def generate_centered_trajs_for_agents(self, center_objects, obj_trajs_past, obj_types, center_indices, sdc_index, timestamps, obj_trajs_future):
        """[summary]

        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_trajs_past (num_objects, num_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            obj_types (num_objects):
            center_indices (num_center_objects): the index of center objects in obj_trajs_past
            centered_valid_time_indices (num_center_objects), the last valid time index of center objects
            timestamps ([type]): [description]
            obj_trajs_future (num_objects, num_future_timestamps, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        Returns:
            ret_obj_trajs (num_center_objects, num_objects, num_timestamps, num_attrs):
            ret_obj_valid_mask (num_center_objects, num_objects, num_timestamps):
            ret_obj_trajs_future (num_center_objects, num_objects, num_timestamps_future, 4):  [x, y, vx, vy]
            ret_obj_valid_mask_future (num_center_objects, num_objects, num_timestamps_future):
        """
        if (self.include_goals): 
            assert obj_trajs_past.shape[-1] == 12
            assert center_objects.shape[-1] == 12
        else: 
            assert obj_trajs_past.shape[-1] == 10
            assert center_objects.shape[-1] == 10
        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        # transform to cpu torch tensor
        center_objects = torch.from_numpy(center_objects).float()
        obj_trajs_past = torch.from_numpy(obj_trajs_past).float()
        timestamps = torch.from_numpy(timestamps)

        # transform coordinates to the centered objects
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        # print(f"obj trajs shape after transform: {obj_trajs.shape}")

        ## generate the attributes for each object
        object_onehot_mask = torch.zeros((num_center_objects, num_objects, num_timestamps, 5))

        object_onehot_mask[:, obj_types == 'TYPE_VEHICLE', :, 0] = 1
        object_onehot_mask[:, obj_types == 'TYPE_PEDESTRAIN', :, 1] = 1  # TODO: CHECK THIS TYPO
        object_onehot_mask[:, obj_types == 'TYPE_CYCLIST', :, 2] = 1

        object_onehot_mask[torch.arange(num_center_objects), center_indices, :, 3] = 1
        object_onehot_mask[:, sdc_index, :, 4] = 1

        object_time_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        object_time_embedding[:, :, torch.arange(num_timestamps), torch.arange(num_timestamps)] = 1
        object_time_embedding[:, :, torch.arange(num_timestamps), -1] = timestamps

        object_heading_embedding = torch.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]  # (num_centered_objects, num_objects, num_timestamps, 2)
        vel_pre = torch.roll(vel, shifts=1, dims=2)
        acce = (vel - vel_pre) / 0.1  # (num_centered_objects, num_objects, num_timestamps, 2)
        acce[:, :, 0, :] = acce[:, :, 1, :]

        ret_obj_trajs = torch.cat((
            obj_trajs[:, :, :, 0:6], 
            object_onehot_mask,
            object_time_embedding, 
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9], 
            acce,
        ), dim=-1)

        # print(f"ret obj trajs: {ret_obj_trajs.shape}, object_time_embedding: {object_time_embedding.shape}, object_heading_embedding: {object_heading_embedding.shape}")

        ret_obj_valid_mask = obj_trajs[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs[ret_obj_valid_mask == 0] = 0

        ##  generate label for future trajectories
        obj_trajs_future = torch.from_numpy(obj_trajs_future).float()
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )
        # (x, y, vx, vy, length, width, heading)
        ret_obj_trajs_future = obj_trajs_future[:, :, :, [0, 1, 7, 8, 3, 4, 6]]
        ret_obj_valid_mask_future = obj_trajs_future[:, :, :, -1]  # (num_center_obejcts, num_objects, num_timestamps_future)  # TODO: CHECK THIS, 20220322
        ret_obj_trajs_future[ret_obj_valid_mask_future == 0] = 0

        return ret_obj_trajs.numpy(), ret_obj_valid_mask.numpy(), ret_obj_trajs_future.numpy(), ret_obj_valid_mask_future.numpy()
    @staticmethod
    def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):
        """
        Args:
            polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

        Returns:
            ret_polylines: (num_polylines, num_points_each_polyline, 7)
            ret_polylines_mask: (num_polylines, num_points_each_polyline)
        """
        point_dim = polylines.shape[-1]

        sampled_points = polylines[::point_sampled_interval]
        sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
        buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
        buffer_points[0, 2:4] = buffer_points[0, 0:2]

        break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
        polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
        ret_polylines = []
        ret_polylines_mask = []

        def append_single_polyline(new_polyline):
            cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
            cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
            cur_polyline[:len(new_polyline)] = new_polyline
            cur_valid_mask[:len(new_polyline)] = 1
            ret_polylines.append(cur_polyline)
            ret_polylines_mask.append(cur_valid_mask)

        for k in range(len(polyline_list)):
            if polyline_list[k].__len__() <= 0:
                continue
            for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
                append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

        ret_polylines = np.stack(ret_polylines, axis=0)
        ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

        ret_polylines = torch.from_numpy(ret_polylines)
        ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

        # # CHECK the results
        # polyline_center = ret_polylines[:, :, 0:2].sum(dim=1) / ret_polyline_valid_mask.sum(dim=1).float()[:, None]  # (num_polylines, 2)
        # center_dist = (polyline_center - ret_polylines[:, 0, 0:2]).norm(dim=-1)
        # assert center_dist.max() < 10
        return ret_polylines, ret_polylines_mask

    def create_map_data_for_center_objects(self, center_objects, map_infos, center_offset):
        """
        Args:
            center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            map_infos (dict):
                all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
            center_offset (2):, [offset_x, offset_y]
        Returns:
            map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
            map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
        """
        num_center_objects = center_objects.shape[0]

        # transform object coordinates by center objects
        def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
            neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
            neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 0:2].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)
            neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
                points=neighboring_polylines[:, :, :, 3:5].view(num_center_objects, -1, 2),
                angle=-center_objects[:, 6]
            ).view(num_center_objects, -1, batch_polylines.shape[1], 2)

            # use pre points to map
            # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
            xy_pos_pre = neighboring_polylines[:, :, :, 0:2]
            xy_pos_pre = torch.roll(xy_pos_pre, shifts=1, dims=-2)
            xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
            neighboring_polylines = torch.cat((neighboring_polylines, xy_pos_pre), dim=-1)

            neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
            return neighboring_polylines, neighboring_polyline_valid_mask

        polylines = torch.from_numpy(map_infos['all_polylines'].copy())
        center_objects = torch.from_numpy(center_objects)

        batch_polylines, batch_polylines_mask = self.generate_batch_polylines_from_map(
            polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
            vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
            num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
        )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

        # collect a number of closest polylines for each center objects
        num_of_src_polylines = self.dataset_cfg.NUM_OF_SRC_POLYLINES

        if len(batch_polylines) > num_of_src_polylines:
            polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
            center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
            center_offset_rot = common_utils.rotate_points_along_z(
                points=center_offset_rot.view(num_center_objects, 1, 2),
                angle=center_objects[:, 6]
            ).view(num_center_objects, 2)

            pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

            dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
            topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
            map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
            map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)
        else:
            map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 1, 1, 1)
            map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 1, 1)

        map_polylines, map_polylines_mask = transform_to_center_coordinates(
            neighboring_polylines=map_polylines,
            neighboring_polyline_valid_mask=map_polylines_mask
        )

        temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].float()).sum(dim=-2)  # (num_center_objects, num_polylines, 3)
        map_polylines_center = temp_sum / torch.clamp_min(map_polylines_mask.sum(dim=-1).float()[:, :, None], min=1.0)  # (num_center_objects, num_polylines, 3)

        map_polylines = map_polylines.numpy()
        map_polylines_mask = map_polylines_mask.numpy()
        map_polylines_center = map_polylines_center.numpy() # This is the center of each batch of batched polylines

        return map_polylines, map_polylines_mask, map_polylines_center

    def generate_prediction_dicts(self, batch_dict):
        """

        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        pred_dict_list = []
        batch_sample_count = batch_dict['batch_sample_count']
        start_obj_idx = 0
        for bs_idx in range(batch_dict['batch_size']):
            cur_scene_pred_list = []
            for obj_idx in range(start_obj_idx, start_obj_idx + batch_sample_count[bs_idx]):
                single_pred_dict = {
                    'scenario_id': input_dict['scenario_id'][obj_idx],
                    'pred_trajs': common_utils.to_numpy(pred_trajs_world[obj_idx, :, :, 0:2]),
                    'pred_scores': common_utils.to_numpy(pred_scores[obj_idx, :]),
                    'object_id': input_dict['center_objects_id'][obj_idx],
                    'object_type': input_dict['center_objects_type'][obj_idx],
                    'gt_trajs': common_utils.to_numpy(input_dict['center_gt_trajs_src'][obj_idx]),
                    'track_index_to_predict': common_utils.to_numpy(input_dict['track_index_to_predict'][obj_idx])
                }
                cur_scene_pred_list.append(single_pred_dict)

            pred_dict_list.append(cur_scene_pred_list)
            start_obj_idx += batch_sample_count[bs_idx]

        assert start_obj_idx == num_center_objects
        assert len(pred_dict_list) == batch_dict['batch_size']

        return pred_dict_list

    def evaluation(self, pred_dicts, eval_method='waymo', **kwargs):
        if eval_method == 'waymo':
            from .waymo_eval import waymo_evaluation
            try:
                num_modes_for_eval = pred_dicts[0][0]['pred_trajs'].shape[0]
            except:
                num_modes_for_eval = 6
            metric_results, result_format_str = waymo_evaluation(pred_dicts=pred_dicts, num_modes_for_eval=num_modes_for_eval)

            metric_result_str = '\n'
            for key in metric_results:
                metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
            metric_result_str += '\n'
            metric_result_str += result_format_str
        else:
            raise NotImplementedError

        return metric_result_str, metric_results


@njit
def rotate_points(points, angle):
    """Rotate a set of points by a given angle (in radians)."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rotation_matrix.T)


@njit
def check_overlap(box1_center, box1_size, box1_angle, box2_center, box2_size, box2_angle):
    """
    Check if two oriented bounding boxes overlap using Separating Axis Theorem (SAT).

    Parameters:
    - box1_center, box2_center: np.ndarray (x, y) representing the center of the bounding box
    - box1_size, box2_size: np.ndarray (width, height) of the bounding box
    - box1_angle, box2_angle: float representing the angle of rotation (in radians) of the bounding box

    Returns:
    - True if the two bounding boxes overlap, False otherwise
    """
    # using L1 distance as a quick check
    if np.abs(box1_center - box2_center).sum() > (box1_size + box2_size).sum():
        return False

    # Calculate the half extents of the boxes
    half_extents1 = box1_size / 2.0
    half_extents2 = box2_size / 2.0

    # Calculate the vertices of the boxes
    vertices1 = np.array([
        [-half_extents1[0], -half_extents1[1]],
        [half_extents1[0], -half_extents1[1]],
        [half_extents1[0], half_extents1[1]],
        [-half_extents1[0], half_extents1[1]]
    ]).astype(np.float32)

    vertices2 = np.array([
        [-half_extents2[0], -half_extents2[1]],
        [half_extents2[0], -half_extents2[1]],
        [half_extents2[0], half_extents2[1]],
        [-half_extents2[0], half_extents2[1]]
    ]).astype(np.float32)

    # Rotate the vertices of the boxes
    rotated_vertices1 = rotate_points(vertices1, box1_angle) + box1_center
    rotated_vertices2 = rotate_points(vertices2, box2_angle) + box2_center

    # Check for overlap on each axis
    axes = [
        rotate_points(np.array([[1, 0]], dtype=np.float32), box1_angle)[0],
        rotate_points(np.array([[0, 1]], dtype=np.float32), box1_angle)[0],
        rotate_points(np.array([[1, 0]], dtype=np.float32), box2_angle)[0],
        rotate_points(np.array([[0, 1]], dtype=np.float32), box2_angle)[0]
    ]

    for axis in axes:
        # Project the vertices onto the axis
        projections1 = np.dot(rotated_vertices1, axis)
        projections2 = np.dot(rotated_vertices2, axis)

        # Check for overlap
        if (np.max(projections1) < np.min(projections2)) or (np.min(projections1) > np.max(projections2)):
            return False

    return True


@njit
def get_interaction_labels_sparse(obj_future_location, obj_future_size, obj_future_velocity, obj_future_heading,
                                  obj_future_mask):
    """
    Args:
        obj_future_location: (num_objects, num_timestamps, 2): [x, y]
        obj_future_size: (num_objects, num_timestamps, 2): [length, width]
        obj_future_velocity: (num_objects, num_timestamps): velocity magnitude
        obj_future_heading: (num_objects, num_timestamps): heading angle in radians, along the length
        obj_future_mask: (num_objects, num_timestamps): binary mask of future state validity

    Returns:

    """
    future_loc = obj_future_location
    future_vel = obj_future_velocity
    future_size = obj_future_size
    future_heading = obj_future_heading

    N, T, _ = future_loc.shape
    labels = np.eye(N, dtype=np.bool_)

    for a in range(1, N):
        for b in range(a):
            # for each (unordered) pairs of vehicles, we check if they are interacting
            # by checking if there is a collision at any pair of future timestamps
            is_coll_mask = np.zeros((T, T), dtype=np.bool_)

            for t1, (loc1, size1, heading1, mask1) in \
                    enumerate(zip(future_loc[a], future_size[a], future_heading[a], obj_future_mask[a])):
                if not mask1:
                    continue

                for t2, (loc2, size2, heading2, mask2) in \
                        enumerate(zip(future_loc[b], future_size[b], future_heading[b], obj_future_mask[b])):
                    if not mask2:
                        continue

                    # only consider the colliding pairs that are within 3 seconds (= 30 timesteps) of each other
                    if abs(t1 - t2) > 30:
                        continue

                    # check if the two vehicles are colliding at this pair of timestamps
                    is_coll_mask[t1, t2] = check_overlap(loc1, size1, heading1, loc2, size2, heading2)

            if not is_coll_mask.any():
                continue

            # [P, 2], first index is a, second is b; P is number of colliding pairs
            coll_ids = np.argwhere(is_coll_mask)

            a_timestep = coll_ids[:, 0]
            b_timestep = coll_ids[:, 1]
            coll_ids_smaller_timestep = np.where(a_timestep <= b_timestep, a_timestep, b_timestep)
            coll_ids_larger_timestep = np.where(a_timestep > b_timestep, a_timestep, b_timestep)

            conflict_time_influencer = coll_ids.min()
            influencer_mask = coll_ids_smaller_timestep == conflict_time_influencer
            candidate_reactors = coll_ids_larger_timestep[influencer_mask]
            conflict_time_reactor = candidate_reactors.min()
            conflict_time_reactor_id = np.argmin(candidate_reactors)

            smallest_conflict_pair = coll_ids[influencer_mask][conflict_time_reactor_id]
            a_is_influencer = smallest_conflict_pair[0] <= smallest_conflict_pair[1]
            if a_is_influencer:
                min_a = conflict_time_influencer
                min_b = conflict_time_reactor
            else:
                min_a = conflict_time_reactor
                min_b = conflict_time_influencer

            # a is the influencer
            if min_a < min_b:
                labels[b, a] = True
            # b is the influencer
            elif min_b < min_a:
                labels[a, b] = True
            else:
                # if both reach the conflict point at the same timestep,
                # the influencer is the vehicle with the higher velocity @ the conflict point.
                if future_vel[a][min_a] > future_vel[b][min_b]:
                    labels[b, a] = True
                elif future_vel[a][min_a] < future_vel[b][min_b]:
                    labels[a, b] = True

    return labels


@njit
def check_overlap_with_causal_agents(causal_agents_mask, obj_loc, obj_size, obj_heading, obj_mask):
    """
    Args:
        causal_agents_mask: (num_objects, ): binary mask of causal agents
        obj_loc: (num_objects, num_timestamps, 2): [x, y]
        obj_size: (num_objects, num_timestamps, 2): [length, width]
        obj_heading: (num_objects, num_timestamps): heading angle in radians, along the length
        obj_mask: (num_objects, num_timestamps): binary mask of future state validity

    Returns:

    """
    causal_loc = obj_loc[causal_agents_mask]
    causal_size = obj_size[causal_agents_mask]
    causal_heading = obj_heading[causal_agents_mask]
    causal_mask = obj_mask[causal_agents_mask]

    num_objects = obj_loc.shape[0]
    num_causal_agents = causal_loc.shape[0]
    labels = np.zeros(num_objects, dtype=np.bool_)

    for a in range(num_objects):
        for b in range(num_causal_agents):
            for loc1, size1, heading1, mask1 in zip(obj_loc[a], obj_size[a], obj_heading[a], obj_mask[a]):
                if not mask1:
                    continue

                for loc2, size2, heading2, mask2 in zip(causal_loc[b], causal_size[b], causal_heading[b], causal_mask[b]):
                    if not mask2:
                        continue

                    # check if the two vehicles are colliding at this pair of timestamps
                    if check_overlap(loc1, size1, heading1, loc2, size2, heading2):
                        labels[a] = True
                        break

                if labels[a]:
                    break

            if labels[a]:
                break

    return labels

