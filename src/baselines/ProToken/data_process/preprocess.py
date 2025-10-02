# ==============================================================================
# Copyright 2024 Changping Laboratory & Peking University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ==============================================================================

import jax
import numpy as np
import tensorflow as tf
import os

from .residue_constants import restype_3to1, restype_order, restype_num, atom_type_num, atom_types, atom_order, \
                                     order_restype_with_x_and_gap, chi_angles_mask, chi_angles_atoms, restypes, restype_1to3
from .geometry import rigids_from_3_points, vecs_from_tensor, rots_from_tensor_np, rigids_mul_rots
from data_process.protein import Protein, from_pdb_string, to_pdb

protoken_dtype_dic = {
    "aatype": np.int32,
    "seq_mask": np.int32,
    "residue_index": np.int32,
    "ca_pos": np.float32,
    "backbone_atom_masks": np.int32,
    "backbone_atom_positions": np.float32,
    "backbone_affine_tensor": np.float32,
    "torsion_angles_sin_cos": np.float32,
    "torsion_angles_mask": np.int32,
    }

protoken_overall_features = ["aatype", "seq_mask", "residue_index",
                             "ca_pos", "backbone_atom_masks", "backbone_atom_positions",
                             "backbone_affine_tensor", "torsion_angles_sin_cos", "torsion_angles_mask",]

protoken_encoder_input_features = ["seq_mask", "residue_index", "backbone_atom_masks", "backbone_atom_positions",
                                   "ca_pos", "backbone_affine_tensor", "torsion_angles_sin_cos", "torsion_angles_mask"]

protoken_decoder_input_features = ["protoken_index", "protoken_mask", "residue_index"]

def encoder_warmup_input_generator(seq_len):
    if seq_len <= 512:
        NUM_RES = 512
    elif seq_len <= 1024:
        NUM_RES = 1024
    elif seq_len <= 2048:
        NUM_RES = 2048
    else:
        raise ValueError("Sequence length is too long.")
    warmup_inputs = {}
    warmup_inputs['seq_mask'] = np.ones(NUM_RES, dtype=np.int32)
    warmup_inputs['residue_index'] = np.arange(NUM_RES, dtype=np.int32)
    warmup_inputs['backbone_atom_masks'] = np.ones((NUM_RES, 37), dtype=np.int32)
    warmup_inputs['backbone_atom_positions'] = np.zeros((NUM_RES, 37, 3), dtype=np.float32)
    warmup_inputs['ca_pos'] = np.zeros((NUM_RES, 3), dtype=np.float32)
    warmup_inputs['backbone_affine_tensor'] = np.zeros((NUM_RES, 7), dtype=np.float32)
    warmup_inputs['torsion_angles_sin_cos'] = np.zeros((NUM_RES, 6), dtype=np.float32)
    warmup_inputs['torsion_angles_mask'] = np.ones((NUM_RES, 3), dtype=np.int32)
    return [warmup_inputs[n] for n in protoken_encoder_input_features]

def decoder_warmup_input_generator(seq_len):
    if seq_len <= 512:
        NUM_RES = 512
    elif seq_len <= 1024:
        NUM_RES = 1024
    elif seq_len <= 2048:
        NUM_RES = 2048
    else:
        raise ValueError("Sequence length is too long.")
    warmup_inputs = {}
    warmup_inputs['protoken_index'] = np.ones((NUM_RES,), dtype=np.int32)
    warmup_inputs['protoken_mask'] = np.arange(NUM_RES, dtype=np.int32)
    warmup_inputs['residue_index'] = np.ones((NUM_RES, ), dtype=np.int32)

    return [warmup_inputs[n] for n in protoken_decoder_input_features]

def init_protoken_model(seq_len_total, base_path='./'):
    if tf.config.list_physical_devices('GPU'):
        print(f"Found GPU, will use GPU for prediction")
    else:
        print(f"Did not find GPU, will use CPU for prediction")

    if seq_len_total <= 512:
        NUM_RES = 512
    elif seq_len_total <= 1024:
        NUM_RES = 1024
    elif seq_len_total <= 2048:
        NUM_RES = 2048
    else:
        raise ValueError('The sequence length is too long')
    
    model = tf.saved_model.load(f"{base_path}/./models/{NUM_RES}")
    warmup_inputs_encoder = encoder_warmup_input_generator(seq_len_total)
    warmup_inputs_decoder = decoder_warmup_input_generator(seq_len_total)
    warmup_results_encoder = model.encoder(*warmup_inputs_encoder)
    warmup_results_decoder = model.decoder(*warmup_inputs_decoder)

    return model

def protoken_decoder_preprocess(protoken_index, task_mode='single'):
    # protoken_index: (N,) for single mode, [(N1,), (N2,)] for multi mode
    # task_mode: 'single' or 'multi'
    # return: decoder_inputs: [protoken_index, protoken_mask, residue_index]

    if task_mode == 'single':
        if not isinstance(protoken_index, np.ndarray):
            raise ValueError("protoken_index should be a numpy array")
        protoken_index = np.asarray(protoken_index)
        seq_len = protoken_index.shape[0]
        if seq_len <= 512:
            NUM_RES = 512
        elif seq_len <= 1024:
            NUM_RES = 1024
        elif seq_len <= 2048:
            NUM_RES = 2048
        else:
            raise ValueError("ProToken length is too long.")
        protoken_index = np.pad(protoken_index, (0, NUM_RES - seq_len), 'constant', constant_values=0)
        protoken_mask = np.ones(seq_len, dtype=np.int32)
        protoken_mask = np.pad(protoken_mask, (0, NUM_RES - seq_len), 'constant', constant_values=0)
        residue_index = np.arange(seq_len, dtype=np.int32)
        residue_index = np.pad(residue_index, (0, NUM_RES - seq_len), 'constant', constant_values=0)
    elif task_mode == 'multi':
        if not isinstance(protoken_index, list):
            raise ValueError("protoken_index should be a list of numpy arrays")
        protoken_index_list = [np.asarray(i) for i in protoken_index]
        seq_len_list = [len(i) for i in protoken_index_list]
        if sum(seq_len_list) <= 512:
            NUM_RES = 512
        elif sum(seq_len_list) <= 1024:
            NUM_RES = 1024
        elif sum(seq_len_list) <= 2048:
            NUM_RES = 2048
        else:
            raise ValueError("ProToken length is too long.")
        protoken_index = np.concatenate(protoken_index_list, axis=0)
        protoken_index = np.pad(protoken_index, (0, NUM_RES - len(protoken_index)), 'constant', constant_values=0)
        protoken_mask = np.ones(sum(seq_len_list), dtype=np.int32)
        protoken_mask = np.pad(protoken_mask, (0, NUM_RES - len(protoken_mask)), 'constant', constant_values=0)
        residue_index_list = []
        for i in range(len(protoken_index_list)):
            if i == 0:
                residue_index_list.append(np.arange(seq_len_list[i], dtype=np.int32))
            else:
                residue_index_list.append(np.arange(seq_len_list[i], dtype=np.int32) + residue_index_list[i-1][-1] + 32)
        residue_index = np.concatenate(residue_index_list, axis=0)
        residue_index = np.pad(residue_index, (0, NUM_RES - len(residue_index)), 'constant', constant_values=0)
    else:
        raise ValueError("task_mode should be either 'single' or 'multi'")
    decoder_inputs = [protoken_index, protoken_mask, residue_index]
    return decoder_inputs


def protoken_encoder_preprocess(pdb_or_mmcif_input_dir, task_mode='single', chain_id=None):
    assert task_mode == "single", "only single needed in StrucTokenBenchmark"
    # incase input a protein file
    if not os.path.isdir(pdb_or_mmcif_input_dir):
        pdbs = [os.path.basename(pdb_or_mmcif_input_dir)]
        pdb_or_mmcif_input_dir = os.path.dirname(pdb_or_mmcif_input_dir)
    else: 
        # to avoid sweeping all files if assigned a single file
        pdbs = [i for i in os.listdir(pdb_or_mmcif_input_dir) if i.endswith('.pdb') or i.endswith('.cif')]
    #print(f"Found {len(pdbs)} pdb files in the input directory.\nWARNING: The input pdb files should be single-chain pdb files.")
    if task_mode == 'single':
        assert len(pdbs) == 1, "Please provide only one pdb file in the input directory with task_mode='single"
        pdb_or_mmcif_input_path = os.path.join(pdb_or_mmcif_input_dir, pdbs[0])
        aux_inputs, seq_len_total = protoken_basic_generator(pdb_or_mmcif_input_path, chain_id)
    elif task_mode == 'multi':
        assert 0, "not modified accordingly"
        assert len(pdbs) > 1, "Please provide more than one single-chain pdb files in the input directory with task_mode='multi"
        pdb_path_list = [os.path.join(pdb_or_mmcif_input_dir, i) for i in pdbs]
        NGAP = 32
        chain_feature_list = []
        chain_length_list = []
        for pdb_path in pdb_path_list:
            feature, seq_len = protoken_basic_generator(pdb_path)
            chain_feature_list.append(feature)
            chain_length_list.append(seq_len)
        TOTAL_NRES = 0
        seq_len_total = sum(chain_length_list)
        if seq_len_total <= 512:
            TOTAL_NRES = 512
        elif seq_len_total <= 1024:
            TOTAL_NRES = 1024
        elif seq_len_total <= 2048:
            TOTAL_NRES = 2048
        else:
            raise ValueError('The total sequence length is too long')
        for i_feat in range(len(chain_feature_list)):
            seq_len_tmp = chain_length_list[i_feat]
            for f_k in protoken_overall_features:
                chain_feature_list[i_feat][f_k] =  chain_feature_list[i_feat][f_k][:seq_len_tmp, ...]
        aux_inputs = {}
        for f_k in protoken_overall_features:
            if not f_k == 'residue_index':
                aux_inputs[f_k] = np.concatenate([chain_feature_list[i][f_k] for i in range(len(chain_feature_list))], axis=0)
            else:
                res_idx_curr = -NGAP
                res_idx_list = []
                for i in range(len(chain_feature_list)):
                    tmp_res_idx = chain_feature_list[i][f_k]
                    tmp_res_idx = tmp_res_idx + np.ones_like(tmp_res_idx) * (res_idx_curr - np.min(tmp_res_idx) + NGAP)
                    res_idx_list.append(tmp_res_idx)
                    res_idx_curr = np.max(tmp_res_idx) + 1
                aux_inputs[f_k] = np.concatenate(res_idx_list, axis=0)
        aux_inputs = _pad(aux_inputs, NUM_RES=TOTAL_NRES)
        chain_length_info = {}
        for k in range(len(pdbs)):
            chain_length_info[k] = {"pdb_name": pdbs[k],
                                    "seq_len": chain_length_list[k],
                                    "start_idx": int(np.sum(chain_length_list[:k]))}
        aux_inputs['chain_length_info'] = chain_length_info
    else:
        raise ValueError("task_mode should be either 'single' or 'multi'")

    inputs = [aux_inputs[name] for name in protoken_encoder_input_features]
    return inputs, aux_inputs, seq_len_total

def protoken_basic_generator(pdb_or_mmcif_path, chain_id=None):
    
    prot_pdb = from_pdb_string(pdb_or_mmcif_path, chain_id=chain_id)

    aatype = prot_pdb.aatype
    residue_index = prot_pdb.residue_index
    seq_len = len(aatype)
    if seq_len <= 512:
        NUM_RES = 512
    elif seq_len <= 1024:
        NUM_RES = 1024
    elif seq_len <= 2048:
        NUM_RES = 2048
    else:
        raise ValueError("Sequence length is too long.")

    ### create input features:
    # seq_mask & aatype & residue_index
    input_features_nopad = {}
    input_features_nopad['aatype'] = aatype
    input_features_nopad['seq_mask']  = np.ones(aatype.shape, dtype=np.float32)
    input_features_nopad['residue_index'] = residue_index
    # backbone_affine_tensor
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)
    backbone_affine_tensor = atom37_to_backbone_affine_tensor_np(aatype, atom37_positions, atom37_mask)
    input_features_nopad['backbone_affine_tensor'] = backbone_affine_tensor
    # all_atom_masks & all_atom_positions & ca_pos
    GLY_MASK_ATOM37 = np.array([1,1,1,0,1]+[0]*32).astype(np.float32)
    GLY_MASK_ATOM14 = np.array([1,1,1,1,]+[0]*10).astype(np.float32)
    masking_aatype = np.ones(aatype.shape, dtype=np.int64)*7
    backbone_atom37_positions = atom37_positions * GLY_MASK_ATOM37.reshape(1,-1,1)
    backbone_atom37_mask = atom37_mask * GLY_MASK_ATOM37.reshape(1,-1)
    ca_pos, ca_pos_mask = pseudo_beta_fn_np(masking_aatype, backbone_atom37_positions, backbone_atom37_mask)
    input_features_nopad['backbone_atom_positions'] = backbone_atom37_positions
    input_features_nopad['backbone_atom_masks'] = backbone_atom37_mask
    input_features_nopad['ca_pos'] = ca_pos  # all set to CA in current situation
    # torsion_angles_sin_cos, torsion_angles_mask
    angle_sin_cos, anlge_mask = get_ppo_angles_sin_cos(backbone_atom37_positions)
    input_features_nopad['torsion_angles_sin_cos'] = angle_sin_cos
    input_features_nopad['torsion_angles_mask'] = anlge_mask
    
    # input_features_pad = {}
    # for k, v in input_features_nopad.items():
    #     pad_shape = list(v.shape)
    #     pad_shape[0] = NUM_RES - pad_shape[0]
    #     pad_value = np.zeros(pad_shape, dtype=v.dtype)
    #     if k == 'backbone_affine_tensor':
    #         pad_value[...,0] = 1.0
    #     input_features_pad[k] = np.concatenate([v, pad_value], axis=0).astype(protoken_dtype_dic[k])[None,...]
    input_features_pad = _pad(input_features_nopad, NUM_RES)

    input_feature_dict = {x: input_features_pad[x] for x in protoken_overall_features}
    return input_feature_dict, seq_len

def _pad(feat_dict, NUM_RES):
    for k, v in feat_dict.items():
        pad_shape = list(v.shape)
        pad_shape[0] = NUM_RES - pad_shape[0]
        pad_value = np.zeros(pad_shape, dtype=v.dtype)
        if k == 'backbone_affine_tensor':
            pad_value[...,0] = 1.0
        feat_dict[k] = np.concatenate([v, pad_value], axis=0).astype(protoken_dtype_dic[k])
    return feat_dict

# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2024 Changping Laboratory & Peking University.
def atom37_to_backbone_affine_tensor_np(
        aatype,
        all_atom_positions,
        all_atom_mask,
):
    r"""
    """
    flat_aatype = np.reshape(aatype, [-1])
    all_atom_positions = np.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = np.reshape(all_atom_mask, [-1, 37])

    rigid_group_names_res = np.full([21, 8, 3], '', dtype=object)

    # group 0: backbone frame
    rigid_group_names_res[:, 0, :] = ['C', 'CA', 'N']

    # group 3: 'psi'
    rigid_group_names_res[:, 3, :] = ['CA', 'C', 'O']

    # group 4,5,6,7: 'chi1,2,3,4'
    for restype, letter in enumerate(restypes):
        restype_name = restype_1to3[letter]
        for chi_idx in range(4):
            if chi_angles_mask[restype][chi_idx]:
                atom_names = chi_angles_atoms[restype_name][chi_idx]
                rigid_group_names_res[restype, chi_idx + 4, :] = atom_names[1:]

    lookup_table = atom_order.copy()
    lookup_table[''] = 0
    rigid_group_atom37_idx_restype = np.vectorize(lambda x: lookup_table[x])(
        rigid_group_names_res)

    rigid_group_atom37_idx_residx = np_gather_ops(
        rigid_group_atom37_idx_restype, flat_aatype)
    #print("... ", rigid_group_atom37_idx_restype.dtype, flat_aatype.dtype)

    base_atom_pos = np_gather_ops(
        all_atom_positions,
        rigid_group_atom37_idx_residx,
        batch_dims=1)

    gt_frames = rigids_from_3_points(
        point_on_neg_x_axis=vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=vecs_from_tensor(base_atom_pos[:, :, 2, :]))

    rotations = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rotations[0, 0, 0] = -1
    rotations[0, 2, 2] = -1
    gt_frames = rigids_mul_rots(gt_frames, rots_from_tensor_np(rotations))

    rotation = [[gt_frames[0][0], gt_frames[0][1], gt_frames[0][2]],
                [gt_frames[0][3], gt_frames[0][4], gt_frames[0][5]],
                [gt_frames[0][6], gt_frames[0][7], gt_frames[0][8]]]
    translation = [gt_frames[1][0], gt_frames[1][1], gt_frames[1][2]]
    backbone_affine_tensor = to_tensor(rotation, translation)[:, 0, :]
    return backbone_affine_tensor

def to_tensor(rotation, translation):
    """get affine based on rotation and translation"""
    quaternion = rot_to_quat(rotation)
    return np.concatenate(
        [quaternion] +
        [np.expand_dims(x, axis=-1) for x in translation],
        axis=-1)

def rot_to_quat(rot, unstack_inputs=False):
    """transfer the rotation matrix to quaternion matrix"""
    if unstack_inputs:
        rot = [np.moveaxis(x, -1, 0) for x in np.moveaxis(rot, -2, 0)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy],
         [zy - yz, xx - yy - zz, xy + yx, xz + zx],
         [xz - zx, xy + yx, yy - xx - zz, yz + zy],
         [yx - xy, xz + zx, yz + zy, zz - xx - yy]]

    k = (1. / 3.) * np.stack([np.stack(x, axis=-1) for x in k],
                             axis=-2)
    # compute eigenvalues
    _, qs = np.linalg.eigh(k)
    return qs[..., -1]

def gather(params, indices, axis=0):
    """gather operation"""
    func = lambda p, i: np.take(p, i, axis=axis)
    return func(params, indices)

def np_gather_ops(params, indices, axis=0, batch_dims=0, is_multimer=False):
    """np gather operation"""
    if is_multimer:
        assert axis < 0 or axis - batch_dims >= 0
        ranges = []
        for i, s in enumerate(params.shape[:batch_dims]):
            r = np.arange(s)
            r = np.resize(r, (1,) * i + r.shape + (1,) * (len(indices.shape) - i - 1))
            ranges.append(r)
        remaining_dims = [slice(None) for _ in range(len(params.shape) - batch_dims)]
        remaining_dims[axis - batch_dims if axis >= 0 else axis] = indices
        ranges.extend(remaining_dims)
        return params[tuple(ranges)]

    if batch_dims == 0:
        return gather(params, indices)
    result = []
    if batch_dims == 1:
        for p, i in zip(params, indices):
            axis = axis - batch_dims if axis - batch_dims > 0 else 0
            r = gather(p, i, axis=axis)
            result.append(r)
        return np.stack(result)
    for p, i in zip(params[0], indices[0]):
        r = gather(p, i, axis=axis)
        result.append(r)
    res = np.stack(result)
    return res.reshape((1,) + res.shape)

# Copyright 2021 DeepMind Technologies Limited,
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2024 Changping Laboratory & Peking University.
def pseudo_beta_fn_np(aatype, all_atom_positions, all_atom_mask):
  """Create pseudo beta features."""
  
  ca_idx = atom_order['CA']
  cb_idx = atom_order['CB']

  is_gly = np.equal(aatype, restype_order['G'])
  is_gly_tile = np.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3])
  pseudo_beta = np.where(is_gly_tile, all_atom_positions[..., ca_idx, :], all_atom_positions[..., cb_idx, :])

  if all_atom_mask is not None:
    pseudo_beta_mask = np.where(is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx])
    pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
    return pseudo_beta, pseudo_beta_mask
  else:
    return pseudo_beta
  
def get_ppo_angles_sin_cos(atom37_positions):
    n_pos = atom37_positions[:, 0, :]
    ca_pos = atom37_positions[:, 1, :]
    c_pos = atom37_positions[:, 2, :]
    # phi: CA(n), C(n), N(n), CA(n+1)
    a1, a2, a3, a4 = c_pos[:-1], n_pos[1:], ca_pos[1:], c_pos[1:]
    phi_angle_values = calculate_dihedral_angle_np(a1, a2, a3, a4)
    phi_angle_values = np.concatenate([np.zeros(1), phi_angle_values])
    # psi: N(n), CA(n), C(n), N(n+1)
    a1, a2, a3, a4 = n_pos[:-1], ca_pos[:-1], c_pos[:-1], n_pos[1:]
    psi_angle_values = calculate_dihedral_angle_np(a1, a2, a3, a4)
    psi_angle_values = np.concatenate([psi_angle_values, np.zeros(1)])
    # omega: CA(n), C(n+1), N(n+1), CA(n+1)
    a1, a2, a3, a4 = ca_pos[:-1], c_pos[:-1], n_pos[1:], ca_pos[1:]
    omega_angle_values = calculate_dihedral_angle_np(a1, a2, a3, a4)
    omega_angle_values = np.concatenate([omega_angle_values, np.zeros(1)])
    
    ppo_angle_tensor = np.stack([phi_angle_values, psi_angle_values, omega_angle_values], axis=1)
    ppo_angle_sin_cos = np.concatenate([np.sin(ppo_angle_tensor),  np.cos(ppo_angle_tensor)], axis=1)
    ppo_anlge_mask = np.ones(ppo_angle_tensor.shape, dtype=np.int32)
    ppo_anlge_mask[0, 0] = 0
    ppo_anlge_mask[-1, 1] = 0
    ppo_anlge_mask[-1, 2] = 0
    return ppo_angle_sin_cos, ppo_anlge_mask

def calculate_dihedral_angle_np(A, B, C, D):
    a = B-A
    b = C-B
    c = D-C
    n1, n2 = np.cross(a, b), np.cross(b, c)
    b_ = np.cross(n1, n2)
    mask_ = np.sum(b*b_, axis=-1)
    mask = mask_ > 0
    angles_candidate_1 = np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2)/\
            (np.maximum(np.linalg.norm(n1, axis=1)*np.linalg.norm(n2, axis=1), 1e-6)), -1.0, 1.0))
    angles_candidate_2 = -np.arccos(np.clip(np.einsum('ij,ij->i', n1, n2)/\
            (np.maximum(np.linalg.norm(n1, axis=1)*np.linalg.norm(n2, axis=1), 1e-6)), -1.0, 1.0))
    angles = np.where(mask, angles_candidate_1, angles_candidate_2)
    return angles


def save_pdb_from_aux(aux, filename=None):
  aux = jax.tree_map(np.asarray, aux)
  p = {k:aux[k] for k in ["aatype","residue_index","atom_positions","atom_mask"]}        
  # p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][...,None]
  p["b_factors"] = p["atom_mask"] * aux["plddt"][...,None]
  Ls = [len(aux['aatype'])]

  def to_pdb_str(x, n=None):
    p_str = to_pdb(Protein(**x))
    p_str = "\n".join(p_str.splitlines()[1:-2])
    if n is not None:
      p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
    return p_str

  if p["atom_positions"].ndim == 4:
    if p["aatype"].ndim == 3: p["aatype"] = p["aatype"].argmax(-1)
    p_str = ""
    for n in range(p["atom_positions"].shape[0]):
      p_str += to_pdb_str(jax.tree_map(lambda x:x[n],p), n+1)
    p_str += "END\n"
  else:
    if p["aatype"].ndim == 2: p["aatype"] = p["aatype"].argmax(-1)
    p_str = to_pdb_str(p)
  if filename is None: 
    return p_str, Ls
  else: 
    with open(filename, 'w') as f:
      f.write(p_str)

# Copyright 2021 DeepMind Technologies Limited,
# SPDX-License-Identifier: Apache-2.0
def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.

    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """

    assert len(predicted_points.shape) == 3
    assert predicted_points.shape[-1] == 3
    assert true_points_mask.shape[-1] == 1
    assert len(true_points_mask.shape) == 3

    # Compute true and predicted distance metrices.
    dmat_true = np.sqrt(1e-10 + np.sum(
        (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

    dmat_predicted = np.sqrt(1e-10 + np.sum(
        (predicted_points[:, :, None] -
        predicted_points[:, None, :])**2, axis=-1))

    dists_to_score = (
        (dmat_true < cutoff).astype(np.float32) * true_points_mask *
        np.transpose(true_points_mask, [0, 2, 1]) *
        (1. - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = np.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                    (dist_l1 < 1.0).astype(np.float32) +
                    (dist_l1 < 2.0).astype(np.float32) +
                    (dist_l1 < 4.0).astype(np.float32))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))

    return score