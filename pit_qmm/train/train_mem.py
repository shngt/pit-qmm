# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
sys.path.append('/work/09030/shngt/ls6/pit-qmm')
print(sys.path)

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

import torch

import transformers
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from torch.utils.data import Dataset
from pit_qmm.train.mplug_owl2_trainer import MPLUGOwl2Trainer
from pit_qmm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from pit_qmm import conversation as conversation_lib
from pit_qmm.model import *
from pit_qmm.mm_utils import tokenizer_image_token
# from .gen_distortions import distort_octant_with_type

from PIL import Image
from icecream import ic
import open3d as o3d
import numpy as np
import faiss
import fpsample

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    pc_data_path: Optional[str] = field(default=None)
    use_two_scale_pc: Optional[bool] = field(default=False)
    use_fp_pc: Optional[bool] = field(default=True)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    
    tune_visual_abstractor: bool = field(default=False)
    freeze_vision_model: bool = field(default=True)

    fix_llm: bool = field(default=True, metadata={"help": "Whether to fix the LLM."})
    fix_pointnet: bool = field(default=False, metadata={"help": "Whether to fix the PointNet."})

    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 128 #128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    visual_abstractor_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    save_safetensors: bool = False
    # * point backbone ckpt path
    point_backbone_ckpt: str = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=True) # * set True when pre-training, and false when fine-tuning


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['visual_abstractor']
    for name, module in model.named_modules():
        if not any(mm_keyword in name for mm_keyword in multimodal_keywords):
            # print(name)
            if ("v_proj.multiway.1" in name or "q_proj" in name or 'attn.qkv' in name) and 'drop' not in name: # or 'attn.proj' in name
                # print(name, ' added to lora')
                lora_module_names.add(name)
            # if "v_proj.multiway.1" in name or "q_proj" in name:
            #     lora_module_names.add(name)
            else:
                continue
        else:
            continue
            if "query" in name or "value" in name:
                lora_module_names.add(name)
            else:
                continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    ls = list(lora_module_names)
    print(ls)
    return ls


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                  ):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict

        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    point_indicator: str = '<point>',
    num_pc_samples: int = 3
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    point_backbone_config = data_args.point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
               
            replace_token = DEFAULT_IMAGE_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            replace_token = default_point_patch_token * point_token_len * num_pc_samples
            if point_backbone_config['mm_use_point_start_end']:
                replace_token = point_backbone_config['default_point_start_token'] + replace_token + point_backbone_config['default_point_end_token']
            sentence["value"] = sentence["value"].replace(point_indicator, replace_token)
        
    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # print(roles)

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # print('conversations: ', conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # print(input_ids.shape)
    targets = input_ids.clone()
    # print(tokenizer.decode(input_ids[0][-4:]))
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO or conv.sep_style == conversation_lib.SeparatorStyle.TWO_NO_SYS

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # print(total_len)
        rounds = conversation.split(conv.sep2)
        # print(rounds)
        cur_len = 1 + 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # print(parts, sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 3
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            round_len -= 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            # print(tar/get)

            cur_len += round_len
            # print('cur_len', cur_len)
        target[cur_len:] = IGNORE_INDEX
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]
    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.num_pc_samples = 3
        self.pc_data_path = data_args.pc_data_path
        
        if self.data_args.use_two_scale_pc:
            rank0_print("Using two scale point cloud sampling...")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list


    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         sources = self.list_data_dict[i]
#         if isinstance(i, int):
#             sources = [sources]
#         assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
#         if 'image' in sources[0]:
#             image_file = self.list_data_dict[i]['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result
#                 image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in self.list_data_dict[i]))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         return data_dict
    
    def next_rand(self):
        import random
        return random.randint(0,len(self)-1)

    def _load_point_cloud(self, object_id, type='objaverse', evaluate=False):
        if type == 'objaverse':
            # return self._load_objaverse_point_cloud(object_id) 
            return self._load_objaverse_point_cloud_sp(object_id, return_full=evaluate)

    # def _load_objaverse_point_cloud(self, object_id):
    #     point_cloud = self.pc_io.load_pointcloud(os.path.join(self.data_path, f'{object_id}.ply'))
    #     point_cloud = point_cloud.subsample(8192)
    #     points, vertices = point_cloud.points_list()[0], point_cloud.features_list()[0]
    #     # points, vertices = point_cloud.points_list()[0][:8192, :], point_cloud.features_list()[0][:8192, :]
    #     # print(points.shape, vertices.shape)
    #     point_cloud = torch.cat((points, vertices), axis=1).numpy()
    #     # print(point_cloud.shape)
    #     if not self.use_color:
    #         point_cloud = point_cloud[:, :3]
    #     # print('Shape of point cloud...', point_cloud.shape)
    #     return point_cloud

    def _load_objaverse_point_cloud_sp(self, object_id, return_full=False):
        # print('filling with zeroes')
        # return torch.zeros((3, 8192, 6))
        # print(object_id)
        point_cloud_orig = o3d.io.read_point_cloud(os.path.join(self.pc_data_path, f'{object_id}.ply'))
        # point_cloud_orig = o3d.io.read_point_cloud(os.path.join(self.pc_data_path, f'{object_id.split("_")[0]}.ply'))
        # octant = int(object_id.split('_')[-1])
        # distortion_name = ' '.join(object_id.split('_')[1:-1])
        # point_cloud = distort_octant_with_type(point_cloud, distortion_name, octant)

        points, colors = np.asarray(point_cloud_orig.points).astype(np.float32), np.asarray(point_cloud_orig.colors).astype(np.float32)
        
        points = self.pc_norm(points)
        point_cloud = np.concatenate((points, colors), axis=1)

        if return_full:
            return torch.from_numpy(point_cloud)
        
        sampled_pcs = []
        if self.data_args.use_fp_pc:
            sampled_pcs.append(point_cloud[fpsample.bucket_fps_kdline_sampling(point_cloud, 8192, h=9)])
        # print(sampled_pcs[0].shape)

        index = faiss.IndexFlatL2(point_cloud.shape[1])  # L2 distance
        index.add(point_cloud)

        if not self.data_args.use_two_scale_pc:
            if self.data_args.use_fp_pc:
                for _ in range(self.num_pc_samples - 1): # range(self.num_pc_samples - 1): # 
                    idx = np.random.choice(point_cloud.shape[0])
                    seed = point_cloud[idx].reshape(1, -1)
                    _, indices = index.search(seed, 8192)
                    sampled_pcs.append(point_cloud[indices[0]])
            else:
                # print('not using fp')
                for _ in range(self.num_pc_samples): # range(self.num_pc_samples - 1): # 
                    idx = np.random.choice(point_cloud.shape[0])
                    seed = point_cloud[idx].reshape(1, -1)
                    _, indices = index.search(seed, 8192)
                    sampled_pcs.append(point_cloud[indices[0]])
        else:
            for _ in range(self.num_pc_samples // 2): # range(self.num_pc_samples - 1): # 
                idx = np.random.choice(point_cloud.shape[0])
                seed = point_cloud[idx].reshape(1, -1)
                _, indices = index.search(seed, 8192)
                sampled_pcs.append(point_cloud[indices[0]])

            # Half scale point cloud
            point_cloud_half = point_cloud_orig.uniform_down_sample(2)

            points, colors = np.asarray(point_cloud_half.points).astype(np.float32), np.asarray(point_cloud_half.colors).astype(np.float32)
            
            points = self.pc_norm(points)
            point_cloud = np.concatenate((points, colors), axis=1)

            index = faiss.IndexFlatL2(point_cloud.shape[1])
            index.add(point_cloud)

            for _ in range(self.num_pc_samples // 2): # range():
                idx = np.random.choice(point_cloud.shape[0])
                seed = point_cloud[idx].reshape(1, -1)
                _, indices = index.search(seed, 8192)
                sampled_pcs.append(point_cloud[indices[0]])

        sampled_pcs = np.stack(sampled_pcs)
        sampled_pcs = torch.from_numpy(sampled_pcs)
        # if not self.data_args.use_fp_pc:
        #     print(sampled_pcs.shape)
        return sampled_pcs

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m if m != 0 else xyz

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print('fetching sample', i)
        while True:
            try:
                sources = self.list_data_dict[i]
                # print(sources)
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
                point_cloud = None
                if 'object_id' in sources[0]:
                    # print('ct')
                    object_id = self.list_data_dict[i]['object_id']
                    point_cloud = self._load_point_cloud(object_id)
                    if 'image' not in sources[0]:
                        sources[0]['image'] = f'{object_id.split("/")[-1]}.mp4'
                        self.list_data_dict[i]['image'] = f'{object_id.split("/")[-1]}.mp4'
                    # print(len(point_cloud))
                if 'image' in sources[0]:
                    image_file = self.list_data_dict[i]['image']

                    image_folder = self.data_args.image_folder
                    processor = self.data_args.image_processor
                    from pathlib import Path
                    #if not Path(os.path.join(image_folder, image_file)).exists():
                    #    i = self.next_rand()
                    #    continue
                    # print(image_file)
                    if isinstance(image_file, list):
                        # Multiple Images as Input
                        try:
                            image = [Image.open(os.path.join(image_folder, imfile)).convert('RGB') for imfile in image_file]
                        except Exception as ex:
                            print(ex)
                            i = self.next_rand()
                            continue
                        if self.data_args.image_aspect_ratio == 'pad':
                            image = [expand2square(img, tuple(int(x*255) for x in processor.image_mean)) for img in image]
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                        else:
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                    elif os.path.join(image_folder, image_file).endswith("mp4"):
                        # Video as Input
                        # print('Video')
                        image = load_video(os.path.join(image_folder, image_file))
                        # print(len(image))
                        if self.data_args.image_aspect_ratio == 'pad':
                            image = [expand2square(img, tuple(int(x*255) for x in processor.image_mean)) for img in image]
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                        else:
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                        # print(image)
                    else:
                        try:
                            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                        except Exception as ex:
                            print(ex)
                            i = self.next_rand()
                            continue
                        if self.data_args.image_aspect_ratio == 'pad':
                            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                        else:
                            image = processor.preprocess(image, return_tensors='pt')['pixel_values']
                    sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args,
                        num_pc_samples=self.num_pc_samples
                    )
                else:
                    sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(
                    sources,
                    self.tokenizer,
                    has_image=('image' in self.list_data_dict[i]))
                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0],
                                    labels=data_dict["labels"][0])

                # image exist in the data
                if 'image' in self.list_data_dict[i]:
                    data_dict['image'] = image
                    data_dict['point_cloud'] = point_cloud
                elif self.data_args.is_multimodal:
                    # image does not exist in the data, but the model is multimodal
                    print('Warning: image not found in data')
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
                    data_dict['point_cloud'] = torch.zeros(3, 8192, 6)
                return data_dict
            except Exception as ex:
                print('tt', ex)
                i = self.next_rand()
                continue


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print('collating inputs')
        # # print(instances)
        # print('image', instances[0]['image'].max(), instances[0]['image'].min())
        # print('point cloud', instances[0]['point_cloud'].max(), instances[0]['point_cloud'].min())
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # print([True if x != y else False for x, y in zip(input_ids[0].tolist(), labels[0].tolist())])
        # print('max length', self.tokenizer.model_max_length)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        # print(input_ids[0].tolist())
        # print(labels[0].tolist())
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'point_cloud' in instances[0]:
            point_clouds = [instance['point_cloud'] for instance in instances]
            if all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds): # * point_clouds have different shapes
                batch['point_clouds'] = torch.stack(point_clouds)
            else:
                batch['point_clouds'] = point_clouds # * return as lists

        return batch

def compute_metrics(pred):
    print(pred)
    return {}


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    if data_args.eval_data_path is not None:
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                             data_path=data_args.eval_data_path,
                                             data_args=data_args)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print(compute_dtype)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            #device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # breakpoint()
    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=compute_dtype,
        **bnb_model_from_pretrained_args
    )
    print(model.config)

    if not training_args.deepspeed:
        model.get_model().load_point_backbone_checkpoint(training_args.point_backbone_ckpt)
    model.initialize_tokenizer_point_backbone_config(tokenizer=tokenizer, device=training_args.device, fix_llm=training_args.fix_llm)

    point_backbone_config = model.get_model().point_backbone_config

    data_args.point_token_len = point_backbone_config['point_token_len']
    data_args.mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']
    data_args.point_backbone_config = point_backbone_config

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    if training_args.fix_llm:
        # * This will fix all the parameters
        logging.info("LLM is fixed. Fix_llm flag is set to True")
        # * fix llama, lm_head, pointnet, projection layer here
        model.get_model().requires_grad_(False)
        model.get_model().fix_llm = True
        model.get_model().point_proj.requires_grad_(True) 
        model.get_model().point_backbone.requires_grad_(True) # * set as True for fsdp, use fix_pointnet flag to control
    else:
        model.get_model().fix_llm = False
        logging.warning("LLM is trainable. Fix_llm flag is set to False")

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        
        model = get_peft_model(model, lora_config)
        # print('PEFT model loaded')
    
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if not training_args.fix_pointnet:
        # * not fix pointnet
        rank0_print("Point backbone is trainable. Fix_pointnet flag is set to False, pointnet grad will be recorded.")
        model.get_model().fix_pointnet = False
    else:
        rank0_print("Point backbone is fixed. Fix_pointnet flag is set to True, pointnet grad will not be recorded.")
        model.get_model().fix_pointnet = True # * use with torch.inference_mode to control, not requires_grad for fsdp for second stage
        # if not training_args.stage_2:
        #     logging.info("Set requires_grad of point backbone to False")
        model.get_model().point_backbone.requires_grad_(False) # * fix pointnet for first stage, need for fsdp in stage2


    if not training_args.freeze_vision_model and training_args.bits in [4, 8]:
        model.get_model().vision_model.to(dtype=compute_dtype, device=training_args.device)
    else:
        vision_tower = model.get_model().vision_model
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    
    if training_args.tune_visual_abstractor and training_args.bits in [4, 8]:
        model.get_model().visual_abstractor.to(dtype=compute_dtype, device=training_args.device)
    else:
        visual_abstractor = model.get_model().visual_abstractor
        visual_abstractor.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.tune_visual_abstractor = model_args.tune_visual_abstractor = training_args.tune_visual_abstractor
    print(training_args.tune_visual_abstractor)
    for n, p in model.named_parameters():
        if training_args.lora_enable:
            p.requires_grad = True if "lora_" in n else False
        else:
            p.requires_grad = True
        # if "lm_head" in n:
        #     print(n)
        #     p.requires_grad = True

    if training_args.tune_visual_abstractor:
        #model.requires_grad_(False)
        for n, p in model.get_model().visual_abstractor.named_parameters():
            p.requires_grad = True

    if training_args.tune_mm_mlp_adapter:
        # * not fix the projection layer
        # * may need to set the embed_tokens to require_grad = True if added new tokens
        # * this is done in initialize_tokenizer_point_backbone_confi
        model.get_model().point_proj.requires_grad_(True)
        logging.info("Point projection layer is trainable.")
    else:
        model.get_model().point_proj.requires_grad_(False)
        logging.info("Point prejcetion layer is fixed.")

    if not training_args.fix_pointnet:
        model.get_model().point_backbone.requires_grad_(True)

    model.config.freeze_vision_model = training_args.freeze_vision_model
    print('vision model frozen', training_args.freeze_vision_model)
    # if training_args.freeze_vision_model:
    #     for p in model.get_model().vision_model.parameters():
    #         p.requires_grad = False
    for p in model.get_model().vision_model.parameters():
        p.requires_grad = not training_args.freeze_vision_model
    if training_args.lora_enable:        
        model.print_trainable_parameters()     
    model.config.visual_abstractor_lr = training_args.visual_abstractor_lr


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # for name, param in model.named_parameters():
    #     print(f'Processing {name}')
    #     print(param)
    #     if torch.any(torch.isnan(param)):
    #         breakpoint()
    # exit(0)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    # steps_per_epoch = len(data_module['train_dataset']) // (training_args.per_device_train_batch_size * torch.cuda.device_count())
    # save_steps = steps_per_epoch * 10

    # training_args.save_steps = save_steps
    # training_args.evaluation_strategy = 'epoch'


    trainer = MPLUGOwl2Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    compute_metrics=compute_metrics,
                    **data_module)

    print('Trainer initialized')
    # Check which parameters require gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    
    # TODO I dont like auto resume << REMOVE IT AND UNCOMMENT THE ABOVE CODE
    trainer.train()

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,
                                       )


if __name__ == "__main__":
    train()
