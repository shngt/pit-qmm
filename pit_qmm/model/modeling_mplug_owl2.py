#    Copyright 2023 Haotian Liu & Qinghao Ye (Modified from LLaVA)
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

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import copy
import os
import sys
import open3d as o3d

import yaml
from easydict import EasyDict

# * add logger
import logging
logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mplug_owl2 import MPLUGOwl2Config, MplugOwlVisionConfig, MplugOwlVisualAbstractorConfig
from .visual_encoder import MplugOwlVisionModel, MplugOwlVisualAbstractorModel
from .modeling_llama2 import replace_llama_modality_adaptive
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"
from icecream import ic

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids if len(chunk) > 0 else [] for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def expand2square(pil_img, background_color):
        from PIL import Image
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

class MPLUGOwl2MetaModel:
    def __init__(self, config):
        super(MPLUGOwl2MetaModel, self).__init__(config)
        self.vision_model = MplugOwlVisionModel(
            MplugOwlVisionConfig(**config.visual_config["visual_model"])
        )
        self.visual_abstractor = MplugOwlVisualAbstractorModel(
            MplugOwlVisualAbstractorConfig(**config.visual_config["visual_abstractor"]), config.hidden_size
        )
    
    def get_vision_tower(self):
        vision_model = getattr(self, 'vision_model', None)
        if type(vision_model) is list:
            vision_model = vision_model[0]
        return vision_model

    def get_visual_abstractor(self):
        visual_abstractor = getattr(self, 'visual_abstractor', None)
        if type(visual_abstractor) is list:
            visual_abstractor = visual_abstractor[0]
        return visual_abstractor

class MPLUGOwl2MetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def encode_images(self, images):
        image_features = self.get_model().vision_model(images).last_hidden_state
        image_features = self.get_model().visual_abstractor(encoder_hidden_states=image_features).last_hidden_state
        return image_features

    def encode_point_clouds(self, point_clouds):
        # print('point_clouds', point_clouds)
        point_features = self.get_model().point_backbone(point_clouds)
        # print('point_backbone', point_features)
        point_features = self.get_model().point_proj(point_features)
        # print('point_proj', point_features)
        return point_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, point_clouds
    ):
        if images is None or point_clouds is None or input_ids.shape[1] == 1:
            if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            multiway_indices = torch.zeros_like(input_ids).long().to(self.device)
            return input_ids, multiway_indices, attention_mask, past_key_values, None, labels
        
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)


        # print(images.shape)
        # print(image_features[0].shape)

        num_pc_samples = point_clouds.shape[1]
        point_clouds = torch.flatten(point_clouds, start_dim=0, end_dim=1)

        point_features = self.encode_point_clouds(point_clouds)

        point_backbone_config = self.get_model().point_backbone_config

        dummy_point_features = torch.zeros(point_backbone_config['point_token_len'], point_backbone_config['backbone_output_dim'], device=self.device, dtype=self.dtype)
        dummy_point_features = self.get_model().point_proj(dummy_point_features)

        # print(labels.tolist())

        new_input_embeds = []
        new_modality_indicators = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        cur_point_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # print('orig', cur_input_ids.tolist())
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # raise Error
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                
                cur_modality_indicators = torch.zeros(len(cur_input_embeds)).long().to(self.device)
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            if (cur_input_ids == point_backbone_config['point_patch_token']).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0. * dummy_point_features).sum() # * seems doing nothing
                new_input_embeds.append(cur_input_embeds)
                cur_modality_indicators = torch.zeros(len(cur_input_embeds)).long().to(self.device)
                new_modality_indicators.append(cur_modality_indicators)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_point_idx += 1
                continue
            # print(cur_point_idx)
            # print(point_features.shape)
            cur_point_features = point_features[cur_point_idx * num_pc_samples:(cur_point_idx + 1) * num_pc_samples].to(device=self.device)
            # print(cur_point_features.shape)
            cur_point_features = cur_point_features.flatten(start_dim=0, end_dim=1) # custom
            num_patches = cur_point_features.shape[0] # * number of point tokens
            # print(num_patches)

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_modality_indicators = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                
                # Add modality indicator
                assert image_token_start == len(cur_input_ids[:image_token_start])
                cur_modality_indicators.append(torch.zeros(len(cur_input_ids[:image_token_start])).long())
                cur_modality_indicators.append(torch.ones(len(cur_image_features)).long())
                
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])   
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            # if cur_input_ids.numel() > 0:
            #     cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
            #     cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
            #     if labels is not None:
            #         cur_new_labels.append(cur_labels)
            # cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            
            # # Modality
            # cur_modality_indicators = [x.to(device=self.device) for x in cur_modality_indicators]
            # cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)
            
            # print('after image', cur_modality_indicators)
            # if labels is not None:
            #     cur_new_labels = torch.cat(cur_new_labels, dim=0)
            #     # new_labels.append(cur_new_labels)
            
            # print(cur_new_labels.shape)

            # print('cur_point_features', cur_point_features)

            if point_backbone_config['mm_use_point_start_end']:
                if (cur_input_ids == point_backbone_config["point_start_token"]).sum() != (cur_input_ids == point_backbone_config["point_end_token"]).sum():
                    raise ValueError("The number of point start tokens and point end tokens should be the same.")
                point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]
                # print(point_start_tokens)
                # print(cur_input_ids)
                assert point_start_tokens.numel() == 1, 'More point cloud indicators than expected'
                # for point_start_token_pos in point_start_tokens:
                while point_start_tokens.numel() > 0:
                    point_start_token_pos = point_start_tokens[0]
                    # print(point_start_token_pos, 'point_start')
                    if cur_input_ids[point_start_token_pos + num_patches + 1] != point_backbone_config["point_end_token"]:
                        raise ValueError("The point end token should follow the image start token.")
                    if self.get_model().orig_embeds_params is not None: # * will not update the original embeddings except for IMAGE_START_TOKEN and IMAGE_END_TOKEN
                        # cur_new_input_embeds = torch.cat(
                        #     (
                        #         cur_new_input_embeds[:point_start_token_pos].detach(), 
                        #         cur_new_input_embeds[point_start_token_pos:point_start_token_pos+1], 
                        #         cur_point_features, 
                        #         cur_new_input_embeds[point_start_token_pos + num_patches + 1:point_start_token_pos + num_patches + 2], 
                        #         cur_new_input_embeds[point_start_token_pos + num_patches + 2:].detach()
                        #     ), 
                        #     dim=0
                        # )
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:point_start_token_pos]).detach())
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[point_start_token_pos:point_start_token_pos+1]))
                        cur_new_input_embeds.append(cur_point_features)
                    else:
                        # cur_new_input_embeds = torch.cat(
                        #     (
                        #         cur_new_input_embeds[:point_start_token_pos+1], 
                        #         cur_point_features, 
                        #         cur_new_input_embeds[point_start_token_pos + num_patches + 1:]
                        #     ), 
                        #     dim=0
                        # )
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:point_start_token_pos+1]))
                        cur_new_input_embeds.append(cur_point_features)

                    # Add modality indicator
                    assert point_start_token_pos == len(cur_input_ids[:point_start_token_pos])
                    # cur_modality_indicators = torch.cat(
                    #     (
                    #         cur_modality_indicators[:point_start_token_pos+1],
                    #         torch.ones(cur_point_features.shape[0], device=labels.device).long() * 2,
                    #         cur_modality_indicators[point_start_token_pos + num_patches + 1:]
                    #     ),
                    #     dim=0
                    # )
                    cur_modality_indicators.append(torch.zeros(len(cur_input_ids[:point_start_token_pos+1])))
                    cur_modality_indicators.append(torch.ones(len(cur_point_features)).long())
                    # cur_modality_indicators.append(torch.zeros(len(cur_point_features)))

                    if labels is not None:
                        # cur_new_labels = torch.cat(
                        #     (
                        #         cur_new_labels[:point_start_token_pos+1],
                        #         torch.full((cur_point_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype),
                        #         cur_new_labels[point_start_token_pos + num_patches + 1:]
                        #     ),
                        #     dim=0
                        # )
                        cur_new_labels.append(cur_labels[:point_start_token_pos+1])
                        cur_new_labels.append(torch.full((cur_point_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[point_start_token_pos+num_patches+1:]

                    cur_point_idx += 1
                    cur_input_ids = cur_input_ids[point_start_token_pos+num_patches+1:]
                    point_start_tokens = torch.where(cur_input_ids == point_backbone_config["point_start_token"])[0]

                # print('after pc', cur_modality_indicators)

                if cur_input_ids.numel() > 0:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                    cur_modality_indicators.append(torch.zeros(len(cur_input_ids)).long())
                    if labels is not None:
                        cur_new_labels.append(cur_labels)

                cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

                # Modality
                cur_modality_indicators = [x.to(device=self.device) for x in cur_modality_indicators]
                cur_modality_indicators = torch.cat(cur_modality_indicators, dim=0)

                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    
                # print('final', cur_modality_indicators.tolist())
                # print('labels', cur_new_labels.tolist())
                
                new_input_embeds.append(cur_new_input_embeds)
                new_modality_indicators.append(cur_modality_indicators)

                if labels is not None:
                    new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            
            # Embedding
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            
            # Modality
            new_modality_indicators_align = []
            for cur_modality_indicator in new_modality_indicators:
                cur_new_embed = torch.cat((cur_modality_indicator, torch.zeros(max_len - cur_modality_indicator.shape[0], dtype=cur_modality_indicator.dtype, device=cur_modality_indicator.device)), dim=0)
                new_modality_indicators_align.append(cur_new_embed)
            new_modality_indicators = torch.stack(new_modality_indicators_align, dim=0)
            
            # Label
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            
            # Attention Mask
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            new_modality_indicators = torch.stack(new_modality_indicators, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                # print('attn mask', attention_mask)
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, new_modality_indicators, attention_mask, past_key_values, new_input_embeds, new_labels



class MPLUGOwl2LlamaModel(MPLUGOwl2MetaModel, LlamaModel):
    config_class = MPLUGOwl2Config

    def __init__(self, config: MPLUGOwl2Config):
        super(MPLUGOwl2LlamaModel, self).__init__(config)

        self.orig_embeds_params = None

        # Point cloud
        self.point_backbone_type = config.point_backbone
        logger.info(f"Using {self.point_backbone_type}.")

        self.view_positional_embeddings = nn.Embedding(6, 768)

        if self.point_backbone_type == "PointBERT":
            from pit_qmm.model import PointTransformer
            # address of config file, in the same dir of this file
            point_bert_config_name = getattr(config, "point_backbone_config_name", "PointTransformer_8192point_2layer") # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
            point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
            print(f"Loading PointBERT config from {point_bert_config_addr}.")
            point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
            if getattr(config, "use_color", False):
                point_bert_config.model.point_dims = 6
            use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
            
            self.point_backbone = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool)
            logger.info(f"Using {self.point_backbone.point_dims} dim of points.")

            self.point_backbone_config = {
                "point_cloud_dim": point_bert_config.model.point_dims,
                "backbone_output_dim": point_bert_config.model.trans_dim if not use_max_pool else point_bert_config.model.trans_dim * 2,
                "project_output_dim": self.config.hidden_size,
                "point_token_len": point_bert_config.model.num_group + 1 if not use_max_pool else 1, # * number of output features, with cls token
                "mm_use_point_start_end": self.config.mm_use_point_start_end,
                "projection_hidden_layer": point_bert_config.model.get('projection_hidden_layer', 0),
                "use_max_pool": use_max_pool
            }
            # print(self.point_backbone_config)
            if point_bert_config.model.get('projection_hidden_layer', 0) > 0:
                self.point_backbone_config["projection_hidden_dim"] = point_bert_config.model.projection_hidden_dim # a list
            
            logger.info(f"Use max pool is {use_max_pool}. Number of point token is {self.point_backbone_config['point_token_len']}.")

        # * print relevant info with projection layers
        backbone_output_dim = self.point_backbone_config["backbone_output_dim"]
        logger.info(f"Point backbone output dim: {backbone_output_dim}.")
        logger.info(f"Use {self.point_backbone_config['projection_hidden_layer']} projection hiddent layers.")
        if self.point_backbone_config['projection_hidden_layer'] > 0:
            # Add projection layer with linear layers and GELU activation
            projection_layers = []
            last_dim = backbone_output_dim
            for i in range(point_bert_config.model.projection_hidden_layer):
                projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["projection_hidden_dim"][i]))
                projection_layers.append(nn.GELU())
                last_dim = self.point_backbone_config["projection_hidden_dim"][i]

            projection_layers.append(nn.Linear(last_dim, self.point_backbone_config["project_output_dim"]))
            self.point_proj = nn.Sequential(*projection_layers)
            logger.info(f"Each layer with {point_bert_config.model.projection_hidden_dim} hidden units.")
        else:
            # Single layer
            self.point_proj = nn.Linear(backbone_output_dim, self.point_backbone_config['project_output_dim'])
        logger.info(f"Point projector output dim: {self.point_backbone_config['project_output_dim']}.")

        self.fix_pointnet = False

        # End point cloud
            
    def load_point_backbone_checkpoint(self, checkpoint_path=None):
        # self.point_backbone.load_model_from_ckpt(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)
        # print('loading checkpoint...')
        self.point_backbone.load_checkpoint(self.config.point_backbone_ckpt if checkpoint_path is None else checkpoint_path)
        # for param_name, param_tensor in self.point_backbone.state_dict().items():
        #     print(param_tensor)



def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config

class MPLUGOwl2LlamaForCausalLM(LlamaForCausalLM, MPLUGOwl2MetaForCausalLM):
    config_class = MPLUGOwl2Config

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MPLUGOwl2LlamaModel(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained("q-future/one-align")
        self.image_processor = CLIPImageProcessor.from_pretrained("q-future/one-align")

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.preferential_ids_ = [id_[1] for id_ in self.tokenizer(["excellent","good","fair","poor","bad"])["input_ids"]]

        # Initialize weights and apply final processing
        self.post_init()
        

    def get_model(self):
        return self.model
    
    def score(self, images, 
              task_: str = "quality",
              input_: str = "image",
             ):
        if not hasattr(self, "weight_tensor"):
            self.weight_tensor = torch.Tensor([5.,4.,3.,2.,1.]).half().to(self.device)
        prompt = "USER: How would you rate the {} of this {}?\n<|image|>\nASSISTANT: The {} of the {} is".format(task_, input_, input_, task_)
        if input_ == "image":
            images = [expand2square(img, tuple(int(x*255) for x in self.image_processor.image_mean)) for img in images]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            with torch.inference_mode():
                image_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"].half().to(self.device)
                output_logits = self(input_ids.repeat(image_tensor.shape[0], 1),
                                images=image_tensor)["logits"][:,-1, self.preferential_ids_]
                return torch.softmax(output_logits, -1) @ self.weight_tensor
        else:
            video = [[expand2square(frame, tuple(int(x*255) for x in self.image_processor.image_mean)) for frame in vid] for vid in images]
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
            with torch.inference_mode():
                video_tensors = [self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].half().to(self.model.device) for vid in video]
                output_logits = self(input_ids.repeat(len(video_tensors), 1),
                            images=video_tensors)["logits"][:,-1, self.preferential_ids_]
                return torch.softmax(output_logits, -1) @ self.weight_tensor
        
    def initialize_tokenizer_point_backbone_config(self, tokenizer, device, fix_llm=True):

        config = self.config
        point_backbone_config = self.get_model().point_backbone_config
        mm_use_point_start_end = point_backbone_config['mm_use_point_start_end'] = config.mm_use_point_start_end

        default_point_patch_token = config.DEFAULT_POINT_PATCH_TOKEN
        point_backbone_config['default_point_patch_token'] = default_point_patch_token
        tokenizer.add_tokens([default_point_patch_token], special_tokens=True) # * no need to update embed since it will be replaced
        self.resize_token_embeddings(len(tokenizer)) # ! resize_token_embeddings will make the tokens trainable again
        point_backbone_config['point_patch_token'] = tokenizer.convert_tokens_to_ids([default_point_patch_token])[0]

        if mm_use_point_start_end:
            default_point_start_token = config.DEFAULT_POINT_START_TOKEN
            default_point_end_token = config.DEFAULT_POINT_END_TOKEN
            point_backbone_config['default_point_start_token'] = default_point_start_token
            point_backbone_config['default_point_end_token'] = default_point_end_token

            num_new_tokens = tokenizer.add_tokens([default_point_start_token, default_point_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            point_backbone_config["point_start_token"] = tokenizer.convert_tokens_to_ids([default_point_start_token])[0]
            point_backbone_config["point_end_token"] = tokenizer.convert_tokens_to_ids([default_point_end_token])[0]

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

                # need to update the input embeding, but no need to update the output embedding
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if fix_llm:
                    self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)] # * only tuning the new embeddings
                    for p in self.get_output_embeddings().parameters(): # * the llm head
                        p.requires_grad = False
                    print(f"Setting output embeddings fixed and {num_new_tokens} new tokens' input embeddings trainable.")
                else:
                    self.get_model().orig_embeds_params = None
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = True
                    print("Setting output embeddings and all input embeddings trainable.")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        # modality_indicators: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids, modality_indicators, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, point_clouds)
        # print('forward input embeds', torch.nonzero(torch.isnan(inputs_embeds)).tolist())

        # print('forward', modality_indicators.tolist())
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            modality_indicators=modality_indicators,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        # print('forward hidden states', outputs[0])
        # if torch.any(torch.isnan(hidden_states)):
        #     breakpoint()
        #     exit(0)
        logits = self.lm_head(hidden_states)
        # print('forward', logits)
        # print('forward', labels)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "point_clouds": kwargs.get("point_clouds", None),
            }
        )
        return model_inputs

AutoConfig.register("mplug_owl2", MPLUGOwl2Config)
AutoModelForCausalLM.register(MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM)

replace_llama_modality_adaptive()

if __name__ == "__main__":
    config = MPLUGOwl2Config.from_pretrained('q-future/one-align')
    from icecream import ic
    # config = MPLUGOwl2Config()
    model =  AutoModelForCausalLM(config)
    
    images = torch.randn(2, 3, 448, 448)
    input_ids = torch.cat([
        torch.ones(8).long(), torch.tensor([-1]*1).long(), torch.ones(8).long(), torch.tensor([-1]*1).long(), torch.ones(8).long()
    ], dim=0).unsqueeze(0)
    labels = input_ids.clone()
    labels[labels < 0] = -100
    
    # image_feature = model.encode_images(images)
    # ic(image_feature.shape)
    
    output = model(images=images, input_ids=input_ids, labels=labels)
    ic(output.loss)
    ic(output.logits.shape)
    
    model.save_pretrained('/cpfs01/shared/public/test/tmp_owl')