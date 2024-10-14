import argparse
import torch

import sys
sys.path.append('/home1/09030/shngt/work/pit-qmm/pit-qmm')

from pit_qmm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from pit_qmm.conversation import conv_templates, SeparatorStyle
from pit_qmm.model.builder import load_pretrained_model
from pit_qmm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from pit_qmm.train.train_mem import LazySupervisedDataset

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from scipy.stats import spearmanr, pearsonr

import json
from tqdm import tqdm
from collections import defaultdict

import os
import numpy as np

def wa5(logits):
    logprobs = np.array([logits["excellent"], logits["good"], logits["fair"], logits["poor"], logits["bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.point_backbone_ckpt, args.load_8bit, args.load_4bit, device=args.device)
    # breakpoint()
    
    import json

    
    # image_paths = [
    #     # "playground/data/",
    #     # "playground/data/",
    #     # "playground/data/KoNViD_1k_videos/",
    #     # "playground/data/maxwell/",
    #     # "/home1/09030/shngt/work/lspcqa_views/"
    #     "/work/09030/shngt/ls6/pcq-lmm/point_qa_datagen/sjtu_pcqa_views/",
    #     # "/work/09030/shngt/ls6/pcq-lmm/point_qa_datagen/wpc_views/",
    # ]

    if 'wpc' in args.model_path:
        if 'mm' in args.model_path:
            image_paths = ['/scratch/09030/shngt/wpc_projections']
        else:
            image_paths = ["/work/09030/shngt/pit-qmm/point_qa_datagen/wpc_views/"]
        dataset = 'wpc'
        args.pc_data_path = '/scratch/09030/shngt/the_WPC_database/distorted_PCs'
    elif 'sjtu' in args.model_path:
        if 'mm' in args.model_path:
            image_paths = ["/work/09030/shngt/ls6/pcq-lmm/point_qa_datagen/sjtu_projections/"]
        else:
            image_paths = ["/work/09030/shngt/ls6/pcq-lmm/point_qa_datagen/sjtu_pcqa_views/"]
        dataset = 'sjtu_pcqa'
        args.pc_data_path = '/scratch/09030/shngt/SJTU-PCQA/distortion'
    elif 'lspcqaf' in args.model_path:
        image_paths = ["/scratch/09030/shngt/lspcqa_full_views_b_bg"]
        dataset = 'lspcqa_full'
        args.pc_data_path = '/scratch/09030/shngt/lspcqa_full/all'
    elif 'lspcqa' in args.model_path:
        image_paths = ["/home1/09030/shngt/work/lspcqa_views/"]
        dataset = 'lspcqa'

    if 'short-desc' in args.model_path:
        dataset += '_short_desc'
    elif 'desc' in args.model_path:
        dataset += '_desc'

    split_num = args.model_path.split('-')[-1]
    jsons = [f'/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/{dataset}/{args.type}_split_{split_num}.json']

    # json_prefix = "playground/data/test_jsons/"s
    # jsons = [
    #     # json_prefix + "test_lsvq.json",
    #     # json_prefix + "test_lsvq_1080p.json",
    #     # json_prefix + "konvid.json",
    #     # json_prefix + "maxwell_test.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/point/test_split_1.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/point/test_split_2.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/point/test_split_3.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/point/test_split_4.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/point/test_split_5.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/sjtu_pcqa/test_split_1.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/sjtu_pcqa/test_split_2.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/sjtu_pcqa/test_split_3.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/sjtu_pcqa/test_split_4.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/sjtu_pcqa/test_split_5.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/wpc/test_split_1.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/wpc/test_split_2.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/wpc/test_split_3.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/wpc/test_split_4.json",
    #     # "/work/09030/shngt/ls6/pcq-lmm/pit-qmm/playground/data/ft/wpc/test_split_5.json",
    #     # "/proj/esv-summer-interns/home/eguhpas/pit-qmm/playground/data/ft/point/test_split_2.json",
    #     # "/proj/esv-summer-interns/home/eguhpas/pit-qmm/playground/data/ft/point/test_split_3.json",
    #     # "/proj/esv-summer-interns/home/eguhpas/pit-qmm/playground/data/ft/point/test_split_4.json",
    #     # "/proj/esv-summer-interns/home/eguhpas/pit-qmm/playground/data/ft/point/test_split_5.json"
    # ]

    # Just need a method from this, others are dummy arguments
    ld = LazySupervisedDataset(jsons[0], tokenizer, args)
    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']

    os.makedirs(f"results/{args.model_path}/", exist_ok=True)

    conv_mode = "mplug_owl2"
    
    inp = "This is a point cloud rated for quality. " \
        + "Following are views of the point cloud." \
        + "How would you rate the quality of this point cloud?"
    # inp = "How would you rate the quality of this point cloud?"
    # inp = "Can you identify the nature and location of the distortion?"
        
    conv = conv_templates[conv_mode].copy()
    inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + point_backbone_config['default_point_start_token'] + default_point_patch_token * point_token_len * 3 + point_backbone_config['default_point_end_token']
    conv.append_message(conv.roles[0], inp)
    image = None
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the point cloud is"
    # prompt = conv.get_prompt() + ""
    
    toks = ["good", "poor", "high", "fair", "low", "excellent", "bad", "fine", "moderate",  "decent", "average", "medium", "acceptable"]
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    for image_path, json_ in zip(image_paths, jsons):
        with open(json_) as f:
            iqadata = json.load(f) 
            prs, gts = [], []
            for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
                try:
                    try:
                        filename = llddata["img_path"]
                    except:
                        try:
                            filename = llddata["image"]
                        except:
                            filename = f'{llddata["object_id"].split("/")[-1]}.mp4'
                    llddata["logits"] = defaultdict(float)
                    # breakpoint()
                    image = load_video(os.path.join(image_path, filename))
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
                    image = [expand2square(img, tuple(int(x*255) for x in image_processor.image_mean)) for img in image]
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(args.device)

                    point_cloud = ld._load_point_cloud(llddata['object_id'], evaluate=False).to(args.device)
                    # point_clouds = torch.split(point_cloud, 3 * 8192)
                    point_clouds = [point_cloud, point_cloud]
                    # breakpoint()

                    if True:
                        with torch.inference_mode():
                            for pc in point_clouds[:-1]:
                                pc = pc.reshape((3, 8192, -1))
                                output_logits = model(input_ids,
                                    images=[image_tensor], point_clouds=pc.unsqueeze(0))["logits"][:,-1]
                                for tok, id_ in zip(toks, ids_):
                                    llddata["logits"][tok] += output_logits.mean(0)[id_].item()
                            # print(llddata["logits"], llddata["gt_score"])
                            # for tok, id_ in zip(toks, ids_):
                            #     llddata["logits"][tok] /= len(point_clouds)
                            llddata["score"] = wa5(llddata["logits"])
                            # print(llddata)
                            prs.append(llddata["score"])
                            gts.append(llddata["gt_score"])
                            # print(llddata)
                            json_ = json_.replace("combined/", "combined-")
                            with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "a") as wf:
                                json.dump(llddata, wf)

                    if i > 0 and i % 200 == 0:
                        print(spearmanr(prs,gts)[0], pearsonr(prs,gts)[0])
                except Exception as e:
                    print(e)
                    breakpoint()
                    continue
            print(json_)
            print("Spearmanr", spearmanr(prs,gts)[0], "Pearson", pearsonr(prs,gts)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="q-future/one-align")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--point-backbone-ckpt", type=str, default="checkpoints/point_bert_v1.2.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    # parser.add_argument("--pc-data-path", type=str, required=True)
    parser.add_argument("--use-two-scale-pc", type=bool, required=True)
    parser.add_argument("--use-fp-pc", type=bool, required=True)
    parser.add_argument("--type", type=str, default='test')
    args = parser.parse_args()
    main(args)
