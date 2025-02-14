import argparse
import torch

from pit_qmm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from pit_qmm.conversation import conv_templates, SeparatorStyle
from pit_qmm.model.builder import load_pretrained_model
from pit_qmm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from scipy.stats import spearmanr, pearsonr


import json
from tqdm import tqdm
from collections import defaultdict

import os

def wa5(logits):
    import numpy as np
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


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    
    import json

    
    image_path = "playground/data/"
                  

    json_prefix = "playground/data/test_jsons/"
    jsons = [
        json_prefix + "test_ava.json",
    ]

    os.makedirs(f"results/{args.model_path}/", exist_ok=True)


    conv_mode = "mplug_owl2"
    
    inp = "How would you rate the aesthetics of this image?"
        
    conv = conv_templates[conv_mode].copy()
    inp = DEFAULT_IMAGE_TOKEN + inp
    conv.append_message(conv.roles[0], inp)
    image = None
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The aesthetics of the image is"
    
    toks = ["good", "poor", "high", "fair", "low", "excellent", "bad", "fine", "moderate",  "decent", "average", "medium", "acceptable"]
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    for json_ in jsons:
        with open(json_) as f:
            iqadata = json.load(f)  

            image_tensors = []
            batch_data = []
            prs, gts = [], []
            for i, llddata in enumerate(tqdm(iqadata, desc="Evaluating [{}]".format(json_.split("/")[-1]))):
                filename = llddata["image"]
                llddata["logits"] = defaultdict(float)
                
                
                
                image = load_image(image_path + filename)
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
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)

                image_tensors.append(image_tensor)
                batch_data.append(llddata)

                if i % 8 == 7 or i == len(iqadata) - 1:                     
                    with torch.inference_mode():
                        output_logits = model(input_ids.repeat(len(image_tensors), 1),
                            images=torch.cat(image_tensors, 0))["logits"][:,-1]
    
                    for j, xllddata in enumerate(batch_data):
                        for tok, id_ in zip(toks, ids_):
                            xllddata["logits"][tok] += output_logits[j,id_].item()
                        xllddata["score"] = wa5(xllddata["logits"])
                        # print(llddata)
                        prs.append(xllddata["score"])
                        gts.append(xllddata["gt_score"])
                        json_ = json_.replace("combined/", "combined-")
                        with open(f"results/{args.model_path}/{json_.split('/')[-1]}", "a") as wf:
                            json.dump(xllddata, wf)

                    image_tensors = []
                    batch_data = []
                
                #if i > 0 and i % 200 == 0:
                #    print(spearmanr(prs,gts)[0], pearsonr(prs,gts)[0])
            print("Spearmanr", spearmanr(prs,gts)[0], "Pearson", pearsonr(prs,gts)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="q-future/one-align")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)