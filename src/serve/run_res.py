from transformers import TextStreamer
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria
import requests
from io import BytesIO
import argparse
import warnings
from pathlib import Path
import os
import json

import sys
sys.path.insert(0, "/home/opc/Phi3-Vision-Finetune")
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<|image_1|>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    images = Path(args.images_file).glob('*.png')

    use_flash_attn = True

    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_path=args.model_path, model_base=args.model_base,
                                             model_name=model_name, device_map=args.device,
                                             load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                             device=args.device, use_flash_attn=use_flash_attn
                                             )
    for image_path in images:
        res = list()
        messages = [
            # {"role": "system", "content": "You are an AI assistant monitoring traffic situations through surveillance systems to support drivers in emergency situations."},
        ]

        image = load_image(image_path)
        image_name = str(image_path).split('/')[-1]
        generation_args = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": True if args.temperature > 0 else False,
            "repetition_penalty": args.repetition_penalty,
        }
        #################################################################
        input_text = """Summarize the chart. The summary should include the following key information:
                        1. What is the title of the chart?
                        2. What does the x-axis represent?
                        3. What does the y-axis represent?
                        4.Provide a brief summary of the chart.
                        5. Provide a brief analysis of the chart, including trend analysis and outlier analysis.
                        6. The labels shown in the chart (if no specific labels are shown, skip this part).
                        7. The legend shown in the chart (if no specific legend is shown, skip this part).
                        8. The annotations shown in the chart (if no specific annotations are shown, skip this part).
                        9. The source shown in the chart (if no specific source is shown, skip this part)."""
        #################################################################
        if image is not None and len(messages) < 2:
                # only putting the image token in the first turn of user.
                # You could just uncomment the system messages or use it.
            inp = DEFAULT_IMAGE_TOKEN + '\n' + input_text
        messages.append({"role": "user", "content": inp})

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, image, return_tensors="pt").to(args.device)

        stop_str = "<|end|>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, processor.tokenizer, inputs["input_ids"])
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                streamer=streamer,
                **generation_args,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
                stopping_criteria=[stopping_criteria]
                )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res = res.append({'image': image_name, "prompt": input_text, "completion" : outputs})
        messages.append({"role": "assistant", "content": outputs})

        # save result
        json_path = os.path.join(args.images_file, image_name + ".json")
        with open(json_path, "w") as outfile:
            json.dump(res, outfile, indent=4)

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="output/vision_merge")
    parser.add_argument("--model-base", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--images-file", type=str, default="data/golden_dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)