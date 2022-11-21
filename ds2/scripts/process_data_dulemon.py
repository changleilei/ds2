# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/10/20 11:32 上午
==================================="""
import argparse
import glob
import json
import os
from collections import defaultdict
from functools import partial

import jieba
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import tqdm
from transformers import BertTokenizer

from ds2.datasets.data_loader import DSTDataset


def read_data(data_path, description=None):

    datas = []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f, desc=f"process {description} data..."):
            d = json.loads(line)
            text = "".join(d['text'].split(" "))
            user_inform = "\n".join(d['User信息'])
            bot_inform = "\n".join(d['Bot信息'])
            user_inform = "".join(user_inform.split(" "))
            bot_inform = "".join(bot_inform.split(" "))
            input_text = text
            output_text = f"User信息：{user_inform}\nBot信息：{bot_inform}"
            datas.append({"input_text": input_text, "output_text": output_text})

    return datas

def collate_fn(tokenizer):
    def _collate(batch):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [d[key] for d in batch]

        input_batch = tokenizer(
            batch_data["input_text"],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            verbose=False,
            truncation=True,
            max_length=1000,
        )

        if "output_text" not in batch_data:
            batch_data["output_text"] = batch_data['output_text']

        batch_data["encoder_input"] = input_batch["input_ids"]
        batch_data["attention_mask"] = input_batch["attention_mask"]
        batch_data["decoder_output"] = tokenizer(
            batch_data["output_text"],
            padding=True,
            return_tensors="pt", # non-padded return List[List[Int]]
            return_attention_mask=False,
            truncation=True,
            max_length=200,
        ).input_ids

        return batch_data
    return _collate


def prepare_dulemon_data(args, tokenizer):
    paths = {
        k: f"{args['data_path']}/{k}.json" for k in ("train", "dev", "test")
    }

    datasets = {
        k: DSTDataset(read_data(path, description=k), args)
        for k, path in paths.items()
    }
    if args["debug_code"]:
        datasets["train"] = datasets["train"][:50]
        datasets["dev"] = datasets["dev"][:50]
        datasets["test"] = datasets["test"][:50]
    dataloaders = {
        k: DataLoader(
            dataset,
            batch_size=args[f"{k}_batch_size"],
            shuffle=(k == "train"),
            collate_fn=collate_fn(tokenizer=tokenizer),
            num_workers=10
        ) for k, dataset in datasets.items()
    }
    return dataloaders


def process_persona(personas: list, persona_dict: dict):
    for b_persona in personas:
        splits = b_persona.split(':')
        key, value = splits[0], splits[1]
        persona_dict[key] = value


def text_2_summary(data_path, target_path):
    paths = {
        k: f"{data_path}/{k}.json" for k in ("train", "dev", "test")
    }
    # if not os.path.exists(target_path):
    #     os.mkdir(target_path)
    target_paths = {
        k: f"{target_path}/{k}.json" for k in ("train", "dev", "test")
    }
    counter = defaultdict(int)

    for k, value in paths.items():
        with open(value, 'r', encoding='utf8') as f, open(target_paths[k], 'w', encoding='utf8') as wf:
            for line in f:
                datas = json.loads(line)
                # get summary
                bot_persona = datas['bot_persona']
                bot_persona_dict = {}
                process_persona(bot_persona, bot_persona_dict)
                user_persona_dict = {}
                user_said_persona = datas['user_said_persona']
                user_not_said_persona = datas['user_no_said_persona']
                process_persona(user_said_persona+user_not_said_persona, user_persona_dict)

                # conversation
                conversations = datas['conversation']
                dialogue_history = ''
                result_dict = {'text': '', 'User信息': [], 'Bot信息': []}
                user_inform = OrderedDict()
                bot_inform = OrderedDict()
                for turn in range(0, len(conversations)-2, 2):

                    sentences = conversations[turn:turn+2]
                    user, bot = sentences[0], sentences[1]
                    template = '对话摘要：\n{}'
                    usr_persona_temp = ''
                    bot_persona_temp = ''
                    user = user.replace('Usr', 'User')
                    if '\t' in user:

                        splits = user.split('\t')
                        user = splits[0]
                        if splits[1] in user_persona_dict:
                            usr_persona_temp = user_persona_dict[splits[1]]
                        if splits[1] in bot_persona_dict:
                            bot_persona_temp = bot_persona_dict[splits[1]]

                    if '\t' in bot:
                        splits = bot.split('\t')
                        bot = splits[0]
                        if splits[1] in user_persona_dict:
                            usr_persona_temp = user_persona_dict[splits[1]]
                        if splits[1] in bot_persona_dict:
                            bot_persona_temp = bot_persona_dict[splits[1]]
                    dialogue_history += user + '\n' + bot + '\n'
                    template_result = template.format(dialogue_history)
                    result_dict['text'] = template_result
                    user_inform[usr_persona_temp] = 1
                    bot_inform[bot_persona_temp] = 1

                    temp_user_inform = [k.strip() for k in list(user_inform.keys()) if k != ""]
                    if not temp_user_inform:
                        result_dict['User信息'] = ["空"]
                    else:
                        result_dict['User信息'] = temp_user_inform
                    temp_bot_inform = [k.strip() for k in list(bot_inform.keys()) if k != ""]
                    if not temp_bot_inform:
                        result_dict['Bot信息'] = ["空"]
                    else:
                        result_dict['Bot信息'] = temp_bot_inform

                    wf.write(json.dumps(result_dict, ensure_ascii=False)+'\n')
                    counter[k] += 1
    print('Done!')
    print(counter)  # {'train': 17037, 'dev': 2107, 'test': 2117})

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process data')
    parser.add_argument('--data_path', type=str, default='DuLeMon/both')
    parser.add_argument('--target_path', type=str, default='DuLeMon/summary')
    args = parser.parse_args()
    text_2_summary(args.data_path, args.target_path)









