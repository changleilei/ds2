# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/10/24 11:05 上午
==================================="""
import itertools
import json
import time

import pytorch_lightning as pl
import nltk; nltk.download('punkt')
import numpy as np
import rouge

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AdamW


class DuLeMonModel(pl.LightningModule):
    def __init__(self, args, tokenizer, sum_model, qa_model):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.sum_model = sum_model
        self.lr = args["lr"]
        self.blank = "____"

        self.evaluator = rouge.Rouge(
            metrics=['rouge-n'],
            max_n=4,
            limit_length=True,
            length_limit=100,
            length_limit_type='words',
            apply_avg=False,
            apply_best=True,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True
        )

    def training_step(self, batch, batch_idx):
        self.sum_model.train()

        outputs = self.sum_model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        )

        return {'loss': outputs.loss, 'log': {'train_loss': outputs.loss.detach()}}

    def eval_step(self, batch, batch_idx):
        self.sum_model.eval()

        outputs = self.sum_model(
            input_ids=batch["encoder_input"],
            attention_mask=batch["attention_mask"],
            labels=batch["decoder_output"],
        )

        return {"loss": outputs.loss.item(), "raw_text": batch['input_text']}

    def pred_step(self, batch, batch_idx):
        # 带上预测结果 rouge相关
        self.sum_model.eval()
        pred_summary_token = self.sum_model.generate(
            batch["encoder_input"],
            num_beams=self.args["num_beams"],
            min_length=5,
            max_length=100,
            early_stopping=True,
        )

        return {
            "raw_text": batch['input_text'],
            "pred_summary_token": pred_summary_token,
            "gold_summary": batch["output_text"]
        }

    def eval_epoch_end(self, outputs):
        res = {}
        res["loss"] = np.mean(outputs)
        print(res)
        return res

    def pred_epoch_end(self, outputs, mode="val"):
        outputs = {k: list(itertools.chain(*[o[k] for o in outputs])) for k in outputs[0]}

        pred_summary = [
            self.tokenizer.decode(_sum, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _sum in outputs["pred_summary_token"]
        ]
        res = {}

        rouge_score = self.evaluator.get_scores(pred_summary, outputs["gold_summary"])["rouge-1"]["f"]
        bleu_score = [
            sentence_bleu(
                [ref.split()],
                hyp.split(),
                smoothing_function=SmoothingFunction().method1
            )
            for ref, hyp in zip(outputs["gold_summary"], pred_summary)
        ]
        res.update({
            'rouge': rouge_score,
            'bleu': np.mean(bleu_score),
        })
        if 'raw_text' in outputs:
            samples = {"input_text": outputs['raw_text'], "gold_summary": outputs["gold_summary"], "pred_summary": pred_summary}
        else:
            samples = {"gold_summary": outputs["gold_summary"], "pred_summary": pred_summary}

        self.save_samples(samples, f'{str(round(res["bleu"], 4))}_{mode}')

        return res

    def validation_step(self, batch, batch_idx):
        if self.args["eval_loss_only"]:
            return self.eval_step(batch, batch_idx)['loss']
        else:
            return self.pred_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        if self.args["eval_loss_only"]:
            res = {f'val_{k}': v for k, v in self.eval_epoch_end(outputs).items()}
        else:
            res = {f'val_{k}': v for k, v in self.pred_epoch_end(outputs, "val").items()}
        self.log_dict(res)
        return res

    def test_step(self, batch, batch_idx):
        # 测试的一小步
        return self.pred_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        res = {f'test_{k}': v for k, v in self.pred_epoch_end(outputs, "test").items()}
        self.log_dict(res)
        return res

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)

    def save_samples(self, samples, name):
        if self.args["save_samples"] > 0:
            output_fields = ['fewshot', 'grad_acc_steps', 'train_batch_size']
            output_name = '_'.join([str(self.args[k]) for k in output_fields]) + '_' + name + '_' + str(round(time.time()))
            filename = f'./samples_data/{output_name}.json'
            with open(filename, 'w') as f:
                for gold, pred in zip(samples['gold_summary'], samples['pred_summary']):
                    f.write(json.dumps({"gold_summary": gold, "pred_summary": pred}, ensure_ascii=False)+'\n')
                # json.dump({k: v[:self.args['save_samples']] for k, v in samples.items()}, f, ensure_ascii=False)
