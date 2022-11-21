import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Tokenizer
)
# os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from ds2.datasets.data_loader import (
    prepare_data,
)
from ds2.configs.config import get_args
from ds2.models.DS2 import DS2
from ds2.models.DuleMon import DuLeMonModel
from ds2.scripts.process_data_dulemon import prepare_dulemon_data, T5PegasusTokenizer


def fine_tune(args, *more):
    args = vars(args)
    seed_everything(args["seed"])
    print(args)
    if args["model_checkpoint"] == "../chinese_t5_pegasus_base":
        tokenizer = T5PegasusTokenizer.from_pretrained(args["model_checkpoint"])
        tokenizer.pad_token = '<pad>'

    else:
        tokenizer = AutoTokenizer.from_pretrained(args["model_checkpoint"])
    model = AutoModelForSeq2SeqLM.from_pretrained(args["model_checkpoint"])
    dds = tokenizer("User：我真帅啊\nBot：对啊，你真的飘亮".replace('\n', '[SEP]'))
    dd = tokenizer("我 真 帅 啊")
    s = tokenizer.decode(dds['input_ids'])


    dataloaders = prepare_dulemon_data(
        args, tokenizer
    )
    print("Created dataloaders")

    # logging
    exp_name = args["exp_name"]
    if not os.path.exists("/data/cll/ds2/ds2/logs"):
        os.mkdir("/data/cll/ds2/ds2/logs")
    log_path = f"/data/cll/ds2/ds2/logs{exp_name}"
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    if args["load_pretrained"]:
        pretrain_ckpt_path = os.path.join(args["load_pretrained"], "pretrain")
        pretrain_ckpts = [
            _ckpt for _ckpt in os.listdir(pretrain_ckpt_path)
            if ".ckpt" in _ckpt
        ]
        assert len(pretrain_ckpts) == 1
        ckpt = pretrain_ckpts[0]
        print("load pretrained model from: ", os.path.join(pretrain_ckpt_path, ckpt))
        dst_model = DuLeMonModel.load_from_checkpoint(
            os.path.join(pretrain_ckpt_path, ckpt),
            args=args,
            tokenizer=tokenizer,
            sum_model=model,
            qa_model=None,
        )
    else:
        dst_model = DuLeMonModel(args, tokenizer, model, None)

    print("Created Model")

    dir_path = os.path.join(log_path, args["mode"])
    if not args["do_test_only"]:
        earlystopping_callback = EarlyStopping(
            monitor="val_loss",
            # min_delta=0.00,
            patience=args["patience"],
            verbose=False,
            mode="min" if args["eval_loss_only"] else "max",
        )
        print(f"model saved in {dir_path}")
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_path,
            filename="{val_loss:.3f}" if args["eval_loss_only"] else "{val_bleu:.3f}",
            save_top_k=1,
            monitor="val_loss" if args["eval_loss_only"] else "val_bleu",
            mode="min" if args["eval_loss_only"] else "max",
        )
        callbacks = [earlystopping_callback, checkpoint_callback]
    else:
        callbacks = None

    # profiler = PyTorchProfiler(export_to_chrome=True)
    trainer = Trainer(
        accumulate_grad_batches=args["grad_acc_steps"],
        gradient_clip_val=args["max_norm"],
        max_epochs=args["n_epochs"],
        callbacks=callbacks,
        gpus=args["GPU"],
        deterministic=True,
        # accelerator="ddp",
        val_check_interval=args["val_check_interval"],
        logger=CSVLogger(dir_path, f"seed_{args['seed']}") if not args["do_test_only"] else None,
        resume_from_checkpoint=args["resume_from_ckpt"],
        # limit_val_batches=0.05,
    )

    if not args["do_test_only"]:
        trainer.fit(dst_model, dataloaders["train"], dataloaders["dev"])

    if not args["do_train_only"]:
        print("test start...")
        # evaluate modelcd
        args["num_beams"] = args["test_num_beams"]
        if args["do_test_only"]:
            ckpts = [_ckpt for _ckpt in os.listdir(dir_path) if ".ckpt" in _ckpt]
            assert len(ckpts) == 1
            ckpt = ckpts[0]
            print("load pretrained model from: ", os.path.join(dir_path, ckpt))
            ckpt_path = os.path.join(dir_path, ckpt)
        else:
            ckpt_path = checkpoint_callback.best_model_path

        print(f"Best checkpoints {ckpt_path}")
        dst_model = DuLeMonModel.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            args=args,
            tokenizer=tokenizer,
            sum_model=model,
            qa_model=None
        )
        trainer.test(dst_model, dataloaders["test"])


if __name__ == "__main__":
    args = get_args()
    fine_tune(args)
