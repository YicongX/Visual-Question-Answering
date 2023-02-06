"""Base experiment runner class for VQA experiments."""
import argparse
import os

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models import BaselineNet, BaselineNetDeep, TransformerNet
from vqa_dataset import VQADataset


class Trainer:
    """Train/test models on manipulation."""

    def __init__(self, model, data_loaders, args):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args

        self.writer = SummaryWriter('runs/' + args.tensorboard_dir)

        self.optimizer = Adam(
            model.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8
        )

        self._id2answer = {
            v: k
            for k, v in data_loaders['val'].dataset.answer_to_id_map.items()
        }
        self._id2answer[len(self._id2answer)] = 'Other'

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def run(self):
        # Set parameters
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load checkpoint
        if os.path.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()

        # Evaluate model
        if self.args.eval or start_epoch >= self.args.epochs:
            self.model.eval()
            self.train_test_loop('val')
            return self.model

        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            print("\nValidation")
            self.model.eval()
            with torch.no_grad():
                val_acc = self.train_test_loop('val', epoch)

            # Store checkpoints
            if val_acc >= val_acc_prev_best:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": val_acc
                }, self.args.ckpnt)
                val_acc_prev_best = val_acc
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(self.args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, self.args.ckpnt)

        return self.model

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        return start_epoch, val_acc_prev_best

    def train_test_loop(self, mode='train', epoch=1000):
        n_correct, n_samples = 0, 0
        pred_sum = np.zeros(5217)
        gt_sum = np.zeros(5217)
        for step, data in tqdm(enumerate(self.data_loaders[mode])):

            # Forward pass
            scores = self.model(
                data['image'].to(self.args.device),
                data['question']
            )
            answers = data['answers'].to(self.args.device)

            # Losses
            
            loss = self.criterion(scores,answers)

            # Update
            if mode == 'train':
                
                # optimize loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            # Accuracy
            n_samples += len(scores)
            found = (
                F.one_hot(scores.argmax(1), scores.size(1))
                * answers
            ).sum(1)
            n_correct += (
                F.one_hot(scores.argmax(1), scores.size(1))
                * answers
            ).sum().item()  # checks if argmax matches any ground-truth

            # Logging
            self.writer.add_scalar(
                'Loss/' + mode, loss.item(),
                epoch * len(self.data_loaders[mode]) + step
            )
            if mode == 'val' and step == 10:  # change this to show other images
                _n_show = 3  # how many images to plot
                for i in range(_n_show):
                    self.writer.add_image(
                        'Image%d' % i, data['orig_img'][i].cpu().numpy(),
                        epoch * _n_show + i, dataformats='CHW'
                    )
                    self.writer.add_text('Question %d' % i,data['question'][i])
                    gt_answer = torch.where(data['answers'][i]==1)[0].cpu().numpy()
                    for j in range(len(gt_answer)):
                        self.writer.add_text('GT Answer %d' % i,self._id2answer[gt_answer[j]])
                    pred_answer = scores.argmax(1).cpu().numpy()[i]
                    if pred_answer not in gt_answer:
                        self.writer.add_text('Predicted Answer %d' % i,self._id2answer[pred_answer])
            # class frequency
            batch_pred = F.one_hot(scores.argmax(1), scores.size(1))
            pred_sum = np.add(pred_sum,batch_pred.sum(0).cpu().numpy()) 
            gt_sum = np.add(gt_sum, answers.sum(0).cpu().numpy())

        acc = n_correct / n_samples
        print(acc)
        self.writer.add_scalar(
            'Accuracy/' + mode, acc,
            epoch
        )
        gt_freq = gt_sum/gt_sum.sum()
        
        pred_freq = pred_sum/pred_sum.sum()
        
        cls = np.arange(0,5218,1,dtype=int)
        cls = cls.astype(str)
        gt_dic = {}
        pred_dic = {}
        for A, B in zip(cls, gt_freq):
            gt_dic[A] = B
        gt_sorted = {k: v for k, v in sorted(gt_dic.items(), key=lambda item: item[1],reverse=True)}
        for A, B in zip(cls, pred_freq):
            pred_dic[A] = B
        pred_sorted = {k: v for k, v in sorted(pred_dic.items(), key=lambda item: item[1],reverse=True)}
        # plt.bar(range(len(pred_sorted)),pred_sorted.values())
        # plt.xlabel('Answers')
        # plt.title("Prdicted answer frequency")
        # plt.ylabel('Answer Frequency')
        # plt.savefig('Prdicted_answer_frequency.png')

        # pred_lable = []
        # gt_lable = []
        # for i in range(10):
        #     pred_lable.append(self._id2answer[int(list(pred_sorted.keys())[i])])
        #     gt_lable.append(self._id2answer[int(list(gt_sorted.keys())[i])])
        # plt.figure()
        # plt.bar(pred_lable,list(pred_sorted.values())[:10])
        # plt.xlabel('Answers')
        # plt.title("Prdicted answer frequency")
        # plt.ylabel('Answer Frequency')
        # plt.savefig('BPrdicted_answer_frequency.png')
        
        # plt.figure()
        # plt.bar(gt_lable,list(gt_sorted.values())[:10])
        # plt.xlabel('Answers')
        # plt.title("GT answer frequency")
        # plt.ylabel('Answer Frequency')
        # plt.savefig('BGT_answer_frequency.png')
        return acc


def main():
    """Run main training/test pipeline."""
    
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    parser.add_argument('--ckpnt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    data_path = args.data_path

    # Other variables
    args.train_image_dir = data_path + 'train2014/'
    args.train_q_path = data_path + 'OpenEnded_mscoco_train2014_questions.json'
    args.train_anno_path = data_path + 'mscoco_train2014_annotations.json'
    args.test_image_dir = data_path + 'val2014/'
    args.test_q_path = data_path + 'OpenEnded_mscoco_val2014_questions.json'
    args.test_anno_path = data_path + 'mscoco_val2014_annotations.json'
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.model
    if args.ckpnt is None:
        args.ckpnt = args.model + '.pt'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Loaders
    train_dataset = VQADataset(
        image_dir=args.train_image_dir,
        question_json_file_path=args.train_q_path,
        annotation_json_file_path=args.train_anno_path,
        image_filename_pattern="COCO_train2014_{}.jpg"
    )
    val_dataset = VQADataset(
        image_dir=args.test_image_dir,
        question_json_file_path=args.test_q_path,
        annotation_json_file_path=args.test_anno_path,
        image_filename_pattern="COCO_val2014_{}.jpg",
        answer_to_id_map=train_dataset.answer_to_id_map
    )
    print(len(train_dataset), len(val_dataset))
    data_loaders = {
        mode: DataLoader(
            train_dataset if mode == 'train' else val_dataset,
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
        )
        for mode in ('train', 'val')
    }

    # Models
    if args.model == "simple":
        model = BaselineNet()
    elif args.model == "deep":
        model = BaselineNetDeep()
    elif args.model == "transformer":
        model = TransformerNet()
    else:
        raise ModuleNotFoundError()

    trainer = Trainer(model.to(args.device), data_loaders, args)
    trainer.run()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
