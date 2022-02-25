import argparse
import utils
import time
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import auc
import json
from model import BERTModel

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = AsymmetricLossOptimized(gamma_neg=3)
        bert_param_ids = list(map(id, model.bert.parameters()))
        base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        self.optimizer = optim.AdamW([
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': args.bert_lr,
             'weight_decay': args.wd},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': args.bert_lr,
             'weight_decay': 0.0},
            {'params': base_params, 'weight_decay': args.wd}
        ], lr=args.lr, weight_decay=args.wd)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train(self, data_loader):
        self.model.train()
        loss_list = []
        for i, data_batch in enumerate(data_loader):
            if args.use_gpu:
                data_batch = [data.cuda() for data in data_batch[:-1]]
            doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, intrain_mask,\
            doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask = data_batch

            outputs = model(doc_inputs, psn_inputs, ner_inputs, dis_inputs,
                               doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask)

            rel_mask = ent2ent_mask.clone()
            rel_mask[:, range(rel_mask.size(1)), range(rel_mask.size(2))] = 0

            loss = self.criterion(outputs[rel_mask], rel_labels[rel_mask])
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # a_losses.append(a_loss.cpu().item())
            loss_list.append(loss.cpu().item())
        self.scheduler.step()
        print("Loss {:.4f}".format(np.mean(loss_list)))

    def eval(self, data_loader):
        self.model.eval()

        test_result = []
        label_result = []
        intrain_list = []
        total_recall = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                if args.use_gpu:
                    data_batch = [data.cuda() for data in data_batch[:-1]]
                doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, intrain_mask,\
                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask = data_batch

                outputs = model(doc_inputs, psn_inputs, ner_inputs, dis_inputs,
                                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask)
                outputs = torch.sigmoid(outputs)

                rel_mask = ent2ent_mask.clone()
                rel_mask[:, range(rel_mask.size(1)), range(rel_mask.size(2))] = 0

                labels = rel_labels[..., 1:][rel_mask].contiguous().view(-1)
                outputs = outputs[..., 1:][rel_mask].contiguous().view(-1)
                intrain_mask = intrain_mask[..., 1:][rel_mask].contiguous().view(-1)
                label_result.append(labels)
                test_result.append(outputs)
                intrain_list.append(intrain_mask)
                total_recall += labels.sum().item()

        label_result = torch.cat(label_result)
        test_result = torch.cat(test_result)
        test_result, indices = torch.sort(test_result, descending=True)
        correct = np.cumsum(label_result[indices].cpu().numpy(), dtype=np.float)
        pr_x = correct / total_recall
        pr_y = correct / np.arange(1, len(correct)+1)

        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        # f1 = f1_arr.max()
        # f1_pos = f1_arr.argmax()
        # theta = test_result[f1_pos].cpu().item()

        auc_score = auc(x=pr_x, y=pr_y)

        intrain_list = torch.cat(intrain_list)
        intrain = np.cumsum(intrain_list[indices].cpu().numpy(), dtype=np.int)
        nt_pr_y = (correct - intrain)
        nt_pr_y[nt_pr_y != 0] /= (np.arange(1, len(correct)+1) - intrain)[nt_pr_y != 0]

        nt_f1_arr = (2 * pr_x * nt_pr_y / (pr_x + nt_pr_y + 1e-20))
        nt_f1 = nt_f1_arr.max()
        nt_f1_pos = nt_f1_arr.argmax()
        theta = test_result[nt_f1_pos].cpu().item()

        logger.info( 'ALL : NT F1 {:3.4f} | F1 {:3.4f} | Precision {:3.4f} | Recall {:3.4f} | AUC {:3.4f} | THETA {:3.4f}'.format(nt_f1, f1_arr[nt_f1_pos], pr_x[nt_f1_pos], pr_y[nt_f1_pos], auc_score, theta))
        return nt_f1, theta

    def test(self, data_loader, theta):
        self.model.eval()

        test_result = []
        with torch.no_grad():

            for i, data_batch in enumerate(data_loader):
                title = data_batch[-1]
                if args.use_gpu:
                    data_batch = [data.cuda() for data in data_batch[:-1]]
                doc_inputs, psn_inputs, ner_inputs, dis_inputs, rel_labels, _, \
                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask = data_batch

                outputs = model(doc_inputs, psn_inputs, ner_inputs, dis_inputs,
                                doc2ent_mask, doc2men_mask, men2ent_mask, ent2ent_mask, men2men_mask)
                outputs = torch.sigmoid(outputs)
                outputs = outputs.cpu().numpy()

                rel_mask = ent2ent_mask.clone()
                rel_mask[:, range(rel_mask.size(1)), range(rel_mask.size(2))] = 0

                for j in range(doc_inputs.size(0)):
                    L = torch.sum(ent2ent_mask[j, 0]).item()
                    for h_idx in range(L):
                        for t_idx in range(L):
                            if h_idx != t_idx:
                                for r in range(1, 97):
                                    test_result.append((float(outputs[j, h_idx, t_idx, r]), title[j], vocab.id2rel[r], h_idx, t_idx, r))

        test_result.sort(key=lambda x: x[0], reverse=True)

        w = 0
        for i, item in enumerate(test_result):
            if item[0] > theta:
                w = i

        output = [{'h_idx': x[3], 't_idx': x[4], 'r_idx': x[-1], 'r': x[-4], 'title': x[1]}
                  for x in test_result[:w + 1]]
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(output, f)


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-log_dir', type=str, default='./log/')

    parser.add_argument('-tok_emb_size', type=int, default=768)
    parser.add_argument('-ner_emb_size', type=int, default=20)
    parser.add_argument('-pos_emb_size', type=int, default=20)
    parser.add_argument('-dis_emb_size', type=int, default=20)

    parser.add_argument('-hid_size', type=int, default=768)
    parser.add_argument('-channels', type=int, default=128)
    parser.add_argument('-layers', type=int, default=3)
    parser.add_argument('-chunk', type=int, default=16)
    parser.add_argument('-dropout1', type=float, default=0.5)
    parser.add_argument('-dropout2', type=float, default=0.33)
    parser.add_argument('-epochs', type=int, default=25)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-freeze_epochs', type=int, default=50)

    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-bert_lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('-wd', type=float, default=1e-5, help='weight deacy')

    parser.add_argument('-use_gpu', type=int, default=1)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1)
    args = parser.parse_args()
    logger = utils.get_logger(args.log_dir + "Baseline_{}.txt".format(time.strftime("%m-%d_%H-%M-%S")))
    logger.info(args)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.set_device(args.device)


    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


    logger.info("Loading Data")
    train_set, dev_set, test_dataset, vocab = utils.load_data(True)

    train_loader, dev_loader = (
        DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn, shuffle=i == 0, num_workers=4, drop_last=i == 0)
        for i, dataset in enumerate([train_set, dev_set])
    )
    updates_total = len(train_loader) // args.batch_size * args.epochs
    logger.info("Loading Embedding")
    logger.info("Building Model")
    model = BERTModel(vocab_size=len(vocab),
                      tok_emb_size=args.tok_emb_size,
                      ner_emb_size=args.ner_emb_size,
                      pos_emb_size=args.pos_emb_size,
                      dis_emb_size=args.dis_emb_size,
                      hid_size=args.hid_size,
                      channels=args.channels,
                      layers=args.layers,
                      dropout1=args.dropout1,
                      dropout2=args.dropout2,
                      chunk=args.chunk,
                      )

    if args.use_gpu:
        model = model.cuda()

    trainer = Trainer(model)

    best_F1 = 0
    input_theta = 0
    for i in range(args.epochs):
        print("Epoch: {}".format(i))
        trainer.train(train_loader)
        interval = 1
        if (i + 1) % interval == 0:
            f1, theta = trainer.eval(dev_loader)
            if f1 > best_F1:
                best_F1 = f1
                input_theta = theta
                trainer.save("model.pt")
    logger.info("Best F1: {:3.4f}".format(best_F1))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn, num_workers=4)
    trainer.load("model.pt")
    logger.info(input_theta)
    trainer.test(test_loader, input_theta)