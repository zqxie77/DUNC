from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def mse_loss(label, alpha, c, lambda2):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    C = lambda2 * KL(alp, c)
    return (A + B) + C


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.step = 0
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward_sims(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        i_feats = i_feats / i_feats.norm(dim=1, keepdim=True)
        t_feats = t_feats / t_feats.norm(dim=1, keepdim=True)

        return i_feats.mm(t_feats.t())

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        if 'hinge' in self.current_task:
            ret.update({'hinge_loss': objectives.compute_hinge(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'decl' in self.current_task:
            i_feats = i_feats / i_feats.norm(dim=1, keepdim=True)
            t_feats = t_feats / t_feats.norm(dim=1, keepdim=True)
            self.tau = 0.05
            sims = i_feats.mm(t_feats.t())
            if self.epoch <= 2:
                loss_edl, loss_rdh = self.warmup_batch(sims)
            else:
                pred = batch['preds'].cuda()
                loss_edl, loss_rdh = self.train_batch(sims, pred)
            ret.update({'loss_edl': loss_edl})
            ret.update({'loss_rdh': loss_rdh})

        return ret

    def RDH_loss(self, scores, neg=None):
        if neg is None:
            neg = 5
        margin = 0.2
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (margin + scores - d1).clamp(min=0)
        cost_im = (margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        top_neg_row = torch.topk(cost_s, k=neg, dim=1).values
        top_neg_column = torch.topk(cost_im.t(), k=neg, dim=1).values
        return (top_neg_row.sum(dim=1) + top_neg_column.sum(dim=1)) / neg  # (K,1)

    def warmup_batch(self, sims):
        self.step += 1
        batch_length = sims.size(0)
        neg = max(int(64 - 0.001 * self.step), 5)
        if batch_length < neg:
            neg = batch_length - 1

        alpha_i2t = (sims / self.tau).exp() + 1
        alpha_t2i = (sims.t() / self.tau).exp() + 1
        sims_tanh = sims

        batch_labels = torch.eye(batch_length).cuda().long()
        loss_edl = mse_loss(batch_labels, alpha_i2t, batch_length, 0.1)
        loss_edl += mse_loss(batch_labels, alpha_t2i, batch_length, 0.1)
        loss_edl = torch.mean(loss_edl)

        loss_rdh = self.RDH_loss(sims_tanh, neg=neg)
        loss_rdh = loss_rdh.sum() * 0.8

        return loss_edl, loss_rdh

    def train_batch(self, sims, preds):
        self.step += 1
        batch_length = sims.size(0)
        neg = max(int(64 - 0.001 * self.step), 5)
        if batch_length < neg:
            neg = batch_length - 1

        alpha_i2t = (sims / self.tau).exp() + 1
        alpha_t2i = (sims.t() / self.tau).exp() + 1
        sims_tanh = sims

        preds = preds.cuda()
        batch_labels = torch.eye(batch_length)
        n_idx = (1 - preds).nonzero().view(1, -1)[0].tolist()
        c_idx = preds.nonzero().view(1, -1)[0].tolist()
        for i in n_idx:
            batch_labels[i][i] = 0
        batch_labels = batch_labels.cuda().long()
        loss_edl = mse_loss(batch_labels, alpha_i2t, batch_length, 0.1)
        loss_edl += mse_loss(batch_labels, alpha_t2i, batch_length, 0.1)
        loss_edl = torch.mean(loss_edl)
        if len(c_idx) == 0:
            loss_rdh = torch.tensor(0).cuda()

        else:
            loss_rdh = self.RDH_loss(sims_tanh, neg=neg)
            loss_rdh = loss_rdh[c_idx]
            loss_rdh = loss_rdh.sum() * 0.8

        return loss_edl, loss_rdh


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
