
from PIL import Image
from typing import List
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import numpy as np
import os

def shuffle(nums,noisy_inx,noisy_rate):
    inx = np.arange(nums)
    np.random.shuffle(inx)
    c_noisy_inx = inx[0: int(noisy_rate * nums)]
    shuffle_noisy_inx = np.array(c_noisy_inx)
    np.random.shuffle(shuffle_noisy_inx)
    noisy_inx[c_noisy_inx] = shuffle_noisy_inx
    return noisy_inx

def inject_dual_noise(dataset, noisy_rate,noisy_file =None,noisy_type='TNL'):
    logger = logging.getLogger("0")
    dataset_copy = dataset.copy()
    nums = len(dataset_copy)
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy] # start from 0

    label_noise = np.array(pids.copy())
    correspondence_noise =  np.arange(nums)
    if noisy_rate > 0:
        # NL: generate the random IDs of images proportionally
        # random.seed(1)
        # class_number = len(set(pids))
        # image_number = len(set(image_ids))
        # image_id_set = list(set(image_ids))
        # labels = np.array(pids)
        # image_ids = np.array(image_ids)   
        # inx = np.arange(image_number)
        # np.random.shuffle(inx)
        # c_noisy_inx = inx[0: int(noisy_rate * image_number)]

        # for idx in c_noisy_inx.tolist():
        #     idx_sub = np.argwhere(image_ids==image_id_set[idx])[:,0]
        #     p_id = labels[idx_sub[0]]
        #     s_id = random.sample(labels.tolist(),1)[0]
        #     while p_id == s_id:
        #         s_id = random.sample(labels.tolist(),1)[0]
        #     label_noise[idx_sub] = s_id

        # random.seed(1)
        # class_number = len(set(pids))
        # image_number = len(set(image_ids))
        # image_id_set = list(set(image_ids))
        # labels = np.array(pids)
        # image_ids = np.array(image_ids)   
        # inx = np.arange(image_number)
        # np.random.shuffle(inx)
        # c_noisy_inx = inx[0: int(noisy_rate * image_number)]

        # for idx in c_noisy_inx.tolist():
        #     idx_sub = np.argwhere(image_ids==image_id_set[idx])[:,0]
        #     p_id = labels[idx_sub[0]]
        #     s_id = random.sample(labels.tolist(),1)[0]
        #     while p_id == s_id:
        #         s_id = random.sample(labels.tolist(),1)[0]
        #     label_noise[idx_sub] = s_id
            
        random.seed(1)
        class_number = len(set(pids))
        labels = np.array(pids)
        image_ids = np.array(image_ids) 
        ids_set = list(set(pids))
    
        # for idx in range(class_number):
        #     idx_sub = np.argwhere(labels==idx)[:,0]
        #     s_img_ids = list(set(image_ids[idx_sub]))
        #     idx_nsy = random.sample(s_img_ids, int(np.round(len(s_img_ids)*noisy_rate)))
        #     for i in idx_nsy:
        #         idx_sub_ = np.argwhere(image_ids==i)[:,0]    
        #         s_i = random.sample(ids_set,1)[0]
        #         while idx==s_i:
        #             s_i = random.sample(ids_set,1)[0]
        #         label_noise[idx_sub_] = s_i
   
        # label_noise = label_noise.tolist()
        # image_ids = image_ids.tolist()

        random.seed(1)
        class_number = len(set(pids))
        label = np.array(pids)
        for idx in range(class_number):
            idx_sub = np.argwhere(label==idx)[:,0]
            idx_nsy = np.array(random.sample(idx_sub.tolist(), int(np.round(len(idx_sub)*noisy_rate))))
            for i in idx_nsy:
                tar_lbl = random.sample((list(range(idx))+list(range(idx+1,class_number))),1)
                label_noise[i] = tar_lbl[0]
        label_noise = label_noise.tolist()
        image_ids = image_ids.tolist()

        # NC: generate the random captions of each person images proportionally
        random.seed(2)
        correspondence = correspondence_noise.copy()
        for idx in range(class_number):
            idx_sub = np.argwhere(labels==idx)[:,0]
            idx_nsy = np.array(random.sample(idx_sub.tolist(), int(np.round(len(idx_sub)*noisy_rate)))).tolist()
            correspondence_noise[idx_nsy] = \
                np.array(random.sample([i for i in correspondence.tolist() if i not in idx_sub.tolist()], len(idx_nsy)))

        # random.seed(10)
        # inx = np.arange(nums)
        # np.random.shuffle(inx)
        # c_noisy_inx = inx[0: int(noisy_rate * nums)]
        # for i in c_noisy_inx.tolist():
        #     idx_sub = np.argwhere(labels==labels[i])[:,0]
        #     tar_lbl = random.sample(labels.tolist(),1)[0]
        #     while tar_lbl in idx_sub:
        #         tar_lbl = random.sample(labels.tolist(),1)[0]
        #     correspondence_noise[i] = tar_lbl


        # NC    
        # random.seed(10)
        # inx = np.arange(nums)
        # np.random.shuffle(inx)
        # c_noisy_inx = inx[0: int(noisy_rate * nums)]
        # shuffle_noisy_inx = np.array(c_noisy_inx)
        # np.random.shuffle(shuffle_noisy_inx)
        # correspondence_noise[c_noisy_inx] = shuffle_noisy_inx


    real_correspondeces = np.ones(nums)
    real_labels = np.ones(nums)
    if noisy_type == 'DNL':
        for i in range(nums):
            if  correspondence_noise[i] != i:
                real_correspondeces[i] = 0
            if  label_noise[i] != dataset[i][0]:
                 real_labels[i] = 0
            # pid, real_pid, image_id, image_path, text
            dataset[i] =  (label_noise[i], image_ids[i], images[i], captions[correspondence_noise[i]])
        logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))
        logger.info('=>Noisy rate: {},  clean labels: {}, noisy labels: {}, total labels: {}'.format(noisy_rate, np.sum(real_labels),nums-np.sum(real_labels), nums))

    elif noisy_type == 'NC':
        for i in range(nums):
            if  correspondence_noise[i] != i:
                real_correspondeces[i] = 0
            # pid, real_pid, image_id, image_path, text
            dataset[i] =  (pids[i], image_ids[i],images[i], captions[correspondence_noise[i]])
        logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))
   
    elif noisy_type == 'NL':
        for i in range(nums):
            if  label_noise[i] != dataset[i][0]:
                real_labels[i] = 0
            dataset[i] = (label_noise[i], image_ids[i],images[i], captions[i])
        logger.info('=>Noisy rate: {},  clean labels: {}, noisy labels: {}, total labels: {}'.format(noisy_rate, np.sum(real_labels),nums-np.sum(real_labels), nums))

    del dataset_copy

    logger.info(label_noise[0:10])
    logger.info(correspondence_noise)
    return dataset, real_correspondeces, real_labels

def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    logger = logging.getLogger("IRRA.dataset")
    nums = len(dataset)
    captions  = [i[3] for i in dataset]
    images    = [i[2] for i in dataset]
    image_ids = [i[1] for i in dataset]
    pids      = [i[0] for i in dataset]

    noisy_inx = np.arange(nums)

    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i]== i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))

    return dataset, np.array(real_correspondeces)


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.dataset,self.real_correspondences = inject_noisy_correspondence(dataset,args.noisy_rate,args.noisy_file)
 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset,args.noisy_rate,args.noisy_file)
 
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        # mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.clone().cpu().numpy())
        
        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'index':index
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)