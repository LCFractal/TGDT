import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer

from models.loss import ContrastiveLoss, PermInvMatchingLoss, AlignmentContrastiveLoss
from models.text import EncoderTextBERT, EncoderText
from models.visual import TransformerPostProcessing, EncoderImage

from .utils import l2norm, PositionalEncodingImageBoxes, PositionalEncodingText, Aggregator, generate_square_subsequent_mask
from nltk.corpus import stopwords, words as nltk_words


class JointTextImageTransformerEncoder(nn.Module):
    """
    This is a bert caption encoder - transformer image encoder (using bottomup features).
    If process the encoder outputs through a transformer, like VilBERT and outputs two different graph embeddings
    """
    def __init__(self, config):
        super().__init__()
        self.txt_enc = EncoderText(config)
        self.istest = False
        visual_feat_dim = config['image-model']['feat-dim']
        caption_feat_dim = config['text-model']['word-dim']
        dropout = config['model']['dropout']
        layers = config['model']['layers']
        embed_size = config['model']['embed-size']
        self.order_embeddings = config['training']['measure'] == 'order'
        self.img_enc = EncoderImage(config)

        self.img_proj = nn.Linear(visual_feat_dim, embed_size)
        self.cap_proj = nn.Linear(caption_feat_dim, embed_size)
        self.embed_size = embed_size
        self.shared_transformer = config['model']['shared-transformer']

        transformer_layer_1 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                       dim_feedforward=2048,
                                                       dropout=dropout, activation='relu')
        self.transformer_encoder_1 = nn.TransformerEncoder(transformer_layer_1,
                                                           num_layers=layers)
        if not self.shared_transformer:
            transformer_layer_2 = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                             dim_feedforward=2048,
                                                             dropout=dropout, activation='relu')
            self.transformer_encoder_2 = nn.TransformerEncoder(transformer_layer_2,
                                                               num_layers=layers)

        self.text_aggregation = Aggregator(embed_size, aggregation_type=config['model']['text-aggregation'])
        self.image_aggregation = Aggregator(embed_size, aggregation_type=config['model']['image-aggregation'])
        self.text_aggregation_type = config['model']['text-aggregation']
        self.img_aggregation_type = config['model']['image-aggregation']

        ## add
        self.text_agg_emb = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size, embed_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size//2, embed_size)
        )
        self.img_agg_emb = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size, embed_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size//2, embed_size)
        )

        transformer_layer_text = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                         dim_feedforward=2048,
                                                         dropout=dropout, activation='relu')
        self.transformer_encoder_text = nn.TransformerEncoder(transformer_layer_text,
                                                           num_layers=layers)

        transformer_layer_img = nn.TransformerEncoderLayer(d_model=embed_size, nhead=4,
                                                         dim_feedforward=2048,
                                                         dropout=dropout, activation='relu')
        self.transformer_encoder_img = nn.TransformerEncoder(transformer_layer_img,
                                                           num_layers=layers)
        ## add

    def switch_eval(self):
        self.txt_enc.eval()
        self.img_enc.eval()
        self.cap_proj.eval()
        self.transformer_encoder_1.eval()
        self.transformer_encoder_2.eval()
        self.istest = False

    def switch_test(self):
        # self.txt_enc.eval()
        # self.img_enc.eval()
        # self.cap_proj.eval()
        # self.transformer_encoder_1.eval()
        # self.transformer_encoder_2.eval()
        self.istest = True

    def switch_train(self):
        self.txt_enc.train()
        self.img_enc.train()
        self.cap_proj.train()
        self.transformer_encoder_1.train()
        self.transformer_encoder_2.train()
        self.istest = False

    def forward(self, features, captions, feat_len, cap_len, boxes):
        # process captions by using bert
        full_cap_emb_aggr, c_emb = self.txt_enc(captions, cap_len)     # B x S x cap_dim

        # process image regions using a two-layer transformer
        full_img_emb_aggr, i_emb = self.img_enc(features, feat_len, boxes)     # B x S x vis_dim
        # i_emb = i_emb.permute(1, 0, 2)                             # B x S x vis_dim

        bs = features.shape[0]

        # forward the captions
        if self.text_aggregation_type is not None:
            c_emb = self.cap_proj(c_emb)

            mask = torch.zeros(bs, max(cap_len)).bool()
            mask = mask.to(features.device)
            for m, c_len in zip(mask, cap_len):
                m[c_len:] = True
            full_cap_emb = self.transformer_encoder_1(c_emb.permute(1, 0, 2), src_key_padding_mask=mask)  # S_txt x B x dim
            full_cap_emb = full_cap_emb.detach()
            full1_cap_emb = self.transformer_encoder_text(full_cap_emb)
            full_cap_emb_aggr = self.text_aggregation(full1_cap_emb, cap_len, mask)
        # else use the embedding output by the txt model
        else:
            full_cap_emb = None
            full1_cap_emb = None

        # forward the regions
        if self.img_aggregation_type is not None:
            i_emb = self.img_proj(i_emb)

            mask = torch.zeros(bs, max(feat_len)).bool()
            mask = mask.to(features.device)
            for m, v_len in zip(mask, feat_len):
                m[v_len:] = True
            if self.shared_transformer:
                full_img_emb = self.transformer_encoder_1(i_emb.permute(1, 0, 2), src_key_padding_mask=mask)  # S_txt x B x dim
            else:
                full_img_emb = self.transformer_encoder_2(i_emb.permute(1, 0, 2), src_key_padding_mask=mask)  # S_txt x B x dim
            full_img_emb = full_img_emb.detach()
            full1_img_emb = self.transformer_encoder_img(full_img_emb)
            full_img_emb_aggr = self.image_aggregation(full1_img_emb, feat_len, mask)
        else:
            full_img_emb = None
            full1_img_emb = None

        ### add
        # full_cap_emb_aggr = self.text_agg_emb(full_cap_emb_aggr)
        # full_img_emb_aggr = self.img_agg_emb(full_img_emb_aggr)

        # full_cap_emb = self.transformer_encoder_text(full_cap_emb)
        # full_img_emb = self.transformer_encoder_img(full_img_emb)
        ###

        full_cap_emb_aggr = l2norm(full_cap_emb_aggr)
        full_img_emb_aggr = l2norm(full_img_emb_aggr)

        if self.istest:
            # local task
            # normalize even every vector of the set
            full_img_emb = F.normalize(full_img_emb, p=2, dim=2)
            full_cap_emb = F.normalize(full_cap_emb, p=2, dim=2)
        else:
            # global task
            # normalize even every vector of the set
            full_img_emb = F.normalize(full1_img_emb, p=2, dim=2)
            full_cap_emb = F.normalize(full1_cap_emb, p=2, dim=2)

        if self.order_embeddings:
            full_cap_emb_aggr = torch.abs(full_cap_emb_aggr)
            full_img_emb_aggr = torch.abs(full_img_emb_aggr)

        full_img_emb[0]=full_img_emb_aggr
        full_cap_emb[0]=full_cap_emb_aggr

        return full_img_emb_aggr, full_cap_emb_aggr, full_img_emb, full_cap_emb


class BASELINE(torch.nn.Module):
    def __init__(self, config):
        # Build Models
        super().__init__()
        self.img_txt_enc = JointTextImageTransformerEncoder(config)
        if torch.cuda.is_available():
            self.img_txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        loss_type = config['training']['loss-type']
        if 'alignment' in loss_type:
            self.alignment_criterion = AlignmentContrastiveLoss(margin=config['training']['margin'],
                                                                measure=config['training']['measure'],
                                                                max_violation=config['training']['max-violation'], aggregation=config['training']['alignment-mode'])
        if 'matching' in loss_type:
            self.matching_criterion = ContrastiveLoss(margin=config['training']['margin'],
                                                      measure=config['training']['measure'],
                                                      max_violation=config['training']['max-violation'])

        embed_size = config['model']['embed-size']
        self.img_first_emb = nn.Linear(embed_size, embed_size)
        self.cap_first_emb = nn.Linear(embed_size, embed_size)

        self.Eiters = 0
        self.config = config

        if 'exclude-stopwords' in config['model'] and config['model']['exclude-stopwords']:
            self.en_stops = set(stopwords.words('english'))
            self.tokenizer = BertTokenizer.from_pretrained(config['text-model']['pretrain'])
        else:
            self.tokenizer = None

    def forward_emb(self, images, captions, img_len, cap_len, boxes):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            boxes = boxes.cuda()

        # Forward
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats = self.img_txt_enc(images, captions, img_len, cap_len, boxes)

        if self.tokenizer is not None:
            # remove stopwords
            # keep only word indexes that are not stopwords
            good_word_indexes = [[i for i, (tok, w) in enumerate(zip(self.tokenizer.convert_ids_to_tokens(ids), ids)) if
                                  tok not in self.en_stops or w == 0] for ids in captions]  # keeps the padding
            cap_len = [len(w) - (cap_feats.shape[0] - orig_len) for w, orig_len in zip(good_word_indexes, cap_len)]
            min_cut_len = min([len(w) for w in good_word_indexes])
            good_word_indexes = [words[:min_cut_len] for words in good_word_indexes]
            good_word_indexes = torch.LongTensor(good_word_indexes).to(cap_feats.device) # B x S
            good_word_indexes = good_word_indexes.t().unsqueeze(2).expand(-1, -1, cap_feats.shape[2]) # S x B x dim
            cap_feats = cap_feats.gather(dim=0, index=good_word_indexes)

        return img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, cap_len

    def get_parameters(self):
        lr_multiplier = 1.0 if self.config['text-model']['fine-tune'] else 0.0

        ret = []
        params = list(self.img_txt_enc.img_enc.parameters())
        params += list(self.img_txt_enc.img_proj.parameters())
        params += list(self.img_txt_enc.cap_proj.parameters())
        params += list(self.img_txt_enc.transformer_encoder_1.parameters())

        params += list(self.img_txt_enc.image_aggregation.parameters())
        params += list(self.img_txt_enc.text_aggregation.parameters())

        if not self.config['model']['shared-transformer']:
            params += list(self.img_txt_enc.transformer_encoder_2.parameters())

        params += list(self.img_first_emb.parameters())
        params += list(self.cap_first_emb.parameters())
        params += list(self.img_txt_enc.text_agg_emb.parameters())
        params += list(self.img_txt_enc.img_agg_emb.parameters())
        params += list(self.img_txt_enc.transformer_encoder_text.parameters())
        params += list(self.img_txt_enc.transformer_encoder_img.parameters())

        ret.append(params)

        ret.append(list(self.img_txt_enc.txt_enc.parameters()))

        return ret, lr_multiplier

    def forward_loss(self, img_emb, cap_emb, img_emb_set, cap_emb_seq, img_lengths, cap_lengths):
        """Compute the loss given pairs of image and caption embeddings
        """
        # bs = img_emb.shape[0]
        losses = {}

        if 'matching' in self.config['training']['loss-type']:
            matching_loss = self.matching_criterion(img_emb, cap_emb)
            losses.update({'matching-loss': matching_loss})
            self.logger.update('matching_loss', matching_loss.item(), img_emb.size(0))

        if 'alignment' in self.config['training']['loss-type']:
            img_emb_set = img_emb_set.permute(1, 0, 2)
            cap_emb_seq = cap_emb_seq.permute(1, 0, 2)
            alignment_loss = self.alignment_criterion(img_emb_set, cap_emb_seq, img_lengths, cap_lengths)
            losses.update({'alignment-loss': alignment_loss})
            self.logger.update('alignment_loss', alignment_loss.item(), img_emb_set.size(0))

        # self.logger.update('Le', matching_loss.item() + alignment_loss.item(), img_emb.size(0) if img_emb is not None else img_emb_set.size(1))
        return losses

    def forward(self, images, targets, img_lengths, cap_lengths, boxes=None, ids=None, *args):
        """One training step given images and captions.
        """
        # assert self.training()
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets
            wembeddings = self.img_txt_enc.txt_enc.word_embeddings(captions.cuda() if torch.cuda.is_available() else captions)

        # compute the embeddings
        img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, cap_lengths = self.forward_emb(images, text, img_lengths, cap_lengths, boxes)
        # NOTE: img_feats and cap_feats are S x B x dim

        loss_dict = self.forward_loss(img_emb_aggr, cap_emb_aggr, img_feats, cap_feats, img_lengths, cap_lengths)
        return loss_dict
