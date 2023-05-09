from __future__ import print_function

import numpy

from data import get_test_loader
import time
import numpy as np
import torch
import tqdm
from collections import OrderedDict
from utils import dot_sim
from utils import get_model as get_model
from evaluate_utils.dcg import DCG
from models.loss import order_sim, AlignmentContrastiveLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.add_scalar(prefix + k, v.val, global_step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    img_lengths = []
    cap_lengths = []

    # compute maximum lenghts in the whole dataset
    max_cap_len = 88
    max_img_len = 37
    # for _, _, img_length, cap_length, _, _ in data_loader:
    #     max_cap_len = max(max_cap_len, max(cap_length))
    #     max_img_len = max(max_img_len, max(img_length))

    for i, (images, targets, img_length, cap_length, boxes, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        if type(targets) == tuple or type(targets) == list:
            captions, features, wembeddings = targets
            # captions = features  # Very weird, I know
            text = features
        else:
            text = targets
            captions = targets
            wembeddings = model.img_txt_enc.txt_enc.word_embeddings(captions.cuda() if torch.cuda.is_available() else captions)

        # compute the embeddings
        with torch.no_grad():
            img_emb_aggr, cap_emb_aggr, img_emb, cap_emb, cap_length = model.forward_emb(images, text, img_length, cap_length, boxes)

            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = torch.zeros((len(data_loader.dataset), max_img_len, img_emb.size(2)))
                cap_embs = torch.zeros((len(data_loader.dataset), max_cap_len, cap_emb.size(2)))

            # preserve the embeddings by copying from gpu and converting to numpy
            img_embs[ids, :img_emb.size(0), :] = img_emb.cpu().permute(1, 0, 2)
            cap_embs[ids, :cap_emb.size(0), :] = cap_emb.cpu().permute(1, 0, 2)
            img_lengths.extend(img_length)
            cap_lengths.extend(cap_length)

            # measure accuracy and record loss
            # model.forward_loss(None, None, img_emb, cap_emb, img_length, cap_length)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions

    # p = np.random.permutation(len(data_loader.dataset) // 5) * 5
    # p = np.transpose(np.tile(p, (5, 1)))
    # p = p + np.array([0, 1, 2, 3, 4])
    # p = p.flatten()
    # img_embs = img_embs[p]
    # cap_embs = cap_embs[p]

    return img_embs, cap_embs, img_lengths, cap_lengths


def evalrank(config, checkpoint, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    # checkpoint = torch.load(model_path)
    data_path = config['dataset']['data']
    measure = config['training']['measure']

    # construct model
    model = get_model(config)
    try:
        model.img_txt_enc.switch_test()
        print("======switch_test======")
    except:
        print("======pass_switch_test======")

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)

    print('Loading dataset')
    data_loader = get_test_loader(config, workers=4, split_name=split)

    # initialize ndcg scorer
    ndcg_val_scorer = DCG(config, len(data_loader.dataset), split, rank=25, relevance_methods=['rougeL', 'spice'])

    # initialize similarity matrix evaluator
    sim_matrix_fn = AlignmentContrastiveLoss(aggregation=config['training']['alignment-mode'], return_similarity_mat=True) if config['training']['loss-type'] == 'alignment' else None

    print('Computing results...')
    img_embs, cap_embs, img_lenghts, cap_lenghts = encode_data(model, data_loader)
    torch.cuda.empty_cache()

    # if checkpoint2 is not None:
    #     # construct model
    #     model2 = get_model(config2)
    #     # load model state
    #     model2.load_state_dict(checkpoint2['model'], strict=False)
    #     img_embs2, cap_embs2 = encode_data(model2, data_loader)
    #     print('Using 2-model ensemble')
    # else:
    #     img_embs2, cap_embs2 = None, None
    #     print('Using NO ensemble')

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        r, rt = i2t(img_embs, cap_embs, img_lenghts, cap_lenghts,
                    return_ranks=True, ndcg_scorer=ndcg_val_scorer, sim_function=sim_matrix_fn, cap_batches=5)
        ri, rti = t2i(img_embs, cap_embs, img_lenghts, cap_lenghts,
                    return_ranks=True, ndcg_scorer=ndcg_val_scorer, sim_function=sim_matrix_fn, im_batches=5)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                         img_lenghts[i * 5000:(i + 1) * 5000], cap_lenghts[i * 5000:(i + 1) * 5000],
                         return_ranks=True, ndcg_scorer=ndcg_val_scorer, fold_index=i, sim_function=sim_matrix_fn, cap_batches=1)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f ndcg_spice=%.4f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000], cap_embs[i * 5000:(i + 1) * 5000],
                           img_lenghts[i * 5000:(i + 1) * 5000], cap_lenghts[i * 5000:(i + 1) * 5000],
                           return_ranks=True, ndcg_scorer=ndcg_val_scorer, fold_index=i, sim_function=sim_matrix_fn, im_batches=1)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, ndcg_rouge=%.4f, ndcg_spice=%.4f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[16] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[14])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[15])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[7:12])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

# @profile
def i2t(images, captions, img_lenghts, cap_lenghts, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0, measure='dot', sim_function=None, cap_batches=1):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    rougel_ndcgs = numpy.zeros(npts)
    spice_ndcgs = numpy.zeros(npts)

    # captions = captions.cuda()
    images_cpu = images.cpu()
    images = images.cuda()
    for index in tqdm.trange(npts):

        # Get query image
        im = images_cpu[5 * index].reshape(1, images.shape[1], images.shape[2])
        im_len = [img_lenghts[5 * index]]

        # global dis
        d_g = torch.mm(im[:, 0, :], captions[:, 0, :].t())
        d_g = d_g.cpu().numpy().flatten()
        g_inds = numpy.argsort(d_g)[::-1]
        top_g_inds = list(g_inds[0:50])
        d_m = d_g[top_g_inds]

        # local dis
        im = images[5 * index].reshape(1, images.shape[1], images.shape[2])
        captions_now = captions[top_g_inds]
        captions_now = captions_now.cuda()
        cap_lenghts_now = list(numpy.array(cap_lenghts)[top_g_inds])
        d_l = sim_function(im, captions_now, im_len, cap_lenghts_now)
        d_l = d_l.cpu().numpy().flatten()
        d_f = d_l+d_m
        l_inds = numpy.argsort(d_f)[::-1]
        inds = g_inds[l_inds]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        tmp_inss = []
        for i in range(5 * index, 5 * index + 5, 1):
            tmp_ins = list(numpy.where(inds == i))
            if len(tmp_ins[0])<=0:
                continue
            tmp_inss.append(tmp_ins[0][0])
        if len(tmp_inss)<=0:
            tmp_inss.append(0)
        tmp = min(tmp_inss)
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if ndcg_scorer is not None:
            rougel_ndcgs[index], spice_ndcgs[index] = ndcg_scorer.compute_ndcg(npts, index, inds.astype(int),
                                                                               fold_index=fold_index,
                                                                               retrieval='sentence').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = np.mean(rougel_ndcgs[rougel_ndcgs != 0])
    mean_spice_ndcg = np.mean(spice_ndcgs[spice_ndcgs != 0])
    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)


def t2i(images, captions, img_lenghts, cap_lenghts, npts=None, return_ranks=False, ndcg_scorer=None, fold_index=0, measure='dot', sim_function=None, im_batches=1):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = torch.stack([images[i] for i in range(0, len(images), 5)], dim=0)
    # ims = ims.cuda()
    ims_len = [img_lenghts[i] for i in range(0, len(images), 5)]

    ranks = numpy.zeros(5 * npts)
    top50 = numpy.zeros((5 * npts, 5))
    rougel_ndcgs = numpy.zeros(5 * npts)
    spice_ndcgs = numpy.zeros(5 * npts)

    images_per_batch = ims.shape[0] // im_batches
    ims = ims.cuda()
    # captions = captions.cuda()
    for index in tqdm.trange(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]
        queries_len = cap_lenghts[5 * index:5 * index + 5]
        queries = queries.cuda()

        # Compute scores
        d_g = torch.mm(queries[:, 0, :], ims[:, 0, :].t())
        d_g = d_g.cpu().numpy()

        inds = numpy.zeros((len(d_g),100))
        for i in range(len(d_g)):
            di_g = d_g[i]
            g_inds = numpy.argsort(di_g)[::-1]
            cap_inds = list(g_inds[0:100])
            quer = queries[i]
            quer = quer.unsqueeze(0)
            quer_len = [queries_len[i]]
            ims_now = ims[cap_inds]
            ims_len_now = list(numpy.array(ims_len)[cap_inds])
            d_l = sim_function(ims_now, quer, ims_len_now, quer_len).t()
            d_l = d_l.cpu().numpy()[0]
            d_m = di_g[cap_inds]
            d_f = d_l+d_m
            l_inds = numpy.argsort(d_f)[::-1]
            inds[i] = g_inds[l_inds]
            r_r = numpy.where(inds[i] == index)[0]
            if len(r_r)<=0:
                ranks[5 * index + i]=0
            else:
                ranks[5 * index + i] = r_r[0]
            top50[5 * index + i] = inds[i][0:5]
            if ndcg_scorer is not None:
                rougel_ndcgs[5 * index + i], spice_ndcgs[5 * index + i] = \
                    ndcg_scorer.compute_ndcg(npts, 5 * index + i, inds[i].astype(int),
                                             fold_index=fold_index, retrieval='image').values()

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    mean_rougel_ndcg = np.mean(rougel_ndcgs)
    mean_spice_ndcg = np.mean(spice_ndcgs)

    if return_ranks:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg), (ranks, top50)
    else:
        return (r1, r5, r10, medr, meanr, mean_rougel_ndcg, mean_spice_ndcg)

