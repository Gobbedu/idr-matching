import torch
from torch.utils import data
from torch.autograd import Variable
import numpy
from PIL import Image

import graf
import fiim
import pool
import proc
from tide import time_spent
import file_handling

import tools
import morph

import os
import bob.ip.gabor as big

from sklearn.metrics import accuracy_score, precision_score, classification_report, auc, roc_curve, plot_roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

subset_name = '_j-smol/'
base_dir = './data/'
input_dir = '1-roi/'
seg_subdir = '2-seg/'
wave_subdir = '3-wave/'
seg_model_file = "./models/x_grey_3000.pth"

def test_seg(data_loader, model):
    for i, data_batch in enumerate(data_loader):
        print(data_batch['image'].size())
        print(data_batch['image'][0])
        yield seg_image(data_batch['image'], model)

@time_spent
def seg_image(image, model):
    with torch.no_grad():
        image = Variable(image)
        pred = model(image)
        pred = numpy.squeeze(torch.sigmoid(pred).cpu().data.numpy())
        multi_fuse = 255 * pred
        return multi_fuse


if __name__ == "__main__":
    data = file_handling.get_list_of_files(base_dir + input_dir + subset_name)
    train_im, test_im, train_im_gray, test_im_gray, train_labels, test_labels, train_paths, test_paths = fiim.split_same(data)

    print("# 2.1. prepare segmentation model.")
    seg_model = pool.build_model("resnet")
    seg_model.load_state_dict(torch.load(seg_model_file, map_location=torch.device('cuda')), strict=False)
    seg_model.eval()

    print("# 2.2. segment images.")
    train_seg, test_seg = [], []
    for i in range(len(train_im)):
        path_endpart = os.path.splitext(train_paths[i].split('/')[-2] + '/' + train_paths[i].split('/')[-1])[0] + '.png'  # changes input filepath extension to png
        final_filepath = base_dir + seg_subdir + subset_name + path_endpart
        if os.path.isfile(final_filepath):  # check if segmented image already exists
            seg = numpy.array(Image.open(final_filepath))
        else:
            seg = seg_image(train_im[i], seg_model)
            seg = tools.normalize(seg)
            seg = tools.binarize(seg, method='adapt')
            seg = tools.watershed(numpy.asarray(Image.fromarray(seg).convert('L')))
            seg = morph.morph(numpy.asarray(Image.fromarray(seg).convert('1')), 'skel_closing')
            Image.fromarray(seg).save(final_filepath)
        train_seg.append(seg)

    print("# 2.2. segment images. - test")
    for i in range(len(test_im)):
        path_endpart = os.path.splitext(test_paths[i].split('/')[-2] + '/' + test_paths[i].split('/')[-1])[0] + '.png'  # changes input filepath extension to png
        final_filepath = base_dir + seg_subdir + subset_name + path_endpart
        if os.path.isfile(final_filepath):  # check if segmented image already exists
            seg = numpy.array(Image.open(final_filepath))
        else:
            seg = seg_image(test_im[i], seg_model)
            seg = fiim.thnim(proc.binarize(seg))
            Image.fromarray(seg).save(final_filepath)
        test_seg.append(seg)

    print("len train_seg", len(train_seg))
    print("len test_seg", len(test_seg))

    print("# 3. get graphs")
    tup = graf.get_graphs(train_seg)
    train_gr = tup[0]
    train_dc = tup[1]
    train_nodes = [list(x.keys()) for x in tup[1]]
    train_jts = []
    train_edges = []
    for tu, im in zip(tup[1], train_seg):  # for tu, im in zip(tup[1], train_im_gray):
        tr = graf.gabor_wavelet_transform(numpy.asarray(Image.fromarray(im).convert('L')))  # tr = graf.gabor_wavelet_transform(im)
        train_jts.append(graf.graph_jets(tr, list(tu.keys())))
        train_edges.append([x for sub in list(tu.values()) for x in sub])

    print("train_jts type", type(train_jts[0]))
    print("train_edges type", type(train_edges[0]))

    print("# 4. get similarity")
    pred, preds = [], []
    cur = []
    for j in range(len(test_seg)):  # for j in range(len(test_im)):
        print('current animal: %s' % test_labels[j])
        preds = -1
        best = -1
        tr = graf.gabor_wavelet_transform(numpy.asarray(Image.fromarray(test_seg[j]).convert('L')))  # tr = graf.gabor_wavelet_transform(test_im_gray[j])
        
        #path_endpart = os.path.splitext(train_paths[i].split('/')[-2] + '/' + train_paths[i].split('/')[-1])[0] + '.png'  # changes input filepath extension to png
        #final_filepath = base_dir + wave_subdir + subset_name + path_endpart
        #Image.fromarray(tr).save(final_filepath)
        print(tr.shape)
        
        for i in range(len(train_seg)):  # for i in range(len(train_im)):
            sim = graf.elastic_matching(train_nodes[i], train_edges[i], train_jts[i], tr, train_im[i].shape[-1])
            graf.draw_graph(train_gr[i], None).save("grftrain" + str(i) + ".png")
            print('%.3f %s' % (sim, train_labels[i]))
            if sim > best:
                best = sim
                preds = i
        pred.append(train_labels[preds])
        cur.append(test_labels[j])
        print(cur[-1], pred[-1])
    print(pred, cur)
    print(accuracy_score(pred, cur))

    # TODO: add ROC.py

    exit(0)
