import torch
from torch.utils import data
from torch.autograd import Variable
import numpy
from PIL import Image

#import graf
#import fiim
import gen_graph               # import added by loi
import bovine_matcher          # import added by loi

import pool
import proc
import file_handling

import tools
import morph

import os
#import bob.ip.gabor as big

from sklearn.metrics import accuracy_score, precision_score, classification_report, auc, roc_curve, plot_roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, f1_score
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#subset_name = '_j-smol/'
#base_dir = './data/'
#input_dir = '1-roi/'
#seg_subdir = '2-seg/'
#wave_subdir = '3-wave/'
#seg_model_file = "./models/x_grey_3000.pth"

testeImg = './data/Jersey_SMix/J71/J71_S2_0.png'
testeImg2 = './data/Jersey_SMix/J71/J71_S1_0.png'

def amain():
    GraficoTeste = gen_graph.graph_routine(testeImg)

    file = open('results/testeNovoGrafico.txt', 'w')

    for Vertex in range(len(GraficoTeste.vertexes)) :
        file.write('Vertice %d :\n' % Vertex)
        file.write('    [%d : %d] :\n' % (GraficoTeste.vertexes[Vertex].yx[1] , GraficoTeste.vertexes[Vertex].yx[0]))
        file.write('Vizinhos %d :\n' % len(GraficoTeste.vertexes[Vertex].neighs))
        for neigh in range(len(GraficoTeste.vertexes[Vertex].neighs)) :
            file.write('      i: %d ||' % GraficoTeste.vertexes[Vertex].neighs[neigh].i)
            file.write(' [%d : %d] :  ' % (GraficoTeste.vertexes[GraficoTeste.vertexes[Vertex].neighs[neigh].i].yx[1], GraficoTeste.vertexes[GraficoTeste.vertexes[Vertex].neighs[neigh].i].yx[0]))
            file.write('ang: %f | dist: %f\n' % (GraficoTeste.vertexes[Vertex].neighs[neigh].ang , GraficoTeste.vertexes[Vertex].neighs[neigh].dist))
        file.write('-----===-----\n\n')
    file.close()

    return


def main():
    MatchTeste = gen_graph.graph_routine(testeImg)
    MatchTeste2 = gen_graph.graph_routine(testeImg2)
    
    ResultadoMatch = bovine_matcher.match_recursive(MatchTeste, MatchTeste2)
    
    file = open('results/testeNovoGrafico.txt', 'w')
    
    for match in ResultadoMatch :
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (match[0][0] , match[0][1], match[1][0], match[1][1]))
    file.write('-----==AGORA O RANSAC==-----\n\n')
    
    
    ResultadoRansac = bovine_matcher.ransac_filter(ResultadoMatch)
    
    for match in range(len(ResultadoRansac[0])) :
        file.write('Ransac: %d  ||  ' % ResultadoRansac[0][match])
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (ResultadoRansac[1][match][0], ResultadoRansac[1][match][1], ResultadoRansac[2][match][0], ResultadoRansac[2][match][1]))
    file.write('-----==END==-----')
    
    file.close()

    return
