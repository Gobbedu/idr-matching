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


bug_test = './data/Jersey_SMix/J357/J357_S1_1.png'

testeImg = './data/Jersey_SMix/J71/J71_S1_0.png'
testeImg2 = './data/Jersey_SMix/J71/J71_S2_0.png'





# test gen_graph
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



# testando RANSAC output
def main():
    MatchTeste = gen_graph.graph_routine(testeImg)
    MatchTeste2 = gen_graph.graph_routine(testeImg2)
    ResultadoMatch = bovine_matcher.match_recursive(MatchTeste, MatchTeste2)
    
    
    file = open('results/RANSACconsole.txt', 'w')
    
    for match in ResultadoMatch :
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (match[0][1] , match[0][0], match[1][1], match[1][0]))
    file.write('-----==AGORA O RANSAC==-----\n\n')
    
    number_of_ransac_confirmations = 0
    ResultadoRansac = bovine_matcher.ransac_filter(ResultadoMatch)
    
    for match in range(len(ResultadoRansac[0])) :
        if (ResultadoRansac[0][match]) :
            number_of_ransac_confirmations += 1
        file.write('Ransac: %d  ||  ' % ResultadoRansac[0][match])
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (ResultadoRansac[1][match][1], ResultadoRansac[1][match][0], ResultadoRansac[2][match][1], ResultadoRansac[2][match][0]))
    file.write('soma do ransac: %d -----==END==-----\n\n\n' % number_of_ransac_confirmations)
    
    file.close()
    
    bovine_matcher.luiz_draw(testeImg, testeImg2, ResultadoRansac)
    
    return



# test and save both match types individually
def lelelmain():
    MatchTeste = gen_graph.graph_routine(testeImg)
    MatchTeste2 = gen_graph.graph_routine(testeImg2)
    
    ResultadoMatchRecursivo = bovine_matcher.match_recursive(MatchTeste, MatchTeste2)
    ResultadoMatch = bovine_matcher.simple_match(MatchTeste, MatchTeste2)
    
    
    file = open('results/testeMatcher.txt', 'w')
    
    for match in ResultadoMatchRecursivo :
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (match[0][1] , match[0][0], match[1][1], match[1][0]))
    file.write('-----==AGORA O RANSAC==-----\n\n')
    
    number_of_ransac_confirmations = 0
    ResultadoRansac = bovine_matcher.ransac_filter(ResultadoMatchRecursivo)
    
    for match in range(len(ResultadoRansac[0])) :
        if (ResultadoRansac[0][match]) :
            number_of_ransac_confirmations += 1
        file.write('Ransac: %d  ||  ' % ResultadoRansac[0][match])
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (ResultadoRansac[1][match][1], ResultadoRansac[1][match][0], ResultadoRansac[2][match][1], ResultadoRansac[2][match][0]))
    file.write('soma do ransac: %d -----==END DO RECURSIVO==-----\n\n\n' % number_of_ransac_confirmations)
    
    
    
    for match in ResultadoMatch :
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (match[0][1] , match[0][0], match[1][1], match[1][0]))
    file.write('-----==AGORA O RANSAC==-----\n\n')
    
    number_of_ransac_confirmations = 0
    ResultadoRansac = bovine_matcher.ransac_filter(ResultadoMatch)
    
    for match in range(len(ResultadoRansac[0])) :
        if (ResultadoRansac[0][match]) :
            number_of_ransac_confirmations += 1
        file.write('Ransac: %d  ||  ' % ResultadoRansac[0][match])
        file.write('[%3d : %3d]  [%3d : %3d]\n' % (ResultadoRansac[1][match][1], ResultadoRansac[1][match][0], ResultadoRansac[2][match][1], ResultadoRansac[2][match][0]))
    file.write('soma do ransac: %d -----==END==-----\n\n\n' % number_of_ransac_confirmations)
    
    file.close()

    return



# compare both match types
def lolmain():
    MatchTeste = gen_graph.graph_routine(testeImg)
    MatchTeste2 = gen_graph.graph_routine(testeImg2)
    
    ResultadoMatchRecursivo = bovine_matcher.match_recursive(MatchTeste, MatchTeste2)
    ResultadoMatch = bovine_matcher.simple_match(MatchTeste, MatchTeste2)
    
    amout_of_repetition = 100
    
    sum_recursivo = 0
    sum_normal = 0
    
    i = 0
    while i < amout_of_repetition :
        print('Next Batch ', i)
        i += 1
        temp = 0
        
        Ransac1 = bovine_matcher.ransac_filter(ResultadoMatchRecursivo)
        for match in range(len(Ransac1[0])) :
            if (Ransac1[0][match]) :
                temp += 1
        print('ransac recursivo resultado: %d' % temp)
        sum_recursivo += temp
        
        temp = 0
        Ransac2 = bovine_matcher.ransac_filter(ResultadoMatch)
        for match in range(len(Ransac2[0])) :
            if (Ransac2[0][match]) :
                temp += 1
        print('ransac normal resultado: %d' % temp)
        sum_normal += temp
    
    print('soma dos recursivos: %d' % sum_recursivo)
    print('soma dos normais: %d' % sum_normal)
    
    print('media dos recursivos: %f' % (sum_recursivo/amout_of_repetition))
    print('media dos normais: %f' % (sum_normal/amout_of_repetition))
    

    return



# bug FINDER
def mmain() :
    ImageTeste = gen_graph.graph_routine(bug_test)
    
    file = open('results/testeNovoGrafico_Bug.txt', 'w')

    for Vertex in range(len(ImageTeste.vertexes)) :
        file.write('Vertice %d :\n' % Vertex)
        file.write('    [%d : %d] :\n' % (ImageTeste.vertexes[Vertex].yx[1] , ImageTeste.vertexes[Vertex].yx[0]))
        file.write('Vizinhos %d :\n' % len(ImageTeste.vertexes[Vertex].neighs))
        for neigh in range(len(ImageTeste.vertexes[Vertex].neighs)) :
            file.write('      i: %d ||' % ImageTeste.vertexes[Vertex].neighs[neigh].i)
            file.write(' [%d : %d] :  ' % (ImageTeste.vertexes[ImageTeste.vertexes[Vertex].neighs[neigh].i].yx[1], ImageTeste.vertexes[ImageTeste.vertexes[Vertex].neighs[neigh].i].yx[0]))
            file.write('ang: %f | dist: %f\n' % (ImageTeste.vertexes[Vertex].neighs[neigh].ang , ImageTeste.vertexes[Vertex].neighs[neigh].dist))
        file.write('-----===-----\n\n')
    file.close()
    
    return

if __name__ == "__main__":
    main()
