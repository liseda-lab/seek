import random
import os
import sys
import argparse
import numpy as np
import time
import copy
import gc
import pandas as pd

import rdflib
from rdflib.namespace import RDF, OWL, RDFS

import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_array
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stat
from sklearn import metrics
from operator import itemgetter

import time
import pickle

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

def _identity(x): return x

def _rdflib_to_networkx_graph(
        graph,
        nxgraph,
        calc_weights,
        edge_attrs,
        transform_s=_identity, transform_o=_identity):
    """Helper method for multidigraph, digraph and graph.
    Modifies nxgraph in-place!
    Arguments:
        graph: an rdflib.Graph.
        nxgraph: a networkx.Graph/DiGraph/MultiDigraph.
        calc_weights: If True adds a 'weight' attribute to each edge according
            to the count of s,p,o triples between s and o, which is meaningful
            for Graph/DiGraph.
        edge_attrs: Callable to construct edge data from s, p, o.
           'triples' attribute is handled specially to be merged.
           'weight' should not be generated if calc_weights==True.
           (see invokers below!)
        transform_s: Callable to transform node generated from s.
        transform_o: Callable to transform node generated from o.
    """
    assert callable(edge_attrs)
    assert callable(transform_s)
    assert callable(transform_o)
    import networkx as nx
    for s, p, o in graph:
        if p == RDFS.subClassOf or p==rdflib.term.URIRef('http://hasAnnotation'):
            ts, to = transform_s(s), transform_o(o)  # apply possible transformations
            data = nxgraph.get_edge_data(ts, to)
            if data is None or isinstance(nxgraph, nx.MultiDiGraph):
                # no edge yet, set defaults
                data = edge_attrs(s, p, o)
                if calc_weights:
                    data['weight'] = 1
                nxgraph.add_edge(ts, to, **data)
            else:
                # already have an edge, just update attributes
                if calc_weights:
                    data['weight'] += 1
                if 'triples' in data:
                    d = edge_attrs(s, p, o)
                    data['triples'].extend(d['triples'])

def process_GO_annotations(annotations_file_path):

    file_annot = open(annotations_file_path, 'r')
    file_annot.readline()
    dic_annotations = {}
    for annot in file_annot:
        list_annot = annot.split('\t')
        id_prot, GO_term = list_annot[1], list_annot[4]

        url_GO_term = "http://purl.obolibrary.org/obo/GO_" + GO_term.split(':')[1]

        if url_GO_term == 'http://purl.obolibrary.org/obo/GO_0044212':
            print(annot)

        url_prot = id_prot

        if url_prot not in dic_annotations:
            dic_annotations[url_prot] = [url_GO_term]
        else:
            dic_annotations[url_prot] = dic_annotations[url_prot] + [url_GO_term]
    file_annot.close()
    return dic_annotations


def process_HP_annotations(annotations_file_path):
    file_annot = open(annotations_file_path, 'r')
    file_annot.readline()
    dic_annotations = {}
    for annot in file_annot:
        list_annot = annot[:-1].split('\t')
        id_ent, HPO_term = list_annot[0], list_annot[1]

        url_HPO_term = 'http://purl.obolibrary.org/obo/HP_' + HPO_term
        url_ent = id_ent

        if url_ent not in dic_annotations:
            dic_annotations[url_ent] = [url_HPO_term]
        else:
            dic_annotations[url_ent] = dic_annotations[url_ent] + [url_HPO_term]

    file_annot.close()
    return dic_annotations


def add_annotations(g, dic_annotations):
    for ent in dic_annotations:
        for a in dic_annotations[ent]:
            g.add((rdflib.term.URIRef(ent), rdflib.term.URIRef('http://hasAnnotation'),rdflib.term.URIRef(a)))
    return g


def construct_kg(ontology_file_path, annotations_file_path, type_dataset='PPI'):
    if type_dataset == "PPI":
        dic_annotations = process_GO_annotations(annotations_file_path)
    elif type_dataset == "GDA":
        dic_annotations = process_HP_annotations(annotations_file_path)
    g_ontology = rdflib.Graph()
    g_ontology.parse(ontology_file_path, format="xml")
    dic_labels_classes = {}
    for (sub, pred, obj) in g_ontology.triples((None, RDFS.label, None)):
        dic_labels_classes[str(sub)] = str(obj)
    return add_annotations(g_ontology, dic_annotations), dic_labels_classes


def process_indexes_partition(file_partition):
    """
    Process the partition file and returns a list of indexes.
    :param file_partition: partition file path (each line is a index);
    :return: list of indexes.
    """
    file_partitions = open(file_partition, 'r')
    indexes_partition = []
    for line in file_partitions:
        indexes_partition.append(int(line[:-1]))
    file_partitions.close()
    return indexes_partition


def process_dataset(path_dataset_file):
    """
    """
    list_labels = []
    with open(path_dataset_file, 'r') as dataset:
        for line in dataset:
            split1 = line.split('\t')
            ent1, ent2 = split1[0], split1[1]
            label = int(split1[2][:-1])
            list_labels.append([(ent1, ent2), label])
    return list_labels


def bar_plot(df, df_without, df_only, n_ancestors, output):
    cm = 1 / 2.54
    fig, ax = plt.subplots()
    #plt.figure(figsize=((1 * n_ancestors + 1.5) * cm, 9 * cm))
    fig.set_size_inches(10, 8)
    #plt.figure(figsize=(5, n_ancestors*2+3))
    sns.set_color_codes("pastel")

    df_without['value'] = [x-0.5 if x>0.5 else -(1-x)+0.5 for x in df_without['prob class 1']]
    list_colors_without=[]
    for x in df_without['value']:
        if x < 0:
            z='#FF5050'
        elif x==0:
            z="#FFFFFF"
        else:
            z='#70BB83'
        list_colors_without.append(z)
    df_without['colors'] = list_colors_without

    df_only['value'] = [float(x)-0.5 if x > 0.5 else -(1-x)+0.5 for x in df_only['prob class 1']]
    list_colors_only = []
    for x in df_only['value']:
        if x < 0:
            z = '#FF5050'
        elif x== 0:
            z = "#FFFFFF"
        else:
            z = '#70BB83'
        list_colors_only.append(z)
    df_only['colors'] = list_colors_only

    df['value'] = [float(x)-0.5 if x > 0.5 else -(1-x)+0.5 for x in df['prob class 1']]
    df['colors'] = ['#FF5050' if x < 0 else '#70BB83' for x in df['value']]

    ax.hlines(y=df.dca, xmin=0, xmax=df.value, color=df['colors'], alpha=0.8, linewidth=2.5)
    for x, y, tex in zip(df.value, df.dca, df.value):
        if len(y) > 2:
            t = ax.text(x, y, round(abs(tex) + 0.5, 2), horizontalalignment='right' if x < 0 else 'left',
                        verticalalignment='center',
                        fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

    ax.hlines(y=df_without.dca, xmin=0, xmax=df_without.value, color=df_without['colors'], alpha=0.8, linewidth=2.5)
    for x, y, tex in zip(df_without.value, df_without.dca, df_without.value):
        if len(y)>2:
            t = ax.text(x, y, round(abs(tex) + 0.5, 2), horizontalalignment='right' if x < 0 else 'left',
                        verticalalignment='center',
                        fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

    ax.hlines(y=df_only.dca, xmin=0, xmax=df_only.value, color=df_only['colors'], alpha=0.8, linewidth=2.5)
    for x, y, tex in zip(df_only.value, df_only.dca, df_only.value):
        if len(y)>2:
            t = ax.text(x, y, round(abs(tex) + 0.5, 2), horizontalalignment='right' if x < 0 else 'left',
                        verticalalignment='center',
                        fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

    ax.set_xlim(-0.7,0.7)
    ax.set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
    ax.set_xticklabels(("1","0.75", "0.5", "0.75", "1"))

    fig.show()
    fig.savefig(output, bbox_inches='tight')
    plt.close('all')


def process_representation_file(path_file_representation):
    """
    """
    dict_representation = {}
    with open(path_file_representation, 'r') as file_representation:
        for line in file_representation:
            line = line[:-1]
            split1 = line.split('\t')
            ent1 = split1[0].split('/')[-1]
            ent2 = split1[1].split('/')[-1]
            feats = split1[2:]
            feats_floats = [float(i) for i in feats]
            dict_representation[(ent1, ent2)] =  feats_floats
    return dict_representation


def read_representation_dataset_file(path_file_representation, path_dataset_file):
    """
    """
    list_representation, labels, list_ents= [], [], []
    dict_representation = process_representation_file(path_file_representation)
    list_labels = process_dataset(path_dataset_file)
    for (ent1, ent2), label in list_labels:
        representation_floats = dict_representation[(ent1, ent2)]
        list_ents.append([ent1, ent2])
        list_representation.append(representation_floats)
        labels.append(label)
    return list_ents, list_representation , labels


def run_save_model(path_file_representation, path_dataset_file, path_file_model, algorithms):

    pairs, X_train, y_train = read_representation_dataset_file(path_file_representation, path_dataset_file)

    for alg in algorithms:

        ensure_dir(path_file_model + alg + '/')
        
        if alg == "XGB":
            ml_model = xgb.XGBClassifier()
            ml_model.fit(np.array(X_train), np.array(y_train))
        if alg == "RF":
            ml_model = RandomForestClassifier()
            ml_model.fit(X_train, y_train)

        if alg == "MLP":
            ml_model = MLPClassifier()
            ml_model.fit(X_train, y_train)

        pickle.dump(ml_model, open(path_file_model + alg + "/Model_" + alg + ".pickle", "wb"))


def run_save_graph(ontology_file_path, annotations_file_path, path_graph, path_label_classes, type='PPI'):

    g, dic_labels_classes = construct_kg(ontology_file_path, annotations_file_path, type)
    G = nx.DiGraph()
    _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})

    nx.write_gpickle(G, path_graph)
    with open(path_label_classes, 'wb') as path_file:
        pickle.dump(dic_labels_classes, path_file)



def getExplanations(path_graph, path_label_classes, path_embedding_classes, target_pair, alg, path_file_model, path_explanations, n_embeddings=100, type='PPI'):

    # g, dic_labels_classes = construct_kg(ontology_file_path, annotations_file_path, type)
    # G = nx.DiGraph()
    # _rdflib_to_networkx_graph(g, G, calc_weights=False, edge_attrs=lambda s, p, o: {})
    G = nx.read_gpickle(path_graph)
    with open(path_label_classes, 'rb') as label_classes:
        dic_labels_classes = pickle.load(label_classes)

    dic_emb_classes = eval(open(path_embedding_classes, 'r').read())

    ml_model = pickle.load(open(path_file_model + alg + "/Model_" + alg + ".pickle", "rb"))

    ent1, ent2 = target_pair

    start = time.time()

    ensure_dir(path_explanations + alg + "/")
    file_predictions = open(path_explanations + alg + "/" + ent1 + "-" + ent2 + ".txt", 'w')
    file_predictions.write('DCA\tRemoving\tPredicted-label\tProb-class0\tProb-class1\n')

    all_common_ancestors = list(nx.descendants(G, rdflib.term.URIRef(ent1)) & nx.descendants(G, rdflib.term.URIRef(ent2)))
    disjoint_common_ancestors, parents = [], {}
    for anc in all_common_ancestors:
        parents[anc] = list(nx.descendants(G, anc))
    for ancestor in all_common_ancestors:
        parent = False
        for anc2 in all_common_ancestors:
            if anc2 != ancestor:
                if ancestor in parents[anc2]:
                    parent = True
        if parent == False:
            if str(ancestor) in dic_emb_classes:
                disjoint_common_ancestors.append(ancestor)

    necessary_explan, sufficient_explan = [], []
    results, results_without, results_only = [], [], []

    all_vectors = []
    for dca in disjoint_common_ancestors:
        all_vectors.append(dic_emb_classes[str(dca)])
    if len(all_vectors) == 0:
        all_avg_vectors = np.array([0 for i in range(n_embeddings)])
    elif len(all_vectors) == 1:
        all_avg_vectors = np.array(all_vectors[0])
    else:
        all_array_vectors = np.array(all_vectors)
        all_avg_vectors = np.average(all_array_vectors, 0)

    X_test_original = [all_avg_vectors.tolist()]
    pred_original = ml_model.predict(X_test_original).tolist()[0]
    proba_pred_original = ml_model.predict_proba(X_test_original).tolist()[0]
    file_predictions.write('All' + '\t' + 'NA' + '\t' + str(pred_original) + '\t' + str(proba_pred_original[0]) + '\t' + str(proba_pred_original[1]) + '\n')

    for dca in disjoint_common_ancestors:

        vectors = []
        for dca2 in disjoint_common_ancestors:
            if dca2 != dca:
                vectors.append(dic_emb_classes[str(dca2)])

        if len(vectors) == 0:
            avg_vectors = np.array([0 for i in range(n_embeddings)])
        elif len(vectors) == 1:
            avg_vectors = np.array(vectors[0])
        else:
            array_vectors = np.array(vectors)
            avg_vectors = np.average(array_vectors, 0)


        X_test_without_dca = [avg_vectors.tolist()]
        pred_without_dca = ml_model.predict(X_test_without_dca).tolist()[0]
        proba_pred_without_dca = ml_model.predict_proba(X_test_without_dca).tolist()[0]

        file_predictions.write(str(dca) + '\t' + 'True' + '\t' + str(pred_without_dca) + '\t' + str(
            proba_pred_without_dca[0]) + '\t' + str(proba_pred_without_dca[1]) + '\n')

        if pred_original != pred_without_dca:
            necessary_explan.append(str(dca))
            results_without.append(["w/o '" + dic_labels_classes[str(dca)] + "'", proba_pred_without_dca[1], proba_pred_without_dca[0]])

        X_test_only_dca = [dic_emb_classes[str(dca)]]
        pred_only_dca = ml_model.predict(X_test_only_dca).tolist()[0]
        proba_pred_only_dca = ml_model.predict_proba(X_test_only_dca).tolist()[0]

        file_predictions.write(str(dca) + '\t' + 'False' + '\t' + str(pred_only_dca) + '\t' + str(
            proba_pred_only_dca[0]) + '\t' + str(proba_pred_only_dca[1]) + '\n')

        if pred_original == pred_only_dca:
            sufficient_explan.append(str(dca))
            results_only.append(["only '" + dic_labels_classes[str(dca)] + "'", proba_pred_only_dca[1], proba_pred_only_dca[0]])

    vectors_withsufficient = []
    for suf in sufficient_explan:
        vectors_withsufficient.append(dic_emb_classes[suf])
    if len(sufficient_explan) == 0:
        x_withsufficient = np.array([0 for j in range(n_embeddings)])
    elif len(sufficient_explan) == 1:
        x_withsufficient = np.array(vectors_withsufficient[0])
    else:
        x_withsufficient = np.average(np.array(vectors_withsufficient), 0)
    predicted_label_withsufficient = ml_model.predict(x_withsufficient.reshape(1, -1))[0]
    proba_withsufficient = ml_model.predict_proba(x_withsufficient.reshape(1, -1))[0]

    vectors_withoutnecessary = []
    for not_nec in disjoint_common_ancestors:
        if str(not_nec) not in necessary_explan:
            vectors_withoutnecessary.append(dic_emb_classes[str(not_nec)])
    if len(necessary_explan) == len(disjoint_common_ancestors):
        x_withoutnecessary = np.array([0 for j in range(n_embeddings)])
    elif len(vectors_withoutnecessary) == 1:
        x_withoutnecessary = np.array(vectors_withoutnecessary[0])
    else:
        x_withoutnecessary = np.average(np.array(vectors_withoutnecessary), 0)
    predicted_label_withoutnecessary = ml_model.predict(x_withoutnecessary.reshape(1, -1))[0]
    proba_withoutnecessary = ml_model.predict_proba(x_withoutnecessary.reshape(1, -1))[0]

    file_predictions.close()

    end = time.time()
    print(end - start)

    results_without.sort(key=itemgetter(2))
    results_only.sort(key=itemgetter(2))

    results_without.append(['  ', 0.5, 0.5])

    results.append(['global', proba_pred_original[1], proba_pred_original[0]])
    results.append(['w/o necessary', proba_withoutnecessary[1], proba_withoutnecessary[0]])
    results.append(['only sufficient', proba_withsufficient[1], proba_withsufficient[0]])
    results.append([' ', 0.5, 0.5])

    df_without = pd.DataFrame(results_without, columns=["dca", "prob class 1", "prob class 0"])
    df_only = pd.DataFrame(results_only, columns=["dca", "prob class 1", "prob class 0"])
    df = pd.DataFrame(results, columns=["dca", "prob class 1", "prob class 0"])
    cols_only = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results_only]
    cols_without = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results_without]
    cols = ['#FF5050' if x[1] < 0.5 else '#70BB83' for x in results]

    bar_plot(df, df_without, df_only, len(disjoint_common_ancestors), path_explanations + alg + "/Plot_" + ent1 + "-" + ent2 + ".png")


if __name__== '__main__':

    ####################################### PPI prediction

    path_file_representation = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"
    path_file_model = "./PPI/Models/RF/Model_RF.pickle
    alg = "RF"
    path_graph = "./PPI/KG.gpickle"
    path_label_classes = "PPI/Labelclasses.pkl"
    path_explanations = "./PPI/Explanations/"
    path_embedding_classes = "PPI/Embeddings/Emb_classes_maxdepth4_nwalks100_disjointcommonancestor.txt"
    target_pair = ('P25398','P46783')

    getExplanations(path_graph, path_label_classes, path_embedding_classes, target_pair, alg, path_file_model, path_explanations)
