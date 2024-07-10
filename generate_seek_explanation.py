import os
import numpy as np
import pandas as pd
import rdflib
import networkx as nx
import pickle
from operator import itemgetter

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def bar_plot(df, df_without, df_only, n_ancestors, output):
    """
    Generate a bar plot with an explanation and save it to an output file.
    Parameters:
        df (DataFrame): Dataframe containing proba predictions using the representations.
        df_without (DataFrame): Dataframe containing proba predictions obtained without necessary semantic aspects.
        df_only (DataFrame): Dataframe containing proba predictions obtained with only sufficient semantic aspects.
        n_ancestors (int): Number of disjoint common ancestors.
        output (str): Output file path to save the bar plot.
    Return:
        none
    """

    # Create a figure and axis object using matplotlib
    cm = 1 / 2.54
    fig, ax = plt.subplots()
    #plt.figure(figsize=((1 * n_ancestors + 1.5) * cm, 9 * cm))
    fig.set_size_inches(10, 8)
    #plt.figure(figsize=(5, n_ancestors*2+3))
    sns.set_color_codes("pastel")

    # Adjust values and colors in df_without dataframe
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

    # Adjust values and colors in df_only dataframe
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

    # Adjust values and colors in df dataframe
    df['value'] = [float(x)-0.5 if x > 0.5 else -(1-x)+0.5 for x in df['prob class 1']]
    df['colors'] = ['#FF5050' if x < 0 else '#70BB83' for x in df['value']]

    # Plot horizontal lines with labels for df dataframe
    ax.hlines(y=df.dca, xmin=0, xmax=df.value, color=df['colors'], alpha=0.8, linewidth=2.5)
    for x, y, tex in zip(df.value, df.dca, df.value):
        if len(y) > 2:
            t = ax.text(x, y, round(abs(tex) + 0.5, 2), horizontalalignment='right' if x < 0 else 'left',
                        verticalalignment='center',
                        fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

    # Plot horizontal lines with labels for df_without dataframe
    ax.hlines(y=df_without.dca, xmin=0, xmax=df_without.value, color=df_without['colors'], alpha=0.8, linewidth=2.5)
    for x, y, tex in zip(df_without.value, df_without.dca, df_without.value):
        if len(y)>2:
            t = ax.text(x, y, round(abs(tex) + 0.5, 2), horizontalalignment='right' if x < 0 else 'left',
                        verticalalignment='center',
                        fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

    # Plot horizontal lines with labels for df_only dataframe
    ax.hlines(y=df_only.dca, xmin=0, xmax=df_only.value, color=df_only['colors'], alpha=0.8, linewidth=2.5)
    for x, y, tex in zip(df_only.value, df_only.dca, df_only.value):
        if len(y)>2:
            t = ax.text(x, y, round(abs(tex) + 0.5, 2), horizontalalignment='right' if x < 0 else 'left',
                        verticalalignment='center',
                        fontdict={'color': '#FF5050' if x < 0 else '#70BB83', 'size': 9})

    # Set x-axis limits, x-axis ticks, x-axis tick labels
    ax.set_xlim(-0.7,0.7)
    ax.set_xticks((-0.5, -0.25, 0, 0.25, 0.5))
    ax.set_xticklabels(("1","0.75", "0.5", "0.75", "1"))

    # Save figure to output file with tight bounding box
    fig.savefig(output, bbox_inches='tight')
    plt.close('all')


def getExplanations(path_graph, path_label_classes, path_embedding_classes, target_pair, alg, path_file_model, path_explanations, n_embeddings=100):
    """
    Generates explanations for predictions made by a ML model on pairs of entities based on their disjoint common ancestors.
    Parameters:
        path_graph (str): Path to the GPickle file containing the relationships between ontology classes.
        path_label_classes (str): Path to the file containing the a dictionary where the keys are ontology classes and the values are the corresponding labels.
        path_embedding_classes (str): Path to the file containing the embedding classes dictionary.
        target_pair (tuple): Tuple of two entities for which explanations are to be generated.
        alg (str): Algorithm used for the ML model ('XGB', 'RF', 'MLP').
        path_file_model (str): Path to the file containing the trained ML model (pickle format).
        path_explanations (str): Path to the directory where explanation files and plots will be saved.
        n_embeddings (int, optional): Number of dimensions in the embeddings. Defaults to 100.
    Returns:
        None
    """

    # Load necessary data
    with open(path_graph, 'rb') as kg:
        G = pickle.load(kg)

    with open(path_label_classes, 'rb') as label_classes:
        dic_labels_classes = pickle.load(label_classes)
    dic_emb_classes = eval(open(path_embedding_classes, 'r').read())
    ml_model = pickle.load(open(path_file_model, "rb"))

    ent1, ent2 = target_pair

    ensure_dir(path_explanations + alg + "/")
    file_predictions = open(path_explanations + alg + "/" + ent1 + "-" + ent2 + ".txt", 'w')
    file_predictions.write('DCA\tRemoving\tPredicted-label\tProb-class0\tProb-class1\n')

    # Determine disjoint common ancestors in the knowledge graph for the target pair
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

    # Generate representation for the target pair
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

    # Iterate over disjoint common ancestors to generate explanations
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

        # Identify necessary shared semantic aspects
        if pred_original != pred_without_dca:
            necessary_explan.append(str(dca))
            results_without.append(["w/o '" + dic_labels_classes[str(dca)] + "'", proba_pred_without_dca[1], proba_pred_without_dca[0]])

        X_test_only_dca = [dic_emb_classes[str(dca)]]
        pred_only_dca = ml_model.predict(X_test_only_dca).tolist()[0]
        proba_pred_only_dca = ml_model.predict_proba(X_test_only_dca).tolist()[0]

        file_predictions.write(str(dca) + '\t' + 'False' + '\t' + str(pred_only_dca) + '\t' + str(
            proba_pred_only_dca[0]) + '\t' + str(proba_pred_only_dca[1]) + '\n')

        # Identify sufficient shared semantic aspects
        if pred_original == pred_only_dca:
            sufficient_explan.append(str(dca))
            results_only.append(["only '" + dic_labels_classes[str(dca)] + "'", proba_pred_only_dca[1], proba_pred_only_dca[0]])

    # Generate representations with only sufficient shared semantc aspects
    vectors_withsufficient = []
    for suf in sufficient_explan:
        vectors_withsufficient.append(dic_emb_classes[suf])
    if len(sufficient_explan) == 0:
        x_withsufficient = np.array([0 for j in range(n_embeddings)])
    elif len(sufficient_explan) == 1:
        x_withsufficient = np.array(vectors_withsufficient[0])
    else:
        x_withsufficient = np.average(np.array(vectors_withsufficient), 0)
    proba_withsufficient = ml_model.predict_proba(x_withsufficient.reshape(1, -1))[0]

    # Generate representations without necessary shared semantc aspects
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
    proba_withoutnecessary = ml_model.predict_proba(x_withoutnecessary.reshape(1, -1))[0]

    file_predictions.close()

    # Sort results and prepare data for plotting
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

    # Generate bar plot with sufficent and necessary semantic aspects
    bar_plot(df, df_without, df_only, len(disjoint_common_ancestors), path_explanations + alg + "/Plot_" + ent1 + "-" + ent2 + ".png")


# if __name__== '__main__':

#     path_file_representation = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"
#     path_file_model = "./PPI/Models/RF/Model_RF_1.pickle"
#     alg = "RF"
#     path_graph = "./PPI/KG.gpickle"
#     path_label_classes = "./PPI/Labelclasses.pkl"
#     path_explanations = "./PPI/Explanations/"
#     path_embedding_classes = "./PPI/Embeddings/Emb_classes_maxdepth4_nwalks100.txt"
#     target_pair = ('P25398','P46783')

#     getExplanations(path_graph, path_label_classes, path_embedding_classes, target_pair, alg, path_file_model, path_explanations)
