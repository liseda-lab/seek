import os
import rdflib
import networkx as nx
import numpy
import pickle

def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: a path-like object representing a file system path
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def write_embeddings(embeddings, ents, path_output_embeddings):
    """
    Write embeddings and their corresponding entities to a tab-separated file.
    Parameters:
        embeddings (list of numpy.ndarray): List of embeddings, where each embedding is a numpy array.
        ents (list of str): List of entity pairs.
        path_output_embeddings (str): Path to the output file where embeddings will be saved.
    Returns:
        None
    """
    with open(path_output_embeddings, "w") as embedding_file:
        for i in range(len(ents)):
            ent1, ent2 = ents[i].split('-')
            embedding_file.write(ent1 + '\t' + ent2)
            for elem in embeddings[i]:
                embedding_file.write('\t' + str(elem))
            embedding_file.write('\n')


def run_PairEmbeddings(path_embedding_file, path_representations_file, path_graph, path_dataset_file, vector_size=100):
    """
    Compute pair embeddings based on disjoint common ancestors in a graph and save them to a file.
    Parameters:
        path_embedding_file (str): Path to the file containing embeddings for ontology classes.
        path_representations_file (str): Path where the output pair embeddings file will be saved.
        path_graph (str): Path to the NetworkX graph file.
        path_dataset_file (str): Path to the dataset file containing pairs of entities.
        vector_size (int, optional): Dimensionality of the embeddings. Default is 100.
    Returns:
        None
    """
     # Read pairs of entities from the dataset file
    pairs_ents = [line.strip().split('\t')[:-1] for line in open(path_dataset_file).readlines()]

    # Load embeddings for ontology classes from the embedding file
    dic_emb_classes = eval(open(path_embedding_file, 'r').read())

    # Read the NetworkX graph from the specified path
    with open(path_graph, 'rb') as kg:
        G = pickle.load(kg)

    embeddings_avg, ents = [], []

    i = 0
    for ent1, ent2 in pairs_ents:
        ents.append(ent1 + '-' + ent2)

        # Add nodes for entities ent1 and ent2 to the graph if not already present
        G.add_node(rdflib.term.URIRef(ent1))
        G.add_node(rdflib.term.URIRef(ent2))

         # Find all common ancestors of ent1 and ent2 in the graph
        all_common_ancestors = list(nx.descendants(G, rdflib.term.URIRef(ent1)) & nx.descendants(G, rdflib.term.URIRef(ent2)))
        
        # Filter common ancestors to include only those that are not ancestors of any other common ancestor (disjoint common ancestors)
        common_ancestors, parents = [], {}
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
                    common_ancestors.append(ancestor)

         # Calculate average embeddings for disjoint common ancestors
        if len(common_ancestors) == 0:
            embeddings_avg.append([0 for dim in range(vector_size)])
        else:
            vectors = []
            for common_ancestor in common_ancestors:
                vectors.append(dic_emb_classes[str(common_ancestor)])
            array_vectors = numpy.array(vectors)

            avg_vectors = numpy.average(array_vectors, 0)
            embeddings_avg.append(avg_vectors.tolist())

        i = i + 1
        if i % 100 == 0:
            print(str(i) + '/' + str(len(pairs_ents)))

    # Write computed pair embeddings to the output file
    write_embeddings(embeddings_avg, ents, path_representations_file)



# if __name__== '__main__':

#     vector_size = 100
#     path_representations_file = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"
#     path_embedding_file = "./PPI/Embeddings/Emb_classes_maxdepth4_nwalks100.txt"
#     path_graph = "./PPI/KG.gpickle"
#     path_dataset_file = "./PPI/STRING_pairs.txt"
#     run_PairEmbeddings(path_embedding_file, path_representations_file, path_graph, path_dataset_file, vector_size)


