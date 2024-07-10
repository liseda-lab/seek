import rdflib
from rdflib.namespace import RDF, OWL, RDFS
from gensim.models.word2vec import Word2Vec as W2V

import kg
import walker



def run_word2vec(walks, vector_size, ents):
    """
    Train a Word2Vec model on random walks and return embeddings for specified entities.
    Parameters:
        walks (list): A list of random walks.
        vector_size (int): The dimensionality of the vectors.
        ents (list of str): A list of entities for which to return embeddings.
    Returns:
        embeddings (list of numpy.ndarray): A list of embeddings corresponding to the specified entities.
    """
     # Convert each walk into a list of strings to form the corpus
    corpus = [list(map(str, x)) for x in walks]
    # Train the Word2Vec model on the corpus
    model = W2V(corpus, min_count=1, vector_size=vector_size)
     # Retrieve the embeddings for the specified entities
    embeddings = [model.wv[str(entity)] for entity in ents]
    return embeddings


def write_embeddings_classes(embeddings, ents, path_output_embeddings):
    """
    ""
    Write entities and their corresponding embeddings to a file in dict format.
    Parameters:
        embeddings (list of numpy.ndarray): List of embeddings, where each embedding is a numpy array.
        ents (list of str): List of entities corresponding to each embedding.
        path_output_embeddings (str): Path to the output file where embeddings will be saved .
    Returns:
        dic_emb (dict): A dictionary where keys are ontology classes and values are their corresponding embeddings (as lists).
    """
    dic_emb = {}
    with open(path_output_embeddings, 'w') as file_emb:
        file_emb.write("{")
        first = False
        for i in range(len(ents)):
            dic_emb[ents[i]] = embeddings[i].tolist()
            if first:
                file_emb.write(", '%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
            else:
                file_emb.write("'%s':%s" % (str(ents[i]), str(embeddings[i].tolist())))
                first = True
            file_emb.flush()
        file_emb.write("}")
    return dic_emb


def compute_embeddings_classes(ontology_file_path, path_embedding_classes, vector_size, n_walks, max_depth):
    """
    Compute embeddings for classes in an ontology and save them to a file.
    Parameters:
        ontology_file_path (str): Path to the ontology file.
        path_output (str): Path where the output embeddings file will be saved.
        vector_size (int): Dimensionality of the embeddings.
        n_walks (int): Maximum number of random walks to generate per entity.
        max_depth (int): Maximum depth for random walks.
    Returns:
        dict_emb (dict): A dictionary where keys are ontology classes and values are their corresponding embeddings.
    """
    #Building the graph
    g_ontology = rdflib.Graph()
    g_ontology.parse(ontology_file_path, format="xml")
    kg_graph = kg.rdflib_to_kg(g_ontology)

    #Generating the walks
    ents, walks = [],[]
    for s, p, o in g_ontology.triples((None, RDF.type, OWL.Class)):
       ents.append(str(s))

    for ent in ents:
        walks_ent = walker.extract_walks(kg_graph, str(ent), n_walks, max_depth)
        for walk in walks_ent:
           walks.append(walk)

    #Training the model
    embeddings = run_word2vec(walks, vector_size, ents)

    # Writing embeddings
    dic_emb = write_embeddings_classes(embeddings, ents, path_embedding_classes)

    return dic_emb



# if __name__== '__main__':
    
#     ontology_file_path = "./PPI/go.owl"
#     path_embedding_classes = "./PPI/Embeddings/Emb_classes_maxdepth" + str(max_depth) + "_nwalks" + str(n_walks) + ".txt""
#     vector_size = 100 
#     n_walks = 100
#     max_depth = 4
#     compute_embeddings_classes(ontology_file_path, path_embedding_classes, vector_size, n_walks, max_depth)