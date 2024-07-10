from build_kg import run_save_graph
from train_embedding_model import compute_embeddings_classes
from generate_pair_representations import run_PairEmbeddings
from train_ml import run_ml_model
from generate_seek_explanation import getExplanations


# Define input variables
ontology_file_path = "./PPI/go.owl" 
annotations_file_path = "./PPI/go_annotations.gaf"
path_graph = "./PPI/KG.gpickle"
path_label_classes = "./PPI/Labelclasses.pkl"
path_embedding_classes = "./PPI/Embeddings/Emb_classes_maxdepth4_nwalks100.txt"

path_dataset_file = "./PPI/STRING_pairs.txt"
path_representations_file = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"

n_partition = 10 
path_partition = "./PPI/StratifiedPartitions/Indexes__crossvalidationTest__Run"
path_ML_model = "./PPI/Models/RF/"

path_explanations = "./PPI/Explanations/"


# Generating knowledge graph
run_save_graph(ontology_file_path, annotations_file_path, path_graph, path_label_classes, 'PPI')

# Training embedding model to generate embeddings for ontology classes
vector_size = 100 
n_walks = 100
max_depth = 4
compute_embeddings_classes(ontology_file_path, path_embedding_classes, vector_size, n_walks, max_depth)

# Generating pair representations based on the disjoint common ancestors
run_PairEmbeddings(path_embedding_classes, path_representations_file, path_graph, path_dataset_file, vector_size)

# # Training the ML model
alg = "RF"
run_ml_model(path_representations_file, path_dataset_file, n_partition, path_partition, alg, path_ML_model)

# Generating SEEK explanations
target_pair = ('P25789', 'P17980')
getExplanations(path_graph, path_label_classes, path_embedding_classes, target_pair, alg, path_ML_model + "Model_" + alg + "_1.pickle", path_explanations)