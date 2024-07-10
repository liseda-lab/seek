import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


def ensure_dir(path):
    """
    Check whether the specified path is an existing directory or not. And if is not an existing directory, it creates a new directory.
    :param path: A path-like object representing a file system path.
    """
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def process_representation_file(path_file_representation):
    """
    Process a representation file and store data in a dictionary.
    Parameters:
        path_file_representation (str): Path to the representation file to be processed.
    Returns:
        dict_representation (dict): A dictionary where keys are tuples of entities and values are lists of features (floats).
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


def process_dataset(path_dataset_file):
    """
    Process a dataset file with labels and store data in a list of lists.
    Parameters:
        path_dataset_file (str): Path to the dataset file to be processed.
    Returns:
        list_labels (list): A list of lists, where each list contains a tuple representing a pair of entities and a label.
    """
    list_labels = []
    with open(path_dataset_file, 'r') as dataset:
        for line in dataset:
            split1 = line.split('\t')
            ent1, ent2 = split1[0], split1[1]
            label = int(split1[2][:-1])
            list_labels.append([(ent1, ent2), label])
    return list_labels


def read_representation_dataset_file(path_file_representation, path_dataset_file):
    """
    Read representation and dataset files, and return entities, representations, and labels.
    Parameters:
        path_file_representation (str): Path to the representation file.
        path_dataset_file (str): Path to the dataset file.
    Returns:
        list_ents (list of lists): Each inner list contains the pair of entities [ent1, ent2].
        list_representation (list of lists): Each inner list contains the representation floats for an entity pair.
        labels (list): List of labels corresponding to each entity pair.
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


def process_indexes_partition(file_partition):
    """
    Process the partition file and returns a list of indexes.
    Parameters:
        file_partition (str): Path to the file containing partition indexes partition file path (each line is a index).
    Return: 
        indexes_partition (list): A list of integers representing partition indexes.
    """
    file_partitions = open(file_partition, 'r')
    indexes_partition = []
    for line in file_partitions:
        indexes_partition.append(int(line[:-1]))
    file_partitions.close()
    return indexes_partition


def run_ml_model(path_file_representation, path_dataset_file, n_partition, path_partition, alg, file_output):
    """
    Run ML models on partitioned data and save ML models to files.
    Parameters:
        path_file_representation (str): Path to the representation file.
        path_dataset_file (str): Path to the dataset file.
        n_partition (int): Number of partitions.
        path_partition (str): Path where partition files are located.
        alg (str): Algorithm to use ("XGB" for XGBoost, "RF" for Random Forest, "MLP" for Multi-layer Perceptron).
        file_output (str): Path where output ML models will be saved.
    Returns:
        none
    """

    # Read representations, entities, and labels from files
    list_ents, list_feat, list_labels = read_representation_dataset_file(path_file_representation, path_dataset_file)
    n_pairs = len(list_labels)

    # Iterate through each partition
    for run in range(1, n_partition + 1):

        file_partition = path_partition + str(run) + '.txt'
        test_index = process_indexes_partition(file_partition)
        train_index = list(set(range(0, n_pairs)) - set(test_index))

        list_labels = np.array(list_labels)
        y_train, y_test = list_labels[train_index], list_labels[test_index]
        y_train, y_test = list(y_train), list(y_test)

        list_feat = np.array(list_feat)
        X_train, X_test = list_feat[train_index], list_feat[test_index]
        X_train, X_test = list(X_train), list(X_test)

        # Initialize and train machine learning model based on chosen algorithm
        if alg == "XGB":
            ml_model = xgb.XGBClassifier()
            ml_model.fit(np.array(X_train), np.array(y_train))

        if alg == "RF":
            ml_model = RandomForestClassifier()
            ml_model.fit(X_train, y_train)

        if alg == "MLP":
            ml_model = MLPClassifier()
            ml_model.fit(X_train, y_train)

        # Save trained model to file using pickle
        ensure_dir(file_output)
        pickle.dump(ml_model, open(file_output + "Model_" + alg + "_" + str(run) + ".pickle", "wb"))


# if __name__== '__main__':
    
#     path_file_representation = "./PPI/Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt" 
#     path_dataset_file = "./PPI/STRING_pairs.txt"
#     n_partition = 10 
#     path_partition = "./PPI/StratifiedPartitions/Indexes__crossvalidationTest__Run"
#     alg = "RF" 
#     file_output = "./PPI/Models/RF/"
#     run_ml_model(path_file_representation, path_dataset_file, n_partition, path_partition, alg, file_output)