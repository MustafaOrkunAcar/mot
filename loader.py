import os

import numpy as np
from scipy.sparse import coo_matrix
from spektral.data import Dataset, Graph, DisjointLoader


class MOTDataset(Dataset):

    def __init__(self, data_source, **kwargs):
        self.data_source = data_source
        super().__init__(**kwargs)

    def read(self):

        graphs = []
        root_dir = self.data_source

        subfolders = [ obj.path for obj in os.scandir(root_dir) if obj.is_dir() ]
        for folder in subfolders:

            folder_name = os.path.basename(folder)

            a_file_name = folder_name + '.graph'
            x_file_name = folder_name + '.features'
            e_file_name = folder_name + '.efeatures'
            l_file_name = folder_name + '.labels'

            a_file = os.path.join(folder, a_file_name)
            x_file = os.path.join(folder, x_file_name)
            e_file = os.path.join(folder, e_file_name)
            l_file = os.path.join(folder, l_file_name)
            
            # Adjacency - Scipy sparse matrix, coo format
            with open(a_file, 'r') as f:
                a_lines = f.readlines()
            a_edges = len(a_lines)
            a_maxnodeid = 0
            a_row = []
            a_col = []
            for l in a_lines:
                r, c = l[:-1].split()
                r = int(r)
                c = int(c)
                a_maxnodeid = max(a_maxnodeid, r, c)
                a_row.append(r)
                a_col.append(c)
            # node_ids start from zero
            a_nodes = a_maxnodeid + 1
            data = np.ones((a_edges,))
            a = coo_matrix((data, (a_row, a_col)), shape=(a_nodes, a_nodes))
            # The adjacency matrix is returned as a SparseTensor, regardless of the input.

            # dense np.array
            x = np.loadtxt(x_file, delimiter=' ')
            
            e = np.loadtxt(e_file, delimiter=' ')

            # [n_nodes, ] 
            y = np.loadtxt(l_file)
    
            graph = Graph(a=a, x=x, e=e, y=y)
            graphs.append(graph)

        return graphs

if __name__ == "__main__":

    data_path = "/home/macar20/motgraphs"    
    mot_dataset = MOTDataset(data_path)

    # shuffles the order of graphs in dataset
    # need to shuffle only for minibatch/SGD, no need for batch gradient descent (?) & shuffling needs to be done before batching
    np.random.shuffle(mot_dataset)    
    
    # graph-based split. 4 graph train, 1 graph test
    # len gets len of self.graphs
    split = int(0.8 * len(mot_dataset))

    # __getitem__ on Dataset applies indexing/slicing on self.graphs attribute.
    # self.graphs is populated by return value of read() call.
    train_data, test_data = mot_dataset[:split], mot_dataset[split:] 

    # batch_size = 1 -> each next call will return a single graph
    # when >1 -> batch many graphs are combined to a single graph obj
    # when batch size cannot divide sample size, the last batch will be smaller
    # when batch >= sample, all samples will be in a single graph obj
    batch_size = 3
    # iterators. next() returns a 2-tuple: 
    # (
    #       4-tuple (
    #           x: numpy.ndarray[shape: (n_nodes, n_node_features)], 
    #           a: tensorflow.python.framework.sparse_tensor.SparseTensor[shape: (n_nodes, n_nodes)], 
    #           e: numpy.ndarray[shape: (n_label, n_edge_features)], 
    #           i: numpy.ndarray[shape: (n_nodes, )]
    #       )
    #       y: numpy.ndarray[shape: (1, n_labels)] 
    # )


    # epoch: iterator gives x Epoch times the same data
    train_loader = DisjointLoader(train_data, node_level=True, batch_size=batch_size, epochs=1)   
    test_loader = DisjointLoader(test_data, node_level=True, batch_size=batch_size)              


    

