import dgl
import scipy
import random
import torch
import pandas as pd
import numpy as np
import scipy.io as sio
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder,StandardScaler
from scipy import sparse


def extract_time_features(df):
    df['access_hour'] = pd.to_datetime(df['C'], unit='s').dt.hour
    df['latest_hour'] = pd.to_datetime(df['latest_time'], unit='s').dt.hour
    df['time_diff'] = (df['latest_time'] - df['C']) / 3600
    return df

def load_csv_UNSW_NB15(filepath, train_rate=0.7, val_rate=0.3):
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df['proto'] = le.fit_transform(df['proto'])
    df['service'] = le.fit_transform(df['service'])
    df['state'] = le.fit_transform(df['state'])
    unique_nodes = pd.unique(df[['spkts', 'dpkts']].values.ravel('K'))
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)
    num_edges = df.shape[0]
    rows = df['spkts'].map(node_mapping).values
    cols = df['dpkts'].map(node_mapping).values
    data = np.ones(num_edges)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T  
    adj = sp.csr_matrix(adj)
    features = df[['dur', 'proto', 'service', 'state', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']].values
    feat = sp.lil_matrix(features)
    epsilon = 0.4  
    feat_original = feat  
    feat = feat + epsilon * np.random.normal(loc=0.0, scale=1.0, size=feat.shape)

    labels = df['label'].values
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)

    ano_labels = df['label'].values
    str_ano_labels = ano_labels
    attr_ano_labels = ano_labels  

    num_train = int(num_nodes * train_rate)
    num_val = int(num_nodes * val_rate)
    all_idx = list(range(num_nodes))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def adj_to_dict(adj,hop=1,min_len=8):
    adj = np.array(adj.todense(),dtype=np.float64)
    num_node = adj.shape[0]
    # adj += np.eye(num_node)

    adj_diff = adj
    if hop > 1:
        for _ in range(hop - 1):
            adj_diff = adj_diff.dot(adj)


    dict = {}
    for i in range(num_node):
        dict[i] = []
        for j in range(num_node):
            if adj_diff[i,j] > 0:
                dict[i].append(j)

    final_dict = dict.copy()

    for i in range(num_node):
        while len(final_dict[i]) < min_len:
            final_dict[i].append(random.choice(dict[random.choice(dict[i])]))
    return dict

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.7, val_rate=0.1):
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv

def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()
    num_drop = int(len(row_idx) * percent)

    edge_index = [i for i in range(len(row_idx))]
    edges = dict(zip(edge_index, zip(row_idx, col_idx)))
    drop_idx = random.sample(edge_index, k=num_drop)

    list(map(edges.__delitem__, filter(edges.__contains__, drop_idx)))

    new_edges = list(zip(*list(edges.values())))
    new_row_idx = new_edges[0]
    new_col_idx = new_edges[1]
    data = np.ones(len(new_row_idx)).tolist()

    new_adj = sp.csr_matrix((data, (new_row_idx, new_col_idx)), shape=input_adj.shape)

    row_idx, col_idx = (new_adj.todense() - 1).nonzero()
    no_edges_cells = list(zip(row_idx, col_idx))
    add_idx = random.sample(no_edges_cells, num_drop)
    new_row_idx_1, new_col_idx_1 = list(zip(*add_idx))
    row_idx = new_row_idx + new_row_idx_1
    col_idx = new_col_idx + new_col_idx_1
    data = np.ones(len(row_idx)).tolist()
    new_adj = sp.csr_matrix((data, (row_idx, col_idx)), shape=input_adj.shape)
    return new_adj

def enhance_sparse_graph_NF(graph_matrix, target_dim=None, threshold=0.0):
    if target_dim is None:
        target_dim = min(graph_matrix.shape) - 1  
    
    u, s, vh = scipy.sparse.linalg.svds(graph_matrix, k=target_dim)  
    s_diag = np.diag(s)
    u_s = np.dot(u, s_diag)
    v_s = np.dot(vh.T, s_diag)
    enhanced_graph = np.dot(u_s, v_s.T)
    enhanced_graph[np.abs(enhanced_graph) < threshold] = 0
    enhanced_graph_sparse = scipy.sparse.csr_matrix(enhanced_graph)

    return enhanced_graph_sparse

def calculate_isomorphism_similarity(adj1, adj2):
    adj1 = adj1.cpu().numpy()
    adj2 = adj2.cpu().numpy()
    intersection = np.sum(np.logical_and(adj1, adj2))
    union = np.sum(np.logical_or(adj1, adj2))
    return intersection / union if union != 0 else 0

def calculate_topology_similarity(adj1, adj2):
    adj1 = adj1.cpu().numpy()
    adj2 = adj2.cpu().numpy()

    intersection = np.sum(np.logical_and(adj1, adj2))
    union = np.sum(np.logical_or(adj1, adj2))
    num_nodes1 = adj1.shape[0]
    num_nodes2 = adj2.shape[0]
    node_similarity = min(num_nodes1, num_nodes2) / max(num_nodes1, num_nodes2)
    return (intersection / union if union!= 0 else 0) * node_similarity

def load_csv_resampled_test(filepath, train_rate=0.8, val_rate=0.2, sample_rate=0.5):
    df = pd.read_csv(filepath)
    df = df.sample(frac=sample_rate, random_state=42)
    df = extract_time_features(df)
    le = LabelEncoder()
    df['attack_stage'] = le.fit_transform(df['attack_stage'])
    unique_nodes = pd.unique(df[['alarm_sip', 'attack_sip']].values.ravel('K'))
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)
    num_edges = df.shape[0]
    rows = df['alarm_sip'].map(node_mapping).values
    cols = df['attack_sip'].map(node_mapping).values
    data = np.ones(num_edges)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T  
    adj = sp.csr_matrix(adj)
    selected_features = df[['latitude', 'longitude', 'time_diff', 'access_hour', 'latest_hour', 'location', 'name_type_chain']]
    selected_features = selected_features.apply(pd.to_numeric, errors='coerce')
    selected_features.fillna(0, inplace=True)
    scaler = StandardScaler()
    features = scaler.fit_transform(selected_features)
    feat = sp.lil_matrix(features)
    epsilon = 0 
    feat = feat + epsilon * np.random.normal(loc=0.0, scale=1.0, size=feat.shape)

    labels = df['attack_stage'].values
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)

    ano_labels = df['attack_stage'].values
    str_ano_labels = ano_labels  
    attr_ano_labels = ano_labels 

    num_train = int(num_nodes * train_rate)
    num_val = int(num_nodes * val_rate)
    all_idx = list(range(num_nodes))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def preprocess_features(features):
    rowsum = np.array(features.sum(1), dtype=np.float32)
    rowsum[rowsum == 0] = 1  
    r_inv = np.power(rowsum, -1).flatten()
    r_mat_inv = sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.matrix):
        features = sparse.csr_matrix(features)
    return features.todense(), sparse_to_tuple(features)

def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

def load_NF_CSE_CIC_IDS2018(filepath, train_rate=0.7, val_rate=0.3,sample_fraction=0.1):
    df = pd.read_csv(filepath)
    df = df.sample(frac=sample_fraction, random_state=1) 
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    df['Binary_Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    unique_nodes = pd.unique(df[['Dst Port', 'Protocol']].values.ravel('K'))
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)
    num_edges = df.shape[0]
    rows = df['Dst Port'].map(node_mapping).values
    cols = df['Protocol'].map(node_mapping).values
    data = np.ones(num_edges)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T 
    adj = sp.csr_matrix(adj)
    features = df[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
                   'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 
                   'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 
                   'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 
                   'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 
                   'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
                   'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 
                   'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 
                   'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 
                   'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 
                   'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 
                   'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 
                   'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 
                   'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']]

    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.fillna(0)  

    feat = sp.lil_matrix(features.values)
    epsilon = 0.1 
    feat_original = feat 
    feat = feat + epsilon * np.random.normal(loc=0.0, scale=1.0, size=feat.shape)
    labels = df['Binary_Label'].values
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)
    ano_labels = df['Binary_Label'].values
    str_ano_labels = ano_labels 
    attr_ano_labels = ano_labels 
    num_train = int(num_nodes * train_rate)
    num_val = int(num_nodes * val_rate)
    all_idx = list(range(num_nodes))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]
    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels