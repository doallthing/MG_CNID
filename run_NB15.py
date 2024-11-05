from utils import *
from sklearn.metrics import roc_auc_score, f1_score
import random
import os
import dgl
import argparse
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, precision_score
from model_dy import Model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='MG-CNID')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='NB_15')
parser.add_argument('--lr', type=float, default=1e-4)  
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=128)  
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--num_epoch', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.1, help='how much the first view involves')
parser.add_argument('--beta', type=float, default=0.1, help='how much the second view involves')
args = parser.parse_args()


if __name__ == '__main__':
    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device number: {torch.cuda.current_device()}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_detection_rates = []
    val_detection_rates = []
    train_aucs = []
    train_precisions = []
    val_aucs = []
    val_precisions = []
    prev_train_auc=0
    weight_feature_history = []
    weight_topology_history = []
    dynamic_weight_feature = 0.5
    dynamic_weight_topology = 0.5
    weight_feature_history = []
    weight_topology_history = []

    for run in range(args.runs):
        seed = run + 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dgl.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f'Run: {run} with random seed: {seed} - Before data loading', flush=True)

        train_dataset = r'D:/MG_CNID/used/UNSW_NB15_training-set.csv'
        val_dataset = r'D:/MG_CNID/used/UNSW_NB15_testing-set.csv'

        batch_size = args.batch_size
        subgraph_size = args.subgraph_size
        adj_train, feat_train, labels_train, idx_train_train, idx_val_train, idx_test_train, ano_labels_train, str_ano_labels_train, attr_ano_labels_train = load_csv_UNSW_NB15(
            train_dataset)
        adj_val, feat_val, labels_val, idx_train_val, idx_val_val, idx_test_val, ano_labels_val, str_ano_labels_val, attr_ano_labels_val = load_csv_UNSW_NB15(
            val_dataset)

        print(f'Run: {run} with random seed: {seed} - After data loading', flush=True)

        non_zero_elements_train = adj_train.nnz
        total_elements_train = adj_train.shape[0] * adj_train.shape[1]
        sparsity_train = non_zero_elements_train / total_elements_train
        degrees_train = np.array(adj_train.sum(axis=1)).flatten()
        unique_train, counts_train = np.unique(degrees_train, return_counts=True)
        features_train, _ = preprocess_features(feat_train)
        dgl_graph_train = adj_to_dgl_graph(adj_train)
        nb_nodes_train = features_train.shape[0]
        ft_size_train = features_train.shape[1]
        nb_classes_train = labels_train.shape[1]
        adj_svg_train = enhance_sparse_graph_NF(adj_train) 
        adj_edge_modification_train = aug_random_edge(adj_train, 0.2)  
        adj_train = normalize_adj(adj_train)
        non_zero_sparse_train = adj_train.nnz
        non_zero_dense_train = np.count_nonzero(adj_train.todense())
        adj_train = (adj_train + sp.eye(adj_train.shape[0])).todense()
        adj_hat_train = normalize_adj(adj_edge_modification_train)
        adj_hat_train = (adj_hat_train + sp.eye(adj_hat_train.shape[0])).todense()
        features_train = torch.FloatTensor(features_train[np.newaxis]).to(device)
        adj_train = torch.FloatTensor(adj_train[np.newaxis]).to(device)
        adj_hat_train = torch.FloatTensor(adj_hat_train[np.newaxis]).to(device)
        labels_train = torch.FloatTensor(labels_train[np.newaxis]).to(device)
        idx_train_train = torch.LongTensor(idx_train_train).to(device)
        idx_val_train = torch.LongTensor(idx_val_train).to(device)
        idx_test_train = torch.LongTensor(idx_test_train).to(device)
        all_auc = []
        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        MG = Model(ft_size_train, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                    args.readout).to(device)
        optimiser = torch.optim.Adam(MG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                             pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num_train = nb_nodes_train // batch_size + 1

        for epoch in range(args.num_epoch):
            MG.train()

            subgraphs_train = generate_rwr_subgraph(dgl_graph_train, subgraph_size)

            nb_nodes_train = min(nb_nodes_train, len(subgraphs_train))

            all_idx_train = list(range(nb_nodes_train))
            random.shuffle(all_idx_train)
            total_loss = 0.
            total_correct = 0
            total_samples = 0
            TP_train = 0
            FN_train = 0

            for batch_idx in range(batch_num_train):
                optimiser.zero_grad()
                is_final_batch = (batch_idx == (batch_num_train - 1))
                if not is_final_batch:
                    idx = all_idx_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx_train[batch_idx * batch_size:]

                cur_batch_size = len(idx)
                lbl_patch = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))),
                                           1).to(device)
                lbl_context = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))),
                                              1).to(device)

                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size_train)).to(device)

                for i in idx:
                    cur_adj = adj_train[:, subgraphs_train[i], :][:, :, subgraphs_train[i]]
                    cur_adj_hat = adj_hat_train[:, subgraphs_train[i], :][:, :, subgraphs_train[i]]
                    cur_feat = features_train[:, subgraphs_train[i], :]

                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                if ba:
                    ba = torch.cat(ba)
                    added_adj_zero_row = added_adj_zero_row.expand(ba.shape[0], -1, ba.shape[2])
                    ba = torch.cat((ba, added_adj_zero_row), dim=1)
                    ba = torch.cat((ba, added_adj_zero_col.expand(ba.shape[0], -1, ba.shape[2])), dim=2)
                else:
                    continue

                if ba_hat:
                    ba_hat = torch.cat(ba_hat)
                    added_adj_zero_row = added_adj_zero_row.expand(ba_hat.shape[0], -1, ba_hat.shape[2])
                    ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                    ba_hat = torch.cat((ba_hat, added_adj_zero_col.expand(ba_hat.shape[0], -1, ba_hat.shape[2])), dim=2)
                else:
                    continue

                if bf:
                    bf = torch.cat(bf)
                    added_feat_zero_row = added_feat_zero_row.expand(bf.shape[0], -1, bf.shape[2])
                    bf = torch.cat((bf, added_feat_zero_row), dim=1)
                else:
                    continue

                logits_1, logits_2, subgraph_embed, node_embed = MG(bf, ba)
                logits_1_hat, logits_2_hat, subgraph_embed_hat, node_embed_hat = MG(bf, ba_hat)

                logits_1_all, logits_2_all, graph_embed, node_embed_all = MG(bf, ba)
                logits_1_hat_all, logits_2_hat_all, graph_embed_hat, node_embed_hat_all = MG(bf, ba_hat)

                subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
                subgraph_embed_hat = F.normalize(subgraph_embed_hat, dim=1, p=2)
                sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
                sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
                sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
                temperature = 1.0
                sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
                sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
                sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
                nega_list = np.arange(0, cur_batch_size - 1, 1)
                nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
                sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:,
                                                                                                      nega_list]
                sim_row_sum = torch.diagonal(sim_row_sum)
                sim_diag = torch.diagonal(sim_matrix_one)
                sim_diag_exp = torch.exp(sim_diag / temperature)
                NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
                NCE_loss = torch.mean(NCE_loss)

                loss_all_1 = b_xent_context(logits_1, lbl_context)
                loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)
                loss_1 = torch.mean(loss_all_1)
                loss_1_hat = torch.mean(loss_all_1_hat)

                loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)
                loss_2 = torch.mean(loss_all_2)
                loss_2_hat = torch.mean(loss_all_2_hat)

                loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat 
                loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat 
                isomorphism_similarity_graph = calculate_isomorphism_similarity(adj_train, adj_hat_train)
                isomorphism_similarity_subgraph = calculate_isomorphism_similarity(ba, ba_hat)
                loss =args.beta * loss_1 + (1 - args.beta) * loss_2 + 0.1 * NCE_loss
                loss.backward()
                optimiser.step()
                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss
                with torch.no_grad():
                    preds = torch.sigmoid(logits_1).round().cpu().numpy()
                    labels_np = lbl_context.cpu().numpy()
                    total_correct += (preds == labels_np).sum()
                    total_samples += len(labels_np)
                    train_f1 = f1_score(labels_np, preds, average='macro')
                    train_auc = roc_auc_score(labels_np, preds)
                    train_precision = precision_score(labels_np, preds, average='macro')
                    TP_train += ((preds == 1) & (labels_np == 1)).sum()
                    FN_train += ((preds == 0) & (labels_np == 1)).sum()

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes_train
            train_losses.append(mean_loss)
            train_accuracies.append(total_correct / total_samples)
            train_f1_scores.append(train_f1)
            train_aucs.append(train_auc)
            train_precisions.append(train_precision)
            train_detection_rates.append(TP_train / (TP_train + FN_train))
            dynamic_weight_feature = max(0, min(1, dynamic_weight_feature))
            dynamic_weight_topology = max(0, min(1, dynamic_weight_topology))
            prev_train_auc = train_auc 
            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0  
                torch.save(MG.state_dict(), 'UNSW_NB15_testing.pkl'.format(args.dataset))  
            else:  
                cnt_wait += 1  
            if cnt_wait == args.patience:  
                print('Early stopping!', flush=True)  
                break  
            print('Epoch:{} Loss:{:.8f} Acc:{:.4f} F1:{:.4f} AUC:{:.4f} Precision:{:.4f} DR:{:.4f}'.format(  
                epoch, mean_loss, total_correct / total_samples, train_f1, train_auc, train_precision, train_detection_rates[-1]), flush=True)  
        MG.eval()  
        with torch.no_grad():  
            val_loss = 0.  
            val_correct = 0  
            val_samples = 0  
            val_f1 = 0  
            val_auc = 0  
            val_precision = 0  
            TP_val = 0  
            FN_val = 0  
            features_val, _ = preprocess_features(feat_val)  
            dgl_graph_val = adj_to_dgl_graph(adj_val)  
            adj_val = normalize_adj(adj_val)  
            adj_hat_val = normalize_adj(adj_edge_modification_train)  
            adj_val = (adj_val + sp.eye(adj_val.shape[0])).todense()  
            adj_hat_val = (adj_hat_val + sp.eye(adj_hat_val.shape[0])).todense()  
        
            features_val = torch.FloatTensor(features_val[np.newaxis]).to(device)  
            adj_val = torch.FloatTensor(adj_val[np.newaxis]).to(device)  
            adj_hat_val = torch.FloatTensor(adj_hat_val[np.newaxis]).to(device)  
            labels_val = torch.FloatTensor(labels_val[np.newaxis]).to(device)  
        
            subgraphs_val = generate_rwr_subgraph(dgl_graph_val, subgraph_size)  
            nb_nodes_val = features_val.shape[0]  
            batch_num_val = nb_nodes_val // batch_size + 1  
        
            for batch_idx in range(batch_num_val):  
                is_final_batch = (batch_idx == (batch_num_val - 1))  
                if not is_final_batch:  
                    idx = all_idx_train[batch_idx * batch_size: (batch_idx + 1) * batch_size]  
                else:  
                    idx = all_idx_train[batch_idx * batch_size:]  
                cur_batch_size = len(idx)  
        
                lbl_patch = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)  
                lbl_context = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)  
        
                ba = []  
                ba_hat = []  
                bf = []  
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)  
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)  
                added_adj_zero_col[:, -1, :] = 1.  
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size_train)).to(device)  
        
                for i in idx:  
                    cur_adj = adj_val[:, subgraphs_val[i], :][:, :, subgraphs_val[i]]  
                    cur_adj_hat = adj_hat_val[:, subgraphs_val[i], :][:, :, subgraphs_val[i]]  
                    cur_feat = features_val[:, subgraphs_val[i], :]  
                    ba.append(cur_adj)  
                    ba_hat.append(cur_adj_hat)  
                    bf.append(cur_feat)  
        
                if ba:  
                    ba = torch.cat(ba)  
                    added_adj_zero_row = added_adj_zero_row.expand(ba.shape[0], -1, ba.shape[2])  
                    ba = torch.cat((ba, added_adj_zero_row), dim=1)  
                    ba = torch.cat((ba, added_adj_zero_col.expand(ba.shape[0], -1, ba.shape[2])), dim=2)  
                else:  
                    continue  
        
                if ba_hat:  
                    ba_hat = torch.cat(ba_hat)  
                    added_adj_zero_row = added_adj_zero_row.expand(ba_hat.shape[0], -1, ba_hat.shape[2])  
                    ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)  
                    ba_hat = torch.cat((ba_hat, added_adj_zero_col.expand(ba_hat.shape[0], -1, ba_hat.shape[2])), dim=2)  
                else:  
                    continue  
        
                if bf:  
                    bf = torch.cat(bf)  
                    added_feat_zero_row = added_feat_zero_row.expand(bf.shape[0], -1, bf.shape[2])  
                    bf = torch.cat((bf, added_feat_zero_row), dim=1)  
                else:  
                    continue  
        
                logits_1, logits_2, subgraph_embed, node_embed = MG(bf, ba)  
                logits_1_hat, logits_2_hat, subgraph_embed_hat, node_embed_hat = MG(bf, ba_hat)  
                logits_1_all, logits_2_all, graph_embed, node_embed_all = MG(bf, ba)  
                logits_1_hat_all, logits_2_hat_all, graph_embed_hat, node_embed_hat_all = MG(bf, ba_hat)  
        
                loss_all_1 = b_xent_context(logits_1, lbl_context)  
                loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)  
                loss_1 = torch.mean(loss_all_1)  
                loss_1_hat = torch.mean(loss_all_1_hat)  
                loss_all_2 = b_xent_patch(logits_2, lbl_patch)  
                loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)  
                loss_2 = torch.mean(loss_all_2)  
                loss_2_hat = torch.mean(loss_all_2_hat)  
        
                loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat   
                loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat  
        
                isomorphism_similarity_graph = calculate_isomorphism_similarity(adj_val, adj_hat_val)  
                isomorphism_similarity_subgraph = calculate_isomorphism_similarity(ba, ba_hat)  
        
                loss = dynamic_weight_feature * (args.beta * loss_1 + (1 - args.beta) * loss_2) + dynamic_weight_topology * 0.1 * (  
                    isomorphism_similarity_graph + isomorphism_similarity_subgraph)  
                loss = args.beta * loss_1  
                loss = loss.detach().cpu().numpy()  
        
                if not is_final_batch:  
                    val_loss += loss  
        
                preds = torch.sigmoid(logits_1).round().cpu().numpy()  
                labels_np = lbl_context.cpu().numpy()  
                val_correct += (preds == labels_np).sum()  
                val_samples += len(labels_np)  
                val_f1 = f1_score(labels_np, preds, average='macro')  
                val_auc = roc_auc_score(labels_np, preds)  
                val_precision = precision_score(labels_np, preds, average='macro')  
                
                TP_val += ((preds == 1) & (labels_np == 1)).sum()  
                FN_val += ((preds == 0) & (labels_np == 1)).sum()  
                
                mean_val_loss = (val_loss * batch_size + loss * cur_batch_size) / nb_nodes_val  
                val_losses.append(mean_val_loss)  
                val_accuracies.append(val_correct / val_samples)  
                val_f1_scores.append(val_f1)  
                val_aucs.append(val_auc)  
                val_precisions.append(val_precision)  
                val_detection_rates.append(TP_val / (TP_val + FN_val))  
                
                print('Validation Epoch:{} Loss:{:.8f} Acc:{:.4f} F1:{:.4f} AUC:{:.4f} Precision:{:.4f} DR:{:.4f}'.format(  
                    epoch, mean_val_loss, val_correct / val_samples, val_f1, val_auc, val_precision, val_detection_rates[-1]), flush=True)  
                print("Training complete.\n")  
                

