from utils import *
from sklearn.metrics import roc_auc_score, f1_score
import random
import os
import dgl
import argparse
import torch.nn as nn
import torch.nn.functional as F
from model_dy import Model
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score, precision_score
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='MG-CNID')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='our')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)  
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=128)  
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--num_epoch', type=int, default=1500)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.1, help='how much the first view involves')
parser.add_argument('--beta', type=float, default=0.1, help='how much the second view involves')
parser.add_argument('--c', type=float, default=0.1, help='how much the second view involves')
parser.add_argument('--d', type=float, default=0.1, help='how much the second view involves')

args = parser.parse_args()

if __name__ == '__main__':
    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_losses = []
    test1_losses = []
    train_accuracies = []
    test1_accuracies = []
    train_f1_scores = []
    test1_f1_scores = []
    train_detection_rates = []
    test1_detection_rates = []
    train_aucs=[]
    train_precisions=[]
    test1_aucs=[]
    test1_precisions=[]

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

        dataset = r'D:/MG_CNID/MG_CNID\dataset/resampled_samples_AptsData_副本.csv'

        batch_size = args.batch_size
        subgraph_size = args.subgraph_size

        adj, feat, labels, idx_train, idx_test1, idx_test, ano_labels, str_ano_labels, attr_ano_labels = load_csv_resampled_test(dataset)

        print(f'Run: {run} with random seed: {seed} - After data loading', flush=True)

        non_zero_elements = adj.nnz
        total_elements = adj.shape[0] * adj.shape[1]
        sparsity = non_zero_elements / total_elements
        degrees = np.array(adj.sum(axis=1)).flatten()
        unique, counts = np.unique(degrees, return_counts=True)

        features, _ = preprocess_features(feat)
        dgl_graph = adj_to_dgl_graph(adj)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        nb_classes = labels.shape[1]
        
        adj_svg = enhance_sparse_graph_NF(adj,50,0.05) 

        adj_edge_modification = aug_random_edge(adj_svg, 0.2)
        adnon_zero_sparse = adj.nnzj = normalize_adj(adj)
        
        non_zero_dense = np.count_nonzero(adj.todense())
        adj = (adj + sp.eye(adj.shape[0])).todense()

        adj_hat = normalize_adj(adj_edge_modification)
        adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis]).to(device)
        adj = torch.FloatTensor(adj[np.newaxis]).to(device)
        adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
        labels = torch.FloatTensor(labels[np.newaxis]).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_test1 = torch.LongTensor(idx_test1).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)
        all_auc = []

        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)

        MG = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context, args.readout).to(device)
        optimiser = torch.optim.Adam(MG.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num_train = len(idx_train) // batch_size + 1
        batch_num_test1 = len(idx_test1) // batch_size + 1

        for epoch in range(1,args.num_epoch+1):
            MG.train()

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            total_loss = 0.
            total_correct = 0
            total_samples = 0
            TP_train = 0
            FN_train = 0
            batch_losses = []  
            batch_corrects = [] 
            batch_samples = []  
            for batch_idx in range(batch_num_train):
                optimiser.zero_grad() 
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(idx_train))
                cur_idx = idx_train[start_idx:end_idx]

                cur_batch_size = len(cur_idx)
                lbl_patch = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device) #二元交叉熵损失计算的标签，指导模型如何区分正样本（positive samples）和负样本（negative samples）
                lbl_context = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in cur_idx:  
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]

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
                sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:, nega_list]
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

                loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat  # node-subgraph contrast loss
                loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat  # node-node contrast loss

                isomorphism_similarity_graph = calculate_topology_similarity(adj, adj_hat)
                isomorphism_similarity_subgraph = calculate_topology_similarity(ba, ba_hat)
                loss = args.beta * loss_1 + (1 - args.beta) * loss_2 + args.c * NCE_loss + args.d * (isomorphism_similarity_graph + isomorphism_similarity_subgraph)
                loss.backward()  
                optimiser.step()  

                loss = loss.detach().cpu().numpy()
                batch_losses.append(loss) 
                with torch.no_grad():
                    preds = torch.sigmoid(logits_1).round().cpu().numpy()
                    labels_np = lbl_context.cpu().numpy()
                    correct = (preds == labels_np).sum()
                    batch_corrects.append(correct)  
                    batch_samples.append(len(labels_np)) 
                    train_f1 = f1_score(labels_np, preds, average='macro')
                    train_auc = roc_auc_score(labels_np, preds)
                    train_precision = precision_score(labels_np, preds, average='macro')

                    TP_train += ((preds == 1) & (labels_np == 1)).sum()
                    FN_train += ((preds == 0) & (labels_np == 1)).sum()

            mean_loss = np.mean(batch_losses)  
            total_correct = np.sum(batch_corrects) 
            total_samples = np.sum(batch_samples)  
            train_losses.append(mean_loss)
            train_accuracies.append(total_correct / total_samples)
            train_f1_scores.append(train_f1)
            train_aucs.append(train_auc)
            train_precisions.append(train_precision)
            train_detection_rates.append(TP_train / (TP_train + FN_train))

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(MG.state_dict(), 'our.pkl'.format(args.dataset))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f} Acc:{:.4f} F1:{:.4f} AUC:{:.4f} Precision:{:.4f} DR:{:.4f}'.format(
                epoch, mean_loss, total_correct / total_samples, train_f1, train_auc, train_precision, train_detection_rates[-1]), flush=True)
        print('Loading {}th epoch'.format(best_t), flush=True)
        MG.load_state_dict(torch.load('our.pkl'.format(args.dataset)))

        test1_accuracies = []
        test1_f1_scores = []
        test1_aucs = []
        test1_precisions = []
        test1_detection_rates = []
        
        MG.eval()
        with torch.no_grad():
            test1_correct = 0
            test1_samples = 0
            test1_f1 = 0
            test1_auc = 0
            test1_precision = 0
            TP_test1 = 0
            FN_test1 = 0

            for batch_idx in range(batch_num_test1):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(idx_test1))
                cur_idx = idx_test1[start_idx:end_idx]

                cur_batch_size = len(cur_idx)
                lbl_patch = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)
                lbl_context = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in cur_idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]

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

                preds = torch.sigmoid(logits_1).round().cpu().numpy()
                labels_np = lbl_context.cpu().numpy()
                test1_correct += (preds == labels_np).sum()
                test1_samples += len(labels_np)
                test1_f1 = f1_score(labels_np, preds, average='macro')
                test1_auc = roc_auc_score(labels_np, preds)
                test1_precision = precision_score(labels_np, preds, average='macro')

                TP_test1 += ((preds == 1) & (labels_np == 1)).sum()
                FN_test1 += ((preds == 0) & (labels_np == 1)).sum()

            test1_accuracies.append(test1_correct / test1_samples)
            test1_f1_scores.append(test1_f1)
            test1_aucs.append(test1_auc)
            test1_precisions.append(test1_precision)
            test1_detection_rates.append(TP_test1 / (TP_test1 + FN_test1))

            print('test Acc:{:.4f} F1:{:.4f} AUC:{:.4f} Precision:{:.4f} DR:{:.4f}'.format(
                test1_correct / test1_samples, test1_f1, test1_auc, test1_precision, test1_detection_rates[-1]), flush=True)
        print("complete.\n")
        print("\nAll runs complete. Exiting.")

