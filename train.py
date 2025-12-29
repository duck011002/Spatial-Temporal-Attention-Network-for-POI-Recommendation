import argparse
import os
from load import *
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
from models import *
import logging
from datetime import datetime
import sys
from config import Config

# Define Logger
class Logger:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write(f"Log started at {datetime.now()}\n")
            
    def log(self, message):
        print(message)
        with open(self.filename, 'a') as f:
            f.write(message + '\n')

def calculate_acc(prob, label):
    # prob (N, L) or (N, K) containing indices
    # label (N)
    acc_train = [0, 0, 0, 0]
    
    if prob.shape[1] <= 20 and (prob.dtype == torch.long or prob.dtype == torch.int64):
        # prob contains top-K indices
        topk_predict_batch = prob
        for i, k in enumerate([1, 5, 10, 20]):
            current_k_preds = topk_predict_batch[:, :k]  # (N, k)
            for j, pred_row in enumerate(to_npy(current_k_preds)):
                if to_npy(label)[j] in pred_row:
                    acc_train[i] += 1
    else:
        # prob contains scores
        for i, k in enumerate([1, 5, 10, 20]):
            # topk_batch (N, k)
            _, topk_predict_batch = torch.topk(prob, k=k)
            for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
                # topk_predict (k)
                if to_npy(label)[j] in topk_predict:
                    acc_train[i] += 1

    return np.array(acc_train)


def sample_candidates(label, l_max, num_neg):
    # label: (N) 0-based
    # l_max: int
    # num_neg: int
    N = label.shape[0]
    K = 1 + num_neg
    
    cand_locs = torch.zeros((N, K), dtype=torch.long, device=label.device)
    cand_locs[:, 0] = label + 1 # 1-based positive
    
    for i in range(N):
        true_loc = label[i].item() + 1
        neg = []
        while len(neg) < num_neg:
            r = random.randint(1, l_max)
            if r != true_loc:
                neg.append(r)
        cand_locs[i, 1:] = torch.tensor(neg, device=label.device)
        
    target = torch.zeros(N, dtype=torch.long, device=label.device)
    return cand_locs, target


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


class Trainer:
    def __init__(self, model, record, config):
        self.model = model.to(device)
        self.records = record
        self.config = config
        self.start_epoch = record['epoch'][-1] if load else 1
        
        # Load from config
        self.num_neg = config['train']['num_neg']
        self.interval = 1000
        self.batch_size = config['train']['batch_size']
        self.learning_rate = config['train']['lr']
        self.num_epoch = config['train']['num_epoch']
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0

        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.poi_coords, self.mat2t, self.label, self.len = \
            trajs, mat1, poi_coords, mat2t, labels, lens
            
        # Ensure poi_coords is tensor and padded
        if not isinstance(self.poi_coords, torch.Tensor):
            self.poi_coords = torch.FloatTensor(self.poi_coords)
            
        padded_coords = torch.zeros((self.poi_coords.shape[0] + 1, 2))
        padded_coords[1:] = self.poi_coords
        self.poi_coords = padded_coords.to(device)

        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def predict_topk_in_chunks(self, input_args, k=20, chunk_size=2048):
        # input_args: (train_input, train_m1, poi_coords, train_m2t, train_len)
        train_input, train_m1, poi_coords, train_m2t, train_len = input_args
        N = train_input.shape[0]
        l_max = poi_coords.shape[0] - 1
        
        all_topk_scores = []
        all_topk_indices = []

        for start in range(1, l_max + 1, chunk_size):
            end = min(start + chunk_size, l_max + 1)
            # 1-based candidates
            cand_chunk = torch.arange(start, end, device=device).unsqueeze(0).expand(N, -1) #(N, chunk)
            
            with torch.no_grad():
                out = self.model(train_input, train_m1, poi_coords, train_m2t, train_len, cand_chunk)
                if isinstance(out, tuple): # Handle MoE return (score, aux)
                    scores = out[0]
                else:
                    scores = out
            
            k_prime = min(k, scores.shape[1])
            c_scores, c_indices = torch.topk(scores, k=k_prime, dim=1)
            c_indices = c_indices + start # shift to global 1-based
            
            all_topk_scores.append(c_scores)
            all_topk_indices.append(c_indices)

        all_scores = torch.cat(all_topk_scores, dim=1)
        all_indices = torch.cat(all_topk_indices, dim=1)
        
        final_scores, idx_in_all = torch.topk(all_scores, k=k, dim=1)
        final_indices = torch.gather(all_indices, 1, idx_in_all)
        
        return final_indices

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            total_train_loss = 0.0
            total_rec_loss = 0.0
            total_lb1 = 0.0
            total_lb2 = 0.0
            gate1_usage_accum = None
            train_batches = 0

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    # prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        
                        cand_locs, target = sample_candidates(train_label, l_max, self.num_neg)
                        
                        out = self.model(train_input, train_m1, self.poi_coords, train_m2t, train_len, cand_locs)
                        
                        if isinstance(out, tuple):
                            score, aux = out
                        else:
                            score, aux = out, None
                        
                        loss_rec = F.cross_entropy(score, target)
                        loss_train = loss_rec
                        
                        # Add Aux Loss
                        if aux:
                            lambda1 = self.config['loss']['lambda_lb1']
                            lambda2 = self.config['loss']['lambda_lb2']
                            loss_train = loss_train + lambda1 * aux['lb1'] + lambda2 * aux['lb2']
                            
                            total_lb1 += aux['lb1'].item() if hasattr(aux['lb1'], 'item') else aux['lb1']
                            total_lb2 += aux['lb2'].item() if hasattr(aux['lb2'], 'item') else aux['lb2']
                            
                            if gate1_usage_accum is None:
                                gate1_usage_accum = aux['gate1_usage']
                            else:
                                gate1_usage_accum += aux['gate1_usage']

                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        
                        total_train_loss += loss_train.item()
                        total_rec_loss += loss_rec.item()
                        train_batches += 1

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        
                        topk_indices = self.predict_topk_in_chunks(
                            [train_input, train_m1, self.poi_coords, train_m2t, train_len],
                            k=20, chunk_size=2048
                        )
                        
                        # Fix: train_label is 0-based, topk_indices are 1-based. Align them.
                        acc_valid += calculate_acc(topk_indices, train_label + 1)

                    elif mask_len == person_traj_len[0]:  # only test
                        test_size += person_input.shape[0]
                        
                        topk_indices = self.predict_topk_in_chunks(
                            [train_input, train_m1, self.poi_coords, train_m2t, train_len],
                            k=20, chunk_size=2048
                        )
                        acc_test += calculate_acc(topk_indices, train_label + 1)

                bar.update(self.batch_size)
            bar.close()

            avg_loss = total_train_loss / train_batches if train_batches > 0 else 0.0
            avg_rec = total_rec_loss / train_batches if train_batches > 0 else 0.0
            
            # Log results
            log_msg = 'epoch:{}, time:{:.2f}, total_loss:{:.4f}, rec_loss:{:.4f}'.format(
                self.start_epoch + t, time.time() - start, avg_loss, avg_rec)
            
            if total_lb1 > 0:
                log_msg += ', lb1:{:.4f}, lb2:{:.4f}'.format(total_lb1/train_batches, total_lb2/train_batches)
            
            if gate1_usage_accum is not None:
                total_usage = gate1_usage_accum.sum()
                if total_usage > 0:
                    pct = (gate1_usage_accum / total_usage).cpu().numpy().round(3) * 100
                    log_msg += f', g1_usage%:{pct}'

            if hasattr(self, 'logger'):
                self.logger.log(log_msg)
            else:
                print(log_msg)

            acc_valid = np.array(acc_valid) / valid_size
            if hasattr(self, 'logger'):
                self.logger.log('epoch:{}, time:{:.2f}, valid_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_valid))
            else:
                print('epoch:{}, time:{:.2f}, valid_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_valid))

            acc_test = np.array(acc_test) / test_size
            if hasattr(self, 'logger'):
                self.logger.log('epoch:{}, time:{:.2f}, test_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_test))
            else:
                print('epoch:{}, time:{:.2f}, test_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_test))

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           'best_hdg_moe_' + dname + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory (e.g. data/TSMC_NYC)')
    parser.add_argument('dataset_name', type=str, help='Dataset name (e.g. NYC)')
    parser.add_argument('--part', type=int, default=100, help='Partition size of data to use (default: 100)')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--config', type=str, default=None, help='Path to custom config json')
    args = parser.parse_args()
    
    # Load Config
    config = Config(args.config)
    
    # Override config with CLI args
    config.config['train']['num_epoch'] = args.epoch
    config.config['train']['device'] = args.gpu
    if args.part is not None:
         # Note: part is handled below, not in config object usually, but good to be consistent
         pass

    # Logging Setup
    if not os.path.exists('log'):
        os.makedirs('log')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"log/{args.dataset_name}_ep{args.epoch}_p{args.part}_{timestamp}.txt"
    logger = Logger(log_filename)
    
    logger.log(f"Command: {' '.join(sys.argv)}")
    logger.log(f"Args: {args}")
    logger.log(f"Config: {config.config}")

    # load data
    dname = args.dataset_name
    dataset_path = args.dataset_path
    
    device = args.gpu if torch.cuda.is_available() and args.gpu == 'cuda' else 'cpu'
    
    data_pkl = os.path.join(dataset_path, dname + '_data.pkl')
    logger.log(f"Loading data from {data_pkl}")
    
    file = open(data_pkl, 'rb')
    file_data = joblib.load(file)
    # tensor(NUM, M, 3), np(NUM, M, M, 2), np(L, 2), np(NUM, M, M), tensor(NUM, M), np(NUM)
    # New order: [trajs, mat1, poi_coords, mat2t, labels, lens, u_max, l_max]
    [trajs, mat1, poi_coords, mat2t, labels, lens, u_max, l_max] = file_data
    
    mat1, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2t), torch.LongTensor(lens)
    
    # Check if category/admin in trajs
    if trajs.shape[-1] > 3:
         logger.log(f"Detected extended features (shape {trajs.shape}). Enabling Cat/Admin.")
         config.config['data']['use_category'] = True
         config.config['data']['use_admin'] = True
         cat_max = int(trajs[:, :, 3].max())
         admin_max = int(trajs[:, :, 4].max())
         config.config['data']['category_vocab_size'] = cat_max + 1
         config.config['data']['admin_vocab_size'] = admin_max + 1
         logger.log(f"Vocab sizes: Cat {cat_max+1}, Admin {admin_max+1}")
    else:
         logger.log(f"Standard features detected (shape {trajs.shape}). HDG-MoE will run with zeros for missing feats.")
         # Force disable to avoid model init error if config has them as True
         config.config['data']['use_category'] = False
         config.config['data']['use_admin'] = False
    
    part = args.part
    if part > len(trajs):
        part = len(trajs)
        logger.log(f"Warning: requested partition {args.part} > data size {part}. Using full data.")
    
    trajs, mat1, mat2t, labels, lens = \
        trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]

    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()

    # Model instantiation
    model_type = config.config['model']['model_type']
    logger.log(f"Initializing Model Type: {model_type}")
    
    if model_type == 'hdg_moe':
        stan = ModelHDGMoE(
            t_dim=hours+1, 
            l_dim=l_max+1, 
            u_dim=u_max+1, 
            embed_dim=config.config['model']['embed_dim'], 
            ex=ex, 
            config=config.config
        )
    else:
        stan = Model(
            t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=50, ex=ex, dropout=0
        )
        
    num_params = 0
    for param in stan.parameters():
        num_params += param.numel()
    logger.log(f"Num of params: {num_params}")

    load = False

    if load:
        checkpoint = torch.load('best_stan_win_' + dname + '.pth')
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
        start = time.time()

    trainer = Trainer(stan, records, config.config)
    trainer.logger = logger # Attach logger
    trainer.train()
