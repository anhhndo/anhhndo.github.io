import torch as T
import numpy as np
import tqdm
import pickle
import time
import ot
import argparse
import random



parser = argparse.ArgumentParser()


# Setup
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument('--r', default= 1e-3  , type=float)
parser.add_argument("--ndim", default=4, type=int)
# Random Seed:
parser.add_argument("--seed", default=169, type=int, help="Max iterations for Sinkhorn-Knopp algorithm")
# OT regularization
parser.add_argument("--reg_coef", default=1e-3, type=float, help="Coefficient for regularisation term for main objective loss")
parser.add_argument("--cost_type", default='shortest', type=str, choices= ['shortest',"transition_probability" ],  help="Shortest path or inversely proportional to transition probability") #inverse_prob
parser.add_argument("--sinkhorn_lambd", default=1e-3, type=float, help="Coefficient for regularisation term to compute OT cost")
parser.add_argument("--cost_func", default='C', type=str, help="Type of cost function between two states")
parser.add_argument("--numItermax", default=30, type=int, help="Max iterations for Sinkhorn-Knopp algorithm")
#parser.add_argument("--masking_cost", default=False, type=bool, help="Masking Cost Matrix")

# Sample OT
parser.add_argument("--sampling_OT", default=False, type=bool, help="If True: sampling edge to compute OT")
parser.add_argument("--percentage_of_sampling", default=0.25, type=float, help="percentage of sampling edge to compute OT")

# Cost function to compute cost between two states
def cost_func_A(prob):
    return T.exp(-prob)

def cost_func_B(prob):
    return 1.0/(prob+1e-12)

def cost_func_C(prob):  
    return -T.log(prob+1e-12)

def main(args):
    device = T.device('cuda')

    horizon = args.horizon
    ndim = args.ndim

    n_hid = 256
    n_layers = 2

    bs = 16

    random.seed(args.seed)
    T.manual_seed(args.seed)
    np.random.seed(args.seed)


    f = {'A': cost_func_A,
         'B': cost_func_B,
         'C': cost_func_C,
    }[args.cost_func]

    detailed_balance = False
    uniform_pb = False
    SliceOT = True

    print('loss is', 'DB' if detailed_balance else 'TB')
    def mode(x):
        ax = abs(x / (horizon-1) * 2 - 1)
        return ((ax < 0.8) * (ax > 0.6)).prod(-1)

    def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
        return T.nn.Sequential(*(sum(
            [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
            for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

    def log_reward(x):
        ax = abs(x / (horizon-1) * 2 - 1)
        return ((ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + args.r).log()

    j = T.zeros((horizon,)*ndim+(ndim,))
    for i in range(ndim):
        jj = T.linspace(0,horizon-1,horizon)
        for _ in range(i): jj = jj.unsqueeze(1)
        j[...,i] = jj

    truelr = log_reward(j)
    print('total reward', truelr.view(-1).logsumexp(0))
    true_dist = truelr.flatten().softmax(0).cpu().numpy()

    def toin(z):
        return T.nn.functional.one_hot(z,horizon).view(z.shape[0],-1).float()

    Z = T.zeros((1,)).to(device)
    terminal_state = T.zeros((bs,ndim*horizon)).to(device)
    if SliceOT:
        terminal_state = T.zeros((ndim*horizon)).to(device)
        model = make_mlp([ndim*horizon] + [n_hid] * n_layers + [2*ndim+1]).to(device)
        opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[terminal_state], 'lr':0.01}, {'params':[Z], 'lr':0.1} ])
        Z.requires_grad_()
        terminal_state.requires_grad_()
        temperature = 1.
    else:
        model = make_mlp([ndim*horizon] + [n_hid] * n_layers + [2*ndim+1]).to(device)
        opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
        Z.requires_grad_()

    losses = []
    zs = []
    all_visited = []
    first_visit = -1 * np.ones_like(true_dist)
    l1log = []
    GFN_losses = [] 
    OT_losses = [] 
    
    mode_remain = 16
    mode_visit = []
    for it in tqdm.trange(62501):
        
        opt.zero_grad()
        
        z = T.zeros((bs,ndim), dtype=T.long).to(device)
        
        done = T.full((bs,), False, dtype=T.bool).to(device)
        
        
        ot_loss = T.zeros((bs,), dtype=T.float).to(device) #239
        action = None
        
     
        ll_diff = T.zeros((bs,)).to(device)
        ll_diff += Z
        
        i = 0
        a = 0
        while T.any(~done):
            terminal_state_emb = model(terminal_state)

            #Forward Policy
 
            # Learn metrix for states in GfloNet
            cur_state = z[~done]
            childs_cur_state = cur_state.unsqueeze(1)+T.diag_embed(cur_state!=horizon-1)

            cur_state_emb = model(toin(z[~done]))
            childs_cur_state_emb = model(toin(childs_cur_state.reshape(-1,ndim))).reshape(childs_cur_state.size()[0], ndim, -1)

            # Compute gaussian kernel as flow
            cur_childs_log_kernel = -T.linalg.norm(childs_cur_state_emb-cur_state_emb.unsqueeze(1),ord = 2,dim = -1)**2/temperature
            cur_terminate_log_kernel = (-T.linalg.norm(terminal_state_emb-cur_state_emb,ord = 2,dim = 1)**2/temperature).unsqueeze(-1)
            forward_log_kernel = T.cat([cur_childs_log_kernel,cur_terminate_log_kernel], -1)
            edge_mask = T.cat([ (z[~done]==horizon-1).float(), T.zeros(((~done).sum(),1), device=device) ], 1) # (bs,ndim + 1)
            
            logits = (forward_log_kernel - 1000000000*edge_mask).log_softmax(1)
            
            cur_poward_policy = (logits[...,:ndim]).softmax(1)

            #Backward Policy
            if action is not None:
                parents_cur_state = (cur_state.unsqueeze(1)-T.diag_embed((cur_state!=0).float())).long()
                parents_cur_state_emb = model(toin(parents_cur_state.reshape(-1,ndim))).reshape(parents_cur_state.size()[0], ndim, -1)

            # Compute gaussian kernel as flow
                cur_parents_log_kernel = -T.linalg.norm(parents_cur_state_emb-cur_state_emb.unsqueeze(1),ord = 2,dim = -1)**2/temperature
                init_edge_mask = (z[~done]== 0).float()
                back_logits = ( (0 if uniform_pb else 1)*cur_parents_log_kernel - 1000000000*init_edge_mask).log_softmax(1)
                        
            if action is not None: 
                ll_diff[~done] -= back_logits.gather(1, action[action!=ndim].unsqueeze(1)).squeeze(1)
                
            exp_weight= 0.
            temp = 1
            sample_ins_probs = (1-exp_weight)*(logits/temp).softmax(1) + exp_weight*(1-edge_mask) / (1-edge_mask+0.0000001).sum(1).unsqueeze(1)
            
            action = sample_ins_probs.multinomial(1)

            
            ll_diff[~done] += logits.gather(1, action).squeeze(1)

            terminate = (action==ndim).squeeze(1)
            for x in z[~done][terminate]: 
                state = (x.cpu()*(horizon**T.arange(ndim))).sum().item()
                if first_visit[state]<0: first_visit[state] = it
                all_visited.append(state)
          
            done[~done] |= terminate

            with T.no_grad():
                z[~done] = z[~done].scatter_add(1, action[~terminate], T.ones(action[~terminate].shape, dtype=T.long, device=device))

            i += 1

        lens = z.sum(1)+1
        
        lr = log_reward(z.float())
        ll_diff -= lr
        
        #ot_loss_bs = T.sum(ot_loss)/bs
   
        GFN_loss = (ll_diff**2).sum()/(lens.sum() if detailed_balance else bs)
        loss = (ll_diff**2).sum()/(lens.sum() if detailed_balance else bs)  #+ args.reg_coef*ot_loss_bs
            
        loss.backward()

        opt.step()

        losses.append(loss.item())
        GFN_losses.append(GFN_loss.item())
        #OT_losses.append(ot_loss_bs.item())
        zs.append(Z.item())    


        if sum(mode(j).flatten()*first_visit < 0) == (mode_remain - 1):
            print(it*16)
            mode_remain -= 1
            mode_visit.append(it*16)
        
        if mode_remain == 0:
            print(mode_visit)
            break

        if it%100==0: 
            print("time 100 iteration:", time.time()-a)
            a = time.time()
            print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
            print('GFN loss =', np.array(GFN_losses[-100:]).mean())
           #print('OT loss =', np.array(OT_losses[-100:]).mean())
            emp_dist = np.bincount(all_visited[-200000:], minlength=len(true_dist)).astype(float)
            emp_dist /= emp_dist.sum()
            l1 = np.abs(true_dist-emp_dist).mean()
            print('L1 =', l1)
            l1log.append((len(all_visited), l1))


    #pickle.dump([losses,GFN_losses,OT_losses, zs,all_visited,first_visit,l1log], open(f'out.pkl','wb'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)