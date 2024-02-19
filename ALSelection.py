import argparse
import numpy as np

import ActiveLearning.adaptive_active_learning as AAL

if __name__ == '__main__':
    parser = argparse.ArgumentParser('argument for active learning selection')

    parser.add_argument('--dataset_path', type=str, default='RiverPIXELS/train_RiverPIXELS_ebd_features.npy',
                        help='path to the embedding features (.npy or .npz) file.')
    parser.add_argument('--acc_eps', type=float, default=0.01,
                        help='the epsilon for the accuracy terminal condition')
    parser.add_argument('--eps_adjust_rate', type=str, default="None",
                        help='the epsilon adjust rate')
    parser.add_argument('--al_batch_size', type=int, default=15,
                        help='the batch size of local max batch active learning')
    parser.add_argument('--sample_batch_size', type=int, default=65536,
                        help='the batch size of the sample process')
    parser.add_argument('--initial_num', type=int, default=15,
                        help='the size of the initial coreset')
    parser.add_argument('--max_new_samples', type=int, default=3000,
                        help='the maximum number of the new samples in each sample step.')
    parser.add_argument('--save_path', type=str, default='RiverPIXELS/train_AAL.npy',
                        help='saving path.')
    parser.add_argument('--acq_fun', type=str, default='uc',
                        choices=['uc', 'supmc', 'supuc'],
                        help='acquisition function for the active learning.')
    parser.add_argument('--use_prior', action='store_true',
                        help='use class prior in the active learning process.')
    parser.add_argument('--class_balance', action='store_true',
                        help='use class accuracy to balance the acq_fun values.')

    opt = parser.parse_args()

    if opt.eps_adjust_rate == 'None':
        opt.eps_adjust_rate = None
    else:
        opt.eps_adjust_rate = float(opt.eps_adjust_rate)

    dataset = np.load(opt.dataset_path, allow_pickle=True).item()
    features = dataset['feature']
    labels = dataset['label']

    select_fvecs, select_labels = AAL.adaptive_active_learning(features,
                                                               labels,
                                                               acc_eps=opt.acc_eps,
                                                               acc_eps_adjust_rate=opt.eps_adjust_rate,
                                                               al_batch_size=opt.al_batch_size,
                                                               sample_batch_size=opt.sample_batch_size,
                                                               initial_num=opt.initial_num,
                                                               use_pp_coreset=False,
                                                               prev_samples=True,
                                                               max_new_samples=opt.max_new_samples,
                                                               acq_fun=opt.acq_fun,
                                                               use_prior=opt.use_prior,
                                                               class_acc_balance=opt.class_balance)

    output_dic = {'feature': select_fvecs, 'label': select_labels}
    np.save(opt.save_path, output_dic)
