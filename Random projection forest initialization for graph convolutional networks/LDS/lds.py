#
#        LDS-GNN
#
#   File:     lds_gnn/lds.py
#   Authors:  Luca Franceschi (luca.franceschi@iit.it)
#             Xiao He
#             Mathias Niepert (mathias.niepert@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2019, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#


"""THe module contains the main algorithm and a demo script to run it"""
import RPTree

import gcn.metrics
from sklearn.neighbors import kneighbors_graph

import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

try:
    from data import ConfigData, UCI, EdgeDelConfigData
    from models import dense_gcn_2l_tensor
    from utils import *
    from hyperparams import *
    from plot import *
except ImportError as e:
    # noinspection PyUnresolvedReferences
    from utils import *
    # noinspection PyUnresolvedReferences
    from hyperparams import *
    # noinspection PyUnresolvedReferences
    from data import ConfigData, UCI, EdgeDelConfigData
    # noinspection PyUnresolvedReferences
    from models import dense_gcn_2l_tensor


def from_svd(svd, train=False, ss=None):
    svd['config'].train = train
    _vrs = eval('Methods.' + svd['method'])(svd['data_config'], svd['config'])
    if not train: restore_from_svd(svd, ss)
    return _vrs


def empirical_mean_model(S, sample_vars, model_out, *what, fd=None, ss=None):
    """ Computes the tensors in `what` using the empirical mean output of the model given by
    `model_out`, sampling `S` times the stochastic variables in `sample_vars`"""
    if ss is None: ss = tf.get_default_session()
    smp = [sample(h) for h in far.utils.as_list(sample_vars)]
    mean_out = []

    # # plot adj_mat -------
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(111)
    # G_1 = nx.from_numpy_matrix(smp[0].eval())
    # pos = np.genfromtxt('features.csv', delimiter=",")
    # nx.draw(G_1, pos, node_size=20, alpha=0.75)
    # date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    # plt.savefig(date_string + '-sample.png', bbox_inches='tight')
    # plt.close()
    # # end plot -----------

    for i in range(S):
        ss.run(smp)
        mean_out.append(ss.run(model_out, fd))
    mean_out = np.mean(mean_out, axis=0)
    lst = ss.run(what, {**fd, model_out: mean_out})
    return lst[0] if len(lst) == 1 else lst


class ConfigMethod(Config):
    def __init__(self, method_name=None, **kwargs):
        self.method_name = method_name
        self.seed = 1979
        self.train = True
        super().__init__(**kwargs)

    def execute(self, data_conf, **kwargs):
        return eval('Methods.' + self.method_name)(data_config=data_conf, config=self, **kwargs)


class LDSConfig(ConfigMethod):

    def __init__(self, method_name='lds', **kwargs):
        self.est_p_edge = 0.0  # initial estimation for the probability of an edge
        self.io_lr = 0.02  # learning rate for the inner optimizer (either hyperparameter sigmoid
        # or fixed value if float
        self.io_opt = 'far.AdamOptimizer'  # name of the inner objective optimizer (should be a far.Optimizer)
        self.oo_lr = 1.  # learning rate of the outer optimizer # decreasing learning rate if this is a tuple
        self.oo_opt = 'far.GradientDescentOptimizer'  # name of the outer  objective optimizer
        self.io_steps = 5  # number of steps of the inner optimization dynamics before an update (hyper batch size)
        self.io_params = (0.001, 20, 400)  # minimum decrease coeff, patience, maxiters
        self.pat = 20  # patience for early stopping
        self.n_sample = 16  # number of samples to compute early stopping validation accuracy and test accuracy
        self.l2_reg = 5.e-4  # l2 regularization coefficient (as in Kipf, 2017 paper)
        self.keep_prob = 1.  # also this is probably not really needed
        self.num_layer = 2

        super().__init__(method_name=method_name, _version=2, **kwargs)

    def io_optimizer(self) -> far.Optimizer:
        if isinstance(self.io_lr, tuple):
            c, a, b = self.io_lr  # starting value, minimum, maximum  -> re-parametrized as a + t * (b-a) with
            # t = sigmoid(\lambda)
            lr = a + tf.sigmoid(far.get_hyperparameter('lr', -tf.log((b - a) / (c - a) - 0.99))) * (b - a)
        else:
            lr = tf.identity(self.io_lr, name='lr')
        lr = tf.identity(lr, name='io_lr')
        return eval(self.io_opt)(lr)

    def oo_optimizer(self, multiplier=1.):
        opt_f = eval(self.oo_opt)
        if isinstance(self.oo_lr, float):
            lr = tf.identity(multiplier * self.oo_lr, name='o_lrd')
            return opt_f(lr)
        elif isinstance(self.oo_lr, tuple):
            gs = new_gs()
            lrd = tf.train.inverse_time_decay(multiplier * self.oo_lr[0], gs, self.oo_lr[1],
                                              self.oo_lr[2], name='o_lrd')
            return opt_f(lrd)
        else:
            raise AttributeError('not understood')


class KNNLDSConfig(LDSConfig):
    def __init__(self, method_name='lds', **kwargs):
        """Configuration instance for the method kNN-LDS.
        The arguments are the same as LDSConfig plus

        `k` (10): number of neighbours

        `metric` (cosine): metric function to use"""
        self.k = 10
        self.metric = 'minkowski'#'cosine'
        super().__init__(method_name, **kwargs)


def lds(data_conf: ConfigData, config: LDSConfig):
    """
    Runs the LDS algorithm on data specified by `data_conf` with parameters
    specified in `config`.

    :param data_conf: Configuration for the data. Please see `ConfigData` for documentation
    :param config: Configuration for the method's parameters. Please see `LDSConfig` for documentation
    :return: a triplet: - the local variable dictionary (as returned by vars()),
                        - the best `early stopping` accuracy
                        - the test accuracy on the iteration that achieved the best `early stopping` accuracy
    """
    ss = setup_tf(config.seed)

    adj, adj_mods, features, ys, train_mask, val_mask, es_mask, test_mask = data_conf.load()
    plc = Placeholders(features, ys)

    if isinstance(config, KNNLDSConfig):
        # # start knn adj matrix ---------------------------------------
        # g = kneighbors_graph(features, config.k, metric=config.metric)
        # g = np.array(g.todense(), dtype=np.float32)
        # adj_mods = g
        # # end knn adj matrix ---------------------------------------

        # start rpTree adj matrix ---------------------------------------
        NumOfTrees = 10
        adj_mods = np.zeros((features.shape[0], features.shape[0]), dtype=np.float32)
        for r in range(NumOfTrees):
            tree = RPTree.BinaryTree(features)
            features_index = np.arange(features.shape[0])
            tree_root = tree.construct_tree(tree, features_index)
            # get the indices of points in leaves
            leaves_array = tree_root.get_leaf_nodes()

            # connect points in the same leaf node
            edgeList = []
            for i in range(len(leaves_array)):
                x = leaves_array[i]
                n = x.size
                perm = np.empty((n, n, 2), dtype=x.dtype)
                perm[..., 0] = x[:, None]
                perm[..., 1] = x
                perm1 = np.reshape(perm, (-1, 2))
                if i == 0:
                    edgeList = perm1
                else:
                    edgeList = np.vstack((edgeList, perm1))

            # assign one as edge weight
            edgeList = edgeList[edgeList[:, 0] != edgeList[:, 1]]
            edgeList = np.hstack((edgeList, np.ones((edgeList.shape[0], 1), dtype=int)))

            # convert edges list to adjacency matrix
            shape = tuple(edgeList.max(axis=0)[:2] + 1)
            adjMatRPTree = sparse.coo_matrix((edgeList[:, 2], (edgeList[:, 0], edgeList[:, 1])), shape=shape,
                                              dtype=edgeList.dtype)
            adjMatRPTree = np.array(adjMatRPTree.todense(), dtype=np.float32)

            # an adjacency matrix holding weights accumulated from all rpTrees
            adj_mods = adj_mods + (adjMatRPTree / NumOfTrees)
        # end rpTree adj matrix ---------------------------------------

        # np.savetxt("adj.csv", adj_mods, delimiter=",")
        
        # # plot adj_mods -------
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111)
        # G_1 = nx.from_numpy_matrix(adj_mods)
        # #pos = np.genfromtxt('data/ring238_Instances.csv', delimiter=",")
        # np.savetxt("features.csv", features, delimiter=",")
        # nx.draw(G_1, features, node_size=20, alpha=0.75)
        # date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        # #plt.savefig(date_string+'.png', dpi=150, bbox_inches='tight')
        # plt.savefig(date_string + '-original.png', bbox_inches='tight')
        # plt.close()
        # # end plot -----------

    constraint = upper_tri_const(adj.shape)
    adj_hyp = get_stc_hyperparameter(
        'adj_hyp', initializer=adj_mods,#constraint(adj_mods + config.est_p_edge * tf.ones(adj.shape)),
        constraints=constraint, sample_func=symm_adj_sample)

    out, ws, rep = dense_gcn_2l_tensor(plc.X, adj_hyp, plc.Y, num_layer=config.num_layer,
                                       dropout=plc.keep_prob)

    error = tf.identity(gcn.metrics.masked_softmax_cross_entropy(out, plc.Y, plc.label_mask), 'error')
    tr_error = error + config.l2_reg * tf.nn.l2_loss(ws[0])

    acc = tf.identity(gcn.metrics.masked_accuracy(out, plc.Y, plc.label_mask), 'accuracy')

    tr_fd, val_fd, es_fd, test_fd = plc.fds(train_mask, val_mask, es_mask, test_mask)
    tr_fd = {**tr_fd, **{plc.keep_prob: config.keep_prob}}

    svd = init_svd(data_conf, config)  # initialize data structure for saving statistics and perform early stopping

    def _test_on_accept(_t):  # execute this when a better parameter configuration is found
        accft = empirical_mean_model(config.n_sample, adj_hyp, out, acc, fd=test_fd)
        update_append(svd, oa_t=_t, oa_act=accft)
        print('iteration', _t, ' - mean test accuracy: ', accft)
        global test_acc
        test_acc = accft

    # initialize hyperparamter optimization method
    ho_mod = far.HyperOptimizer(StcReverseHG())
    io_opt, oo_opt = config.io_optimizer(), config.oo_optimizer()
    ho_step = ho_mod.minimize(error, oo_opt, tr_error, io_opt, global_step=get_gs())

    # run the method
    tf.global_variables_initializer().run()
    es_gen = early_stopping_with_save(config.pat, ss, svd, on_accept=_test_on_accept)

    if config.train:
        try:
            for _ in es_gen:  # outer optimization loop
                e_es, a_es = empirical_mean_model(config.n_sample, adj_hyp, out, error, acc, fd=es_fd)

                es_gen.send(a_es)  # new early stopping accuracy

                # records some statistics -------
                etr, atr = ss.run([error, acc], tr_fd)
                eva, ava = ss.run([error, acc], val_fd)
                ete, ate = ss.run([error, acc], test_fd)
                iolr = ss.run(io_opt.optimizer_params_tensor)
                n_edgs = np.sum(adj_hyp.eval())

                update_append(svd, etr=etr, atr=atr, eva=eva, ava=ava, ete=ete, ate=ate, iolr=iolr,
                              e_es=e_es, e_ac=a_es, n_edgs=n_edgs, olr=ss.run('o_lrd:0'))

                if _ > 0:
                    with open('Results.csv', 'a') as my_file:
                        # atr, ava, ate, n_edgs
                        my_file.write("\n")
                        my_file.write(str(np.round(atr, 4))+","+str(np.round(ava, 4))+","+str(np.round(ate, 4))+","+str(np.round(n_edgs, 4)))
                # end record ----------------------

                # # plot adj_mat -------
                # fig = plt.figure(figsize=(6, 6))
                # ax = fig.add_subplot(111)
                # G_1 = nx.from_numpy_matrix(adj_hyp.eval())
                # #pos = np.genfromtxt('data/ring238_Instances.csv', delimiter=",")
                # nx.draw(G_1, features, node_size=20, alpha=0.75)
                # date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                # plt.savefig(date_string + '--test-'+str(ate)+'.png', bbox_inches='tight')
                # plt.close()
                # # end plot -----------

                steps, rs = ho_mod.hypergradient.min_decrease_condition(
                    config.io_params[0], config.io_params[1], config.io_steps,
                    feed_dicts=tr_fd, verbose=False, obj=None)
                # inner optimization loop
                for j in range(config.io_params[2] // config.io_steps):  
                    if rs['pat'] == 0: break  # inner objective is no longer decreasing
                    ho_step(steps(), tr_fd, val_fd, online=j)  # do one hypergradient optimization step
        except KeyboardInterrupt:
            print('Interrupted.', file=sys.stderr)
            return vars()
    return vars(), svd['es final value'], test_acc


def main(data, method, seed, missing_percentage):

    data = 'iris'
    method = 'knnlds'
    seed = 1
    missing_percentage = 50
    if data == 'iris':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=False)
    elif data == 'wine':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=True)
    elif data == 'breast_cancer':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=True)
    elif data == 'digits':
        data_config = UCI(seed=seed, dataset_name=data, n_train=50, n_val=50, n_es=50, scale=False)
    elif data == 'ring238':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=False)
    elif data == '3rings299':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=False)
    elif data == 'sparse622':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=False)
    elif data == 'sparse303':
        data_config = UCI(seed=seed, dataset_name=data, n_train=10, n_val=10, n_es=10, scale=False)        
    elif data == '20newstrain':
        data_config = UCI(seed=seed, dataset_name=data, n_train=200, n_val=200, n_es=200, scale=False)
    elif data == '20news10':
        data_config = UCI(seed=seed, dataset_name=data, n_train=100, n_val=100, n_es=100, scale=False)
    elif data == 'cora' or data == 'citeseer':
        data_config = EdgeDelConfigData(prob_del=missing_percentage, seed=seed, enforce_connected=False,
                                        dataset_name=data)
    elif data == 'fma':
        data_config = UCI(seed=seed, dataset_name=data, n_train=160, n_val=160, n_es=160, scale=False)
    else:
        raise AttributeError('Dataset {} not available'.format(data))

    if method == 'knnlds':
        configs = KNNLDSConfig.grid(pat=20, seed=seed, io_steps=[5, 16], keep_prob=0.5,
                                    io_lr=(2.e-2, 1.e-4, 0.05),
                                    oo_lr=[(1., 1., 1.e-3), (.1, 1., 1.e-3)],
                                    metric=['minkowski', 'minkowski'], k=[10, 20])
                                    # metric=['minkowski', 'minkowski'], k=[2, 2])
                                    # metric = ['cosine', 'minkowski'], k = [10, 20])
    elif method == 'lds':
        configs = LDSConfig.grid(pat=20, seed=seed, io_steps=[1, 5, 20],
                                 io_lr=(2.e-2, 1.e-4, 0.05), keep_prob=0.5,
                                 oo_lr=[(1., 1., 1.e-3), (.1, 1., 1.e-3)])
    else:
        raise NotImplementedError('Method {} unknown. Choose between `knnlds` and `lds`'.format(method))

    best_valid_acc = 0
    best_test_acc = 0
    print(data_config)
    for cnf in configs:
        print(cnf)
        vrs, valid_acc, test_acc = lds(data_config, cnf)
        if best_valid_acc <= valid_acc:
            print('Found a better configuration:', valid_acc)
            best_valid_acc = valid_acc
            best_test_acc = test_acc

        print('Test accuracy of the best found model:', best_test_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='method')
    parser.add_argument('-d', default='breast_cancer', type=str,
                        help='The evaluation dataset: iris, wine, breast_cancer, digits, 20newstrain, 20news10, ' +
                        'cora, citeseer, fma. Default: breast_cancer')
    parser.add_argument('-m', default='knnlds', type=str,
                        help='The method: lds or knnlds. Default: knnlds')
    parser.add_argument('-s', default=1, type=int,
                        help='The random seed. Default: 1')
    parser.add_argument('-e', default=50, type=int,
                        help='The percentage of missing edges (valid only for cora and citeseer dataset): Default 50. - ' +  
                        'PLEASE NOTE THAT the x-axes of Fig. 2  in the paper reports the percentage of retained edges rather ' +
                        'than that of missing edges.')
    args = parser.parse_args()

    _data, _method, _seed, _missing_percentage = args.d, args.m, args.s, args.e/100

    main(_data, _method, _seed, _missing_percentage)
