from __future__ import absolute_import, print_function, division

import sys
from collections import defaultdict, deque

import tensorflow as tf
from tensorflow.python.training import slot_creator
from tensorflow.contrib.opt import ScipyOptimizerInterface

from far_ho import utils
from far_ho.optimizer import OptimizerDict
from far_ho.utils import dot, maybe_add, reduce_all_sums

RAISE_ERROR_ON_DETACHED = False


class HyperGradient(object):
    def __init__(self, name):
        self._optimizer_dicts = set()
        self._inner_objectives = None
        self._hypergrad_dictionary = defaultdict(list)  # dictionary (hyperparameter, list of hypergradients)
        self._ts = None

        self._initialization = None
        self._iteration = None
        self._state = None
        self._name = name

    _ERROR_NOT_OPTIMIZER_DICT = """
    Looks like {} is not an `OptimizerDict`. Use optimizers in far_ho.optimizers for obtaining an OptimizerDict.
    """

    _ERROR_HYPER_DETACHED = """
    Hyperparameter {} is detached from this optimization dynamics.
    """

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        # Doesn't do anything useful here. To be overridden.
        """
        Function overridden by specific methods.

        :param optimizer_dict: OptimzerDict object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the hyperparameters (scalar tensor)
        :param hyper_list: Optional list of hyperparameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.

        :return: list of hyperparameters involved in the computation
        """
        assert isinstance(optimizer_dict, OptimizerDict), HyperGradient._ERROR_NOT_OPTIMIZER_DICT.format(optimizer_dict)
        self._optimizer_dicts.add(optimizer_dict)

        if hyper_list is None:  # get default hyperparameters
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)
        return hyper_list

    @property
    def initialization(self):
        if self._initialization is None:
            self._initialization = [opt_dict.initialization for opt_dict in sorted(self._optimizer_dicts)]
        return self._initialization

    @property
    def iteration(self):
        if self._iteration is None:
            self._iteration = [opt_dict.iteration for opt_dict in sorted(self._optimizer_dicts)]
        return self._iteration

    @property
    def state(self):
        for opt_dict in sorted(self._optimizer_dicts):
            for v in opt_dict.state:
                yield v

    @property
    def inner_objectives(self):
        if self._inner_objectives is None:
            self._inner_objectives = [opt.objective if hasattr(opt, 'objective') else tf.constant(False)
                                      for opt in sorted(self._optimizer_dicts)]
        return self._inner_objectives

    @property
    def ts(self):
        if self._ts is None:
            self._ts = tf.group(*[opt_dict.ts for opt_dict in sorted(self._optimizer_dicts)])
        return self._ts

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, callback=None):
        """
        Runs the inner optimization dynamics for T iterations (T_or_generator can be indeed a generator) and computes
        in the meanwhile.

        :param T_or_generator: integer or generator that should yield a step. Express either a total number of
                                iterations of inner objective optimization dynamics, or could implement a stopping
                                condition, or variables number of steps.
        :param inner_objective_feed_dicts: Optional feed dictionary for the inner objective
        :param outer_objective_feed_dicts: Optional feed dictionary for the outer objective
                                            (note that this is not used in ForwardHG since hypergradients are not
                                            variables)
        :param initializer_feed_dict: Optional feed dictionary for the inner objective
        :param global_step: Optional global step for the
        :param session: Optional session (otherwise will take the default session)
        :param online: Performs the computation of the hypergradient in the online (or "real time") mode. Note that
                        `ReverseHG` and `ForwardHG` behave differently.
        :param callback: callback funciton for the forward optimization

        """
        raise NotImplementedError()

    def hgrads_hvars(self, hyper_list=None, aggregation_fn=None, process_fn=None):
        """
        Method for getting hypergradient and hyperparameters as required by apply_gradient methods from tensorflow 
        optimizers.
        
        :param hyper_list: Optional list of hyperparameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.
        :param aggregation_fn: Optional operation to aggregate multiple hypergradients (for the same hyperparameter),
                                by default reduce_mean
        :param process_fn: Optional operation like clipping to be applied.
        :return: 
        """
        if hyper_list is None:
            hyper_list = utils.hyperparameters(tf.get_variable_scope().name)

        assert all([h in self._hypergrad_dictionary for h in hyper_list]), 'FINAL ERROR!'

        if aggregation_fn is None:
            aggregation_fn = lambda hgrad_list: tf.reduce_mean(hgrad_list, axis=0)

        def _aggregate_process_manage_collection(_hg_lst):
            if len(_hg_lst) == 1:  # avoid useless operations...
                aggr = _hg_lst[0]
            else:
                with tf.name_scope(_hg_lst[0].op.name):
                    aggr = aggregation_fn(_hg_lst) if len(_hg_lst) > 1 else _hg_lst[0]
            if process_fn is not None:
                with tf.name_scope('process_gradients'):
                    aggr = process_fn(aggr)
            tf.add_to_collection(utils.GraphKeys.HYPERGRADIENTS, aggr)
            return aggr

        return [(_aggregate_process_manage_collection(self._hypergrad_dictionary[h]),
                 h) for h in hyper_list]

    @property
    def name(self):
        return self._name

    @staticmethod
    def need_scalar_hyperparameters():
        return False

    # noinspection PyMethodMayBeStatic
    def _make_callback(self):
        """
        Template for callbacks
        """
        values = []

        # noinspection PyUnusedLocal
        def _callback(t, feed_dcit, session):
            values.append(0)  # these should not depend from any feed dictionary

        return values, _callback

    def __str__(self):
        return self._name


class ReverseHG(HyperGradient):

    def __init__(self, history=None, name='ReverseHG'):
        super(ReverseHG, self).__init__(name)
        self._alpha_iter = tf.no_op()
        self._reverse_initializer = tf.no_op()
        self._history = history if history is not None else []

    @staticmethod
    def truncated(reverse_iterations, name='TruncatedReverseHG'):
        """
        Utility method to initialize truncated reverse HG (not necessarily online),
        see the paper Truncated Back-propagation for Bilevel Optimization:
        https://arxiv.org/abs/1810.10667

        :param reverse_iterations: Maximum number of iterations that will be stored
        :param name: a name for the operations and variables that will be created
        :return: ReverseHG object
        """
        return ReverseHG(deque(maxlen=reverse_iterations + 1), name=name)

    # noinspection SpellCheckingInspection
    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        """
        Function that adds to the computational graph all the operations needend for computing
        the hypergradients in a "dynamic" way, without unrolling the entire optimization graph.
        The resulting computation, while being roughly 2x more expensive then unrolling the
        optimizaiton dynamics, requires much less (GPU) memory and is more flexible, allowing
        to set a termination condition to the parameters optimizaiton routine.

        :param optimizer_dict: OptimzerDict object resulting from the inner objective optimization.
        :param outer_objective: A loss function for the hyperparameters (scalar tensor)
        :param hyper_list: Optional list of hyperparameters to consider. If not provided will get all variables in the
                            hyperparameter collection in the current scope.

        :return: list of hyperparameters involved in the computation
        """
        hyper_list = super(ReverseHG, self).compute_gradients(outer_objective, optimizer_dict, hyper_list)

        # derivative of outer objective w.r.t. state
        with tf.variable_scope(outer_objective.op.name):  # for some reason without this there is a cathastrofic
            # failure...
            doo_ds = tf.gradients(outer_objective, list(optimizer_dict.state))

            alphas = self._create_lagrangian_multipliers(optimizer_dict, doo_ds)

            alpha_vec = utils.vectorize_all(alphas)
            dyn_vec = utils.vectorize_all(list(optimizer_dict.dynamics))
            lag_phi_t = utils.dot(alpha_vec, dyn_vec, name='iter_wise_lagrangian_part1')
            # TODO outer_objective might be a list... handle this case

            # iterative computation of hypergradients
            alpha_dot_B = tf.gradients(lag_phi_t, hyper_list)
            # check that optimizer_dict has initial ops (phi_0)
            if optimizer_dict.init_dynamics is not None:
                lag_phi0 = utils.dot(alpha_vec, utils.vectorize_all([d for (s, d) in optimizer_dict.init_dynamics]))
                alpha_dot_B0 = tf.gradients(lag_phi0, hyper_list)
            else:
                alpha_dot_B0 = [None] * len(hyper_list)

            # here, if some of this is None it may mean that the hyperparameter compares inside phi_0: check that and
            # if it is not the case raise error...
            hyper_grad_vars, hyper_grad_step = [], tf.no_op()
            for dl_dh, a_d_b0, hyper in zip(alpha_dot_B, alpha_dot_B0, hyper_list):
                assert dl_dh is not None or a_d_b0 is not None, HyperGradient._ERROR_HYPER_DETACHED.format(hyper)
                hgv = None
                if dl_dh is not None:  # "normal hyperparameter"
                    hgv = self._create_hypergradient(outer_objective, hyper)

                    hyper_grad_step = tf.group(hyper_grad_step, hgv.assign_add(dl_dh))
                if a_d_b0 is not None:
                    hgv = hgv + a_d_b0 if hgv is not None else a_d_b0
                    # here hyper_grad_step has nothing to do...
                hyper_grad_vars.append(hgv)  # save these...

            with tf.control_dependencies([hyper_grad_step]):  # first update hypergradinet then alphas.
                _alpha_iter = tf.group(*[alpha.assign(dl_ds) for alpha, dl_ds
                                         in zip(alphas, tf.gradients(lag_phi_t, list(optimizer_dict.state)))])
            self._alpha_iter = tf.group(self._alpha_iter, _alpha_iter)  # put all the backward iterations toghether

            [self._hypergrad_dictionary[h].append(hg) for h, hg in zip(hyper_list, hyper_grad_vars)]

            self._reverse_initializer = tf.group(self._reverse_initializer,
                                                 tf.variables_initializer(alphas),
                                                 tf.variables_initializer([h for h in hyper_grad_vars
                                                                           if hasattr(h, 'initializer')]))  # some ->
            # hypergradients (those coming form initial dynamics) might be just tensors and not variables...

            return hyper_list

    @staticmethod
    def _create_lagrangian_multipliers(optimizer_dict, doo_ds):
        lag_mul = [slot_creator.create_slot(v.initialized_value(), utils.val_or_zero(der, v), 'alpha') for v, der
                   in zip(optimizer_dict.state, doo_ds)]
        [tf.add_to_collection(utils.GraphKeys.LAGRANGIAN_MULTIPLIERS, lm) for lm in lag_mul]
        utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, *lag_mul)
        # this prevents the 'automatic' initialization with tf.global_variables_initializer.
        return lag_mul

    @staticmethod
    def _create_hypergradient_from_dodh(hyper, doo_dhypers):
        """
        Creates one hyper-gradient as a variable. doo_dhypers:  initialization, that is the derivative of
        the outer objective w.r.t this hyper
        """
        hgs = slot_creator.create_slot(hyper, utils.val_or_zero(doo_dhypers, hyper), 'hypergradient')
        utils.remove_from_collection(utils.GraphKeys.GLOBAL_VARIABLES, hgs)
        return hgs

    @staticmethod
    def _create_hypergradient(outer_obj, hyper):
        return ReverseHG._create_hypergradient_from_dodh(hyper, tf.gradients(outer_obj, hyper)[0])

    def _state_feed_dict_generator(self, history, T_or_generator):
        for t, his in zip(utils.solve_int_or_generator(T_or_generator), history):
            yield t, utils.merge_dicts(
                *[od.state_feed_dict(h) for od, h in zip(sorted(self._optimizer_dicts), his)]
            )

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, callback=None):
        # callback may be a pair, first for froward pass, second for reverse pass
        callback = utils.as_tuple_or_list(callback)
        # same thing for T
        T_or_generator = utils.as_tuple_or_list(T_or_generator)

        ss = session or tf.get_default_session()

        self._history.clear()

        def _adjust_step(_t):
            if online:
                _T = utils.maybe_eval(global_step, ss)
                if _T is None:
                    _T = 0
                tot_t = T_or_generator[0]
                if not isinstance(tot_t, int): return _t  # when using a generator there is little to do...
                return int(_t + tot_t*_T)
            else: return _t

        if not online:
            _fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
            self._save_history(ss.run(self.initialization, feed_dict=_fd))

        # else:  # not totally clear if i should add this
        #     self._save_history(ss.run(list(self.state)))

        T = 0  # this is useful if T_or_generator is indeed a generator...
        for t in utils.solve_int_or_generator(T_or_generator[0]):
            # nonlocal t  # with nonlocal would not be necessary the variable T... not compatible with 2.7

            _fd = utils.maybe_call(inner_objective_feed_dicts, _adjust_step(t))
            self._save_history(ss.run(self.iteration, feed_dict=_fd))
            T = t

            utils.maybe_call(callback[0], _adjust_step(t), _fd, ss)  # callback

        # initialization of support variables (supports stochastic evaluation of outer objective via global_step ->
        # variable)
        # TODO (maybe tf bug or oddity) for some strange reason, if some variable's initializer depends on
        # a placeholder, then the initializer of alpha SEEMS TO DEPEND ALSO ON THAT placeholder,
        # as if the primary variable should be reinitialized as well, but, I've checked, the primary variable is NOT
        # actually reinitialized. This doesn't make sense since the primary variable is already initialized
        # and Tensorflow seems not to care... should maybe look better into this issue
        reverse_init_fd = utils.maybe_call(outer_objective_feed_dicts, utils.maybe_eval(global_step, ss))
        # now adding also the initializer_feed_dict because of tf quirk...
        maybe_init_fd = utils.maybe_call(initializer_feed_dict, utils.maybe_eval(global_step, ss))
        reverse_init_fd = utils.merge_dicts(reverse_init_fd, maybe_init_fd)
        ss.run(self._reverse_initializer, feed_dict=reverse_init_fd)

        del self._history[-1]  # do not consider last point

        for pt, state_feed_dict in self._state_feed_dict_generator(reversed(self._history), T_or_generator[-1]):
            # this should be fine also for truncated reverse... but check again the index t
            t = T - pt - 1  # if T is int then len(self.history) is T + 1 and this numerator
            # shall start at T-1
            _fd = utils.merge_dicts(state_feed_dict, utils.maybe_call(inner_objective_feed_dicts,
                                                                      _adjust_step(t)))
            ss.run(self._alpha_iter, _fd)
            if len(callback) == 2: utils.maybe_call(callback[1], _adjust_step(t), _fd, ss)

    def _save_history(self, weights):
        self._history.append(weights)

    def hypergrad_callback(self, hyperparameter=None, flatten=True):
        """callback that records the partial hypergradients on the reverse pass"""
        values = []
        gs = list(self._hypergrad_dictionary.values()) if hyperparameter is None else \
            self._hypergrad_dictionary[hyperparameter]
        if flatten: gs = utils.vectorize_all(gs)

        # noinspection PyUnusedLocal
        def _callback(_, __, ss):
            values.append(ss.run(gs))  # these should not depend from any feed dictionary

        return values, _callback


class ReverseHg(ReverseHG):

    def __init__(self, history=None):
        print('WARNING, DEPRECATED: please use the class ReverseHG', file=sys.stderr)
        super(ReverseHg, self).__init__(history)


class ForwardHG(HyperGradient):
    def __init__(self, name='ForwardHG'):
        super(ForwardHG, self).__init__(name)
        self._forward_initializer = tf.no_op()
        self._zs = {}  # hyperparameter - zs dictionary
        self._z_iter = tf.no_op()
        self._iteration = None
        self.A_dot_zs = {}

    _HYPER_RANK_ERROR_MESSAGE = """
    ForwardHG: Only scalar hyperparameters accepted.\n
     Hyperparameter tensor {} has rank {}.\n
     Use keyword argument far_ho.get_hyperparameter(..., scalar=True) on hyperparameter creation.
    """

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        hyper_list = super(ForwardHG, self).compute_gradients(outer_objective, optimizer_dict, hyper_list)

        # scalar_hyper_list

        with tf.variable_scope(outer_objective.op.name):
            # dynamics_vec = vectorize_all(optimizer_dict.dynamics)  # in the new implementation there's no need of
            # vectorizing... it might be more efficient since it's better to avoid too many reshaping operations...
            d_oo_d_state = tf.gradients(outer_objective, list(optimizer_dict.state))

            with tf.name_scope('DUMMY'):  # variables to compute forward propagation
                # TODO avoid this computation if optimizer_dict has already been seen.
                aux_vs = [tf.zeros_like(v) for v in optimizer_dict.state]
                dynamics_dot_aux_v = reduce_all_sums(list(optimizer_dict.dynamics), aux_vs)

                der_dynamics_dot_aux_v = tf.gradients(dynamics_dot_aux_v, list(optimizer_dict.state))
                # this is a list of jacobians times aux_vs that have the same dimension of states variables.

                init_dynamics_dot_aux_v = None
                if optimizer_dict.init_dynamics:
                    # init_dynamics_dot_aux_v = dot(vectorize_all(optimizer_dict.init_dynamics), aux_v_vec)  # old impl
                    init_dynamics_dot_aux_v = reduce_all_sums(
                        optimizer_dict.init_dynamics, aux_vs)

            for hyp in hyper_list:
                assert hyp.shape.ndims == 0, ForwardHG._HYPER_RANK_ERROR_MESSAGE.format(hyp, hyp.shape.ndims)

                d_init_dyn_d_hyp = None if init_dynamics_dot_aux_v is None else \
                    tf.gradients(init_dynamics_dot_aux_v, hyp)[0]
                d_dyn_d_hyp = tf.gradients(dynamics_dot_aux_v, hyp)[0]
                d_oo_d_hyp = tf.gradients(outer_objective, hyp)[0]

                # ------------------------------------------------------------
                # check detached hyperparameters (for which hypergradient would be always null)
                hyper_ok = d_init_dyn_d_hyp is not None or d_dyn_d_hyp is not None or d_oo_d_hyp is not None
                if RAISE_ERROR_ON_DETACHED:
                    # try:
                    assert hyper_ok, HyperGradient._ERROR_HYPER_DETACHED.format(hyp)
                    # ex
                else:
                    if not hyper_ok:
                        print(HyperGradient._ERROR_HYPER_DETACHED.format(hyp), file=sys.stderr)
                        hyper_list.remove(hyp)
                # -------------------------------------------------------------

                # UPDATE OF TOTAL DERIVATIVE OF STATE W.R.T. HYPERPARAMETER
                zs = ForwardHG._create_zs(
                    optimizer_dict, hyp, None if d_init_dyn_d_hyp is None else tf.gradients(d_init_dyn_d_hyp, aux_vs)
                )  # this is one z for each variable
                self._zs[hyp] = zs  # store a reference for the total derivatives for easy access
                Bs = tf.gradients(d_dyn_d_hyp, aux_vs)

                A_dot_zs = tf.gradients(reduce_all_sums(der_dynamics_dot_aux_v, zs), aux_vs)

                self.A_dot_zs[hyp] = A_dot_zs

                _z_iter = tf.group(*[
                    z.assign(maybe_add(A_dot_z, B)) for z, A_dot_z, B
                    in zip(zs, A_dot_zs, Bs)
                ])
                self._z_iter = tf.group(self._z_iter, _z_iter)

                # -- HYPERGRADIENT -----
                d_E_T = [dot(d_oo_d_s, z) for d_oo_d_s, z in zip(d_oo_d_state, zs)
                         if d_oo_d_s is not None and z is not None]  # list of dot products
                hg = maybe_add(tf.reduce_sum(d_E_T), d_oo_d_hyp)  # sum the partial dot products and possibly ->
                # adds the ''direct derivative'' term d(E( . , \lambda))/d \lambda

                self._hypergrad_dictionary[hyp].append(hg)
                self._forward_initializer = tf.group(self._forward_initializer,
                                                     tf.variables_initializer(zs))
        return hyper_list

    @staticmethod
    def _create_zs(optimizer_dict, hyper, d_init_dynamics_d_hyper):
        if d_init_dynamics_d_hyper is None: d_init_dynamics_d_hyper = [None] * len(optimizer_dict)
        with tf.variable_scope('Z'):
            z = [slot_creator.create_slot(v, utils.val_or_zero(der, v), hyper.op.name) for v, der
                 in zip(optimizer_dict.state, d_init_dynamics_d_hyper)]
            [tf.add_to_collection(utils.GraphKeys.ZS, lm) for lm in z]
            # in this case it is completely fine to keep zs into the global variable...
            return z

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, callback=None):

        ss = session or tf.get_default_session()

        if online == False:
            self._run_batch_initialization(ss, utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss)))

        elif online == 'wr':
            self._run_wr_initialization(ss, utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss)))

        for t in utils.solve_int_or_generator(T_or_generator):
            _fd = utils.maybe_call(inner_objective_feed_dicts, t)
            self._forward_step(ss, _fd)
            utils.maybe_call(callback, t, _fd, ss)

    def _forward_step(self, ss, _fd):
        ss.run(self._z_iter, _fd)
        ss.run(self.iteration, _fd)

    def _run_batch_initialization(self, ss, fd):
        ss.run(self.initialization, feed_dict=fd)
        ss.run(self._forward_initializer, feed_dict=fd)

    def _run_wr_initialization(self, ss, fd):
        # ss.run(self.initialization, feed_dict=fd)
        ss.run(self._forward_initializer, feed_dict=fd)

    @staticmethod
    def need_scalar_hyperparameters():
        return True

    @property
    def w_dots(self):
        # if hyper: return self._zs[hyper]
        return [{h: self._zs[h][k] for h in self._zs} for k, _ in enumerate(self.state)]

    def z_callback(self, hyperparameter=None, flatten=True):
        zs_values = []
        zs = list(self._zs.values()) if hyperparameter is None else self._zs[hyperparameter]
        if flatten: zs = utils.vectorize_all(zs)

        # noinspection PyUnusedLocal
        def _callback(_, __, ss):
            zs_values.append(ss.run(zs))  # these should not depend from any feed dictionary

        return zs_values, _callback


class ImplicitHG(HyperGradient):
    """
    Implementation follows Pedregosa's algorithm HOAG
    """

    def __init__(self, linear_system_solver_gen=None, tolerance=None, name='ImplicitHG'):
        super(ImplicitHG, self).__init__(name)
        if linear_system_solver_gen is None:
            linear_system_solver_gen = lambda _obj, var_list, _tolerance: ScipyOptimizerInterface(
                _obj, var_list=var_list, options={'maxiter': 100}, method='cg', tol=_tolerance)
        self.linear_system_solver = linear_system_solver_gen

        if tolerance is None:
            tolerance = lambda _k: 0.1 * (0.9 ** _k)
        self.tolerance = tolerance

        self._lin_sys = []
        self._qs = []

    def compute_gradients(self, outer_objective, optimizer_dict, hyper_list=None):
        hyper_list = super(ImplicitHG, self).compute_gradients(outer_objective, optimizer_dict, hyper_list)
        state = list(optimizer_dict.state)

        with tf.variable_scope(outer_objective.op.name):
            g1 = utils.vectorize_all(tf.gradients(outer_objective, state))
            grads_inner_obj_vec = utils.vectorize_all(tf.gradients(optimizer_dict.objective, state))

            q = self._create_q(g1)
            obj = tf.norm(
                utils.vectorize_all(tf.gradients(utils.dot(grads_inner_obj_vec, q), state)) - g1
            )  # using the norm seems to produce better results then squared norm...
            # (even though is more costly)

            self._lin_sys.append(lambda _tolerance: self.linear_system_solver(obj, [q], _tolerance))

            g2s = tf.gradients(outer_objective, hyper_list)
            cross_ders = tf.gradients(utils.dot(grads_inner_obj_vec, q), hyper_list)
            for g2, cd, hyper in zip(g2s, cross_ders, hyper_list):
                assert g2 is not None or cd is not None, HyperGradient._ERROR_HYPER_DETACHED.format(hyper)
                hg = utils.maybe_add(-cd, g2)
                if hg is None:  # this would be strange...
                    print('WARNING, outer objective is only directly dependent on hyperparameter {}. ' +
                          'Direct optimization would be better!'.format(hyper))
                    hg = g2
                self._hypergrad_dictionary[hyper].append(hg)

            return hyper_list

    def _create_q(self, d_oo_d_state):
        self._qs.append(slot_creator.create_zeros_slot(d_oo_d_state, 'q'))
        return self._qs[-1]

    def run(self, T_or_generator, inner_objective_feed_dicts=None, outer_objective_feed_dicts=None,
            initializer_feed_dict=None, global_step=None, session=None, online=False, callback=None):
        ss = session or tf.get_default_session()

        inner_objective_feed_dicts = utils.as_tuple_or_list(inner_objective_feed_dicts)
        if not online:
            self._run_batch_initialization(ss, utils.maybe_call(
                initializer_feed_dict, utils.maybe_eval(global_step, ss)))

        for t in utils.solve_int_or_generator(T_or_generator):
            _fd = utils.maybe_call(inner_objective_feed_dicts[0], t)
            self._forward_step(ss, _fd)
            utils.maybe_call(callback, t, _fd, ss)

        # end of optimization. Solve linear systems.
        tol_val = utils.maybe_call(self.tolerance, utils.maybe_eval(global_step, ss))  # decreasing tolerance (seq.)
        # feed dictionaries (could...in theory, implement stochastic solution of this linear system...)
        _fd = utils.maybe_call(inner_objective_feed_dicts[-1], -1)
        _fd_outer = utils.maybe_call(outer_objective_feed_dicts, utils.maybe_eval(global_step, ss))
        _fd = utils.merge_dicts(_fd, _fd_outer)

        for lin_sys in self._lin_sys:
            lin_sys(tol_val).minimize(ss, _fd)  # implicitly warm restarts with previously found q

    def _forward_step(self, ss, _fd):
        ss.run(self.iteration, _fd)

    def _run_batch_initialization(self, ss, fd):
        ss.run(self.initialization, feed_dict=fd)
