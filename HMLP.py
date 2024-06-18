from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from init_utils import *
from utils import init_params


class HMLP(nn.Module):
    """
    MLP-based Hypernetworks:  
    Adapted from https://github.com/chrhenning/hypnettorch/blob/master/hypnettorch/hnets/mlp_hnet.py
    """

    def __init__(self,
                 target_shapes,  # mnet.param.shapes -> should be set to the largest
                 cond_in_size=8,  # embedding size
                 layers=(100, 100),  # hidden dimension size
                 verbose=True,  # print information
                 activation_fn=torch.nn.ReLU(),  # activation
                 dropout_rate=0.2,  # dropout rate
                 use_batch_norm=False):  # batch norm layer

        nn.Module.__init__(self)

        assert len(target_shapes) > 0
        # conditional embedding size - a layer to preprocess the input (hyperparameters)
        self._cond_in_size = cond_in_size
        self._layers = layers
        self._act_fn = activation_fn
        self._dropout_rate = dropout_rate
        self._use_batch_norm = use_batch_norm
        self._target_shapes = target_shapes
        self._param_shapes = []
        self._param_shapes_meta = []
        self._internal_params = nn.ParameterList()
        self._batchnorm_layers = None
        self._dropout = F.dropout

        # weights
        self._layer_weight_tensors = nn.ParameterList()
        # bias
        self._layer_bias_vectors = nn.ParameterList()

        ### Create conditional weights embedding ###
        assert cond_in_size > 0
        self._internal_params.append(nn.Parameter(data=torch.Tensor(cond_in_size), requires_grad=True))
        torch.nn.init.normal_(self._internal_params[-1], mean=0., std=1.)
        self._param_shapes.append([cond_in_size])
        self._param_shapes_meta.append({'name': 'embedding', 'index': len(self._internal_params) - 1, 'layer': 0})

        ### Create fully-connected hidden-layers ###
        in_size = cond_in_size
        if len(layers) > 0:
            # We use odd numbers starting at 1 as layer indices for hidden layers
            self._add_fc_layers([in_size, *layers[:-1]], layers, False, fc_layers=list(range(1, 2 * len(layers), 2)))
            hidden_size = layers[-1]
        else:
            hidden_size = in_size

        ### Create fully-connected output-layers ###
        self.num_outputs = int(np.sum([np.prod(l) for l in self._target_shapes]))
        self._add_fc_layers([hidden_size], [self.num_outputs], False, fc_layers=[2 * len(layers) + 1])

        ### Finalize construction ###
        self._unconditional_param_shapes_ref = list(range(1, len(self._param_shapes)))
        self._unconditional_params_ref = []
        for idx in self._unconditional_param_shapes_ref:
            meta = self._param_shapes_meta[idx]
            if meta['index'] != -1:
                self._unconditional_params_ref.append(meta['index'])

        self._unconditional_param_shapes = []
        for idx in self._unconditional_param_shapes_ref:
            self._unconditional_param_shapes.append(self._param_shapes[idx])

        if self._unconditional_param_shapes_ref is None:
            self._unconditional_params = None
        else:
            self._unconditional_params = []
            for idx in self._unconditional_params_ref:
                self._unconditional_params.append(self._internal_params[idx])

        ## conditional params
        self._conditional_param_shapes = []
        self._conditional_param_shapes_ref = []
        if self._internal_params is None:
            self._conditional_params = None
        else:
            uc_indices = self._unconditional_params_ref
            if uc_indices is None:
                uc_indices = []
            self._conditional_params = []
            for idx in range(len(self._internal_params)):
                if idx not in uc_indices:
                    self._conditional_params.append(self._internal_params[idx])
                    self._conditional_param_shapes_ref.append(idx)

        for idx in self._conditional_param_shapes_ref:
            self._conditional_param_shapes.append(self._param_shapes[idx])

        ## number of parameters
        self.num_params = int(np.sum([np.prod(l) for l in self._param_shapes]))

        if verbose:
            print('Created MLP Hypernet.')
            print(self)
            print('Meta parameters:')
            print(self._param_shapes_meta)

    def forward(self,
                cond_input=None,
                ret_format='squeezed'):
        """Compute the weights of a target network.
        Args:
        Cond_input: H_container to float_tensor, taken as the conditional input,
        Ret_format: str, default = 'squeezed',
        Returns:
            (list or torch.Tensor): weights generated
        """
        is_batched = True
        if cond_input != None and len(cond_input) == 1:
            is_batched = False
        uncond_input, cond_input, uncond_weights, _ = self._preprocess_forward_args(uncond_input=None,
                                                                                    cond_input=cond_input,
                                                                                    weights=self._internal_params,
                                                                                    ret_format=ret_format)
        ### Prepare hypernet input ###
        assert len(cond_input.shape) == 2 and cond_input.shape[1] == self._cond_in_size
        h = cond_input

        ### Extract layer weights ###
        fc_weights = []
        fc_biases = []
        assert len(uncond_weights) == len(self._unconditional_param_shapes_ref)
        for i, idx in enumerate(self._unconditional_param_shapes_ref):
            meta = self._param_shapes_meta[idx]
            if meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        ### Process inputs through network ###
        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)
            h = F.linear(h, fc_weights[i], bias=fc_biases[i])
            if not last_layer:
                # Batch-norm
                if self._use_batch_norm:
                    batch_mean = torch.mean(h, dim=0).detach()
                    batch_var = torch.var(h, dim=0).detach()
                    h = F.batch_norm(h, batch_mean, batch_var, weight=None, bias=None, training=False)
                # Dropout 
                h = self._dropout(h)
                # Non-linearity
                if self._act_fn is not None:
                    h = self._act_fn(h)

        ### Split output into target shapes ###
        target_weight_matrices = self._flat_to_ret_format(h, ret_format)
        if is_batched:
            return target_weight_matrices
        else:
            return [target_weight_matrices]

    def __str__(self):
        """Print network information."""
        num_uncond = int(np.sum([np.prod(l) for l in self._unconditional_param_shapes]))
        num_cond = int(np.sum([np.prod(l) for l in self._conditional_param_shapes]))
        num_uncond_internal = 0
        num_cond_internal = 0
        if self._unconditional_params is not None:
            num_uncond_internal = int(np.sum([np.prod(l) for l in \
                                              [p.shape for p in self._unconditional_params]]))
        if self._unconditional_params is not None:
            num_cond_internal = int(np.sum([np.prod(l) for l in \
                                            [p.shape for p in self._conditional_params]]))
        msg = 'Hypernetwork with %d weights and %d outputs (compression ' + \
              'ratio: %.2f).\nThe network consists of %d unconditional ' + \
              'weights (%d internally maintained) and %d conditional ' + \
              'weights (%d internally maintained).'
        return msg % (self.num_params, self.num_outputs,
                      self.num_params / self.num_outputs, num_uncond, num_uncond_internal,
                      num_cond, num_cond_internal)

    def _add_fc_layers(self,
                       in_sizes,
                       out_sizes,
                       no_weights,
                       fc_layers=None):
        """Add fully-connected layers to the network.
        This method will set the weight requirements for fully-connected layers
        correctly. During the :meth:`forward` computation, those weights can be
        used in combination with :func:`torch.nn.functional.linear`.
        Args:
            in_sizes (list): List of intergers denoting the input size of each
                added fc-layer.
            out_sizes (list): List of intergers denoting the output size of each
                added fc-layer.
            no_weights (bool): If ``True``, fc-layers will be generated without
                internal parameters :attr:`internal_params`.
            fc_layers (list, optional): See attribute ``cm_layers`` of method
                :meth:`_add_context_mod_layers`.
        """
        assert len(in_sizes) == len(out_sizes)
        assert fc_layers is None or len(fc_layers) == len(in_sizes)
        assert self._param_shapes_meta is not None
        assert not no_weights or self._hyper_shapes_learned_ref is not None

        if self._layer_weight_tensors is None:
            self._layer_weight_tensors = torch.nn.ParameterList()
        if self._layer_bias_vectors is None:
            self._layer_bias_vectors = torch.nn.ParameterList()
        for i, n_in in enumerate(in_sizes):
            n_out = out_sizes[i]
            s_w = [n_out, n_in]
            s_b = [n_out]
            for j, s in enumerate([s_w, s_b]):
                if s is None:
                    continue
                is_bias = True
                if j % 2 == 0:
                    is_bias = False
                if not no_weights:
                    self._internal_params.append(torch.nn.Parameter( \
                        torch.Tensor(*s), requires_grad=True))
                    if is_bias:
                        self._layer_bias_vectors.append( \
                            self._internal_params[-1])
                    else:
                        self._layer_weight_tensors.append( \
                            self._internal_params[-1])
                else:
                    self._hyper_shapes_learned.append(s)
                    self._hyper_shapes_learned_ref.append( \
                        len(self.param_shapes))
                self._param_shapes.append(s)
                self._param_shapes_meta.append({
                    'name': 'bias' if is_bias else 'weight',
                    'index': -1 if no_weights else len(self._internal_params) - 1,
                    'layer': -1 if fc_layers is None else fc_layers[i]
                })
            if not no_weights:
                init_params(self._layer_weight_tensors[-1], self._layer_bias_vectors[-1])

    def _preprocess_forward_args(self,
                                 _input_required=True,
                                 _parse_cond_id_fct=None,
                                 **kwargs):
        """Parse all :meth:`forward` arguments.

        Note:
            This method is currently not considering the arguments
            ``distilled_params`` and ``condition``.

        Args:
            _input_required (bool): Whether at least one of the forward
                arguments ``uncond_input``, ``cond_input`` and ``cond_id`` has
                to be not ``None``.
            _parse_cond_id_fct (func): A function with signature
                ``_parse_cond_id_fct(self, cond_ids, cond_weights)``, where
                ``self`` is the current object, ``cond_ids`` is a ``list`` of
                integers and ``cond_weights`` are the parsed conditional weights
                if any (see return values).
                The function is expected to parse argument ``cond_id`` of the
                :meth:`forward` method. If not provided, we simply use the
                indices within ``cond_id`` to stack elements of
                :attr:`conditional_params`.
            **kwargs: All keyword arguments passed to the :meth:`forward`
                method.

        Returns:
            (tuple): Tuple containing:

            - **uncond_input**: The argument ``uncond_input`` passed to the
              :meth:`forward` method.
            - **cond_input**: If provided, then this is just argument
              ``cond_input`` of the :meth:`forward` method. Otherwise, it is
              either ``None`` or if provided, the conditional input will be
              assembled from the parsed conditional weights ``cond_weights``
              using :meth:`forward` argument ``cond_id``.
            - **uncond_weights**: The unconditional weights :math:`\\theta` to
              be used during forward processing (they will be assembled from
              internal and given weights).
            - **cond_weights**: The conditional weights if tracked be the
              hypernetwork. The parsing is done analoguously as for
              ``uncond_weights``.
        """
        if kwargs['ret_format'] not in ['flattened', 'sequential', 'squeezed']:
            raise ValueError('Return format %s unknown.' % (kwargs['ret_format']))

        # We first parse the weights as they night be needed later to choose
        # inputs via `cond_id`.
        uncond_weights = self._unconditional_params
        cond_weights = self._conditional_params
        if kwargs['weights'] is not None:
            if len(kwargs['weights']) != len(self._param_shapes):
                raise ValueError('The length of argument ' +
                                 '"weights" does not meet the specifications.')
            # In this case, we simply split the given weights into
            # conditional and unconditional weights.
            weights = kwargs['weights']

            assert len(weights) == len(self._param_shapes)

            # Split 'weights' into conditional and unconditional weights.
            up_ref = self._unconditional_param_shapes_ref
            cp_ref = self._conditional_param_shapes_ref

            if up_ref is not None:
                uncond_weights = [None] * len(up_ref)
            else:
                up_ref = []
                uncond_weights = None
            if cp_ref is not None:
                cond_weights = [None] * len(cp_ref)
            else:
                cp_ref = []
                cond_weights = None

            for i in range(len(self._param_shapes)):
                if i in up_ref:
                    idx = up_ref.index(i)
                    assert uncond_weights[idx] is None
                    uncond_weights[idx] = weights[i]
                else:
                    assert i in cp_ref
                    idx = cp_ref.index(i)
                    assert cond_weights[idx] is None
                    cond_weights[idx] = weights[i]

        if _input_required and kwargs['cond_input'] is None and kwargs['uncond_input'] is None:
            raise RuntimeError('No hypernet inputs have been provided!')

        # No further preprocessing required.
        uncond_input = kwargs['uncond_input']

        cond_input = None
        if kwargs['cond_input'] is not None:
            cond_input = kwargs['cond_input']
            if len(cond_input.shape) == 1:
                raise ValueError('Batch dimension for conditional inputs is ' +
                                 'missing.')

        if cond_input is not None and uncond_input is not None:
            # We assume the first dimension being the batch dimension.
            # Note, some old hnet implementations could only process one
            # embedding at a time and it was ok to not have a dedicated
            # batch dimension. To avoid nasty bugs we enforce a separate
            # batch dimension.
            assert len(cond_input.shape) > 1 and len(uncond_input.shape) > 1
            if cond_input.shape[0] != uncond_input.shape[0]:
                # If one batch-size is 1, we just repeat the input.
                if cond_input.shape[0] == 1:
                    batch_size = uncond_input.shape[0]
                    cond_input = cond_input.expand(batch_size,
                                                   *cond_input.shape[1:])
                elif uncond_input.shape[0] == 1:
                    batch_size = cond_input.shape[0]
                    uncond_input = uncond_input.expand(batch_size,
                                                       *uncond_input.shape[1:])
                else:
                    raise RuntimeError('Batch dimensions of hypernet ' +
                                       'inputs do not match!')
            assert cond_input.shape[0] == uncond_input.shape[0]

        return uncond_input, cond_input, uncond_weights, cond_weights

    def _flat_to_ret_format(self, flat_out, ret_format):
        """Helper function to convert flat hypernet output into desired output
        format.

        Args:
            flat_out (torch.Tensor): The flat output tensor corresponding to
                ``ret_format='flattened'``.
            ret_format (str): The target output format. See docstring of method
                :meth:`forward`.

        Returns:
            (list or torch.)
        """
        assert ret_format in ['flattened', 'sequential', 'squeezed']
        assert len(flat_out.shape) == 2
        batch_size = flat_out.shape[0]
        if ret_format == 'flattened':
            return flat_out
        ret = [[] for _ in range(batch_size)]
        ind = 0
        for s in self._target_shapes:
            num = int(np.prod(s))
            W = flat_out[:, ind:ind + num]
            W = W.view(batch_size, *s)
            for bind, W_b in enumerate(torch.split(W, 1, dim=0)):
                W_b = torch.squeeze(W_b, dim=0)
                assert np.all(np.equal(W_b.shape, s))
                ret[bind].append(W_b)
            ind += num
        if ret_format == 'squeezed' and batch_size == 1:
            return ret[0]
        return ret

    def apply_hyperfan_init(self, method='in', use_xavier=False,
                            uncond_var=1., cond_var=1., mnet=None,
                            w_val=None, w_var=None, b_val=None, b_var=None):
        if method not in ['in', 'out', 'harmonic']:
            raise ValueError('Invalid value "%s" for argument "method".' %
                             method)
        if self._unconditional_params is None:
            assert self._no_uncond_weights
            raise ValueError('Hypernet without internal weights can\'t be ' +
                             'initialized.')
        meta = []

        # Heuristical approach to derive meta information from given shapes.
        layer_ind = 0
        for i, s in enumerate(self._target_shapes):
            curr_meta = dict()
            if len(s) > 1:
                curr_meta['name'] = 'weight'
                curr_meta['layer'] = layer_ind
                layer_ind += 1
            else:  # just a heuristic, we can't know
                curr_meta['name'] = 'bias'
                if i > 0 and meta[-1]['name'] == 'weight':
                    curr_meta['layer'] = meta[-1]['layer']
                else:
                    curr_meta['layer'] = -1
            meta.append(curr_meta)
        assert len(meta) == len(self._target_shapes)

        # Mapping from layer index to the corresponding shape.
        layer_shapes = dict()
        # Mapping from layer index to whether the layer has a bias vector.
        layer_has_bias = defaultdict(lambda: False)
        for i, m in enumerate(meta):
            if m['name'] == 'weight' and m['layer'] != -1:
                assert len(self._target_shapes[i]) > 1
                layer_shapes[m['layer']] = self._target_shapes[i]
            if m['name'] == 'bias' and m['layer'] != -1:
                layer_has_bias[m['layer']] = True

        ### Compute input variance ###
        cond_dim = self._cond_in_size
        inp_dim = cond_dim

        input_variance = 0
        if cond_dim > 0:
            input_variance += (cond_dim / inp_dim) * cond_var

        ### Initialize hidden layers to preserve variance ###
        # Note, if batchnorm layers are used, they will simply be initialized to
        # have no effect after initialization. This does not effect the
        # performed whitening operation.
        if self._batchnorm_layers is not None:
            for bn_layer in self._batchnorm_layers:
                if hasattr(bn_layer, 'scale'):
                    nn.init.ones_(bn_layer.scale)
                if hasattr(bn_layer, 'bias'):
                    nn.init.zeros_(bn_layer.bias)

            # Since batchnorm layers whiten the statistics of hidden
            # acitivities, the variance of the input will not be preserved by
            # Xavier/Kaiming.
            if len(self._batchnorm_layers) > 0:
                input_variance = 1.

        # We initialize biases with 0 (see Xavier assumption 4 in the Hyperfan
        # paper). Otherwise, we couldn't ignore the biases when computing the
        # output variance of a layer.
        # Note, we have to use fan-in init for the hidden layer to ensure the
        # property, that we preserve the input variance.
        assert len(self._layers) + 1 == len(self._layer_weight_tensors)
        for i, w_tensor in enumerate(self._layer_weight_tensors[:-1]):
            if use_xavier:
                xavier_fan_in_(w_tensor)
            else:
                torch.nn.init.kaiming_uniform_(w_tensor, mode='fan_in',
                                               nonlinearity='relu')
            nn.init.zeros_(self._layer_bias_vectors[i])

        ### Define default parameters of weight init distributions ###
        w_val_list = []
        w_var_list = []
        b_val_list = []
        b_var_list = []

        for i, m in enumerate(meta):
            def extract_val(user_arg):
                curr = None
                if isinstance(user_arg, (list, tuple)) and \
                        user_arg[i] is not None:
                    curr = user_arg[i]
                elif isinstance(user_arg, (dict)) and \
                        m['name'] in user_arg.keys():
                    curr = user_arg[m['name']]
                return curr

            curr_w_val = extract_val(w_val)
            curr_w_var = extract_val(w_var)
            curr_b_val = extract_val(b_val)
            curr_b_var = extract_val(b_var)

            if m['name'] == 'weight' or m['name'] == 'bias':
                if None in [curr_w_val, curr_w_var, curr_b_val, curr_b_var]:
                    # If distribution not fully specified, then we just fall
                    # back to hyper-fan init.
                    curr_w_val = None
                    curr_w_var = None
                    curr_b_val = None
                    curr_b_var = None
            else:
                assert m['name'] in ['bn_scale', 'bn_shift', 'cm_scale',
                                     'cm_shift', 'embedding']
                if curr_w_val is None:
                    curr_w_val = 0
                if curr_w_var is None:
                    curr_w_var = 0
                if curr_b_val is None:
                    curr_b_val = 1 if m['name'] in ['bn_scale', 'cm_scale'] \
                        else 0
                if curr_b_var is None:
                    curr_b_var = 1 if m['name'] in ['embedding'] else 0

            w_val_list.append(curr_w_val)
            w_var_list.append(curr_w_var)
            b_val_list.append(curr_b_val)
            b_var_list.append(curr_b_var)

        ### Initialize output heads ###
        # Note, that all output heads are realized internally via one large
        # fully-connected layer.
        # All output heads are linear layers. The biases of these linear
        # layers (called gamma and beta in the paper) are simply initialized
        # to zero. Note, that we allow deviations from this below.
        nn.init.zeros_(self._layer_bias_vectors[-1])

        c_relu = 1 if use_xavier else 2

        # We are not interested in the fan-out, since the fan-out is just
        # the number of elements in the main network.
        # `fan-in` is called `d_k` in the paper and is just the size of the
        # last hidden layer.
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out( \
            self._layer_weight_tensors[-1])

        s_ind = 0
        for i, out_shape in enumerate(self._target_shapes):
            m = meta[i]
            e_ind = s_ind + int(np.prod(out_shape))

            curr_w_val = w_val_list[i]
            curr_w_var = w_var_list[i]
            curr_b_val = b_val_list[i]
            curr_b_var = b_var_list[i]

            if curr_w_val is None:
                c_bias = 2 if layer_has_bias[m['layer']] else 1

                if m['name'] == 'bias':
                    m_fan_out = out_shape[0]

                    # NOTE For the hyperfan-out init, we also need to know the
                    # fan-in of the layer.
                    if m['layer'] != -1:
                        m_fan_in, _ = calc_fan_in_and_out( \
                            layer_shapes[m['layer']])
                    else:
                        # FIXME Quick-fix.
                        m_fan_in = m_fan_out

                    var_in = c_relu / (2. * fan_in * input_variance)
                    num = c_relu * (1. - m_fan_in / m_fan_out)
                    denom = fan_in * input_variance
                    var_out = max(0, num / denom)

                else:
                    assert m['name'] == 'weight'
                    m_fan_in, m_fan_out = calc_fan_in_and_out(out_shape)

                    var_in = c_relu / (c_bias * m_fan_in * fan_in * \
                                       input_variance)
                    var_out = c_relu / (m_fan_out * fan_in * input_variance)

                if method == 'in':
                    var = var_in
                elif method == 'out':
                    var = var_out
                elif method == 'harmonic':
                    var = 2 * (1. / var_in + 1. / var_out)
                else:
                    raise ValueError('Method %s invalid.' % method)

                # Initialize output head weight tensor using `var`.
                std = math.sqrt(var)
                a = math.sqrt(3.0) * std
                torch.nn.init._no_grad_uniform_( \
                    self._layer_weight_tensors[-1][s_ind:e_ind, :], -a, a)
            else:
                if curr_w_var == 0:
                    nn.init.constant_(
                        self._layer_weight_tensors[-1][s_ind:e_ind, :],
                        curr_w_val)
                else:
                    std = math.sqrt(curr_w_var)
                    a = math.sqrt(3.0) * std
                    torch.nn.init._no_grad_uniform_( \
                        self._layer_weight_tensors[-1][s_ind:e_ind, :],
                        curr_w_val - a, curr_w_val + a)

                if curr_b_var == 0:
                    nn.init.constant_(
                        self._layer_bias_vectors[-1][s_ind:e_ind],
                        curr_b_val)
                else:
                    std = math.sqrt(curr_b_var)
                    a = math.sqrt(3.0) * std
                    torch.nn.init._no_grad_uniform_( \
                        self._layer_bias_vectors[-1][s_ind:e_ind],
                        curr_b_val - a, curr_b_val + a)
            s_ind = e_ind
