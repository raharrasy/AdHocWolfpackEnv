import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from misc import *
import timeit

class DQN(nn.Module):

    def __init__(self, h, w, hidden, lstm_seq_length, outputs, extended_feature_len=0, conv_kernel_sizes = [4,2],
                 pool_kernel_sizes=[3,2],  conv_strides=[1,1], pool_conv_strides=[1,1],
                 num_channels = 3, device="cpu", mode="full"):
        super(DQN, self).__init__()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes
        self.conv_strides = conv_strides
        self.pool_conv_strides = pool_conv_strides
        self.mode = mode
        self.device = device

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=self.conv_kernel_sizes[0],
                               stride=self.conv_strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d(self.pool_kernel_sizes[0],
                                      stride=self.pool_conv_strides[0])
        self.conv2 = nn.Conv2d(16, 32, kernel_size= self.conv_kernel_sizes[1],
                               stride=self.conv_strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(self.pool_kernel_sizes[1],
                                      stride=self.pool_conv_strides[1])

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        def pooling_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def calculate_output_dim(inp):
            return pooling_size_out(conv2d_size_out(pooling_size_out(
            conv2d_size_out(inp, kernel_size=self.conv_kernel_sizes[0], stride = self.conv_strides[0]),
            kernel_size=self.pool_kernel_sizes[0], stride=self.pool_conv_strides[0]),
            kernel_size=self.conv_kernel_sizes[1], stride = self.conv_strides[1]),
            kernel_size=self.pool_kernel_sizes[1], stride=self.pool_conv_strides[1])

        convw = calculate_output_dim(w)
        convh = calculate_output_dim(h)


        self.lstm_input_dim = convw * convh * 32
        self.hidden_dim = hidden

        self.lstm = nn.LSTM(self.lstm_input_dim, hidden, batch_first=True)
        self.lstm_seq_length = lstm_seq_length
        self.head = nn.Linear(hidden+extended_feature_len, 7)
        #self.head2 = nn.Linear(20, outputs)



    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, extended_feature = None):
        original_inp_size = list(x.size())
        transformed_size = [original_inp_size[0]*original_inp_size[1]]
        transformed_size.extend(original_inp_size[2:])

        x = x.view(transformed_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = x.view(original_inp_size[0],original_inp_size[1],-1)

        hidden = (torch.zeros(1, original_inp_size[0], self.hidden_dim).to(self.device),
                         torch.zeros(1, original_inp_size[0], self.hidden_dim).to(self.device))

        x, hidden = self.lstm(x, hidden)

        input = x[:,-1,:]
        if not extended_feature is None :
            input = torch.cat((x[:,-1,:], extended_feature), dim=-1)
        #action_vals = F.relu(self.head(input))
        action_vals = self.head(input)
        return action_vals

class MADDPGDQN(nn.Module):

    def __init__(self, h, w, hidden, lstm_seq_length, outputs, extended_feature_len=0, extended_feature2_len=0,
                 conv_kernel_sizes = [5,3], pool_kernel_sizes=[5,3],  conv_strides=[2,1], pool_conv_strides=[1,1],
                 num_channels = 3, device="cpu", mode="full"):
        super(MADDPGDQN, self).__init__()

        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes
        self.conv_strides = conv_strides
        self.pool_conv_strides = pool_conv_strides
        self.mode = mode
        self.device = device

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=self.conv_kernel_sizes[0],
                               stride=self.conv_strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d(self.pool_kernel_sizes[0],
                                      stride=self.pool_conv_strides[0])
        self.conv2 = nn.Conv2d(16, 32, kernel_size= self.conv_kernel_sizes[1],
                               stride=self.conv_strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(self.pool_kernel_sizes[1],
                                      stride=self.pool_conv_strides[1])

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        def pooling_size_out(size, kernel_size = 5, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def calculate_output_dim(inp):
            return pooling_size_out(conv2d_size_out(pooling_size_out(
            conv2d_size_out(inp, kernel_size=self.conv_kernel_sizes[0], stride = self.conv_strides[0]),
            kernel_size=self.pool_kernel_sizes[0], stride=self.pool_conv_strides[0]),
            kernel_size=self.conv_kernel_sizes[1], stride = self.conv_strides[1]),
            kernel_size=self.pool_kernel_sizes[1], stride=self.pool_conv_strides[1])

        convw = calculate_output_dim(w)
        convh = calculate_output_dim(h)

        self.lstm_input_dim = convw * convh * 32 + extended_feature_len
        self.hidden_dim = hidden

        self.lstm = nn.LSTM(self.lstm_input_dim, hidden, batch_first=True)
        self.lstm_seq_length = lstm_seq_length
        self.head = nn.Linear(hidden + extended_feature2_len, 50)
        self.head2 = nn.Linear(50, outputs)



    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, extended_feature = None, extended_feature_2 = None):
        original_inp_size = list(x.size())
        transformed_size = [original_inp_size[0]*original_inp_size[1]]
        transformed_size.extend(original_inp_size[2:])

        x = x.view(transformed_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = x.view(original_inp_size[0],original_inp_size[1],-1)

        hidden = (torch.zeros(1, original_inp_size[0], self.hidden_dim).to(self.device),
                         torch.zeros(1, original_inp_size[0], self.hidden_dim).to(self.device))

        x = torch.cat([x,extended_feature], dim=-1)
        x, hidden = self.lstm(x, hidden)

        input = x[:,-1,:]
        if not extended_feature_2 is None :
            input = torch.cat((x[:,-1,:], extended_feature_2), dim=-1)
        action_vals = F.relu(self.head(input))
        action_vals = self.head2(action_vals)
        return action_vals

class MapProcessor(nn.Module):
    def __init__(self, h, w, u_dim, conv_kernel_sizes = [4,2],
                 pool_kernel_sizes=[3,2],  conv_strides=[1,1], pool_conv_strides=[1,1],
                 num_channels = 3, device="cpu"):
        super(MapProcessor, self).__init__()
        self.conv_kernel_sizes = conv_kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes
        self.conv_strides = conv_strides
        self.pool_conv_strides = pool_conv_strides
        self.device = device

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=self.conv_kernel_sizes[0], stride=self.conv_strides[0])
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool1 = nn.MaxPool2d(self.pool_kernel_sizes[0],
                                      stride=self.pool_conv_strides[0])
        self.conv2 = nn.Conv2d(16, 32, kernel_size=self.conv_kernel_sizes[1],
                               stride=self.conv_strides[1])
        self.bn2 = nn.BatchNorm2d(32)
        self.max_pool2 = nn.MaxPool2d(self.pool_kernel_sizes[1],
                                      stride=self.pool_conv_strides[1])

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def pooling_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def calculate_output_dim(inp):
            return pooling_size_out(conv2d_size_out(pooling_size_out(
                conv2d_size_out(inp, kernel_size=self.conv_kernel_sizes[0], stride=self.conv_strides[0]),
                kernel_size=self.pool_kernel_sizes[0], stride=self.pool_conv_strides[0]),
                kernel_size=self.conv_kernel_sizes[1], stride=self.conv_strides[1]),
                kernel_size=self.pool_kernel_sizes[1], stride=self.pool_conv_strides[1])

        convw = calculate_output_dim(w)
        convh = calculate_output_dim(h)

        self.ff_input_dim = convw * convh * 32
        self.u_dim = u_dim

        self.head = nn.Linear(self.ff_input_dim, self.u_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool2(x)
        x = x.view(batch_size,-1)

        out = F.relu(self.head(x))
        return out

class RFMBlock(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, dim_out):
        super(RFMBlock, self).__init__()
        self.fc_edge = nn.Linear(dim_in_edge,hidden_dim)
        self.fc_edge2 = nn.Linear(hidden_dim, dim_out)
        self.fc_node = nn.Linear(dim_in_node, hidden_dim)
        self.fc_node2 = nn.Linear(hidden_dim, dim_out)
        self.fc_u = nn.Linear(dim_in_u, hidden_dim)
        self.fc_u2 = nn.Linear(hidden_dim, dim_out)
        # Check Graph batch

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat'] }

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None,:].repeat(num_edge,1) for g, num_edge
                       in zip(g_repr,edge_nums)], dim=0)
        inp = torch.cat([edges.data['edge_feat'],edges.src['node_feat'],edges.dst['node_feat'], u], dim=-1)
        return {'edge_feat' : self.fc_edge2(F.relu(self.fc_edge(inp)))}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)
        inp = torch.cat([nodes.data['node_feat'], nodes.data['h'], u], dim=-1)
        return {'node_feat' : self.fc_node2(F.relu(self.fc_node(inp)))}

    def compute_u_repr(self, n_comb, e_comb, g_repr):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)
        return self.fc_u2(F.relu(self.fc_u(inp)))

    def forward(self, graph, edge_feat, node_feat, g_repr):
        #graph.register_message_func(self.graph_message_func)
        #graph.register_reduce_func(self.graph_reduce_func)
        node_trf_func = lambda x: self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)
        #graph.register_apply_node_func(node_trf_func)

        graph.edata['edge_feat'] = edge_feat
        graph.ndata['node_feat'] = node_feat
        edge_trf_func = lambda x : self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)

        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)
        #graph.update_all(self.graph_msg, self.graph_reduce, node_trf_func)


        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')

        e_out = graph.edata['edge_feat']
        n_out = graph.ndata['node_feat']


        graph.edata.pop('edge_feat')
        graph.ndata.pop('node_feat')
        graph.ndata.pop('h')

        return e_out, n_out, self.compute_u_repr(n_comb, e_comb, g_repr)

class GraphLSTM(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, dim_out, unbatch_return_feats=True):
        super(GraphLSTM, self).__init__()
        self.lstm_edge = nn.LSTM(dim_in_edge, dim_out, batch_first=True)
        self.lstm_node = nn.LSTM(dim_in_node, dim_out, batch_first=True)
        self.lstm_u = nn.LSTM(dim_in_u, dim_out, batch_first=True)

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')
        self.unbatch_return_feats = unbatch_return_feats

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat'] }

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None, :].repeat(num_edge, 1) for g, num_edge
                       in zip(g_repr, edge_nums)], dim=0)
        inp = torch.cat([edges.data['edge_feat'], edges.src['node_feat'],
                         edges.dst['node_feat'], u], dim=-1)[:,None,:]
        hidden = (edges.data['hidden1'][None,:,:], edges.data['hidden2'][None,:,:])
        out, hidden = self.lstm_edge(inp, hidden)
        out_shape = out.shape
        out = out.view([out_shape[0], out_shape[2]])
        return {'edge_feat': F.relu(out), 'hidden1' : hidden[0][0], 'hidden2' : hidden[1][0]}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)
        inp = torch.cat([nodes.data['node_feat'], nodes.data['h'], u], dim=-1)[:,None,:]
        hidden = (nodes.data['hidden1'][None,:,:], nodes.data['hidden2'][None,:,:])
        out, hidden = self.lstm_node(inp, hidden)
        out_shape = out.shape
        out = out.view([out_shape[0], out_shape[2]])
        return {'node_feat' : F.relu(out), 'hidden1' : hidden[0][0], 'hidden2' : hidden[1][0]}

    def compute_u_repr(self, n_comb, e_comb, g_repr, hidden):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)[:,None,:]
        out, hidden = self.lstm_u(inp, hidden)
        out_shape = out.shape
        out = out.view([out_shape[0], out_shape[2]])
        return F.relu(out), hidden


    def forward(self, graph, edge_feat, node_feat, g_repr, edge_hidden, node_hidden, graph_hidden):
        #graph.register_message_func(self.graph_message_func)
        #graph.register_reduce_func(self.graph_reduce_func)

        graph.edata['edge_feat'] = edge_feat
        graph.ndata['node_feat'] = node_feat
        graph.edata['hidden1'] = edge_hidden[0][0]
        graph.ndata['hidden1'] = node_hidden[0][0]
        graph.edata['hidden2'] = edge_hidden[1][0]
        graph.ndata['hidden2'] = node_hidden[1][0]

        node_trf_func = lambda x : self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)
        #graph.register_apply_node_func(node_trf_func)

        edge_trf_func = lambda x: self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)
        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)
        #graph.update_all(self.graph_msg, self.graph_reduce, node_trf_func)

        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')

        u_out, u_hidden = self.compute_u_repr(n_comb, e_comb, g_repr, graph_hidden)
        unbatched_graphs = dgl.unbatch(graph)


        e_hiddens = [(gr.edata['hidden1'][None,:,:],gr.edata['hidden2'][None,:,:]) for gr in unbatched_graphs]
        n_hiddens = [(gr.ndata['hidden1'][None,:,:],gr.ndata['hidden2'][None,:,:]) for gr in unbatched_graphs]

        e_feat = graph.edata['edge_feat']
        n_feat = graph.ndata['node_feat']

        graph.edata.pop('edge_feat')
        graph.edata.pop('hidden1')
        graph.edata.pop('hidden2')
        graph.ndata.pop('node_feat')
        graph.ndata.pop('hidden1')
        graph.ndata.pop('hidden2')
        graph.ndata.pop('h')

        return e_feat, e_hiddens, n_feat, n_hiddens, \
               u_out, u_hidden

class MRFMessagePassingModule(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, mid_pair,num_acts, device=None):
        super(MRFMessagePassingModule, self).__init__()
        self.joint_factor_model = nn.Linear(dim_in_edge+dim_in_u,num_acts*mid_pair)
        self.indiv_factor_model = nn.Linear(dim_in_node+dim_in_u, num_acts)
        self.joint_util_model = nn.Linear(dim_in_edge+dim_in_u, num_acts*mid_pair)
        self.indiv_util_model = nn.Linear(dim_in_node+dim_in_u, num_acts)

        self.edge_flag_repr = nn.Linear(dim_in_edge+dim_in_u, mid_pair)
        self.edge_classifier = nn.Linear(mid_pair, 2)

        self.mid_pair = mid_pair
        self.num_acts = num_acts
        self.device = device

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def compute_node_data(self, nodes):
        return {'indiv_factor': self.indiv_factor_model(torch.cat([nodes.data['node_feat'],
                                                                   nodes.data['u_feat']], dim=-1)),
                'indiv_util': self.indiv_util_model(torch.cat([nodes.data['node_feat'],
                                                               nodes.data['u_feat']], dim=-1))}

    def compute_edge_data(self, edges, idx_revs):
        inp = edges.data['edge_feat']
        inp_reflected = edges.data['edge_feat_reflected']

        # Compute the factor components.
        factor_comp = self.joint_factor_model(torch.cat([inp,edges.data['u_feat']], dim=-1)).view(-1,
                                                                            self.num_acts, self.mid_pair)
        factor_comp_reflected = self.joint_factor_model(torch.cat([inp_reflected,
                                                                   edges.data['u_feat']], dim=-1)).view(-1,
                                                                   self.num_acts, self.mid_pair).permute(0,2,1)
        # Compute the util components
        util_comp = self.joint_util_model(torch.cat([inp,edges.data['u_feat']],
                                                    dim=-1)).view(-1,self.num_acts, self.mid_pair)
        util_comp_reflected = self.joint_util_model(torch.cat([inp_reflected,edges.data['u_feat']], dim=-1)).view(-1,
                                                                        self.num_acts, self.mid_pair).permute(0,2,1)

        factor_vals = torch.bmm(factor_comp, factor_comp_reflected)
        util_vals = torch.bmm(util_comp, util_comp_reflected)

        # Compute representation of edge for edge existence computation
        edge_flag_repr = self.edge_flag_repr(torch.cat([inp, edges.data['u_feat']], dim=-1))
        edge_flag_repr_ref = self.edge_flag_repr(torch.cat([inp_reflected, edges.data['u_feat']], dim=-1))
        edge_flag_repr_mul = edge_flag_repr * edge_flag_repr_ref

        edge_flags_logits = self.edge_classifier(edge_flag_repr_mul)
        edge_flags = gumbel_softmax(edge_flags_logits, idx_revs, hard=True)
        return {'factor_vals': factor_vals, 'util_vals':util_vals, 'edge_flags':edge_flags}

    def graph_edge_update_func(self, edges, graph):
        # Process softmax that are passed as message
        log_softmax = nn.LogSoftmax(dim=-1)

        # Compute forward message for edge
        aggregated_msg = edges.src['model_msg'] - edges.data['reverse_msg'] + edges.src['indiv_factor']
        m = edges.data['factor_vals'] + aggregated_msg[:,None,:]
        m_sum = log_softmax(torch.log(torch.exp(m).sum(dim=-1)))[:,None,:]
        default_m = torch.zeros_like(m_sum).to(self.device)
        msg = (torch.cat([m_sum,default_m], dim=1) * edges.data['edge_flags'][:,:,None]).sum(dim=1)


        # Compute reverse message for edge
        aggregated_msg_reverse = (edges.dst['model_msg'] - edges.data['msg']) + edges.dst['indiv_factor']
        m_rev = edges.data['factor_vals'].permute(0,2,1) + aggregated_msg_reverse[:,None,:]
        m_rev_sum = log_softmax(torch.log(torch.exp(m_rev).sum(dim=-1)))[:,None,:]
        rev_default_m = torch.zeros_like(m_rev_sum).to(self.device)
        msg_rev = (torch.cat([m_rev_sum,rev_default_m], dim=1) * edges.data['edge_flags'][:,:,None]).sum(dim=1)

        # Compute forward util for edge
        util = (edges.src['util_msg'] -
                          edges.data['reverse_util_msg']) + edges.src['indiv_util']
        total_util_msg = edges.data['util_vals'] + util[:,None,:]
        m_normalized = torch.exp(log_softmax(m))
        util_msg = (total_util_msg * m_normalized).sum(dim=-1)[:,None,:]
        default_util = torch.zeros_like(util_msg).to(self.device)
        util_msg = (torch.cat([util_msg,default_util], dim=1) * edges.data['edge_flags'][:,:,None]).sum(dim=1)

        # Compute backward util for edge
        rev_util = (edges.dst['util_msg'] -
                edges.data['util_msg']) + edges.dst['indiv_util']
        rev_total_util_msg = edges.data['util_vals'].permute(0,2,1) + rev_util[:, None, :]
        m_rev_normalized = torch.exp(log_softmax(m_rev))
        rev_util_msg = (rev_total_util_msg * m_rev_normalized).sum(dim=-1)[:, None, :]
        default_rev_util = torch.zeros_like(rev_util_msg).to(self.device)
        rev_util_msg = (torch.cat([rev_util_msg,
                                   default_rev_util], dim=1) * edges.data['edge_flags'][:, :, None]).sum(dim=1)

        graph.edata['msg'] = msg
        graph.edata['reverse_msg'] = msg_rev
        graph.edata['util_msg'] = util_msg
        graph.edata['reverse_util_msg'] = rev_util_msg

        # return {'msg': msg, 'reverse_msg':msg_rev, 'util_msg':util_msg, "reverse_util_msg":rev_util_msg}
        return {'msg': msg, 'util_msg': util_msg}

    #def graph_message_func(self, edges):
    #    return {'msg': edges.data['msg'], 'util_msg':edges.data['util_msg']}

    def graph_reduce_func(self, nodes):
        mod_msg = torch.sum(nodes.mailbox['msg'], dim=1)
        util_msg = torch.sum(nodes.mailbox['util_msg'], dim=1)
        return {'model_msg': mod_msg, 'util_msg': util_msg}

    def forward(self, graph, edge_feats, node_feats, graph_feats, edge_feat_reflected):
        graph.edata['edge_feat'] = edge_feats
        graph.edata['edge_feat_reflected'] = edge_feat_reflected
        graph.ndata['node_feat'] = node_feats

        graph.edata['u_feat'] = torch.cat([a[None,:].repeat(k,1) for a,k in
                                            zip(graph_feats,graph.batch_num_edges)], dim=0)
        graph.ndata['u_feat'] = torch.cat([a[None, :].repeat(k, 1) for a, k in
                                            zip(graph_feats, graph.batch_num_nodes)], dim=0)

        graph.apply_nodes(self.compute_node_data)
        edge_idxes_0, edge_idxes_1 = graph.edges()
        idx_revs = graph.edge_ids(edge_idxes_1, edge_idxes_0)
        computed_edge_data = lambda x : self.compute_edge_data(x, idx_revs)
        graph.apply_edges(computed_edge_data)

        graph.ndata['model_msg'] = torch.zeros_like(graph.ndata['indiv_factor']).to(self.device)
        graph.edata['reverse_msg'] = torch.zeros(graph.number_of_edges(),self.num_acts).to(self.device)
        graph.edata['msg'] = torch.zeros(graph.number_of_edges(),self.num_acts).to(self.device)

        graph.ndata['util_msg'] = torch.zeros_like(graph.ndata['indiv_factor']).to(self.device)
        graph.edata['util_msg'] = torch.zeros(graph.number_of_edges(),self.num_acts).to(self.device)
        graph.edata['reverse_util_msg'] = torch.zeros(graph.number_of_edges(),self.num_acts).to(self.device)

        edge_update_fun = lambda x : self.graph_edge_update_func(x, graph=graph)
        g_lists = graph.batch_num_nodes
        for a in range(max(g_lists)):
            # graph.apply_edges(self.graph_edge_update_func)
            graph.update_all(message_func=edge_update_fun, reduce_func=self.graph_reduce_func)


        #all_log = graph.ndata['model_msg'] + graph.ndata['indiv_factor']
        #all_utils = graph.ndata['util_msg'] + graph.ndata['indiv_util']

        #softmax = nn.Softmax(dim=-1)
        #probs_acts = softmax(all_log)
        #graph.ndata['model_node_res'] = probs_acts
        #graph.ndata['util_node_res'] = all_utils

        unbatched_graphs = dgl.unbatch(graph)
        returned_values = torch.cat([(g.ndata['util_msg'] + g.ndata['indiv_util'])[0][None,:]
                          for g in unbatched_graphs], dim=0)

        return returned_values




class AdHocWolfpackGNN(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, hidden_dim2, dim_mid, dim_out,
                 dim_lstm_out, fin_mid_dim, act_dims, with_added_u_feat=False, added_u_feat_dim=0,
                 with_rfm=False):
        super(AdHocWolfpackGNN, self).__init__()
        self.dim_lstm_out = dim_lstm_out
        self.MapCNN = MapProcessor(25,25, dim_in_u)
        self.with_added_u_feat = with_added_u_feat
        if not self.with_added_u_feat:
            self.GNBlock = RFMBlock(dim_mid+ dim_in_node + dim_in_u,
                                2 * dim_in_node + dim_in_u + dim_in_edge,
                                2 * dim_mid + dim_in_u, hidden_dim,
                                dim_mid)
        else:
            self.GNBlock = RFMBlock(dim_mid + dim_in_node + dim_in_u + added_u_feat_dim,
                                    2 * dim_in_node + dim_in_u + dim_in_edge + added_u_feat_dim,
                                    2 * dim_mid + dim_in_u + added_u_feat_dim, hidden_dim,
                                    dim_mid)
        self.GNBlock2 = RFMBlock(dim_out+2*dim_mid, 4*dim_mid, 2*dim_out + dim_mid, hidden_dim2, dim_out)
        # here

        self.GraphLSTM = GraphLSTM(dim_lstm_out+2*dim_out, 4*dim_out, 2*dim_lstm_out + dim_out, dim_lstm_out)
        self.with_rfm = with_rfm

        if not self.with_rfm:
            self.pre_q_net = nn.Linear(dim_lstm_out, fin_mid_dim)
            self.q_net = nn.Linear(fin_mid_dim, act_dims)

        else:
            self.q_net = MRFMessagePassingModule(self.dim_lstm_out, self.dim_lstm_out, self.dim_lstm_out, 5, 7)


    def forward(self, graph, edge_feat, node_feat, u_obs, hidden_e, hidden_n, hidden_u, added_u_feat=None):

        g_repr = self.MapCNN.forward(u_obs)

        if self.with_added_u_feat:
            g_repr = torch.cat([g_repr, added_u_feat], dim=-1)
        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock.forward(graph, edge_feat, node_feat, g_repr)

        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock2.forward(graph,
                                                                               updated_e_feat, updated_n_feat,
                                                                               updated_u_feat)

        e_feat, e_hid, n_feat, n_hid, u_out, u_hid = self.GraphLSTM.forward(graph, updated_e_feat, updated_n_feat,
                                                                                updated_u_feat, hidden_e,
                                                                                hidden_n, hidden_u)


        inp = u_out.view(u_out.shape[0],-1)
        if not self.with_rfm:
            out = self.q_net(F.relu(self.pre_q_net(inp)))
        else:
            edges = graph.edges()
            reverse_feats = e_feat[graph.edge_ids(edges[1],edges[0])]
            out = self.q_net(graph, e_feat, n_feat, u_out, reverse_feats)

        return out, e_hid, n_hid, u_hid

class GraphOppoModel(nn.Module):
    def __init__(self,dim_in_node, dim_in_edge, dim_in_u, added_u, hidden_dim, hidden_dim2, dim_mid, dim_out,
                 added_mid_dims, act_dims):
        super(GraphOppoModel, self).__init__()
        self.MapCNN = MapProcessor(25, 25, dim_in_u)
        dim_in_u = dim_in_u + added_u
        self.GNBlock = RFMBlock(dim_mid + dim_in_node + dim_in_u,
                                2 * dim_in_node + dim_in_u + dim_in_edge,
                                2 * dim_mid + dim_in_u, hidden_dim,
                                dim_mid)
        self.GNBlock2 = RFMBlock(dim_out + 2 * dim_mid, 4 * dim_mid, 2 * dim_out + dim_mid, hidden_dim2, dim_out)
        # here

        self.head = nn.Linear(dim_out, added_mid_dims)
        self.head2 = nn.Linear(added_mid_dims, act_dims)

    def forward(self, graph, edge_feat, node_feat, u_obs, added_u):

        g_repr = self.MapCNN.forward(u_obs)
        g_repr = torch.cat((g_repr, added_u), dim=-1)
        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock.forward(graph, edge_feat, node_feat, g_repr)

        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock2.forward(graph,
                                                                               updated_e_feat, updated_n_feat,
                                                                               updated_u_feat)

        out = self.head2(F.relu(self.head(updated_n_feat)))
        return out