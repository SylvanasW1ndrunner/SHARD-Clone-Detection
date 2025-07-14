# 3_model.py (最终分层版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool

# 您可以将这个新模型保存在 model.py 文件中


class HierarchicalGNN(nn.Module):
    def __init__(self, node_feature_dim, gnn_hidden_dim, output_dim, num_gat_layers=3, heads=4):
        super(HierarchicalGNN, self).__init__()

        # --- GAT卷积层 (保持不变) ---
        self.convs = nn.ModuleList()
        gat_output_dim = 0
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            self.convs.append(GATConv(current_dim, gnn_hidden_dim, heads=heads))
            current_dim = gnn_hidden_dim * heads
        gat_output_dim = current_dim # 记录下GAT最终输出的维度

        # --- 分层路径的专用层 (保持不变) ---
        self.func_agg_linear = nn.Linear(gat_output_dim, gat_output_dim)

        # --- 新增：门控网络 ---
        # 门控网络的输入是“分层”和“扁平”两个向量拼接的结果
        gate_input_dim = gat_output_dim * 2
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, gat_output_dim),
            nn.Sigmoid() # Sigmoid函数确保输出的门控值在0到1之间
        )

        self.final_linear = nn.Linear(gat_output_dim, output_dim)

    def forward(self, data):
        # 解包Data对象
        x, edge_index, function_mapping, node_to_graph_batch = \
            data.x, data.edge_index, data.function_mapping, data.batch

        # 确保func_to_graph_batch存在，以备后用
        func_to_graph_batch = data.func_to_graph_batch if hasattr(data, 'func_to_graph_batch') else None

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        block_embeddings = x # 重命名以示清晰

        # --- 层次二：并行池化 ---

        # 路径A: 分层路径 (Hierarchical Path)
        # ------------------------------------
        valid_nodes_mask = function_mapping >= 0
        if hasattr(data, 'global_function_mapping') and torch.any(valid_nodes_mask):
            global_func_map = data.global_function_mapping
            func_embeds = global_add_pool(block_embeddings[valid_nodes_mask], global_func_map)
            func_embeds = F.elu(self.func_agg_linear(func_embeds))
            h_hier = global_mean_pool(func_embeds, func_to_graph_batch, size=data.num_graphs)
        else:
            h_hier = torch.zeros(data.num_graphs, self.func_agg_linear.out_features).to(x.device)

        # 路径B: 扁平化路径 (Flat Path)
        # ------------------------------------
        # 直接将所有块的向量聚合为图向量
        h_flat = global_mean_pool(block_embeddings, node_to_graph_batch)

        # --- 层次三：门控聚合 (Gated Aggregation) ---
        h_combined = torch.cat([h_hier, h_flat], dim=1)

        # 2. 计算门控值gate
        gate = self.gating_network(h_combined)

        # 3. 使用门控值进行加权融合
        # gate * h_hier: “宏观功能结构”应该保留多少信息
        # (1 - gate) * h_flat: “微观代码全貌”应该保留多少信息
        h_aggregated = gate * h_hier + (1 - gate) * h_flat

        # --- 最终输出层 ---
        final_embedding = self.final_linear(h_aggregated)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding

class FlatGNN(nn.Module):
    """
    用于消融实验的“扁平化”GNN模型。
    它不使用分层池化，直接将所有节点聚合为图向量。
    """
    def __init__(self, node_feature_dim, gnn_hidden_dim, output_dim, num_gat_layers=3, heads=4):
        super(FlatGNN, self).__init__()

        self.convs = nn.ModuleList()
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            self.convs.append(GATConv(current_dim, gnn_hidden_dim, heads=heads))
            current_dim = gnn_hidden_dim * heads

        self.final_linear = nn.Linear(current_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GAT卷积层，与分层模型完全一样
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        # x shape: [总节点数, gnn_hidden_dim * heads]

        graph_embedding = global_mean_pool(x, batch)
        # graph_embedding 的形状是正确的 [batch_size, gnn_hidden_dim * heads]

        # 3. 最终线性层和归一化
        final_embedding = self.final_linear(graph_embedding)
        final_embedding = F.normalize(final_embedding, p=2, dim=-1)

        return final_embedding
