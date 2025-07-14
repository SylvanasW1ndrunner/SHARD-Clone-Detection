import pandas as pd
from evm_cfg_builder.cfg import CFG
import json
import os
from typing import Dict, List, Set

from sklearn.model_selection import train_test_split


class ContractCFGGenerator:
    def __init__(self, bytecode_path: str):
        with open(bytecode_path, 'r') as f:
            self.bytecode = f.read().strip()
        self.cfg = CFG(self.bytecode)

        # 提取合约名称（从文件路径）
        self.contract_name = os.path.basename(bytecode_path).replace('.txt', '')

    def generate_files(self, output_dir: str = "output"):
        """生成DOT和JSON文件"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建两个子目录
        cfg_dir = os.path.join(output_dir, "cfg")
        function_dir = os.path.join(output_dir, "function")

        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        if not os.path.exists(function_dir):
            os.makedirs(function_dir)

        # 生成DOT文件（整个合约的CFG）存在cfg目录下
        dot_file = os.path.join(cfg_dir, f"{self.contract_name}_cfg.dot")
        self.generate_dot_file(dot_file)

        # 生成JSON文件（函数映射）存在function目录下
        json_file = os.path.join(function_dir, f"{self.contract_name}_function_mapping.json")
        self.generate_json_file(json_file)

        print(f"生成文件:")
        print(f"  DOT文件: {dot_file}")
        print(f"  JSON文件: {json_file}")
    def _get_block_functions(self, bb):
        """获取包含该基本块的所有函数"""
        belonging_functions = []
        for function in self.cfg.functions:
            if bb in function.basic_blocks:
                belonging_functions.append(function)
        return belonging_functions
    def generate_dot_file(self, output_file: str):
        """生成整个合约的CFG DOT文件"""
        dot_lines = []
        dot_lines.append(f'digraph "{self.contract_name}_CFG" {{')
        dot_lines.append('    rankdir=TB;')
        dot_lines.append('    node [shape=box, style=filled];')
        dot_lines.append('    edge [fontsize=10];')
        dot_lines.append('')

        # 添加合约信息
        dot_lines.append(f'    label="{self.contract_name} Control Flow Graph";')
        dot_lines.append('    labelloc=t;')
        dot_lines.append('    fontsize=16;')
        dot_lines.append('')

        # 收集所有基本块（去重）
        all_basic_blocks = set()
        for function in self.cfg.functions:
            all_basic_blocks.update(function.basic_blocks)

        # 为每个基本块创建节点，使用简单的块号作为节点名
        block_id_map = {}  # 映射：基本块 -> 块号
# 为每个基本块创建节点
        for i, bb in enumerate(sorted(all_basic_blocks, key=lambda x: x.start.pc)):
            block_id = f"block_{i}"
            block_id_map[bb] = block_id
            # 1. label：块号
            label = f"Block {i}"

            instructions_with_operands = []
            for ins in bb.instructions:
                if hasattr(ins, 'operand') and ins.operand is not None:
                    # 将操作数转换为16进制格式
                    if isinstance(ins.operand, int):
                        operand_hex = f"0x{ins.operand:x}"
                    else:
                        operand_hex = str(ins.operand)
                    instructions_with_operands.append(f"{ins.name} {operand_hex}")
                else:
                    instructions_with_operands.append(ins.name)

            instr_text = "\\n".join(instructions_with_operands) if instructions_with_operands else "No Instructions"

            belonging_functions = self._get_block_functions(bb)
            if belonging_functions:
                function_names = []
                for func in belonging_functions:
                    # 处理函数选择器格式
                    if func.hash_id:
                        if isinstance(func.hash_id, int):
                            selector = f"0x{func.hash_id:x}"
                        elif isinstance(func.hash_id, str):
                            selector = func.hash_id if func.hash_id.startswith('0x') else f"0x{func.hash_id}"
                        else:
                            selector = str(func.hash_id)
                        function_names.append(f"{func.name}({selector})")
                    else:
                        function_names.append(func.name)
                function_text = "\\n".join(function_names)
            else:
                function_text = "No Function"

            # 确定节点颜色
            node_color = self._get_node_color(bb)

            dot_lines.append(f'    {block_id} [')
            dot_lines.append(f'        label="Label: {label}\\nInstr: {instr_text}\\nFunction: {function_text}",')
            dot_lines.append(f'        fillcolor="{node_color}",')
            dot_lines.append('        fontsize=9,')
            dot_lines.append('        shape=box')
            dot_lines.append('    ];')

        dot_lines.append('')

        # 添加控制流边
        added_edges = set()  # 避免重复边

        for function in self.cfg.functions:
            for bb in function.basic_blocks:
                if bb not in block_id_map:
                    continue

                source_id = block_id_map[bb]

                for outgoing_bb in bb.outgoing_basic_blocks(function.key):
                    if outgoing_bb not in block_id_map:
                        continue

                    target_id = block_id_map[outgoing_bb]
                    edge_key = (source_id, target_id)

                    if edge_key not in added_edges:
                        added_edges.add(edge_key)

                        # 判断边的类型
                        edge_label = self._get_edge_label(bb)
                        edge_style = self._get_edge_style(bb)

                        if edge_label:
                            dot_lines.append(f'    {source_id} -> {target_id} [label="{edge_label}", {edge_style}];')
                        else:
                            dot_lines.append(f'    {source_id} -> {target_id} [{edge_style}];')

        dot_lines.append('}')

        with open(output_file, 'w') as f:
            f.write('\n'.join(dot_lines))

    def generate_json_file(self, output_file: str):
        """生成函数-基本块映射的JSON文件"""

        # 收集所有基本块并分配块号
        all_basic_blocks = set()
        for function in self.cfg.functions:
            all_basic_blocks.update(function.basic_blocks)

        # 创建基本块到块号的映射
        block_id_map = {}
        block_info_map = {}

        for i, bb in enumerate(sorted(all_basic_blocks, key=lambda x: x.start.pc)):
            block_id = f"block_{i}"
            block_id_map[bb] = block_id

            # 存储基本块的详细信息
            block_info_map[block_id] = {
                "block_id": block_id,
                "block_number": i,
                "start_pc": f"0x{bb.start.pc:x}",
                "end_pc": f"0x{bb.end.pc:x}",
                "start_pc_decimal": bb.start.pc,
                "end_pc_decimal": bb.end.pc,
                "instructions": [
                    {
                        "name": ins.name,
                        "pc": ins.pc,
                        "operand": ins.operand if hasattr(ins, 'operand') else None
                    } for ins in bb.instructions
                ],
                "instruction_count": len(bb.instructions)
            }

        # 构建函数成员关系映射
        function_membership = {}

        for function in sorted(self.cfg.functions, key=lambda x: x.start_addr):
            # 确保函数选择器是16进制格式
            # 确保函数选择器是16进制格式
            function_selector = function.hash_id
            if function_selector:
                if isinstance(function_selector, int):
                    function_selector = f"0x{function_selector:x}"
                elif isinstance(function_selector, str) and not function_selector.startswith('0x'):
                    function_selector = f"0x{function_selector}"

            function_key = function.name if function.name else f"func_0x{function.start_addr:x}"

            # 收集该函数包含的基本块
            function_blocks = []
            function_block_ids = []

            for bb in sorted(function.basic_blocks, key=lambda x: x.start.pc):
                if bb in block_id_map:
                    block_id = block_id_map[bb]
                    function_block_ids.append(block_id)

                    # 添加该块在此函数中的连接信息
                    block_info = block_info_map[block_id].copy()
                    block_info["incoming_blocks"] = [
                        block_id_map[inc_bb]
                        for inc_bb in bb.incoming_basic_blocks(function.key)
                        if inc_bb in block_id_map
                    ]
                    block_info["outgoing_blocks"] = [
                        block_id_map[out_bb]
                        for out_bb in bb.outgoing_basic_blocks(function.key)
                        if out_bb in block_id_map
                    ]
                    function_blocks.append(block_info)

            # 函数信息
# 函数信息
            function_info = {
                "function_name": function.name,
                "function_selector": function_selector,
                "start_address": f"0x{function.start_addr:x}",
                "start_address_decimal": function.start_addr,
                "attributes": list(function.attributes),
                "basic_blocks_count": len(function.basic_blocks),
                "basic_block_ids": function_block_ids,
                "basic_blocks_detail": function_blocks
            }

            function_membership[function_key] = function_info

        # 构建完整的JSON结构
        contract_cfg_data = {
            "contract_name": self.contract_name,
            "total_functions": len(self.cfg.functions),
            "total_basic_blocks": len(all_basic_blocks),
            "all_basic_blocks": block_info_map,  # 所有基本块的信息
            "function_membership": function_membership,
            "metadata": {
                "generated_by": "evm_cfg_builder",
                "description": "Function-to-BasicBlock mapping for smart contract CFG analysis",
                "note": "basic_block_ids shows the simple mapping F = {func: [block_0, block_1, ...]}"
            }
        }

        with open(output_file, 'w') as f:
            json.dump(contract_cfg_data, f, indent=2, ensure_ascii=False)

    def _get_node_color(self, bb) -> str:
        """根据基本块的指令类型确定节点颜色"""
        instructions = [ins.name for ins in bb.instructions]

        if any(ins in ['RETURN', 'REVERT', 'STOP', 'SELFDESTRUCT'] for ins in instructions):
            return 'lightcoral'  # 终止块
        elif any(ins == 'JUMPDEST' for ins in instructions):
            return 'lightblue'   # 跳转目标
        elif any(ins in ['CALL', 'DELEGATECALL', 'STATICCALL', 'CALLCODE'] for ins in instructions):
            return 'lightyellow' # 外部调用
        elif any(ins in ['SSTORE', 'SLOAD'] for ins in instructions):
            return 'lightgreen'  # 存储操作
        else:
            return 'lightgray'   # 普通块

    def _get_edge_label(self, bb) -> str:
        """获取边的标签"""
        if not bb.instructions:
            return ""

        last_instruction = bb.instructions[-1].name
        if last_instruction == 'JUMPI':
            return "conditional"
        elif last_instruction == 'JUMP':
            return "jump"
        elif last_instruction in ['RETURN', 'REVERT', 'STOP']:
            return "exit"
        else:
            return ""

    def _get_edge_style(self, bb) -> str:
        """获取边的样式"""
        if not bb.instructions:
            return 'color=black'

        last_instruction = bb.instructions[-1].name
        if last_instruction == 'JUMPI':
            return 'color=blue, style=dashed'
        elif last_instruction == 'JUMP':
            return 'color=red'
        elif last_instruction in ['RETURN', 'REVERT', 'STOP']:
            return 'color=gray, style=dotted'
        else:
            return 'color=black'

# 使用示例
def process_contract(bytecode_path: str, output_dir: str = "cfg_output"):
    """处理单个合约"""
    generator = ContractCFGGenerator(bytecode_path)
    generator.generate_files(output_dir)

def batch_process_contracts(contracts_dir: str, output_dir: str = "cfg_output"):
    """批量处理合约"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(contracts_dir):
        if filename.endswith('.txt'):
            contract_path = os.path.join(contracts_dir, filename)
            print(f"\n处理合约: {filename}")
            try:
                process_contract(contract_path, output_dir)
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

def data_process(csv_path, type1_csv, type2_csv, type3_csv):
    df = pd.read_csv(csv_path)

    train_list, val_list, test_list = [], [], []

    # ========== Type 0 ==========
    type0 = df[df['type'] == 0]
    type0_train, type0_remain = train_test_split(type0, test_size=0.2, random_state=42)
    type0_val, type0_test = train_test_split(type0_remain, test_size=0.75, random_state=42)  # 剩余的 20% -> 5% val, 15% test

    # ========== Type 4 ==========
    type4 = df[df['type'] == 4]
    type4_train, type4_remain = train_test_split(type4, test_size=0.2, random_state=42)
    type4_val, type4_test = train_test_split(type4_remain, test_size=0.75, random_state=42)

    # ========== Type 1 / 2 / 3 ==========
    for t in [1, 2, 3]:
        type_t = df[df['type'] == t]
        type_t_val, type_t_test = train_test_split(type_t, test_size=0.8, random_state=42)  # 20% val, 80% test
        val_list.append(type_t_val)
        test_list.append(type_t_test)

    # 组装
    train_list.extend([type0_train, type4_train])
    val_list.extend([type0_val, type4_val])
    test_list.extend([type0_test, type4_test])

    train_set = pd.concat(train_list).reset_index(drop=True)
    val_set = pd.concat(val_list).reset_index(drop=True)
    test_set = pd.concat(test_list).reset_index(drop=True)

    # 保存数据
    train_set.to_csv("train.csv", index=False)
    val_set.to_csv("val.csv", index=False)
    test_set.to_csv("test.csv", index=False)


    return train_set, val_set, test_set


def encodercsv_extract(csv_path):
    df = pd.read_csv(csv_path)


    type0 = df[df['type'] == 0.0]
    type4 = df[df['type'] == 4.0]
    print(len(type0), len(type4))

    df = pd.concat([type0, type4]).reset_index(drop=True)
    df.to_csv("encoder_test.csv", index=False)


encodercsv_extract("test.csv")


    # 处理单个合约
    #bytecode_path = "../0_dataset/bytecode_dataset_I/bytecode_dataset_I/contract_bytecode/0x0a0af8a0604ba0c40d81e6b766a0f44aa6616431.txt"
    #batch_process_contracts(contracts_dir=r"C:\Users\zkjg\Downloads\13756064\clone_detection_replication_package\0_dataset\bytecode_dataset_II\bytecode_dataset_II\train_dataset")
##data_process(csv_path="replication.csv", type1_csv="../suppmaterial-18-masanari-smart_contract_cloning-1.0/clone-pairs-per-type/only-type1_clone.csv", type2_csv="../suppmaterial-18-masanari-smart_contract_cloning-1.0/clone-pairs-per-type/only-type2_clone.csv", type3_csv="../suppmaterial-18-masanari-smart_contract_cloning-1.0/clone-pairs-per-type/only-type3_clone.csv")
# csv = pd.read_csv("replication.csv")
# contract_set = set(csv['contract_id'].values.tolist())
# contract_set.update(set(csv['clone_contract_id'].values.tolist()))
# print("set len: ", len(contract_set))
# bytecodefile = "../0_dataset/bytecode_dataset_II/bytecode_dataset_II/replication_dataset"
# for filename in contract_set:
#     try:
#         filepath = os.path.join(bytecodefile, filename + ".txt")
#         process_contract(filepath, output_dir="GNNdata")
#         print("process: ", filename)
#     except Exception as e:
#         print("error in: ", filename)
#         print(e.with_traceback(e.__traceback__))




