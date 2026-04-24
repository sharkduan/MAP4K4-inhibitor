#!/usr/bin/env python3
"""
分子性质计算脚本
从CSV文件中读取SMILES分子，计算各种性质并输出结果
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, Lipinski
from rdkit.Contrib.SA_Score import sascorer
import collections
import math
from scipy import stats
import argparse


def calculate_sa_score(mol):
    """计算SA (Synthetic Accessibility) 分数"""
    try:
        return sascorer.calculateScore(mol)
    except:
        return None


def calculate_logp(mol):
    """计算logP"""
    try:
        return Descriptors.MolLogP(mol)
    except:
        return None


def calculate_qed(mol):
    """计算QED (Quantitative Estimate of Drug-likeness)"""
    try:
        return QED.qed(mol)
    except:
        return None


def calculate_lipinski(mol):
    """计算Lipinski五规则"""
    try:
        properties = {
            'molecular_weight': Descriptors.ExactMolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'h_bond_donors': Lipinski.NumHDonors(mol),
            'h_bond_acceptors': Lipinski.NumHAcceptors(mol),
        }

        # 检查是否违反Lipinski规则
        violations = 0
        if properties['molecular_weight'] > 500:
            violations += 1
        if properties['logp'] > 5:
            violations += 1
        if properties['h_bond_donors'] > 5:
            violations += 1
        if properties['h_bond_acceptors'] > 10:
            violations += 1

        properties['lipinski_violations'] = violations
        properties['pass_lipinski'] = violations <= 1

        return properties
    except:
        return None


def calculate_ring_statistics(mol):
    """计算环统计信息（修复版）"""
    try:
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        # 计算不同大小的环的数量
        ring_sizes = [len(ring) for ring in atom_rings]
        ring_count = collections.Counter(ring_sizes)

        # 计算n元环出现数量（整数）和概率
        total_rings = len(atom_rings)
        ring_probabilities = {}

        for size in range(3, 12):  # 3-11元环
            count = ring_count.get(size, 0)
            ring_probabilities[f'ring_{size}_count'] = count        # 整数
            ring_probabilities[f'ring_{size}_prob'] = count / total_rings if total_rings > 0 else 0.0

        ring_probabilities['total_rings'] = total_rings

        return ring_probabilities
    except:
        return None


def calculate_bond_angle_distribution(mol):
    """计算键角分布概率"""
    try:
        from rdkit.Chem import AllChem
        # 获取3D构象
        mol_with_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDG())

        conf = mol_with_h.GetConformer()

        angles = []
        for atom in range(mol_with_h.GetNumAtoms()):
            neighbors = [n.GetIdx() for n in mol_with_h.GetAtomWithIdx(atom).GetNeighbors()]
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        a1, a2, a3 = neighbors[i], atom, neighbors[j]
                        try:
                            angle = rdMolDescriptors.GetAngleDeg(conf, a1, a2, a3)
                            angles.append(angle)
                        except:
                            continue

        if len(angles) == 0:
            return None

        # 统计键角分布
        angles = np.array(angles)
        angle_stats = {
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'min_angle': np.min(angles),
            'max_angle': np.max(angles),
            'median_angle': np.median(angles),
        }

        # 计算不同角度范围的概率
        angle_ranges = [(0, 60), (60, 90), (90, 120), (120, 150), (150, 180)]
        for start, end in angle_ranges:
            count = np.sum((angles >= start) & (angles < end))
            angle_stats[f'angle_{start}_{end}_prob'] = count / len(angles)

        return angle_stats
    except:
        return None


def process_molecule(smiles):
    """处理单个分子"""
    result = {'smiles': smiles}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result['error'] = 'Invalid SMILES'
        return result

    # 计算基本性质
    result['sa_score'] = calculate_sa_score(mol)
    result['logp'] = calculate_logp(mol)
    result['qed'] = calculate_qed(mol)

    # 新增：重原子数
    result['heavy_atom_count'] = Lipinski.HeavyAtomCount(mol)

    # 计算Lipinski性质
    lipinski_props = calculate_lipinski(mol)
    if lipinski_props:
        result.update(lipinski_props)

    # 计算环统计（修复版）
    ring_stats = calculate_ring_statistics(mol)
    if ring_stats:
        result.update(ring_stats)

    # 计算键角分布
    angle_stats = calculate_bond_angle_distribution(mol)
    if angle_stats:
        result.update(angle_stats)

    return result


def main():
    parser = argparse.ArgumentParser(description='计算分子性质')
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件')
    parser.add_argument('-o', '--output', required=True, help='输出CSV文件')
    parser.add_argument('-c', '--column', default='smiles', help='SMILES列名')
    parser.add_argument('-n', '--nrows', type=int, help='处理的行数')

    args = parser.parse_args()

    # 读取输入文件
    print(f"读取输入文件: {args.input}")
    df = pd.read_csv(args.input)

    if args.nrows:
        df = df.head(args.nrows)

    if args.column not in df.columns:
        print(f"错误: 列 '{args.column}' 不存在于输入文件中")
        return

    # 处理每个分子
    results = []
    total = len(df)

    for idx, smiles in enumerate(df[args.column]):
        if pd.isna(smiles):
            continue

        print(f"处理分子 {idx + 1}/{total}: {smiles[:50]}...")

        result = process_molecule(str(smiles).strip())
        results.append(result)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # ---------------- 新增：全局环统计 ----------------
    # 收集所有环计数
    all_ring_cnt = pd.DataFrame([r for r in results if 'ring_3_count' in r])
    if not all_ring_cnt.empty:
        # 3-11 元环的 count 列
        count_cols = [f'ring_{s}_count' for s in range(3, 12)]
        # 按列求和 → 总个数
        global_counts = all_ring_cnt[count_cols].sum(axis=0)
        total_rings_all = global_counts.sum()
        # 占比
        global_probs = global_counts / total_rings_all if total_rings_all else 0.0

        # 拼两行
        global_row_cnt = {'smiles': 'GLOBAL_COUNTS'}
        global_row_cnt.update(global_counts.to_dict())
        global_row_prb = {'smiles': 'GLOBAL_PROBS'}
        global_row_prb.update(global_probs.to_dict())

        # 追加到结果末尾
        results.append(global_row_cnt)
        results.append(global_row_prb)
    # ----------------------------------------------

    # 保存结果
    print(f"保存结果到: {args.output}")
    result_df.to_csv(args.output, index=False)

    # 打印统计信息
    print("\n处理完成!")
    print(f"总共处理了 {len(result_df)} 个分子")
    if 'error' in result_df.columns:
        error_count = result_df['error'].notna().sum()
        print(f"其中 {error_count} 个分子有错误")


if __name__ == '__main__':
    main()