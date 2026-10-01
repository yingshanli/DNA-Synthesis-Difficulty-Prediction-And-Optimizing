# APP_Paper_PUBLICATION.py
# Publication-aligned Streamlit application for:
# "An interpretable machine learning framework for predicting and optimizing DNA synthesis difficulty"
#
# This version is aligned with the manuscript feature definitions and the final
# xgboost_best_model.pkl file. Predicted synthesis duration is a model estimate
# and should not be interpreted as experimentally confirmed synthesis performance.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import RNA
import os
from collections import Counter
from typing import List, Union, Optional, Tuple, Dict
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib
# ===== 解决中文显示问题 =====
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 定义规范特征列表
# ============================================================
from itertools import product

CANONICAL_FEATURES = (
    ['log_length'] +
    ['A_ratio', 'G_ratio', 'C_ratio', 'T_ratio'] +
    [''.join(p) for p in product('AGCT', repeat=2)] +
    [''.join(p).lower() for p in product('ACGT', repeat=3)] +
    ['max_repeat_len', 'repeat_density'] +
    ['max_stem_length', 'avg_stem_length', 'total_loops', 'max_loop_size',
     'GT_pairs', 'pairing_rate', 'stem_count', 'MFE']
)

# ============================================================
# 1. 加载模型
# ============================================================
@st.cache_resource
def load_model():
    try:
        obj = joblib.load('xgboost_best_model.pkl')
        print("✅ Loaded xgboost_best_model.pkl")
    except FileNotFoundError:
        st.error("❌ 模型文件 'xgboost_best_model.pkl' 未找到")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ 加载模型失败: {e}")
        return None, None, None, None
    
    if isinstance(obj, dict):
        pipeline = obj.get('pipeline')
        full_features = obj.get('feature_columns')
        log_transform_used = obj.get('log_transform_used', False)
        regularized_used = obj.get('regularized', True)
    else:
        pipeline = obj
        full_features = None
        log_transform_used = False
        regularized_used = True
    
    if full_features is None and hasattr(pipeline, 'steps'):
        for step_name, step in pipeline.steps:
            if hasattr(step, 'feature_names_in_'):
                full_features = list(step.feature_names_in_)
                break

    if full_features is None and hasattr(pipeline, 'feature_names_in_'):
        full_features = list(pipeline.feature_names_in_)
    
    if full_features is None:
        full_features = list(CANONICAL_FEATURES)
    
    return pipeline, full_features, log_transform_used, regularized_used

# ============================================================
# 2. 特征计算函数
# ============================================================
def calculate_all_features(seq: str) -> dict:
    """计算单条Sequence的所有特征"""
    
    seq = seq.upper().strip()
    seq = ''.join(seq.split())
    
    if not seq:
        return {f: 0.0 for f in CANONICAL_FEATURES}

    length = len(seq)
    features = {}

    # ---- 1. 基础特征 ----
    # The training workflow used ln(sequence length); all valid DNA sequences have length > 0.
    features['log_length'] = float(np.log(length))

    a = seq.count('A')
    c = seq.count('C')
    g = seq.count('G')
    t = seq.count('T')
    features['A_ratio'] = a / length if length > 0 else 0.0
    features['G_ratio'] = g / length if length > 0 else 0.0
    features['C_ratio'] = c / length if length > 0 else 0.0
    features['T_ratio'] = t / length if length > 0 else 0.0

    # ---- 2. 二核苷酸 ----
    dinucs = ['AA','AG','AC','AT','GA','GG','GC','GT','CA','CG','CC','CT','TA','TG','TC','TT']
    dinuc_count = Counter(seq[i:i+2] for i in range(length-1))
    total_dinuc = max(length-1, 1)
    for d in dinucs:
        features[d] = float(dinuc_count.get(d, 0) / total_dinuc)

    # ---- 3. 三核苷酸（模型中的规范名称为小写） ----
    bases = ['A','C','G','T']
    trinucs = [a+b+c for a in bases for b in bases for c in bases]
    trinuc_count = Counter(seq[i:i+3] for i in range(length-2))
    total_trinuc = max(length-2, 1)
    for tri in trinucs:
        features[tri.lower()] = float(trinuc_count.get(tri, 0) / total_trinuc)

    # ---- 4. 重复Sequence ----
    def find_repeats_optimized(seq, min_len=4):
        repeats = {}
        n = len(seq)
        for l in range(min_len, n // 2):
            substrings = (seq[i:i+l] for i in range(n - l + 1))
            counts = Counter(substrings)
            for k, v in counts.items():
                if v > 1:
                    repeats[k] = v
        max_rep = max((len(k) for k in repeats.keys()), default=0)
        dens = len(repeats) / n if n > 0 else 0.0
        return max_rep, dens

    max_repeat, repeat_density = find_repeats_optimized(seq, min_len=4)
    features['max_repeat_len'] = float(max_repeat)
    features['repeat_density'] = float(repeat_density)
    
    # ---- 5. 二级结构计算（ViennaRNA Python API） ----
    # Publication settings:
    #   - DNA-specific Mathews 2004 parameters
    #   - 37 °C
    #   - DNA sequence retained with T (no T→U conversion)
    structure = ""
    mfe = 0.0

    try:
        # Load the built-in DNA Mathews 2004 parameter set before creating
        # the model details object. ViennaRNA documents that this also
        # switches the default geometric parameters to DNA.
        RNA.params_load_DNA_Mathews2004()
        md = RNA.md()
        md.temperature = 37.0

        fc = RNA.fold_compound(seq, md)
        structure, mfe = fc.mfe()

        structure = str(structure)
        mfe = float(mfe)
    except Exception as e:
        raise RuntimeError(
            "ViennaRNA DNA secondary-structure calculation failed. "
            f"Details: {e}"
        ) from e

    features['MFE'] = float(mfe)

    # ---- 6. 衍生特征 ----
    if structure and len(structure) == length:
        stack = []
        pairs = []
        seq_len = len(structure)

        for idx, ch in enumerate(structure.strip()):
            if ch == '(':
                stack.append(idx)
            elif ch == ')':
                if stack:
                    start = stack.pop()
                    pairs.append((start, idx))

        stems = []
        current_stem = []
        for pair in sorted(pairs):
            if not current_stem:
                current_stem.append(pair)
            else:
                last_pair = current_stem[-1]
                if (pair[0] - last_pair[0] == 1) and (last_pair[1] - pair[1] == 1):
                    current_stem.append(pair)
                else:
                    stems.append(current_stem)
                    current_stem = [pair]
        if current_stem:
            stems.append(current_stem)

        stem_count = len(stems)
        stem_lengths = [len(stem) for stem in stems]

        loop_regions = []
        prev_end = -1
        for stem in stems:
            start = stem[0][0]
            if start > prev_end + 1:
                loop_regions.append((prev_end + 1, start - 1))
            prev_end = stem[-1][1]
        if prev_end < seq_len - 1:
            loop_regions.append((prev_end + 1, seq_len - 1))

        loop_sizes = [end - start + 1 for (start, end) in loop_regions]

        gt_count = 0
        for (i, j) in pairs:
            if i < len(seq) and j < len(seq):
                pair_bases = sorted([seq[i].upper(), seq[j].upper()])
                if pair_bases == ['G', 'T'] or pair_bases == ['G', 'U']:
                    gt_count += 1

        pairing_rate = len(pairs) * 2 / seq_len if seq_len else 0

        features['stem_count'] = float(stem_count)
        features['max_stem_length'] = float(max(stem_lengths)) if stem_lengths else 0.0
        features['avg_stem_length'] = float(np.mean(stem_lengths)) if stem_lengths else 0.0
        features['total_loops'] = float(len(loop_sizes))
        features['max_loop_size'] = float(max(loop_sizes)) if loop_sizes else 0.0
        features['GT_pairs'] = float(gt_count)
        features['pairing_rate'] = float(pairing_rate)
    else:
        features['stem_count'] = 0.0
        features['max_stem_length'] = 0.0
        features['avg_stem_length'] = 0.0
        features['total_loops'] = 0.0
        features['max_loop_size'] = 0.0
        features['GT_pairs'] = 0.0
        features['pairing_rate'] = 0.0

    # ---- 7. 确保所有特征都存在 ----
    for key in CANONICAL_FEATURES:
        if key not in features:
            features[key] = 0.0
        elif features[key] is None:
            features[key] = 0.0
        elif isinstance(features[key], (int, float)):
            features[key] = float(features[key])
        else:
            features[key] = 0.0

    return features


# ============================================================
# 3. 预测函数
# ============================================================
def predict_sequence(seq: str, pipeline, full_features, log_transform_used) -> float:
    """预测Sequence的合成周期"""
    all_feat = calculate_all_features(seq)
    df = pd.DataFrame([all_feat])
    df = df[full_features]
    
    pred_log = pipeline.predict(df)[0]
    
    # ✅ 如果模型使用了 log 变换, 需要转换回来
    if log_transform_used:
        pred = np.expm1(pred_log)
    else:
        pred = pred_log
    
    return max(1.0, min(200.0, pred))


# ============================================================
# 4. 显示特征表格的函数
# ============================================================
def show_feature_table(seq: str, full_features: List[str]):
    """显示计算的特征表格"""
    all_feat = calculate_all_features(seq)
    df = pd.DataFrame([all_feat])
    
    # 只显示模型需要的特征
    df = df[full_features]
    
    # 选择要显示的特征类别
    st.subheader("📊 Calculated features")
    
    # 特征分组
    base_features = ['log_length', 'A_ratio', 'G_ratio', 'C_ratio', 'T_ratio']
    dinuc_features = ['AA','AG','AC','AT','GA','GG','GC','GT','CA','CG','CC','CT','TA','TG','TC','TT']
    trinuc_features = [''.join(p).lower() for p in product('ACGT', repeat=3)]
    repeat_features = ['max_repeat_len', 'repeat_density']
    secondary_features = ['max_stem_length', 'avg_stem_length', 'total_loops', 'max_loop_size', 'GT_pairs', 'pairing_rate', 'stem_count', 'MFE']
    
    # 显示基础特征
    with st.expander("📈 Base features", expanded=True):
        base_df = df[base_features]
        st.dataframe(base_df.style.format("{:.4f}"))
    
    # 显示二核苷酸特征
    with st.expander("🧬 Dinucleotide features"):
        dinuc_df = df[dinuc_features]
        st.dataframe(dinuc_df.style.format("{:.4f}"))
    
    # 显示三核苷酸特征
    with st.expander("🧬 Trinucleotide features"):
        # 只显示前20个三核苷酸特征
        trinuc_display = df[trinuc_features[:20]]
        st.dataframe(trinuc_display.style.format("{:.4f}"))
        if len(trinuc_features) > 20:
            st.caption(f"... 还有 {len(trinuc_features) - 20} 个三核苷酸特征")
    
    # 显示重复Sequence特征
    with st.expander("🔄 Repeat features"):
        repeat_df = df[repeat_features]
        st.dataframe(repeat_df.style.format("{:.4f}"))
    
    # 显示二级结构特征
    with st.expander("🔬 Secondary-structure features"):
        sec_df = df[secondary_features]
        st.dataframe(sec_df.style.format("{:.4f}"))
    
    # 全部特征汇总
    with st.expander("📋 All features"):
        st.dataframe(df.style.format("{:.4f}"))
    
    return df


# ============================================================
# 5. OPTIMIZATION FUNCTIONS
# ============================================================
codon_table = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '*': ['TAA', 'TAG', 'TGA'],
}

valid_codons = set()
for codons in codon_table.values():
    for codon in codons:
        valid_codons.add(codon)

codon_to_aa = {}
for aa, codons in codon_table.items():
    for codon in codons:
        codon_to_aa[codon] = aa


def is_stop_codon(codon: str) -> bool:
    return codon in ['TAA', 'TAG', 'TGA']


def dna_to_protein(dna: str) -> Tuple[str, List[Tuple[int, str]]]:
    dna = dna.upper()
    protein = ''
    stop_codons = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        if codon in codon_to_aa:
            aa = codon_to_aa[codon]
            protein += aa
            if aa == '*':
                stop_codons.append((i, codon))
        else:
            protein += '?'
            stop_codons.append((i, codon))
    return protein, stop_codons


def validate_dna(dna: str) -> Tuple[str, List[Tuple[int, str]]]:
    dna = dna.upper().strip().replace(' ', '').replace('\n', '')
    if not dna:
        raise ValueError("DNA sequence is empty")
    illegal_chars = set(dna) - {'A', 'T', 'C', 'G'}
    if illegal_chars:
        raise ValueError(f"Unsupported characters: {illegal_chars}")
    if len(dna) % 3 != 0:
        raise ValueError(f"长度 {len(dna)} is not divisible by 3")
    unknown = []
    for i in range(0, len(dna)-2, 3):
        codon = dna[i:i+3]
        if codon not in valid_codons:
            unknown.append((i, codon))
    return dna, unknown


def count_inverted_repeats(seq: str, min_len: int = 5, max_len: int = 15, min_gap: int = 4, max_gap: int = 5) -> int:
    length = len(seq)
    if length < 2 * min_len + min_gap:
        return 0

    complement = str.maketrans('ATCG', 'TAGC')
    stem_count = 0

    for stem_len in range(min_len, min(max_len, length // 2) + 1):
        for i in range(0, length - 2 * stem_len - min_gap + 1):
            forward = seq[i:i + stem_len]
            reverse_comp = forward.translate(complement)[::-1]
            for gap in range(min_gap, max_gap + 1):
                start = i + stem_len + gap
                end = start + stem_len
                if end > length:
                    continue
                if seq[start:end] == reverse_comp:
                    stem_count += 1
    return stem_count


def fast_score(seq: str) -> float:
    """Publication loss: 100*|GC-0.5| + 2.5*N_SIR."""
    length = len(seq)
    if length == 0:
        return 1000.0
    gc_ratio = get_gc_ratio(seq)
    n_sir = count_inverted_repeats(seq, min_len=5, max_len=15, min_gap=4, max_gap=5)
    return 100.0 * abs(gc_ratio - 0.5) + 2.5 * n_sir


def get_gc_ratio(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    return (seq.count('G') + seq.count('C')) / len(seq)


def optimize_single_sequence(
    input_dna: str,
    max_iterations: int = 5,
    progress_callback=None
) -> Tuple[Optional[str], Dict]:
    try:
        input_dna, unknown = validate_dna(input_dna)
    except ValueError as e:
        return None, {'error': str(e)}
    
    stop_positions = []
    non_stop = []
    for pos, codon in unknown:
        if is_stop_codon(codon):
            stop_positions.append((pos, codon))
        else:
            non_stop.append((pos, codon))
    if non_stop:
        return None, {'error': f"Non-standard codons: {non_stop[:5]}"}
    
    protein_seq, stop_codons = dna_to_protein(input_dna)
    stop_set = set(pos for pos, _ in stop_codons)
    
    initial_gc = get_gc_ratio(input_dna)
    initial_stems = count_inverted_repeats(input_dna)
    initial_fast_score = fast_score(input_dna)
    
    current_dna = input_dna
    current_score = initial_fast_score
    
    codon_options = []
    for i, aa in enumerate(protein_seq):
        start = i * 3
        if start in stop_set or aa == '*':
            codon_options.append([])
            continue
        current_codon = current_dna[start:start+3]
        options = [c for c in codon_table.get(aa, []) if c != current_codon]
        codon_options.append(options)
    
    for iteration in range(max_iterations):
        changed = False
        improvements = 0
        
        if progress_callback:
            progress_callback(iteration, max_iterations, stage=f"Pass {iteration+1}/{max_iterations}")
        
        for i in range(len(protein_seq)):
            aa = protein_seq[i]
            options = codon_options[i]
            if not options:
                continue
            
            start = i * 3
            end = start + 3
            best_codon = None
            best_score = current_score
            
            for codon in options[:2]:
                new_dna = current_dna[:start] + codon + current_dna[end:]
                new_score = fast_score(new_dna)
                if new_score < best_score:
                    best_score = new_score
                    best_codon = codon
            
            if best_codon is not None and best_score < current_score:
                current_dna = current_dna[:start] + best_codon + current_dna[end:]
                current_score = best_score
                changed = True
                improvements += 1
        
        if not changed:
            break
    
    final_gc = get_gc_ratio(current_dna)
    final_stems = count_inverted_repeats(current_dna)
    final_fast_score = fast_score(current_dna)
    
    final_protein, _ = dna_to_protein(current_dna)
    protein_verified = final_protein == protein_seq
    
    return current_dna, {
        'optimized_sequence': current_dna,
        'initial_gc': initial_gc,
        'final_gc': final_gc,
        'gc_change': final_gc - initial_gc,
        'initial_stems': initial_stems,
        'final_stems': final_stems,
        'stem_change': final_stems - initial_stems,
        'initial_fast_score': initial_fast_score,
        'final_fast_score': final_fast_score,
        'initial_mfe': initial_mfe,
        'final_mfe': final_mfe,
        'mfe_change': final_mfe - initial_mfe,
        'protein_verified': protein_verified,
        'original_protein': protein_seq,
        'optimized_protein': final_protein
    }

def optimize_and_predict_single(
    seq: str, 
    pipeline,                  
    full_features,
    log_transform_used,
    max_iterations: int = 5,
    progress_bar=None,
    status_text=None
) -> Dict:
    orig_pred = predict_sequence(seq, pipeline, full_features, log_transform_used)
    
    if status_text:
        status_text.text("🔄 正在优化Sequence...")
    
    def update_progress(current, total, stage=""):
        if progress_bar:
            progress = 0.3 + (current / total) * 0.6
            progress_bar.progress(progress, text=stage)
    
    optimized, metrics = optimize_single_sequence(
        seq, 
        max_iterations=max_iterations,
        progress_callback=update_progress
    )
    
    if optimized is None:
        return {'error': metrics.get('error', 'Optimization failed')}
    
    if status_text:
        status_text.text("📊 预测优化后Sequence...")
    if progress_bar:
        progress_bar.progress(0.95, text="预测优化后Sequence...")
    
    opt_pred = predict_sequence(optimized, pipeline, full_features, log_transform_used)
    
    if progress_bar:
        progress_bar.progress(1.0, text="✅ Complete!")
    
    return {
        'original_sequence': seq,
        'optimized_sequence': optimized,
        'original_prediction': orig_pred,
        'optimized_prediction': opt_pred,
        'prediction_improvement': orig_pred - opt_pred,
        'metrics': metrics,
        'protein_verified': metrics.get('protein_verified', False)
    }


# ============================================================
# 6. 辅助函数
# ============================================================
def classify_duration(days):
    if days < 30:
        return "🟢 Low"
    elif days < 60:
        return "🟡 Moderate"
    else:
        return "🔴 High"


def read_sequences_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    seqs = []
    
    if file_extension in ['xlsx', 'xls']:
        try:
            df_raw = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), header=0)
            first_col = df_raw.columns[0]
            seqs = df_raw[first_col].astype(str).tolist()
            seqs = [s.strip() for s in seqs if s and s.strip()]
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
        return seqs
    
    raw_data = uploaded_file.getvalue()
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp936']
    content = None
    for enc in encodings:
        try:
            content = raw_data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        st.error("Unable to decode file. Please use UTF-8 or GBK encoding.")
        return []
    
    lines = content.splitlines()
    
    if lines and lines[0].startswith('>'):
        seq = ""
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    seqs.append(seq)
                seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)
        return seqs
    
    if ',' in lines[0] or '\t' in lines[0]:
        try:
            for enc in encodings:
                try:
                    df_raw = pd.read_csv(pd.StringIO(content), header=0, encoding=enc)
                    first_col = df_raw.columns[0]
                    seqs = df_raw[first_col].astype(str).tolist()
                    seqs = [s.strip() for s in seqs if s and s.strip()]
                    return seqs
                except:
                    continue
        except:
            pass
    
    seqs = [line.strip() for line in lines if line.strip()]
    return seqs


# ============================================================
# 7. 可视化函数
# ============================================================
def show_optimization_comparison(result: Dict):
    metrics = result.get('metrics', {})
    
    st.subheader("📊 Before vs. after optimization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    orig_pred = int(round(result['original_prediction']))
    opt_pred = int(round(result['optimized_prediction']))
    improvement = orig_pred - opt_pred
    col1.metric("📅 Predicted duration", f"{orig_pred} → {opt_pred} days", 
                delta=f"-{improvement} 天" if improvement > 0 else f"{improvement} 天")
    
    init_gc = metrics.get('initial_gc', 0) * 100
    final_gc = metrics.get('final_gc', 0) * 100
    gc_delta = final_gc - init_gc
    col2.metric("🧬 GC content", f"{init_gc:.1f}% → {final_gc:.1f}%",
                delta=f"{gc_delta:+.1f}%" if gc_delta != 0 else "✓ 已达标")
    
    init_stems = metrics.get('initial_stems', 0)
    final_stems = metrics.get('final_stems', 0)
    stem_delta = final_stems - init_stems
    col3.metric("🌀 High-risk SIRs", f"{init_stems} → {final_stems}",
                delta=f"{stem_delta:+d}" if stem_delta != 0 else "✓ 已优化")
    
    init_mfe = metrics.get('initial_mfe', 0)
    final_mfe = metrics.get('final_mfe', 0)
    mfe_delta = final_mfe - init_mfe
    if init_mfe != 0 or final_mfe != 0:
        col4.metric("⚡ MFE (kcal/mol)", f"{init_mfe:.2f} → {final_mfe:.2f}",
                    delta=f"{mfe_delta:+.2f}" if mfe_delta != 0 else "")
    else:
        col4.metric("⚡ MFE (kcal/mol)", "N/A")
    
    if init_mfe != 0 or final_mfe != 0:
        st.markdown("---")
        st.subheader("⚡ MFE comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(['Before', 'After'], [init_mfe, final_mfe], 
                      color=['#ff6b6b', '#51cf66'], edgecolor='black', linewidth=1.5)
        ax.set_ylabel('MFE (kcal/mol)', fontsize=12)
        ax.set_title('Minimum free energy (MFE) comparison', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, [init_mfe, final_mfe]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        if mfe_delta > 0:
            ax.annotate(f'↑ Change {mfe_delta:+.2f}', 
                       xy=(1, final_mfe), xytext=(1.3, (init_mfe + final_mfe)/2),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=12, color='green', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    st.subheader("🔬 Protein-sequence verification")
    
    if metrics.get('protein_verified', False):
        st.success("✅ Amino-acid sequence preserved exactly.")
        with st.expander("📝 View protein sequences"):
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Original protein")
                st.code(metrics.get('original_protein', ''), language='text')
            with col2:
                st.caption("Optimized protein")
                st.code(metrics.get('optimized_protein', ''), language='text')
    else:
        st.warning("⚠️ 蛋白质Sequence不一致！请检查输入Sequence是否正确")
    
    st.markdown("---")
    with st.expander("🧬 View full optimized sequence"):
        opt_seq = result.get('optimized_sequence', '')
        st.code(opt_seq, language='text')


# ============================================================
# 8. STREAMLIT 界面
# ============================================================
st.set_page_config(page_title="DNA Synthesis Duration Prediction & Sequence Optimization", page_icon="🧬", layout="centered")

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 22px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 DNA Synthesis Duration Prediction & Sequence Optimization")
st.markdown("Enter a DNA sequence to predict synthesis duration and optionally perform synonymous optimization for protein-coding sequences.")
st.caption("Note: Predictions and optimization results are computational estimates and require experimental validation.")

# 加载模型
pipeline, full_features, log_transform_used, regularized_used = load_model()
if pipeline is None:
    st.stop()

st.success(f"✅ Model loaded successfully! Features: {len(full_features)}")
st.info(f"🔬 Target log transform: {log_transform_used}")

# ===== 侧边栏 =====
st.sidebar.header("⚙️ Settings")
enable_optimization = st.sidebar.checkbox("🧬 Enable sequence optimization", value=True)
if enable_optimization:
    st.sidebar.info("Optimization objectives:\n- GC content → ~50%\n- Reduce predefined high-risk short inverted repeats (SIRs)\n- Preserve the encoded amino-acid sequence")
    max_iter = st.sidebar.slider("Maximum optimization passes", min_value=1, max_value=5, value=5)
    simple_threshold = st.sidebar.slider("Low-difficulty threshold (days)", min_value=10, max_value=60, value=30, 
                                          help="Sequences with predicted duration below this threshold will skip optimization.")
    st.session_state['simple_threshold'] = simple_threshold

with st.sidebar.expander("📊 Model information", expanded=False):
    st.write(f"**Number of features:** {len(full_features)}")
    st.write("**Training data:** 1,593 industrial DNA synthesis records")
    st.write("**Prediction target:** synthesis duration (days)")
    st.write("**Held-out test:** R² = 0.871; RMSE = 4.90 days; MAE = 2.70 days")
    st.write("**二级结构:** ViennaRNA, DNA Mathews 2004, 37 °C")
    st.write(f"**Log Transform (target):** {log_transform_used}")

input_type = st.radio("Input mode", ["Single sequence", "Batch upload"])

# ============================================================
# 单条Sequence模式
# ============================================================
if input_type == "Single sequence":
    seq = st.text_area("Enter DNA sequence (A/C/G/T only)", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("🔮 Predict", use_container_width=True)
    with col2:
        optimize_btn = st.button("🧬 Optimize & predict", use_container_width=True) if enable_optimization else st.button("🧬 Optimize & predict", disabled=True, use_container_width=True)
    
    if predict_btn or optimize_btn:
        if not seq:
            st.warning("Please enter a DNA sequence.")
        else:
            with st.spinner("Calculating..."):
                if predict_btn:
                    try:
                        pred = predict_sequence(seq, pipeline, full_features, log_transform_used)
                        pred_int = int(round(pred))
                        st.metric("Predicted synthesis duration", f"{pred_int} days")
                        st.info(f"Difficulty category: {classify_duration(pred_int)}")
                        
                        # ✅ 显示特征表格
                        show_feature_table(seq, full_features)
                        
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                
                elif optimize_btn and enable_optimization:
                    try:
                        orig_pred = predict_sequence(seq, pipeline, full_features, log_transform_used)
                        orig_pred_int = int(round(orig_pred))
                        threshold = st.session_state.get('simple_threshold', 30)
                        
                        if orig_pred < threshold:
                            st.info(f"📢 该Sequence预测周期为 {orig_pred_int} 天，属于「🟢 简单」级别，模型预测的合成Difficulty较低，通常无需进一步优化。")
                            st.metric("预测合成周期", f"{orig_pred_int} 天")
                            
                            # ✅ 显示特征表格
                            show_feature_table(seq, full_features)
                            
                            st.success("✅ 模型预测该Sequence具有较低的合成Difficulty。")
                        else:
                            progress_container = st.empty()
                            status_text = st.empty()
                            
                            progress_bar = progress_container.progress(0, text="🔄 Initializing optimization...")
                            status_text.text("🔍 分析原始Sequence...")
                            
                            result = optimize_and_predict_single(
                                seq, pipeline, full_features, log_transform_used,
                                max_iterations=max_iter,
                                progress_bar=progress_bar,
                                status_text=status_text
                            )
                            
                            progress_container.empty()
                            status_text.empty()
                            
                            if 'error' in result:
                                st.error(f"Optimization failed: {result['error']}")
                            else:
                                # ✅ 显示特征表格 (优化后)
                                st.subheader("📊 优化后特征")
                                show_feature_table(result['optimized_sequence'], full_features)
                                
                                show_optimization_comparison(result)
                                
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

# ============================================================
# 批量上传模式
# ============================================================
else:
    uploaded_file = st.file_uploader(
        "Upload sequence file (FASTA / CSV / TXT / XLSX)", 
        type=['fasta', 'csv', 'txt', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        seqs = read_sequences_from_file(uploaded_file)
        
        if seqs:
            st.info(f"📄 Loaded {len(seqs)} sequences")
            
            col1, col2 = st.columns(2)
            with col1:
                predict_btn = st.button("🔮 Batch predict", use_container_width=True)
            with col2:
                optimize_btn = st.button("🧬 Batch optimize & predict", use_container_width=True) if enable_optimization else st.button("🧬 Batch optimize & predict", disabled=True, use_container_width=True)
            
            if predict_btn or optimize_btn:
                results = []
                
                if predict_btn:
                    progress_bar = st.progress(0, text="Predicting...")
                    total = len(seqs)
                    for i, seq in enumerate(seqs):
                        try:
                            pred = predict_sequence(seq, pipeline, full_features, log_transform_used)
                            pred_int = int(round(pred))
                        except:
                            pred_int = 0
                        results.append({
                            'Sequence ID': i+1,
                            'Sequence': seq[:50] + '...' if len(seq) > 50 else seq,
                            'Predicted duration (days)': pred_int,
                            'Difficulty': classify_duration(pred_int)
                        })
                        progress_bar.progress((i + 1) / total, text=f"Predicting {i+1}/{total}")
                    progress_bar.empty()
                
                elif optimize_btn and enable_optimization:
                    total_seqs = len(seqs)
                    overall_progress = st.progress(0, text=f"Preparing 1/{total_seqs}...")
                    overall_status = st.empty()
                    
                    for i, seq in enumerate(seqs):
                        overall_status.text(f"🔄 处理第 {i+1}/{total_seqs} 条Sequence...")
                        overall_progress.progress(i / total_seqs, text=f"Processing {i+1}/{total_seqs}...")
                        
                        sub_progress = st.progress(0, text="Initializing...")
                        sub_status = st.empty()
                        
                        try:
                            orig_pred = predict_sequence(seq, pipeline, full_features, log_transform_used)
                            threshold = st.session_state.get('simple_threshold', 30)
                        except:
                            orig_pred = 30
                            threshold = 30
                        
                        if orig_pred < threshold:
                            results.append({
                                'Sequence ID': i+1,
                                '原始Sequence': seq[:50] + '...' if len(seq) > 50 else seq,
                                '优化后Sequence': 'Not optimized (low difficulty)',
                                'Original prediction (days)': int(round(orig_pred)),
                                'Optimized prediction (days)': int(round(orig_pred)),
                                'Improvement (days)': 0,
                                'Protein preserved': '✅'
                            })
                            sub_progress.empty()
                            sub_status.empty()
                            overall_progress.progress((i + 1) / total_seqs, text=f"Completed {i+1}/{total_seqs}")
                            continue
                        
                        result = optimize_and_predict_single(
                            seq, pipeline, full_features, log_transform_used,
                            max_iterations=max_iter,
                            progress_bar=sub_progress,
                            status_text=sub_status
                        )
                        
                        sub_progress.empty()
                        sub_status.empty()
                        
                        if 'error' not in result:
                            results.append({
                                'Sequence ID': i+1,
                                '原始Sequence': seq[:50] + '...' if len(seq) > 50 else seq,
                                '优化后Sequence': result['optimized_sequence'][:50] + '...' if len(result['optimized_sequence']) > 50 else result['optimized_sequence'],
                                'Original prediction (days)': int(round(result['original_prediction'])),
                                'Optimized prediction (days)': int(round(result['optimized_prediction'])),
                                'Improvement (days)': int(round(result['prediction_improvement'])),
                                'Protein preserved': '✅' if result['protein_verified'] else '❌'
                            })
                        else:
                            results.append({
                                'Sequence ID': i+1,
                                '原始Sequence': seq[:50] + '...' if len(seq) > 50 else seq,
                                '优化后Sequence': 'Optimization failed',
                                'Original prediction (days)': int(round(orig_pred)),
                                'Optimized prediction (days)': 'N/A',
                                'Improvement (days)': 'N/A',
                                'Protein preserved': '❌'
                            })
                        
                        overall_progress.progress((i + 1) / total_seqs, text=f"Completed {i+1}/{total_seqs}")
                    
                    overall_progress.empty()
                    overall_status.empty()
                
                st.subheader("📋 Results")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
                
                csv = df_results.to_csv(index=False)
                st.download_button("📥 Download results CSV", csv, "results.csv", "text/csv")
        else:
            st.warning("No valid DNA sequences were detected.")
            
st.markdown("---")
st.caption("💡 Supports FASTA, CSV, TXT, and Excel | Secondary structure: ViennaRNA DNA Mathews 2004, 37 °C | Predictions are computational estimates and require experimental validation.")