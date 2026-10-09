# APP_Paper_PUBLICATION.py
# Publication-aligned Streamlit application for:
# "An interpretable machine learning framework for predicting and optimizing DNA synthesis difficulty"
#
# This version is aligned with the manuscript feature definitions and the final
# dna_synthesis_model_bundle.pkl file. Predicted synthesis duration is a model estimate
# and should not be interpreted as experimentally confirmed synthesis performance.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import RNA
import os
from pathlib import Path
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
    """Load the complete deployment bundle produced by model_bundle_ready_fixed.ipynb."""
    bundle_path = Path(__file__).resolve().parent / "dna_synthesis_model_bundle.pkl"
    try:
        bundle = joblib.load(bundle_path)
    except FileNotFoundError:
        st.error(
            f"❌ Model bundle not found / 模型文件未找到: {bundle_path}"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load model bundle / 模型加载失败: {e}")
        st.stop()

    required = {"pre_scaler", "pipeline", "feature_columns"}
    missing = required.difference(bundle.keys())
    if missing:
        st.error(
            "❌ Invalid model bundle. Missing keys / 模型包缺少字段: "
            + ", ".join(sorted(missing))
        )
        st.stop()

    pre_scaler = bundle["pre_scaler"]
    pipeline = bundle["pipeline"]
    full_features = list(bundle["feature_columns"])
    log_transform_used = bool(bundle.get("target_log_transform", False))

    return bundle, pre_scaler, pipeline, full_features, log_transform_used

# ============================================================
# 2. 特征计算函数
# ============================================================
def calculate_all_features(seq: str) -> dict:
    """计算单条序列的所有特征"""
    
    seq = seq.upper().strip()
    seq = ''.join(seq.split())
    
    if not seq:
        return {f: 0.0 for f in CANONICAL_FEATURES}

    length = len(seq)
    features = {}

    # ---- 1. 基础特征 ----
    # The training workflow used ln(sequence length); all valid DNA sequences have length > 0.
    # IMPORTANT: the currently trained model uses raw sequence length in the
    # feature column named 'log_length'. Keep the historical column name
    # for compatibility with dna_synthesis_model_bundle.pkl.
    features['log_length'] = float(length)

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

    # ---- 4. 重复序列 ----
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
def predict_sequence(
    seq: str,
    pipeline,
    full_features,
    log_transform_used=False,
    pre_scaler=None
) -> float:
    """
    Predict synthesis duration using the exact deployment preprocessing chain:
    raw 95 features -> pre_scaler -> final trained pipeline.
    """
    feature_dict = calculate_all_features(seq)

    X_raw = pd.DataFrame([feature_dict])
    X_raw = X_raw.reindex(columns=full_features)

    if X_raw.isna().any().any():
        missing_cols = X_raw.columns[X_raw.isna().any()].tolist()
        raise ValueError(
            f"Missing or invalid features before prediction: {missing_cols}"
        )

    if pre_scaler is None:
        raise ValueError("pre_scaler is required for bundle-based inference.")

    X_pre_scaled = pre_scaler.transform(X_raw)
    X_pre_scaled = pd.DataFrame(
        X_pre_scaled,
        columns=full_features
    )

    pred = float(pipeline.predict(X_pre_scaled)[0])

    if log_transform_used:
        pred = float(np.expm1(pred))

    return pred
# ============================================================
# 4. 显示特征表格的函数
# ============================================================
def show_feature_table(seq: str, full_features: List[str], zh: bool = False):
    """Display model input features in Chinese or English."""
    def T(en, cn):
        return cn if zh else en

    all_feat = calculate_all_features(seq)
    df = pd.DataFrame([all_feat])
    df = df[full_features]

    st.subheader(T("📊 Calculated features", "📊 计算的特征"))

    base_features = ['log_length', 'A_ratio', 'G_ratio', 'C_ratio', 'T_ratio']
    dinuc_features = ['AA','AG','AC','AT','GA','GG','GC','GT','CA','CG','CC','CT','TA','TG','TC','TT']
    trinuc_features = [''.join(p).lower() for p in product('ACGT', repeat=3)]
    repeat_features = ['max_repeat_len', 'repeat_density']
    secondary_features = [
        'max_stem_length', 'avg_stem_length', 'total_loops', 'max_loop_size',
        'GT_pairs', 'pairing_rate', 'stem_count', 'MFE'
    ]

    with st.expander(T("📈 Base features", "📈 基础特征"), expanded=True):
        base_df = df[base_features].copy()
        # The trained bundle uses raw sequence length in the legacy field 'log_length'.
        # Show the scientifically correct label in the UI without changing the model key.
        base_df = base_df.rename(columns={'log_length': 'length'})
        format_map = {
            'length': '{:.0f}',
            'A_ratio': '{:.4f}',
            'G_ratio': '{:.4f}',
            'C_ratio': '{:.4f}',
            'T_ratio': '{:.4f}'
        }
        st.dataframe(base_df.style.format(format_map))

    with st.expander(T("🧬 Dinucleotide features", "🧬 二核苷酸特征")):
        st.dataframe(df[dinuc_features].style.format("{:.4f}"))

    with st.expander(T("🧬 Trinucleotide features", "🧬 三核苷酸特征")):
        trinuc_display = df[trinuc_features[:20]]
        st.dataframe(trinuc_display.style.format("{:.4f}"))
        if len(trinuc_features) > 20:
            st.caption(T(
                f"... {len(trinuc_features) - 20} additional trinucleotide features",
                f"... 还有 {len(trinuc_features) - 20} 个三核苷酸特征"
            ))

    with st.expander(T("🔁 Repeat features", "🔁 重复序列特征")):
        st.dataframe(df[repeat_features].style.format("{:.4f}"))

    with st.expander(T("🔬 Secondary-structure features", "🔬 二级结构特征")):
        st.dataframe(df[secondary_features].style.format("{:.4f}"))

    with st.expander(T("📋 All features", "📋 所有特征")):
        all_display = df.copy().rename(columns={'log_length': 'length'})
        format_map_all = {c: "{:.4f}" for c in all_display.columns}
        if 'length' in format_map_all:
            format_map_all['length'] = "{:.0f}"
        st.dataframe(all_display.style.format(format_map_all))

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
        raise ValueError("DNA序列为空")
    illegal_chars = set(dna) - {'A', 'T', 'C', 'G'}
    if illegal_chars:
        raise ValueError(f"非法字符: {illegal_chars}")
    if len(dna) % 3 != 0:
        raise ValueError(f"长度 {len(dna)} 不是3的倍数")
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


def generate_synonymous_candidates(
    seed_seq: str,
    protein_seq: str,
    stop_set: set,
    rng: np.random.Generator,
    n_candidates: int = 60,
    min_changes: int = 3,
    max_changes: int = 18
) -> List[str]:
    """Generate diverse protein-preserving synonymous variants."""
    positions = []
    for i, aa in enumerate(protein_seq):
        start = i * 3
        if start in stop_set or aa == '*':
            continue
        current = seed_seq[start:start+3]
        alternatives = [c for c in codon_table.get(aa, []) if c != current]
        if alternatives:
            positions.append((i, alternatives))

    if not positions:
        return []

    max_changes = min(max_changes, len(positions))
    min_changes = min(min_changes, max_changes)

    candidates = set()
    for _ in range(max(n_candidates * 4, 100)):
        if len(candidates) >= n_candidates:
            break

        k = int(rng.integers(min_changes, max_changes + 1))
        selected = rng.choice(len(positions), size=k, replace=False)
        chars = list(seed_seq)

        for idx in selected:
            codon_i, alternatives = positions[idx]
            new_codon = alternatives[int(rng.integers(0, len(alternatives)))]
            s = codon_i * 3
            chars[s:s+3] = list(new_codon)

        candidate = ''.join(chars)
        if candidate != seed_seq:
            candidates.add(candidate)

    return list(candidates)


def proxy_candidate_score(seq: str, original_gc: float) -> float:
    """Cheap prescreening score; final selection is based on the trained model."""
    gc = get_gc_ratio(seq)
    sir = count_inverted_repeats(seq)

    if 0.40 <= gc <= 0.60:
        gc_penalty = 0.0
    else:
        gc_penalty = 100.0 * min(abs(gc - 0.40), abs(gc - 0.60))

    gc_drift = 5.0 * abs(gc - original_gc)
    return 2.5 * sir + gc_penalty + gc_drift


def optimize_single_sequence(
    input_dna: str,
    prediction_callback,
    feature_callback,
    target_reduction: float = 0.10,
    max_rounds: int = 5,
    beam_width: int = 3,
    generated_per_seed: int = 60,
    full_evaluations_per_seed: int = 8,
    random_seed: int = 42,
    progress_callback=None
) -> Tuple[Optional[str], Dict]:
    """
    Model-guided synonymous optimization.

    Success requires:
      1) predicted synthesis duration reduced by at least target_reduction;
      2) protein sequence unchanged;
      3) MFE becomes less negative;
      4) high-risk SIR count does not increase.
    """
    try:
        input_dna, unknown = validate_dna(input_dna)
    except ValueError as e:
        return None, {'error': str(e)}

    non_stop = [(p, c) for p, c in unknown if not is_stop_codon(c)]
    if non_stop:
        return None, {'error': f"Non-standard codons / 非标准密码子: {non_stop[:5]}"}

    protein_seq, stop_codons = dna_to_protein(input_dna)
    stop_set = set(pos for pos, _ in stop_codons)

    rng = np.random.default_rng(random_seed)

    original_prediction = float(prediction_callback(input_dna))
    original_features = feature_callback(input_dna)
    original_mfe = float(original_features.get('MFE', 0.0))
    original_sir = int(count_inverted_repeats(input_dna))
    original_gc = float(get_gc_ratio(input_dna))
    target_prediction = original_prediction * (1.0 - target_reduction)

    beam = [(original_prediction, input_dna, original_features, original_sir)]
    evaluated = {input_dna}
    best_qualifying = None

    history = [{
        'round': 0,
        'predicted_duration': original_prediction,
        'reduction_pct': 0.0,
        'MFE': original_mfe,
        'high_risk_SIRs': original_sir,
        'qualifies': False,
        'sequence': input_dna
    }]

    for round_idx in range(1, max_rounds + 1):
        if progress_callback:
            progress_callback(round_idx - 1, max_rounds, stage=f"Round {round_idx}/{max_rounds}")

        candidates_to_evaluate = []

        for _, seed_seq, _, _ in beam:
            generated = generate_synonymous_candidates(
                seed_seq=seed_seq,
                protein_seq=protein_seq,
                stop_set=stop_set,
                rng=rng,
                n_candidates=generated_per_seed,
                min_changes=3,
                max_changes=min(18, max(8, len(seed_seq) // 90))
            )

            ranked_proxy = sorted(
                ((proxy_candidate_score(c, original_gc), c) for c in generated),
                key=lambda x: x[0]
            )

            for _, c in ranked_proxy[:full_evaluations_per_seed]:
                if c not in evaluated:
                    evaluated.add(c)
                    candidates_to_evaluate.append(c)

        if not candidates_to_evaluate:
            break

        round_results = []

        for candidate in candidates_to_evaluate:
            candidate_protein, _ = dna_to_protein(candidate)
            if candidate_protein != protein_seq:
                continue

            features = feature_callback(candidate)
            pred = float(prediction_callback(candidate))
            mfe = float(features.get('MFE', 0.0))
            sir = int(count_inverted_repeats(candidate))

            reduction = (
                (original_prediction - pred) / original_prediction
                if original_prediction > 0 else 0.0
            )

            # Structural direction is enforced throughout the search:
            # MFE must be less negative than the original sequence, and
            # high-risk SIR count must not increase.
            structural_ok = (mfe > original_mfe) and (sir <= original_sir)
            qualifies = pred <= target_prediction and structural_ok

            result = {
                'round': round_idx,
                'sequence': candidate,
                'features': features,
                'predicted_duration': pred,
                'reduction_pct': 100.0 * reduction,
                'MFE': mfe,
                'high_risk_SIRs': sir,
                'structural_ok': structural_ok,
                'qualifies': qualifies
            }
            round_results.append(result)

            if qualifies:
                if best_qualifying is None or pred < best_qualifying['predicted_duration']:
                    best_qualifying = result

        if not round_results:
            break

        # Only structurally acceptable candidates are allowed to define the
        # reported checkpoint and the next beam. This prevents the search
        # trajectory from drifting toward more-negative MFE values.
        structural_results = [r for r in round_results if r['structural_ok']]

        if not structural_results:
            history.append({
                'round': round_idx,
                'predicted_duration': None,
                'reduction_pct': None,
                'MFE': None,
                'high_risk_SIRs': None,
                'qualifies': False,
                'sequence': None
            })
            break

        best_round = min(
            structural_results,
            key=lambda r: (
                r['predicted_duration'],
                -r['MFE'],
                r['high_risk_SIRs']
            )
        )

        history.append({
            'round': round_idx,
            'predicted_duration': best_round['predicted_duration'],
            'reduction_pct': best_round['reduction_pct'],
            'MFE': best_round['MFE'],
            'high_risk_SIRs': best_round['high_risk_SIRs'],
            'qualifies': best_round['qualifies'],
            'sequence': best_round['sequence']
        })

        if best_qualifying is not None:
            break

        ranked = sorted(
            structural_results,
            key=lambda r: (
                r['predicted_duration'],
                -r['MFE'],
                r['high_risk_SIRs']
            )
        )

        beam = [
            (r['predicted_duration'], r['sequence'], r['features'], r['high_risk_SIRs'])
            for r in ranked[:beam_width]
        ]

    if best_qualifying is None:
        return input_dna, {
            'success': False,
            'optimized_sequence': input_dna,
            'initial_prediction': original_prediction,
            'final_prediction': original_prediction,
            'target_prediction': target_prediction,
            'prediction_reduction_pct': 0.0,
            'initial_gc': original_gc,
            'final_gc': original_gc,
            'gc_change': 0.0,
            'initial_stems': original_sir,
            'final_stems': original_sir,
            'stem_change': 0,
            'initial_mfe': original_mfe,
            'final_mfe': original_mfe,
            'mfe_change': 0.0,
            'protein_verified': True,
            'original_protein': protein_seq,
            'optimized_protein': protein_seq,
            'checkpoint_history': history,
            'target_reduction': target_reduction
        }

    final_seq = best_qualifying['sequence']
    final_pred = float(best_qualifying['predicted_duration'])
    final_mfe = float(best_qualifying['MFE'])
    final_sir = int(best_qualifying['high_risk_SIRs'])
    final_gc = float(get_gc_ratio(final_seq))
    final_protein, _ = dna_to_protein(final_seq)

    return final_seq, {
        'success': True,
        'optimized_sequence': final_seq,
        'initial_prediction': original_prediction,
        'final_prediction': final_pred,
        'target_prediction': target_prediction,
        'prediction_reduction_pct': 100.0 * (original_prediction - final_pred) / original_prediction,
        'initial_gc': original_gc,
        'final_gc': final_gc,
        'gc_change': final_gc - original_gc,
        'initial_stems': original_sir,
        'final_stems': final_sir,
        'stem_change': final_sir - original_sir,
        'initial_mfe': original_mfe,
        'final_mfe': final_mfe,
        'mfe_change': final_mfe - original_mfe,
        'protein_verified': final_protein == protein_seq,
        'original_protein': protein_seq,
        'optimized_protein': final_protein,
        'checkpoint_history': history,
        'target_reduction': target_reduction
    }


def optimize_and_predict_single(
    seq: str,
    pipeline,
    full_features,
    log_transform_used,
    pre_scaler,
    target_reduction: float = 0.10,
    max_iterations: int = 5,
    progress_bar=None,
    status_text=None
) -> Dict:

    def prediction_callback(candidate_seq: str) -> float:
        return float(
            predict_sequence(
                candidate_seq,
                pipeline,
                full_features,
                log_transform_used,
                pre_scaler=pre_scaler
            )
        )

    def feature_callback(candidate_seq: str) -> Dict:
        return calculate_all_features(candidate_seq)

    def update_progress(current, total, stage=""):
        if progress_bar:
            progress = min(0.95, 0.05 + ((current + 1) / max(total, 1)) * 0.90)
            progress_bar.progress(progress, text=stage)
        if status_text:
            status_text.text(f"🔄 {stage}")

    optimized, metrics = optimize_single_sequence(
        input_dna=seq,
        prediction_callback=prediction_callback,
        feature_callback=feature_callback,
        target_reduction=target_reduction,
        max_rounds=max_iterations,
        beam_width=3,
        generated_per_seed=60,
        full_evaluations_per_seed=8,
        random_seed=42,
        progress_callback=update_progress
    )

    if optimized is None:
        return {'error': metrics.get('error', 'Optimization failed')}

    if progress_bar:
        progress_bar.progress(1.0, text="✅ Complete!")

    return {
        'original_sequence': seq,
        'optimized_sequence': optimized,
        'original_prediction': float(metrics['initial_prediction']),
        'optimized_prediction': float(metrics['final_prediction']),
        'prediction_improvement': float(metrics['initial_prediction']) - float(metrics['final_prediction']),
        'metrics': metrics,
        'protein_verified': metrics.get('protein_verified', False),
        'success': metrics.get('success', False)
    }


# ============================================================
# 6. Input parsing and helper functions
# ============================================================

def normalize_dna_sequence(seq: str) -> str:
    """Remove whitespace and normalize a DNA sequence to uppercase."""
    return ''.join(str(seq).upper().split())


def parse_fasta_text(content: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA text and automatically remove header lines beginning with '>'.
    Returns a list of (header, sequence) tuples.
    """
    records = []
    header = None
    chunks = []

    for raw_line in str(content).replace('\ufeff', '').splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith('>'):
            if chunks:
                seq = normalize_dna_sequence(''.join(chunks))
                if seq:
                    records.append((header or '', seq))
                chunks = []
            header = line[1:].strip()
        else:
            chunks.append(line)

    if chunks:
        seq = normalize_dna_sequence(''.join(chunks))
        if seq:
            records.append((header or '', seq))

    return records


def parse_sequence_input(text: str) -> str:
    """
    Accept either a plain DNA sequence or a single FASTA record.
    FASTA header lines beginning with '>' are removed automatically.
    """
    raw = str(text).strip()
    if not raw:
        raise ValueError("No sequence provided. / 未输入序列。")

    # FASTA pasted into the text box
    if any(line.lstrip().startswith('>') for line in raw.splitlines()):
        records = parse_fasta_text(raw)
        if not records:
            raise ValueError("No valid FASTA sequence found. / 未识别到有效 FASTA 序列。")
        if len(records) > 1:
            raise ValueError(
                "Multiple FASTA records were detected. Please use Batch upload mode. "
                "/ 检测到多个 FASTA 记录，请使用批量上传模式。"
            )
        seq = records[0][1]
    else:
        seq = normalize_dna_sequence(raw)

    illegal = set(seq) - {'A', 'C', 'G', 'T'}
    if illegal:
        raise ValueError(
            f"Unsupported characters: {sorted(illegal)}. "
            "Only A/C/G/T are allowed in the sequence body. "
            f"/ 序列正文仅允许 A/C/G/T，检测到非法字符: {sorted(illegal)}"
        )
    return seq


def classify_duration(days):
    if days < 30:
        return "🟢 Low"
    elif days < 60:
        return "🟡 Moderate"
    else:
        return "🔴 High"


def read_sequences_from_file(uploaded_file):
    """Read DNA sequences from FASTA/FA/FNA, CSV/TSV/TXT, or Excel files."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    seqs = []

    if file_extension in ['xlsx', 'xls']:
        try:
            df_raw = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), header=0)
            first_col = df_raw.columns[0]
            candidates = df_raw[first_col].dropna().astype(str).tolist()
            for value in candidates:
                try:
                    seqs.append(parse_sequence_input(value))
                except ValueError:
                    continue
        except Exception as e:
            st.error(f"Failed to read Excel file / 读取 Excel 文件失败: {e}")
        return seqs

    raw_data = uploaded_file.getvalue()
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1', 'cp936']
    content = None
    for enc in encodings:
        try:
            content = raw_data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        st.error("Unable to decode file. Please use UTF-8 or GBK encoding. / 无法解码文件，请使用 UTF-8 或 GBK 编码。")
        return []

    stripped = content.lstrip('\ufeff\r\n\t ')

    # FASTA is detected by file extension or by the presence of header lines.
    if file_extension in ['fasta', 'fa', 'fna', 'fas'] or any(
        line.lstrip().startswith('>') for line in stripped.splitlines()
    ):
        records = parse_fasta_text(content)
        for _, seq in records:
            illegal = set(seq) - {'A', 'C', 'G', 'T'}
            if not illegal and seq:
                seqs.append(seq)
        return seqs

    # CSV / TSV
    if file_extension in ['csv', 'tsv'] or ',' in stripped.splitlines()[0] or '\t' in stripped.splitlines()[0]:
        try:
            sep = '\t' if file_extension == 'tsv' else None
            if sep:
                df_raw = pd.read_csv(io.StringIO(content), sep=sep, header=0)
            else:
                df_raw = pd.read_csv(io.StringIO(content), sep=None, engine='python', header=0)
            first_col = df_raw.columns[0]
            for value in df_raw[first_col].dropna().astype(str).tolist():
                try:
                    seqs.append(parse_sequence_input(value))
                except ValueError:
                    continue
            return seqs
        except Exception:
            pass

    # Plain-text fallback: each non-empty line is interpreted as one sequence.
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            seqs.append(parse_sequence_input(line))
        except ValueError:
            continue
    return seqs



def save_optimization_result(result: Dict, seq: str):
    """Persist optimization output across Streamlit reruns."""
    st.session_state["optimization_result"] = result
    st.session_state["optimization_input_sequence"] = seq


def clear_optimization_result():
    """Clear stale optimization output when a new sequence is submitted."""
    st.session_state.pop("optimization_result", None)
    st.session_state.pop("optimization_input_sequence", None)


def render_optimization_result(
    result: Dict,
    full_features,
    zh: bool = False
):
    """Render persisted optimization results, including selectable checkpoints."""
    T = lambda en, cn: cn if zh else en

    if 'error' in result:
        st.error(T("Optimization failed: ", "优化失败：") + result['error'])
        return

    metrics = result.get('metrics', {})
    history = metrics.get('checkpoint_history', [])
    valid_history = [
        h for h in history
        if h.get('sequence') and h.get('predicted_duration') is not None
    ]

    if not result.get('success', False):
        target_pct = int(round(100 * metrics.get('target_reduction', 0.10)))
        st.warning(T(
            f"No synonymous variant meeting the required ≥{target_pct}% predicted-duration reduction was found. You can still inspect and select any structurally acceptable checkpoint below.",
            f"未找到达到 ≥{target_pct}% 预测周期降低幅度的同义变体，但仍可在下方查看并选择任意满足结构约束的检查点。"
        ))
    else:
        st.success(T(
            f"✅ Predicted duration reduced by {metrics.get('prediction_reduction_pct', 0.0):.1f}%.",
            f"✅ 预测合成周期降低 {metrics.get('prediction_reduction_pct', 0.0):.1f}%。"
        ))
        show_optimization_comparison(result, zh=zh)

    if valid_history:
        history_df = pd.DataFrame([
            {
                'round': h['round'],
                'predicted_duration': h['predicted_duration'],
                'reduction_pct': h['reduction_pct'],
                'MFE': h['MFE'],
                'high_risk_SIRs': h['high_risk_SIRs'],
                'qualifies': h['qualifies']
            }
            for h in valid_history
        ])
        st.dataframe(history_df, use_container_width=True)

        option_labels = []
        for h in valid_history:
            if h['round'] == 0:
                label = T(
                    f"Round 0 — Original | {h['predicted_duration']:.2f} d | {h['reduction_pct']:.2f}%",
                    f"第 0 轮 — 原始序列 | {h['predicted_duration']:.2f} 天 | {h['reduction_pct']:.2f}%"
                )
            else:
                label = T(
                    f"Round {h['round']} | {h['predicted_duration']:.2f} d | {h['reduction_pct']:.2f}% | MFE {h['MFE']:.2f}",
                    f"第 {h['round']} 轮 | {h['predicted_duration']:.2f} 天 | {h['reduction_pct']:.2f}% | MFE {h['MFE']:.2f}"
                )
            option_labels.append(label)

        # Persist selection independently from the optimization button click.
        if "selected_checkpoint_idx" not in st.session_state:
            st.session_state["selected_checkpoint_idx"] = len(valid_history) - 1

        # Clamp stale indices if the number of checkpoints changes.
        st.session_state["selected_checkpoint_idx"] = min(
            st.session_state["selected_checkpoint_idx"],
            len(valid_history) - 1
        )

        selected_idx = st.selectbox(
            T("Select a checkpoint to inspect", "选择一个检查点查看"),
            options=list(range(len(valid_history))),
            index=st.session_state["selected_checkpoint_idx"],
            format_func=lambda i: option_labels[i],
            key="checkpoint_selector"
        )

        st.session_state["selected_checkpoint_idx"] = selected_idx
        selected = valid_history[selected_idx]

        st.subheader(T(
            f"🧬 Selected checkpoint: Round {selected['round']}",
            f"🧬 已选择检查点：第 {selected['round']} 轮"
        ))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            T("Predicted duration", "预测合成周期"),
            T(
                f"{selected['predicted_duration']:.2f} days",
                f"{selected['predicted_duration']:.2f} 天"
            )
        )
        c2.metric(T("Reduction", "降低幅度"), f"{selected['reduction_pct']:.2f}%")
        c3.metric("MFE", f"{selected['MFE']:.2f}")
        c4.metric(T("High-risk SIRs", "高风险 SIR"), str(selected['high_risk_SIRs']))

        st.text_area(
            T("Selected DNA sequence", "所选 DNA 序列"),
            value=selected['sequence'],
            height=220,
            key="selected_checkpoint_sequence_display"
        )

        st.subheader(T("📊 Selected-sequence features", "📊 所选序列特征"))
        show_feature_table(selected['sequence'], full_features, zh=zh)

    st.metric(
        T("Original predicted synthesis duration", "原始预测合成周期"),
        T(
            f"{result.get('original_prediction', 0.0):.1f} days",
            f"{result.get('original_prediction', 0.0):.1f} 天"
        )
    )

# ============================================================
# 7. 可视化函数
# ============================================================
def show_optimization_comparison(result: Dict, zh: bool = False):
    """Display optimization results in Chinese or English."""
    def T(en, cn):
        return cn if zh else en

    metrics = result.get('metrics', {})

    st.subheader(T("📊 Before vs. after optimization", "📊 优化前后指标对比"))

    col1, col2, col3, col4 = st.columns(4)

    orig_pred = float(result['original_prediction'])
    opt_pred = float(result['optimized_prediction'])
    improvement = orig_pred - opt_pred

    col1.metric(
        T("📅 Predicted duration", "📅 预测周期"),
        T(f"{orig_pred:.1f} → {opt_pred:.1f} days", f"{orig_pred:.1f} → {opt_pred:.1f} 天"),
        delta=T(
            f"{-improvement:+.1f} days" if improvement != 0 else "0.0 days",
            f"{-improvement:+.1f} 天" if improvement != 0 else "0.0 天"
        )
    )

    init_gc = metrics.get('initial_gc', 0) * 100
    final_gc = metrics.get('final_gc', 0) * 100
    gc_delta = final_gc - init_gc
    col2.metric(
        T("🧬 GC content", "🧬 GC 含量"),
        f"{init_gc:.1f}% → {final_gc:.1f}%",
        delta=f"{gc_delta:+.1f}%"
    )

    init_stems = int(metrics.get('initial_stems', 0))
    final_stems = int(metrics.get('final_stems', 0))
    stem_delta = final_stems - init_stems
    col3.metric(
        T("🎯 High-risk SIRs", "🎯 高风险 SIR 数"),
        f"{init_stems} → {final_stems}",
        delta=f"{stem_delta:+d}"
    )

    init_mfe = float(metrics.get('initial_mfe', 0))
    final_mfe = float(metrics.get('final_mfe', 0))
    mfe_delta = final_mfe - init_mfe
    if init_mfe != 0 or final_mfe != 0:
        col4.metric(
            "⚡ MFE (kcal/mol)",
            f"{init_mfe:.2f} → {final_mfe:.2f}",
            delta=f"{mfe_delta:+.2f}"
        )
    else:
        col4.metric("⚡ MFE (kcal/mol)", "N/A")

    if init_mfe != 0 or final_mfe != 0:
        st.markdown("---")
        st.subheader(T("⚡ MFE comparison", "⚡ MFE 改善可视化"))

        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ["Before", "After"] if not zh else ["优化前", "优化后"]
        bars = ax.bar(labels, [init_mfe, final_mfe])

        ax.set_ylabel("MFE (kcal/mol)", fontsize=12)
        ax.set_title(
            "Minimum free energy (MFE) comparison" if not zh else "最小自由能（MFE）对比",
            fontsize=14, fontweight='bold'
        )

        for bar, val in zip(bars, [init_mfe, final_mfe]):
            # Put labels inside/near bars in a robust way for negative values.
            y = val + (2 if val < 0 else 0.5)
            ax.text(
                bar.get_x() + bar.get_width()/2,
                y,
                f"{val:.2f}",
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        if mfe_delta > 0:
            annotation = (
                f"↑ Less negative by {mfe_delta:.2f} kcal/mol"
                if not zh else
                f"↑ 增加 {mfe_delta:.2f} kcal/mol"
            )
            ax.annotate(
                annotation,
                xy=(1, final_mfe),
                xytext=(1.08, (init_mfe + final_mfe) / 2),
                arrowprops=dict(arrowstyle='->', lw=1.8),
                fontsize=10,
                fontweight='bold'
            )

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    st.subheader(T("🔬 Protein-sequence verification", "🔬 蛋白质序列验证"))

    if metrics.get('protein_verified', False):
        st.success(T(
            "✅ The encoded amino-acid sequence is preserved exactly.",
            "✅ 编码的氨基酸序列完全保持不变。"
        ))
        with st.expander(T("📝 View protein sequences", "📝 查看蛋白质序列")):
            c1, c2 = st.columns(2)
            with c1:
                st.caption(T("Original protein", "原始蛋白质"))
                st.code(metrics.get('original_protein', ''), language='text')
            with c2:
                st.caption(T("Optimized protein", "优化后蛋白质"))
                st.code(metrics.get('optimized_protein', ''), language='text')
    else:
        st.warning(T(
            "⚠️ Protein sequences differ. Please verify the coding sequence and reading frame.",
            "⚠️ 蛋白质序列不一致，请检查编码序列及阅读框。"
        ))

    st.markdown("---")
    with st.expander(T("🧬 View full optimized sequence", "🧬 查看优化后完整序列")):
        st.code(result.get('optimized_sequence', ''), language='text')


# ============================================================
# 8. STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="DNA Synthesis Duration Prediction & Sequence Optimization",
    page_icon="🧬",
    layout="centered"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 22px !important; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; }
    [data-testid="stMetricDelta"] { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)

# Language selector
language = st.sidebar.selectbox(
    "Language / 语言",
    ["English", "中文"],
    index=0
)

st.sidebar.caption("App version: 2026-10-09 v12-bundlendlendlendlendlendle")
ZH = language == "中文"

def L(en: str, zh: str) -> str:
    return zh if ZH else en

st.title(L(
    "🧬 DNA Synthesis Duration Prediction & Sequence Optimization",
    "🧬 DNA 合成周期预测与序列优化工具"
))
st.markdown(L(
    "Enter a DNA sequence or FASTA record to predict synthesis duration. Protein-coding sequences can also be synonymously optimized.",
    "输入 DNA 序列或 FASTA 记录以预测合成周期；蛋白编码序列还可进行同义优化。"
))
st.caption(L(
    "Note: Predictions and optimization results are computational estimates and require experimental validation.",
    "注：预测值和优化结果均为计算结果，实际合成性能仍需实验验证。"
))

bundle, pre_scaler, pipeline, full_features, log_transform_used = load_model()
if pipeline is None:
    st.stop()

st.success(L(
    f"✅ Model loaded successfully! Features: {len(full_features)}",
    f"✅ 模型加载成功！特征数: {len(full_features)}"
))
st.info(L(
    f"🔬 Target log transform: {log_transform_used}",
    f"🔬 目标值 Log Transform: {log_transform_used}"
))

# Sidebar
st.sidebar.header(L("⚙️ Settings", "⚙️ 功能设置"))
enable_optimization = st.sidebar.checkbox(
    L("🧬 Enable sequence optimization", "🧬 启用序列优化（辅助功能）"),
    value=True
)

if enable_optimization:
    st.sidebar.info(L(
        "Optimization strategy:\n"
        "- Primary objective: ≥10% lower model-predicted duration\n"
        "- MFE should become less negative\n"
        "- High-risk SIRs must not increase\n"
        "- Preserve the encoded amino-acid sequence",
        "优化策略：\n"
        "- 首要目标：模型预测合成周期至少降低 10%\n"
        "- MFE 变得更不负\n"
        "- 高风险 SIR 不增加\n"
        "- 不改变编码的氨基酸序列"
    ))

    max_iter = st.sidebar.slider(
        L("Maximum optimization rounds", "最大优化搜索轮数"),
        min_value=1,
        max_value=8,
        value=5
    )

    target_reduction_pct = st.sidebar.slider(
        L("Required predicted-duration reduction (%)", "要求的预测周期降低幅度（%）"),
        min_value=10,
        max_value=40,
        value=10,
        step=5
    )

    st.session_state['target_reduction'] = target_reduction_pct / 100.0

with st.sidebar.expander(L("📊 Model information", "📊 模型信息"), expanded=False):
    st.write(L(f"**Number of features:** {len(full_features)}", f"**特征数：** {len(full_features)}"))
    st.write(L("**Training data:** 1,593 industrial DNA synthesis records", "**训练数据：** 1,593 条工业 DNA 合成记录"))
    st.write(L("**Prediction target:** synthesis duration (days)", "**预测目标：** 合成周期（天）"))
    st.write("**Held-out test:** R² = 0.871; RMSE = 4.90 days; MAE = 2.70 days")
    st.write(L("**Secondary structure:** ViennaRNA, DNA Mathews 2004, 37 °C", "**二级结构：** ViennaRNA, DNA Mathews 2004, 37 °C"))
    st.write(f"**Target log transform:** {log_transform_used}")
    st.write(L(
        f"**Deployment model:** bundled pre-scaler + final XGBoost pipeline",
        f"**部署模型：** pre-scaler + 最终 XGBoost pipeline 一体化模型包"
    ))
    st.write(L(
        "**Length feature:** true sequence length in bp",
        "**长度特征：** 真实序列长度（bp）"
    ))
    if bundle.get("model_version"):
        st.write(f"**Bundle version:** {bundle['model_version']}")

input_type = st.radio(
    L("Input mode", "选择输入方式"),
    [L("Single sequence", "单条序列"), L("Batch upload", "批量上传")]
)

# ---------------- Single sequence ----------------
if input_type == L("Single sequence", "单条序列"):
    raw_seq = st.text_area(
        L(
            "Enter DNA sequence or paste one FASTA record",
            "输入 DNA 序列，或直接粘贴一条 FASTA 记录"
        ),
        height=180,
        placeholder=L(
            ">example_sequence\nATGGCCTACG...\n\nor simply:\nATGGCCTACG...",
            ">示例序列\nATGGCCTACG...\n\n也可以直接输入：\nATGGCCTACG..."
        )
    )
    st.caption(L(
        "FASTA header lines beginning with '>' are removed automatically.",
        "以 “>” 开头的 FASTA header 会自动去除，无需手动删除。"
    ))

    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button(L("🔮 Predict", "🔮 预测"), use_container_width=True)
    with col2:
        optimize_btn = st.button(
            L("🧬 Optimize & predict", "🧬 优化并预测"),
            disabled=not enable_optimization,
            use_container_width=True
        )

    if predict_btn or optimize_btn:
        try:
            seq = parse_sequence_input(raw_seq)
        except ValueError as e:
            st.error(str(e))
            seq = None

        if seq:
            with st.spinner(L("Calculating...", "计算中...")):
                if predict_btn:
                    try:
                        pred = predict_sequence(seq, pipeline, full_features, log_transform_used, pre_scaler=pre_scaler)
                        pred_int = int(round(pred))
                        st.metric(
                            L("Predicted synthesis duration", "预测合成周期"),
                            L(f"{pred_int} days", f"{pred_int} 天")
                        )
                        category = classify_duration(pred_int)
                        if ZH:
                            category = category.replace("Low", "简单").replace("Moderate", "中等").replace("High", "困难")
                        st.info(L("Difficulty category: ", "难度等级：") + category)
                        show_feature_table(seq, full_features, zh=ZH)
                    except Exception as e:
                        st.error(L("Prediction failed: ", "预测失败：") + str(e))

                elif optimize_btn and enable_optimization:
                    try:
                        progress_container = st.empty()
                        status_text = st.empty()
                        progress_bar = progress_container.progress(
                            0,
                            text=L(
                                "🔄 Initializing model-guided optimization...",
                                "🔄 初始化模型引导优化..."
                            )
                        )

                        result = optimize_and_predict_single(
                            seq,
                            pipeline,
                            full_features,
                            log_transform_used,
                            pre_scaler=pre_scaler,
                            target_reduction=st.session_state.get('target_reduction', 0.10),
                            max_iterations=max_iter,
                            progress_bar=progress_bar,
                            status_text=status_text
                        )

                        progress_container.empty()
                        status_text.empty()

                        # Persist results so checkpoint selection survives Streamlit reruns.
                        save_optimization_result(result, seq)
                        st.session_state["selected_checkpoint_idx"] = (
                            max(
                                0,
                                len(
                                    [
                                        h for h in result.get('metrics', {}).get('checkpoint_history', [])
                                        if h.get('sequence') and h.get('predicted_duration') is not None
                                    ]
                                ) - 1
                            )
                        )

                    except Exception as e:
                        error_result = {'error': str(e)}
                        save_optimization_result(error_result, seq)

            # Render saved optimization results on every rerun.
            # This is essential because interacting with the checkpoint selectbox
            # triggers a Streamlit rerun.
            saved_result = st.session_state.get("optimization_result")
            saved_input = st.session_state.get("optimization_input_sequence")

            if saved_result is not None and saved_input == seq:
                render_optimization_result(
                    saved_result,
                    full_features,
                    zh=ZH
                )


# ---------------- Batch upload ----------------
else:
    uploaded_file = st.file_uploader(
        L(
            "Upload sequence file (FASTA / FA / FNA / CSV / TSV / TXT / XLSX)",
            "上传序列文件（FASTA / FA / FNA / CSV / TSV / TXT / XLSX）"
        ),
        type=['fasta', 'fa', 'fna', 'fas', 'csv', 'tsv', 'txt', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        seqs = read_sequences_from_file(uploaded_file)

        if seqs:
            st.info(L(f"📄 Loaded {len(seqs)} sequences", f"📄 共读取 {len(seqs)} 条序列"))
            st.caption(L(
                "FASTA headers beginning with '>' were removed automatically.",
                "FASTA 中以 “>” 开头的 header 已自动去除。"
            ))

            col1, col2 = st.columns(2)
            with col1:
                predict_btn = st.button(L("🔮 Batch predict", "🔮 批量预测"), use_container_width=True)
            with col2:
                optimize_btn = st.button(
                    L("🧬 Batch optimize & predict", "🧬 批量优化并预测"),
                    disabled=not enable_optimization,
                    use_container_width=True
                )

            if predict_btn or optimize_btn:
                results = []

                if predict_btn:
                    progress_bar = st.progress(0, text=L("Predicting...", "预测中..."))
                    total = len(seqs)
                    for i, seq in enumerate(seqs):
                        try:
                            pred = predict_sequence(seq, pipeline, full_features, log_transform_used, pre_scaler=pre_scaler)
                            pred_int = int(round(pred))
                            difficulty = classify_duration(pred_int)
                            if ZH:
                                difficulty = difficulty.replace("Low", "简单").replace("Moderate", "中等").replace("High", "困难")
                        except Exception:
                            pred_int = 0
                            difficulty = "N/A"

                        results.append({
                            L('Sequence ID', '序列号'): i + 1,
                            L('Sequence', '序列'): seq[:50] + '...' if len(seq) > 50 else seq,
                            L('Predicted duration (days)', '预测周期(天)'): pred_int,
                            L('Difficulty', '难度'): difficulty
                        })
                        progress_bar.progress(
                            (i + 1) / total,
                            text=L(f"Predicting {i+1}/{total}", f"预测中 {i+1}/{total}")
                        )
                    progress_bar.empty()

                elif optimize_btn and enable_optimization:
                    total_seqs = len(seqs)
                    overall_progress = st.progress(
                        0, text=L(f"Preparing 1/{total_seqs}...", f"准备处理 1/{total_seqs}...")
                    )
                    overall_status = st.empty()

                    for i, seq in enumerate(seqs):
                        overall_status.text(L(
                            f"🔄 Processing sequence {i+1}/{total_seqs}...",
                            f"🔄 处理第 {i+1}/{total_seqs} 条序列..."
                        ))

                        sub_progress = st.progress(0, text=L("Initializing...", "初始化..."))
                        sub_status = st.empty()

                        try:
                            result = optimize_and_predict_single(
                                seq,
                                pipeline,
                                full_features,
                                log_transform_used,
                                pre_scaler=pre_scaler,
                                target_reduction=st.session_state.get('target_reduction', 0.10),
                                max_iterations=max_iter,
                                progress_bar=sub_progress,
                                status_text=sub_status
                            )

                            if 'error' in result:
                                raise RuntimeError(result['error'])

                            success = result.get('success', False)

                            results.append({
                                L('Sequence ID', '序列号'): i + 1,
                                L('Original sequence', '原始序列'): seq[:50] + '...' if len(seq) > 50 else seq,
                                L('Optimized sequence', '优化后序列'): (
                                    result['optimized_sequence'][:50] + '...'
                                    if success and len(result['optimized_sequence']) > 50
                                    else (
                                        result['optimized_sequence']
                                        if success
                                        else L('No qualifying variant', '未找到满足条件的变体')
                                    )
                                ),
                                L('Original prediction (days)', '原始预测(天)'): round(result['original_prediction'], 1),
                                L('Optimized prediction (days)', '优化后预测(天)'): (
                                    round(result['optimized_prediction'], 1) if success else 'N/A'
                                ),
                                L('Improvement (%)', '改善(%)'): (
                                    round(result['metrics'].get('prediction_reduction_pct', 0.0), 1)
                                    if success else 0.0
                                ),
                                L('Protein preserved', '蛋白质一致'): '✅' if result['protein_verified'] else '❌'
                            })

                        except Exception as e:
                            results.append({
                                L('Sequence ID', '序列号'): i + 1,
                                L('Original sequence', '原始序列'): seq[:50] + '...' if len(seq) > 50 else seq,
                                L('Optimized sequence', '优化后序列'): L('Optimization failed', '优化失败'),
                                L('Original prediction (days)', '原始预测(天)'): 'N/A',
                                L('Optimized prediction (days)', '优化后预测(天)'): 'N/A',
                                L('Improvement (%)', '改善(%)'): 'N/A',
                                L('Protein preserved', '蛋白质一致'): '❌'
                            })

                        sub_progress.empty()
                        sub_status.empty()

                        overall_progress.progress(
                            (i + 1) / total_seqs,
                            text=L(f"Completed {i+1}/{total_seqs}", f"完成 {i+1}/{total_seqs}")
                        )

                    overall_progress.empty()
                    overall_status.empty()

                st.subheader(L("📋 Results", "📋 结果"))
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)

                csv = df_results.to_csv(index=False)
                st.download_button(
                    L("📥 Download results CSV", "📥 下载结果 CSV"),
                    csv, "results.csv", "text/csv"
                )
        else:
            st.warning(L("No valid DNA sequences were detected.", "未识别到有效 DNA 序列。"))

st.markdown("---")
st.caption(L(
    "💡 Supports pasted FASTA and FASTA/FA/FNA/CSV/TSV/TXT/Excel files | FASTA headers are removed automatically | Secondary structure: ViennaRNA DNA Mathews 2004, 37 °C | Predictions require experimental validation",
    "💡 支持粘贴 FASTA 及 FASTA/FA/FNA/CSV/TSV/TXT/Excel 文件 | FASTA header 自动去除 | 二级结构：ViennaRNA DNA Mathews 2004, 37 °C | 预测结果需实验验证"
))
