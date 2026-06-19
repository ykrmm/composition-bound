#!/bin/bash
set -e

# ── 1. Generate individuals + 1-hop NL biographies ─────────────────────────

python gen_data/generate_graph_bios.py \
    --out_dir      ../graph_bios_data \
    --n_individuals 100000 \
    --n_multi      5

# ── 2. Generate multi-hop QA pairs ─────────────────────────────────────────

python gen_data/generate_graph_qa.py \
    --individuals ../graph_bios_data/individuals.json \
    --out_dir     ../graph_qa_data

# ── 3. Generate 2-hop NL augmentation (P_comp, ids 0–49999) ────────────────

python gen_data/generate_bioG_2hop_nl.py \
    --individuals    ../graph_bios_data/individuals.json \
    --max_individuals 50000 \
    --implicit \
    --out_full       ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt

python gen_data/generate_bioG_2hop_nl.py \
    --individuals    ../graph_bios_data/individuals.json \
    --max_individuals 50000 \
    --out_full       ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt

# ── 4. Generate 2-hop RDF augmentation (P_comp, ids 0–49999) ───────────────

python gen_data/generate_bioG_2hop_triple.py \
    --individuals    ../graph_bios_data/individuals.json \
    --max_individuals 50000 \
    --out_full       ../graph_bios_data/bioG_2hop_triple_implicit_50k_all.txt

python gen_data/generate_bioG_2hop_triple.py \
    --individuals    ../graph_bios_data/individuals.json \
    --max_individuals 50000 \
    --explicit \
    --out_full       ../graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt

# ── 5. Mix and tokenize pretrain corpora (exp2–exp9) ───────────────────────

mkdir -p ../graph_bios_tokens

# exp2: 30% bios + 70% implicit 2-hop NL
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt \
    --ratios  0.3 0.7 \
    --out     ../graph_bios_data/mixed_nl_implicit_30_70_all.txt
python gen_data/tokenizer_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_nl_implicit_30_70_all.txt

# exp3: 30% bios + 70% explicit 2-hop NL
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt \
    --ratios  0.3 0.7 \
    --out     ../graph_bios_data/mixed_nl_explicit_30_70_all.txt
python gen_data/tokenizer_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_nl_explicit_30_70_all.txt

# exp4: 30% bios + 70% implicit 2-hop RDF
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_triple_implicit_50k_all.txt \
    --ratios  0.3 0.7 \
    --out     ../graph_bios_data/mixed_triple_implicit_30_70_all.txt
python gen_data/tokenizer_graph_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_triple_implicit_30_70_all.txt

# exp5: 30% bios + 70% explicit 2-hop RDF
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt \
    --ratios  0.3 0.7 \
    --out     ../graph_bios_data/mixed_triple_explicit_30_70_all.txt
python gen_data/tokenizer_graph_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_triple_explicit_30_70_all.txt

# exp6: 30% bios + 35% implicit 2-hop RDF + 35% explicit 2-hop RDF
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_triple_implicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt \
    --ratios  0.3 0.35 0.35 \
    --out     ../graph_bios_data/mixed_triple_explicit_implicit_30_35_35_all.txt
python gen_data/tokenizer_graph_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_triple_explicit_implicit_30_35_35_all.txt

# exp7: 30% bios + 35% implicit 2-hop NL + 35% explicit 2-hop NL
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt \
    --ratios  0.3 0.35 0.35 \
    --out     ../graph_bios_data/mixed_nl_explicit_implicit_30_35_35_all.txt
python gen_data/tokenizer_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_nl_explicit_implicit_30_35_35_all.txt

# exp8: 30% bios + 17.5% each of (impl NL, expl NL, impl triple, expl triple)
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_triple_implicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt \
    --ratios  0.3 0.175 0.175 0.175 0.175 \
    --out     ../graph_bios_data/mixed_nl_triple_all4_30_175_175_175_175_all.txt
python gen_data/tokenizer_graph_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_nl_triple_all4_30_175_175_175_175_all.txt

# exp9: 15% bios + 15% 1-hop atomic RDF + 17.5% each of 4 2-hop formats
# NOTE: bioG_triple_all.txt (1-hop atomic triples) is not yet generated by this script.
python gen_data/mix_pretrain.py \
    --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
              ../graph_bios_data/bioG_triple_all.txt \
              ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_triple_implicit_50k_all.txt \
              ../graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt \
    --ratios  0.15 0.15 0.175 0.175 0.175 0.175 \
    --out     ../graph_bios_data/mixed_nl_triple_all4_atomic_15_15_175x4_all.txt
python gen_data/tokenizer_graph_bios.py \
    --in_dir  ../graph_bios_data \
    --out_dir ../graph_bios_tokens \
    --files   mixed_nl_triple_all4_atomic_15_15_175x4_all.txt
