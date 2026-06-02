
# generate graph data 
python gen_data/generate_graph_bios.py --out_dir ../graph_bios_data --individuals ../graph_bios_data/individuals.json --n_multi 5 --n_individuals 100000

# generate graph QA data 
python gen_data/generate_graph_qa.py --individuals ../graph_bios_data/individuals.json --out_dir ../graph_qa_data


# generate 2 hop implicit explicit NL 
# 50k individuals (P_comp)
python gen_data/generate_bioG_2hop_nl.py --individuals ../graph_bios_data/individuals.json --max_individuals 50000 --implicit --out_full ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt

python gen_data/generate_bioG_2hop_nl.py --individuals ../graph_bios_data/individuals.json --max_individuals 50000 --out_full ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt

# generate 2 hop implicit explicit RDF 
# 50k individuals (P_comp)

python gen_data/generate_bioG_2hop_triple.py --individuals ../graph_bios_data/individuals.json --max_individuals 50000 --out_full ../graph_bios_data/bioG_2hop_triple_implicit_50k_all.txt


python gen_data/generate_bioG_2hop_triple.py --individuals ../graph_bios_data/individuals.json --max_individuals 50000 --explicit --out_full ../graph_bios_data/bioG_2hop_triple_explicit_50k_all.txt


# Prepare mix for exp 1 to 9 

# exp 1 - 50% bios multi+permute + 50% bioG triple
python gen_data/mix_pretrain.py \
        --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
                  ../graph_bios_data/bioG_triple_all.txt \
        --ratios  0.5 0.5 \
        --out     ../graph_bios_data/exp1.txt

python gen_data/tokenizer_graph_bio.py --input ../graph_bios_data/exp1.txt --output ../graph_bios_data/exp1_tokenized.txt

# exp 2 - 30 % bios multi+permute + 70% implicit NL 

python gen_data/mix_pretrain.py \
        --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
                  ../graph_bios_data/bioG_2hop_nl_implicit_50k_all.txt \
        --ratios  0.3 0.7 \
        --out     ../graph_bios_data/exp2.txt

python gen_data/tokenize_graph_bio.py --in ../graph_bios_data/exp2.txt --out ../graph_bios_tokens/exp2.npy

# exp 2 - 30 % bios multi+permute + 70% explicit NL 

python gen_data/mix_pretrain.py \
        --sources ../graph_bios_data/bios_multi5p_fullname_all.txt \
                  ../graph_bios_data/bioG_2hop_nl_explicit_50k_all.txt \
        --ratios  0.3 0.7 \
        --out     ../graph_bios_data/exp3.txt