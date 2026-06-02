export CODE_DIR=$HOME/physics_llm3 # replace with your path 

cd $CODE_DIR

mkdir -p logs

# generate BioS dataset (100K people, 5 bios each, with permutation)
python generate_bios.py --n_individuals 100000 --n_multi 5 --seed 42 --out_dir bios_data

# tokenize BioS dataset for pretraining

python tokenize_bios.py --in_dir bios_data --out_dir bios_tokens

# generate QA datasets for finetuning/evaluation: N/2 train and the remaining test 

python generate_qa.py --individuals bios_data/individuals.json --out_dir qa_data