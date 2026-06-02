export CODE_DIR=$HOME/physics_llm3 # replace with your path 

cd $CODE_DIR

mkdir -p logs

torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_bios.py \
    --config configs/pretrain_single_fullname.yaml
