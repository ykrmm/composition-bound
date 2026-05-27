export CODE_DIR=$HOME/physics_llm3 # replace with your path 

cd $CODE_DIR

mkdir -p logs

# you can change config with qv16

CKPT=$CODE_DIR/logs/bios_multi5p_fullname/bios_multi5p_fullname_399999.pt
if [ ! -f "$CKPT" ]; then
    echo "ERROR: pretrain checkpoint not found: $CKPT"
    exit 1
fi

echo "--- LoRA finetune (paper canonical: q/v=8, embed=128, cosine→10%, 50K steps) ---"
python finetune_qa.py --config configs/finetune_qa_bios_multi5p_fullname_qv8.yaml

LORA_CKPT=$CODE_DIR/logs/qa_multi5p_fullname_qv8/bios_multi5p_fullname_qa_qv8_lora.pt

echo "--- Eval P_train + P_test ---"
python eval_qa.py \
    --config    configs/finetune_qa_bios_multi5p_fullname_qv8.yaml \
    --lora_ckpt $LORA_CKPT