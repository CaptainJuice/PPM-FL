filename=eval_ModelNet40_deepsets_np
savedirname="result/ModelNet40_deepsets_pretrained_eval_np1024_seed2026"

mkdir -p "$savedirname"
python3 "$filename.py" 2>&1 | tee "$savedirname/eval.log"
