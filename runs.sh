python main.py --config configs/default.yaml --dataset fomc --model openai/o1-mini --batch_size 5
python main.py --config configs/default.yaml --dataset fpb --model openai/o1-mini --batch_size 5
python main.py --config configs/default.yaml --dataset banking77 --model openai/o1-mini --batch_size 5
python main.py --config configs/default.yaml --dataset causal_detection --model openai/o1-mini --batch_size 5
python main.py --config configs/default.yaml --dataset edtsum --model openai/o1-mini --batch_size 5 --max_tokens 128 --mode inference # not finished
python main.py --config configs/default.yaml --dataset finqa --model openai/o1-mini --batch_size 5 --max_tokens 16384 --mode inference
python main.py --config configs/default.yaml --dataset fiqa_task1 --model openai/o1-mini --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset fiqa_task2 --model openai/o1-mini --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset headlines --model openai/o1-mini --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset refind --model openai/o1-mini --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset finred --model openai/o1-mini --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset finbench --model openai/o1-mini --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset subjectiveqa --model openai/o1-mini --batch_size 1 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset finer --model openai/o1-mini --batch_size 5 --max_tokens 8192 --mode inference