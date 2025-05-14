python main.py --config configs/default.yaml --dataset fomc --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5
python main.py --config configs/default.yaml --dataset fpb --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5
python main.py --config configs/default.yaml --dataset banking77 --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5
python main.py --config configs/default.yaml --dataset causal_detection --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5
python main.py --config configs/default.yaml --dataset edtsum --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 128 --mode inference # not finished
python main.py --config configs/default.yaml --dataset finqa --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 16384 --mode inference
python main.py --config configs/default.yaml --dataset fiqa_task1 --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset fiqa_task2 --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset headlines --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset refind --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset finred --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset finbench --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset subjectiveqa --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 1 --max_tokens 4096 --mode inference
python main.py --config configs/default.yaml --dataset finer --model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --batch_size 5 --max_tokens 8192 --mode inference