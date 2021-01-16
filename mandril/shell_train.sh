echo "---------------------------------------------"

echo ">> Mandril with random from best random k=5:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/rand_from_rand_best_5' --expert-type rand_from_rand_best --expert-args '{"max_k": 5}'

echo ">> Mandril with random from best random k=3:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/rand_from_rand_best_3' --expert-type rand_from_rand_best --expert-args '{"max_k": 3}'

echo ">> Mandril with random from best random k=2:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/rand_from_rand_best_2' --expert-type rand_from_rand_best --expert-args '{"max_k": 2}'

echo ">> Mandril with random from best random k=4:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/rand_from_rand_best_4' --expert-type rand_from_rand_best --expert-args '{"max_k": 4}'

echo "---------------------------------------------"

echo ">> Mandril with perfect expert:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/perfect' --expert-type perfect --expert-args '{}'

echo "---------------------------------------------"

echo ">> Mandril with random from best 2:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/rand_from_2_best' --expert-type rand_from_k_best --expert-args '{"k": 2}'

echo ">> Mandril with random from best 3:"
python3 mandril/run_mandril.py --output-folder 'banditk5n10/rand_from_3_best' --expert-type rand_from_k_best --expert-args '{"k": 3}'

echo "---------------------------------------------"
