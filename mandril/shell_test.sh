echo ">> Mandril with perfect expert:"
python3 mandril/test_mandril.py --output-folder 'banditk5n10/perfect' --expert-type perfect --expert-args '{}'

echo ">> Mandril with random from best 2:"
python3 mandril/test_mandril.py --output-folder 'banditk5n10/rand_from_2_best' --expert-type rand_from_k_best --expert-args '{"k": 2}'

echo ">> Mandril with random from best 3:"
python3 mandril/test_mandril.py --output-folder 'banditk5n10/rand_from_3_best' --expert-type rand_from_k_best --expert-args '{"k": 3}'