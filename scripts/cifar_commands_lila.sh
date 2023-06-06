classification_image.py --config configs/cifar_lila.yaml --dataset cifar10 --seed 1 --approx kernel --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar10 --seed 2 --approx kernel --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar10 --seed 3 --approx kernel --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar10 --seed 1 --approx kron --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar10 --seed 2 --approx kron --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar10 --seed 3 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar10 --seed 1 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar10 --seed 2 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar10 --seed 3 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar10 --seed 1 --approx kron --random_flip --single_output --independent_outputs --partial_batch_size 120 --n_hypersteps 45
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar10 --seed 2 --approx kron --random_flip --single_output --independent_outputs --partial_batch_size 120 --n_hypersteps 45
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar10 --seed 3 --approx kron --random_flip --single_output --independent_outputs --partial_batch_size 120 --n_hypersteps 45
classification_image.py --config configs/cifar_lila.yaml --dataset cifar100 --seed 1 --approx kernel --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar100 --seed 2 --approx kernel --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar100 --seed 3 --approx kernel --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar100 --seed 1 --approx kron --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar100 --seed 2 --approx kron --random_flip
classification_image.py --config configs/cifar_lila.yaml --dataset cifar100 --seed 3 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar100 --seed 1 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar100 --seed 2 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar100 --seed 3 --approx kron --random_flip
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar100 --seed 1 --approx kron --random_flip --single_output --independent_outputs --partial_batch_size 120 --n_hypersteps 45
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar100 --seed 2 --approx kron --random_flip --single_output --independent_outputs --partial_batch_size 120 --n_hypersteps 45
classification_image.py --config configs/cifar_lila_baseline.yaml --dataset cifar100 --seed 3 --approx kron --random_flip --single_output --independent_outputs --partial_batch_size 120 --n_hypersteps 45