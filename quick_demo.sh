# Download the CIFAR-10 dataset
bash resource/download_cifar10.sh

# Generate poison data to poison_data/badnets10
python attack/data_poison.py --attack badnets --ratio 0.1

# Train the model
python attack/train_attack.py --poison_data badnets10

# Simple Defense by Fne-tuning on a few clean data
python defense/ft.py --poison_data badnets10 --clean_ratio 0.1

