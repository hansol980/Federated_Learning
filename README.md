# Federated_Learning

1. FedAvg -> IID 상황으로 실험
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedAvg\(IID\).py
Device: cpu
Federated Learning Start: 10 rounds, 5 clients
Round 1/10 - Global Accuracy: 91.29%, | Time: 14.67s
Round 2/10 - Global Accuracy: 94.34%, | Time: 13.97s
Round 3/10 - Global Accuracy: 95.78%, | Time: 13.89s
Round 4/10 - Global Accuracy: 96.72%, | Time: 14.06s
Round 5/10 - Global Accuracy: 97.27%, | Time: 14.66s
Round 6/10 - Global Accuracy: 97.39%, | Time: 15.48s
Round 7/10 - Global Accuracy: 97.69%, | Time: 15.46s
Round 8/10 - Global Accuracy: 97.91%, | Time: 16.25s
Round 9/10 - Global Accuracy: 97.99%, | Time: 18.02s
Round 10/10 - Global Accuracy: 98.07%, | Time: 15.35s
Total Training Time: 151.89s

2. FedAvg -> Non_IID 상황으로 실험
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedAvg\(Non_IID\).py
Device: cpu
Federated Learning Start: 10 rounds, 5 clients
Round 1/10 - Global Accuracy: 39.54%, | Time: 14.19s
Round 2/10 - Global Accuracy: 56.19%, | Time: 13.91s
Round 3/10 - Global Accuracy: 70.28%, | Time: 14.02s
Round 4/10 - Global Accuracy: 74.83%, | Time: 14.51s
Round 5/10 - Global Accuracy: 77.49%, | Time: 14.67s
Round 6/10 - Global Accuracy: 80.82%, | Time: 14.35s
Round 7/10 - Global Accuracy: 82.72%, | Time: 15.85s
Round 8/10 - Global Accuracy: 84.41%, | Time: 15.37s
Round 9/10 - Global Accuracy: 87.91%, | Time: 15.35s
Round 10/10 - Global Accuracy: 88.63%, | Time: 15.39s
Total Training Time: 147.61s


3. FedProx -> Non_IID 상황
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedProx.py 
Device: cpu
Federated Learning (FedProx) Start: 10 rounds, 5 clients
Round 1/10 - Global Accuracy: 37.75%, | Time: 15.53s
Round 2/10 - Global Accuracy: 47.89%, | Time: 15.00s
Round 3/10 - Global Accuracy: 55.79%, | Time: 14.89s
Round 4/10 - Global Accuracy: 63.75%, | Time: 15.61s
Round 5/10 - Global Accuracy: 69.12%, | Time: 14.87s
Round 6/10 - Global Accuracy: 73.63%, | Time: 14.82s
Round 7/10 - Global Accuracy: 77.05%, | Time: 15.19s
Round 8/10 - Global Accuracy: 79.04%, | Time: 18.53s
Round 9/10 - Global Accuracy: 80.68%, | Time: 17.28s
Round 10/10 - Global Accuracy: 80.89%, | Time: 18.96s
Total Training Time: 160.69s

4. Scaffold
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python Scaffold.py 
Device: cpu
Federated Learning Start: 10 rounds, 5 clients
Round 1/10 - Global Accuracy: 40.88%, | Time: 14.08s
Round 2/10 - Global Accuracy: 57.34%, | Time: 13.93s
Round 3/10 - Global Accuracy: 80.42%, | Time: 14.01s
Round 4/10 - Global Accuracy: 87.01%, | Time: 14.28s
Round 5/10 - Global Accuracy: 88.68%, | Time: 14.06s
Round 6/10 - Global Accuracy: 87.98%, | Time: 14.23s
Round 7/10 - Global Accuracy: 87.17%, | Time: 14.09s
Round 8/10 - Global Accuracy: 89.92%, | Time: 14.77s
Round 9/10 - Global Accuracy: 91.15%, | Time: 15.35s
Round 10/10 - Global Accuracy: 90.91%, | Time: 15.52s
Total Training Time: 144.34s

5. FedDyn 
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedDyn.py
Device: cpu
FedDyn Start: 10 rounds, 5 clients, alpha=0.1
Round 1/10 - Global Accuracy: 31.94%, | Time: 17.07s
Round 2/10 - Global Accuracy: 51.13%, | Time: 15.29s
Round 3/10 - Global Accuracy: 61.41%, | Time: 15.33s
Round 4/10 - Global Accuracy: 67.93%, | Time: 15.38s
Round 5/10 - Global Accuracy: 71.23%, | Time: 15.24s
Round 6/10 - Global Accuracy: 73.69%, | Time: 15.40s
Round 7/10 - Global Accuracy: 76.25%, | Time: 15.70s
Round 8/10 - Global Accuracy: 77.36%, | Time: 16.55s
Round 9/10 - Global Accuracy: 78.55%, | Time: 16.58s
Round 10/10 - Global Accuracy: 80.36%, | Time: 16.62s
Total Training Time: 159.16s