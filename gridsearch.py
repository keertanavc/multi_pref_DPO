import json
import subprocess

betas = [0.1, 0.5, 1, 5]
learning_rates = [0.01, 0.05, 0.01]

for beta in betas:
    for learning_rate in learning_rates:
        with open('config.json', 'r') as f:
            config = json.load(f)
        config["BETA"] = beta
        config["LEARNING_RATE"] = learning_rate
        with open('config.json', 'w') as f:
            json.dump(config, f)
        subprocess.run(["python", "-u", "trainer.py"])

