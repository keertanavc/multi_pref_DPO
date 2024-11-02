from striprtf.striprtf import rtf_to_text

def read_file(filename):
    with open(filename, 'r') as infile:
        content = infile.read()
        text = rtf_to_text(content)
    return text.split("\n")

lines = read_file("dpo_policy_0.01_0.1.rtf")
print(len(lines))

def calculate_accuracy(lines):
    policy = []
    rewards = []
    for i in range(len(lines)):
        if i % 4 == 1:
            rewards.append(lines[i][-1])
        if i % 4 == 3:
            policy.append(lines[i][-1])
    count = 0
    print(policy)
    print(rewards)
    for i in range(len(rewards)):
        if rewards[i] == policy[i]:
            count += 1
    return count / len(rewards)

files = ["dpo_policy_0.01_0.1.rtf", "dpo_policy_0.01_0.5.rtf", "dpo_policy_0.01_1.rtf", "dpo_policy_0.01_5.rtf", "dpo_policy_0.05_0.1.rtf", "dpo_policy_0.05_0.5.rtf", "dpo_policy_0.05_1.rtf", "dpo_policy_0.05_5.rtf"]
for file in files:
    lines = read_file(file)
    print(file, calculate_accuracy(lines))
