import pandas as pd
import random
import matplotlib.pyplot as plt

with open('names.txt', 'r') as f:
    names = [line.strip() for line in f.readlines()]

bigram_counts = {}
for name in names:
    name = '^' + name + '$'
    for i in range(len(name) - 1):
        bigram = name[i:i+2]
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1

total_bigrams = sum(bigram_counts.values())
bigram_probabilities = {bigram: count/total_bigrams for bigram, count in bigram_counts.items()}

def generate_name():
    name = '^'
    current_bigram = random.choice(list(bigram_probabilities.keys()))
    while current_bigram[1] != '$':
        name += current_bigram[1]
        possible_bigrams = [bigram for bigram in bigram_probabilities.keys() if bigram[0] == current_bigram[1]]
        current_bigram = random.choices(possible_bigrams, [bigram_probabilities[bigram] for bigram in possible_bigrams])[0]
    return name[1:-1]

def print_bigram_probabilities():
    for bigram, probability in sorted(bigram_probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"{bigram}: {probability:.4f}")

print("Top bigram probabilities:")
print_bigram_probabilities()

print("\nGenerated name:")
print(generate_name())

freq_table = pd.DataFrame(list(bigram_counts.items()), columns=['bigram', 'count'])
freq_table['frequency'] = freq_table['count'] / total_bigrams
freq_table = freq_table.sort_values(by='count', ascending=False)
freq_table.to_excel('bigram_counts.xlsx', index=False)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

ax1.pie(freq_table.head(10)['count'], labels=freq_table.head(10)['bigram'])
ax1.set_title('Top 10 Bigram Frequencies in Names')
ax2.bar(freq_table['bigram'], freq_table['count'])
ax2.set_xlabel('Bigram')
ax2.set_ylabel('Count')
ax2.set_title('Frequency of Bigrams in Names')
ax2.set_xticks([])
plt.show()