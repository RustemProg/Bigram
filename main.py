import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

with open("names.txt", "r") as f:
    names = f.read().splitlines()

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

corpus = nltk.Text(names)
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(corpus)
finder.apply_freq_filter(2)
finder.apply_word_filter(lambda w: len(w) < 3)
bigrams = finder.nbest(bigram_measures.pmi, 100)

input_size = 27  
hidden_size = 128
output_size = 27  
learning_rate = 0.01
all_chars = set(corpus)
char_to_index = {}

for i, char in enumerate(all_chars):
    char_to_index[char] = i

def name_to_tensor(name):
    tensor = torch.zeros(len(name), 1, input_size)
    for i, char in enumerate(name):
        if char in char_to_index:
            tensor[i][0][char_to_index[char]] = 1
    return tensor

class NameGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_to_output = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

model = NameGenerator(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(name_tensor):
    hidden = model.init_hidden()
    model.zero_grad()
    
    loss = 0
    
    for i in range(name_tensor.size()[0] - 1):
        input = name_tensor[i]
        output, hidden = model(input, hidden)
        loss += criterion(output, name_tensor[i+1])
        
    loss.backward()
    optimizer.step()
    
    return output, loss.item() / (name_tensor.size()[0] - 1)


def generate_name():
    if not bigrams:
        return ""
    bigram = random.choice(bigrams)
    name = bigram[0] + bigram[1]
    input_tensor = name_to_tensor(name)
    hidden = model.init_hidden()
    while len(name) < 5:
        output, hidden = model(input_tensor[0], hidden)
        topv, topi = output.topk(1)
        if topi == 26:
            break
        else:
            letter = chr(topi.item() + 97)
            name += letter
            input_tensor = name_to_tensor(name[-2:])
    
    return name

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