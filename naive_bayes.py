from collections import Counter
import numpy as np

training_data = {'e':[], 'j':[], 's':[]}
for num in range(10):
    for lang in ['e', 'j', 's']:
        with open('./languageID/{}{}.txt'.format(lang, num), 'r') as f:
            data = [*f.read()]
            if lang == 'e':
                training_data['e'] += data

            elif lang == 'j':
                training_data['j'] += data
                
            else:
                training_data['s'] += data

vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
char, counts = {}, {}
cond_prob = {'e':[], 'j':[], 's':[]}
for lang in ['e', 'j', 's']:
    char[lang] = list(Counter(training_data[lang]).keys())
    idx = char[lang].index('\n')
    del char[lang][idx]
    counts[lang] = list(Counter(training_data[lang]).values())
    del counts[lang][idx]
    for c in vocab:
        if c not in char[lang]:
            char[lang].append(c)
            counts[lang].append(0)
    print(len(char[lang]))

    


    
#q2

for lang in ['e', 'j', 's']:
    temp = []
    for i in range(len(char[lang])):
        temp.append(round((counts[lang][i]+0.5)/(sum(counts[lang])+27*0.5), 6))
    cond_prob[lang] = dict(zip(char[lang], temp))

print(cond_prob)

with open('./languageID/e10.txt', 'r') as f:
    data = [*f.read()]
    # print(data)
    pe, pj, ps = 0, 0, 0
    for c in data:
        if c != '\n':
            pe += np.log(cond_prob['e'][c])
            pj += np.log(cond_prob['j'][c])
            ps += np.log(cond_prob['s'][c])
print('e, j, s', pe, pj, ps)


