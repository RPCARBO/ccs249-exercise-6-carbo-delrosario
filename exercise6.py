import math
from collections import defaultdict

# Training data
training_sentences = [
    [("The", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("A", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
    [("The", "DET"), ("dog", "NOUN"), ("sleeps", "VERB")],
    [("My", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB"), ("loudly", "ADV")],
    [("Your", "DET"), ("cat", "NOUN"), ("runs", "VERB")],
    [("The", "DET"), ("bird", "NOUN"), ("sings", "VERB"), ("sweetly", "ADV")],
    [("A", "DET"), ("bird", "NOUN"), ("chirps", "VERB")]
]

# Estimate transition probabilities
transition_counts = defaultdict(lambda: defaultdict(int))
state_counts = defaultdict(int)

for sentence in training_sentences:
    prev_tag = "<START>"
    state_counts[prev_tag] += 1
    for word, tag in sentence:
        transition_counts[prev_tag][tag] += 1
        state_counts[tag] += 1
        prev_tag = tag
    # end of sentence transition
    transition_counts[prev_tag]["<END>"] += 1
    state_counts["<END>"] += 1

# Normalize to get probabilities
transition_probs = {}
for prev, nexts in transition_counts.items():
    total = sum(nexts.values())
    transition_probs[prev] = {tag: cnt / total for tag, cnt in nexts.items()}

# Estimate emission probabilities
emission_counts = defaultdict(lambda: defaultdict(int))
tag_totals = defaultdict(int)

for sentence in training_sentences:
    for word, tag in sentence:
        emission_counts[tag][word] += 1
        tag_totals[tag] += 1

emission_probs = {}
for tag, words in emission_counts.items():
    total = tag_totals[tag]
    emission_probs[tag] = {w: c / total for w, c in words.items()}

# Viterbi decoding function
def viterbi(observation, transition_probs, emission_probs):
    states = list(emission_probs.keys())
    V = [{}]
    path = {}

    # Initialization for t = 0
    for st in states:
        tp = transition_probs["<START>"].get(st, 0)
        ep = emission_probs[st].get(observation[0], 0)
        if tp > 0 and ep > 0:
            V[0][st] = math.log(tp) + math.log(ep)
            path[st] = [st]

    # Run through the sentence
    for t in range(1, len(observation)):
        V.append({})
        new_path = {}

        for st in states:
            ep = emission_probs[st].get(observation[t], 0)
            if ep == 0:
                continue
            best_prev, best_score = None, float('-inf')
            for prev_st, prev_score in V[t-1].items():
                tp = transition_probs[prev_st].get(st, 0)
                if tp > 0:
                    score = prev_score + math.log(tp) + math.log(ep)
                    if score > best_score:
                        best_prev, best_score = prev_st, score
            if best_prev is not None:
                V[t][st] = best_score
                new_path[st] = path[best_prev] + [st]

        path = new_path

    # Termination
    best_final, best_score = None, float('-inf')
    for st, score in V[-1].items():
        tp = transition_probs[st].get("<END>", 0)
        if tp > 0:
            final_score = score + math.log(tp)
            if final_score > best_score:
                best_final, best_score = st, final_score

    if best_final is None:
        return [], 0.0
    return path[best_final], math.exp(best_score)

#Test sentences
tests = [
    ["The", "cat", "meows"],
    ["My", "dog", "barks", "loudly"],
    ["A", "cat", "chirps", "fast"], #example
]
if __name__ == "__main__":
    for obs in tests:
        tags, prob = viterbi(obs, transition_probs, emission_probs)
        print(f"Input Sentence : {' '.join(obs)}")
        print(f"Predicted Tags : {tags}")