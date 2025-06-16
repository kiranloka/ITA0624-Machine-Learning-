# prompt: For a given set of training data examples stored in a .CSV file, implement and demonstrate the
# Candidate-Elimination algorithm in python to output a description of the set of all hypotheses
# consistent with the training examples

# Implement the Candidate-Elimination algorithm
def candidate_elimination(df):
    attributes = list(df.columns[:-1])
    n_attributes = len(attributes)

    # Initialize the most general and most specific hypotheses
    G = [['?' for _ in range(n_attributes)]]
    S = [['0' for _ in range(n_attributes)]]

    for i in range(len(df)):
        example = list(df.iloc[i, :-1])
        target = df.iloc[i, -1]

        if target == 'Yes':
            # Process positive examples
            # Remove inconsistent hypotheses from G
            G = [h for h in G if consistent(h, example)]

            # Update S
            for s_idx in range(len(S)):
                s = S[s_idx]
                if not consistent(s, example):
                    # Generalize s to be consistent with the positive example
                    S[s_idx] = generalize_hypothesis(s, example)
            # Remove duplicates and more general hypotheses from S
            S = remove_duplicates_and_general(S)

        else:  # target == 'No'
            # Process negative examples
            # Remove inconsistent hypotheses from S
            S = [h for h in S if consistent(h, example)]

            # Update G
            for g_idx in range(len(G)):
                g = G[g_idx]
                if consistent(g, example):
                    # Specialize g to be inconsistent with the negative example
                    G[g_idx] = specialize_hypothesis(g, example)
            # Remove duplicates and more specific hypotheses from G
            G = remove_duplicates_and_specific(G)


    return S, G

def consistent(hypothesis, example):
    """Checks if a hypothesis is consistent with an example."""
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != example[i]:
            return False
    return True

def generalize_hypothesis(hypothesis, example):
    """Generalizes a hypothesis to be consistent with a positive example."""
    new_hypothesis = hypothesis[:]
    for i in range(len(new_hypothesis)):
        if new_hypothesis[i] == '0':
            new_hypothesis[i] = example[i]
        elif new_hypothesis[i] != example[i]:
            new_hypothesis[i] = '?'
    return new_hypothesis

def specialize_hypothesis(hypothesis, example):
    """Specializes a hypothesis to be inconsistent with a negative example."""
    specialized_hypotheses = []
    for i in range(len(hypothesis)):
        if hypothesis[i] == '?':
            # Replace '?' with the attribute values from the example for all other attributes
            for attribute_value in set(df[df.columns[i]]): # Assuming df is available
                if attribute_value != example[i]:
                    new_hypothesis = hypothesis[:]
                    new_hypothesis[i] = attribute_value
                    specialized_hypotheses.append(new_hypothesis)
    return specialized_hypotheses[0] # Simplistic approach: take the first specialization

def remove_duplicates_and_general(hypotheses):
    """Removes duplicate and more general hypotheses from a list of specific hypotheses."""
    cleaned_hypotheses = []
    for h1 in hypotheses:
        is_duplicate_or_general = False
        for h2 in hypotheses:
            if h1 != h2 and is_more_general(h2, h1):
                is_duplicate_or_general = True
                break
        if not is_duplicate_or_general and h1 not in cleaned_hypotheses:
            cleaned_hypotheses.append(h1)
    return cleaned_hypotheses

def remove_duplicates_and_specific(hypotheses):
    """Removes duplicate and more specific hypotheses from a list of general hypotheses."""
    cleaned_hypotheses = []
    for h1 in hypotheses:
        is_duplicate_or_specific = False
        for h2 in hypotheses:
            if h1 != h2 and is_more_specific(h2, h1):
                is_duplicate_or_specific = True
                break
        if not is_duplicate_or_specific and h1 not in cleaned_hypotheses:
            cleaned_hypotheses.append(h1)
    return cleaned_hypotheses


def is_more_general(h1, h2):
    """Checks if hypothesis h1 is more general than hypothesis h2."""
    for i in range(len(h1)):
        if h1[i] == '?' and h2[i] != '?':
            return True
        if h1[i] != '?' and h1[i] != h2[i]:
            return False
    return False # They are the same or h2 is more general

def is_more_specific(h1, h2):
    """Checks if hypothesis h1 is more specific than hypothesis h2."""
    return is_more_general(h2, h1)

# Run the Candidate-Elimination algorithm
S_hypotheses, G_hypotheses = candidate_elimination(df)

print("\nThe set of all consistent hypotheses:")
print("Most Specific Hypotheses (S):", S_hypotheses)
print("Most General Hypotheses (G):", G_hypotheses)