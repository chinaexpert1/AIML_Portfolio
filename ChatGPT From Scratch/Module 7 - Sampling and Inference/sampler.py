import torch
import numpy as np

'''
Class implementing a sampler for inference on a model. Given the raw logits from
an LLM model, this will sample the next token id.
'''
class Sampler:

	def __init__(
		self,
		top_k=None,
		top_p=None,
		frequency_penalty=1.0,
		presence_penalty=1.0
	):
		'''
		param top_k : (None or int)
			If specified, only the top k logits should be used during sampling
			If this is specified, top_p should be None

		param top_p : (None or int)
			If specified, only the logits representing the probability mass p should be used during sampling.
			Or, if the top token has mass greater than p, the top token is returned.
			If this is specified, top_k should be None

		If top_k and top_p are both None, sample from the whole distribution (same as top_p=1.0)

		param frequency_penalty : (float)
			A penalty applied to tokens that have previously occured in the sequence. Along with
			presence_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.

		param presence_penalty : (float)
			A penalty applied to tokens IF they have previously occured in the sequence. Along with
			frequency_penalty, this adjusts the per-token softmax temperature.
			A penalty of 1.0 indicates no change from normal softmax.
		'''
		assert top_k is None or top_p is None, "Cannot specify both top_k and top_p"
		
		self.top_k = top_k
		self.top_p = top_p
		self.frequency_penalty = frequency_penalty
		self.presence_penalty = presence_penalty


	def make_token_distribution(self, raw_unsorted_logits, previous_token_ids):
		'''
		param: raw_unsorted_logits (float numpy array)
			A one dimensional list of logits representing an unnormalized distribution over next tokens
			These are "unsorted" in the sense that their order aligns with vocabulary order, not with probability.

		param: previous_token_ids (int numpy array)
			A one dimensional list of ids representing the previous tokens, for calculating repetition penalties.

		returns:
			- the final probability distribution that this token is sampled from
			It should be returned back to token-id order (unsorted order) before returning.
		'''

		# Step 1: Initialize temperature array (per-token temperature adjustment)
		temps = np.ones(len(raw_unsorted_logits))
		
		# Step 2: Apply repetition penalties
		# Count frequency of each token in previous sequence
		token_counts = {}
		for token_id in previous_token_ids:
			token_counts[token_id] = token_counts.get(token_id, 0) + 1
		
		# Apply penalties to temperatures
		for token_id, count in token_counts.items():
			# Presence penalty: applied once if token appeared
			# Frequency penalty: applied based on count
			temps[token_id] = self.presence_penalty * (self.frequency_penalty ** count)
		
		# Step 3: Adjust logits to be non-negative for numerical stability
		logits = raw_unsorted_logits.copy()
		logits = logits - np.min(logits)
		
		# Step 4: Apply temperature-adjusted softmax
		# Divide logits by temperature (higher temp = flatter distribution)
		adjusted_logits = logits / temps
		
		# Compute softmax
		exp_logits = np.exp(adjusted_logits - np.max(adjusted_logits))  # subtract max for stability
		probs = exp_logits / np.sum(exp_logits)
		
		# Step 5: Sort the distribution (keep track of indices)
		sorted_indices = np.argsort(probs)[::-1]  # descending order
		sorted_probs = probs[sorted_indices]
		
		# Step 6: Apply top-k or top-p filtering
		if self.top_k is not None:
			# Keep only top k tokens
			cutoff_idx = min(self.top_k, len(sorted_probs))
			filtered_probs = np.zeros_like(sorted_probs)
			filtered_probs[:cutoff_idx] = sorted_probs[:cutoff_idx]
		elif self.top_p is not None:
			# Keep tokens until cumulative probability exceeds p
			cumsum = np.cumsum(sorted_probs)
			cutoff_idx = np.searchsorted(cumsum, self.top_p) + 1
			# Ensure at least one token is kept
			cutoff_idx = max(1, cutoff_idx)
			filtered_probs = np.zeros_like(sorted_probs)
			filtered_probs[:cutoff_idx] = sorted_probs[:cutoff_idx]
		else:
			# No filtering, use all probabilities
			filtered_probs = sorted_probs
		
		# Step 7: Renormalize
		prob_sum = np.sum(filtered_probs)
		if prob_sum > 0:
			filtered_probs = filtered_probs / prob_sum
		else:
			# Fallback: uniform distribution over all tokens
			filtered_probs = np.ones_like(filtered_probs) / len(filtered_probs)
		
		# Step 8: Revert back to original ordering
		undo_indices = np.argsort(sorted_indices)
		final_probs = filtered_probs[undo_indices]
		
		return final_probs


	#==========================
	# for actually sampling the distribution
	def sample_one_token(self, raw_unsorted_logits, previous_token_ids):
		probs = self.make_token_distribution(raw_unsorted_logits, previous_token_ids)
		return np.random.choice(np.arange(len(raw_unsorted_logits)), p=probs)

	# this is also callable
	def __call__(self, raw_unsorted_logits, previous_token_ids):
		return self.sample_one_token(raw_unsorted_logits, previous_token_ids)




if __name__ == "__main__":
    
    # example of using this with dummy data, keeping everything in token ids

    sampler = Sampler(top_p=0.8, frequency_penalty=1.1, presence_penalty=1.1)

    sequence = [1,2,3,4,5]

    for i in range(10):
    	# fake logits for a vocab of size 500
    	logits = np.random.randn(500)

    	# get next token in sequence
        next_token = sampler(logits, sequence)
        sequence.append(next_token)

    print(sequence)