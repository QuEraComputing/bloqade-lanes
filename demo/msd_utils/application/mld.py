"""Legacy MLD helpers removed for the notebook-focused prototype."""

# TODO: Not sure if this helper function is the best abstraction that we want.
# Is the logic here (a) really reused across train_mld_decoder_pair and
# train_mld_decoder_pair_from_task, and (b) is there a cleaner way to write/do
# we need train_mld_decoder_pair_from_task? Maybe the task itself should handle
# the batching/streaming logic. -- see below comment about implementing like a
# DataLoader class.
# TODO: a better fix would to make this independent of a "Task" object and to
# implement some kind of batched dataloader. however, that might be too
# complicated for the first iteration of stdlibs
# TODO: ideally, we don't hardcode to just single qubit logical tomography; but this would require additional refactoring
# Plus, I think our current fidelity path is coupled to single qubit (it's bloch-sphere based).. not sure how it would
# work with multi-qubit logical tomography
# TODO: continue reading here, 4/21 11:56 AM
# TODO: ideally, estimating the scores from the tasks is really a batching
# problem, which should be solved at the dataloader level, NOT by creating an
# extra function.
# NOTE: this function is really just putting different pieces/data together.

__all__: list[str] = []
