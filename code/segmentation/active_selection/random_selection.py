import numpy as np


class RandomSelector:
    def select_next_batch(self, trainer, active_set, selection_count, selection_iter):
        scores = []
        for i in range(len(active_set.pool_dataset.im_idx)):
            scores.append(np.random.random())
        selected_samples = list(zip(*sorted(zip(scores, active_set.pool_dataset.im_idx),
                                            key=lambda x: x[0], reverse=True)))[1][:selection_count]
        active_set.expand_training_set(selected_samples)
