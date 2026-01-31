"""
Custom trainset and prediction types (Surprise-compatible interface).

Replaces scikit-surprise dependency with minimal custom implementation
to support NumPy 2.x and avoid dependency conflicts.
"""

from collections import namedtuple
from typing import List, Tuple, Iterator


# Prediction: (uid, iid, r_ui, est, details) - compatible with Surprise format
# Evaluation code expects pred[0]=uid, pred[1]=iid, pred[2]=true_r, pred[3]=est
Prediction = namedtuple('Prediction', ['uid', 'iid', 'r_ui', 'est', 'details'])


class SimpleTrainset:
    """
    Minimal Trainset-compatible interface for GraphSAGE training.
    
    Mimics Surprise Trainset: all_ratings(), to_raw_uid(), to_raw_iid(),
    n_ratings, n_users, n_items.
    """

    def __init__(self, ratings_list: List[Tuple[str, str, float]]):
        """
        Args:
            ratings_list: List of (user_id, item_id, rating) tuples
        """
        self.ratings = ratings_list
        unique_users = sorted(set(r[0] for r in ratings_list))
        unique_items = sorted(set(r[1] for r in ratings_list))
        self._uid_to_inner = {uid: i for i, uid in enumerate(unique_users)}
        self._iid_to_inner = {iid: i for i, iid in enumerate(unique_items)}
        self._inner_to_uid = {i: uid for uid, i in self._uid_to_inner.items()}
        self._inner_to_iid = {i: iid for iid, i in self._iid_to_inner.items()}

    def all_ratings(self) -> Iterator[Tuple[int, int, float]]:
        """Yield (inner_uid, inner_iid, rating) for each rating."""
        for uid, iid, r in self.ratings:
            if uid in self._uid_to_inner and iid in self._iid_to_inner:
                yield self._uid_to_inner[uid], self._iid_to_inner[iid], r

    def to_raw_uid(self, inner_uid: int) -> str:
        """Map inner user index to raw user ID."""
        return self._inner_to_uid[inner_uid]

    def to_raw_iid(self, inner_iid: int) -> str:
        """Map inner item index to raw item ID."""
        return self._inner_to_iid[inner_iid]

    @property
    def n_ratings(self) -> int:
        """Number of ratings in training set."""
        return len(self.ratings)

    @property
    def n_users(self) -> int:
        """Number of users in training set."""
        return len(self._uid_to_inner)

    @property
    def n_items(self) -> int:
        """Number of items in training set."""
        return len(self._iid_to_inner)
