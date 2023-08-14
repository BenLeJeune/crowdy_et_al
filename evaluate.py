from dataclasses import dataclass
import numpy as np


@dataclass
class Evaluation:
    """Higher Evaluation.value indicates better scenarios and vice versa."""
    structures_damaged = []
    points_destroyed = 0  # enemy structure points destroyed
    damage_dealt = 0  # damage dealt, ignoring overflow!
    points_scored = 0
    truncated = False  # this is true if we stop calculating due to already having calculated this path
    length = 0  # how long this runs for
    self_destruct_order = []

    def __str__(self):
        return f'<{__class__} {self.value}>'

    @property
    def value(self):
        return self.points_scored


class ScoutEvaluation(Evaluation):
    @property
    def value(self):
        return super().value + self.damage_dealt/320 + self.points_destroyed/5


class DemolisherEvaluation(Evaluation):
    @property
    def value(self):
        return (super().value*2 + self.damage_dealt/64 + self.points_destroyed) / 2


class InterceptorEvaluation(Evaluation):
    @property
    def value(self):
        return super().value
