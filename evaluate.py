from dataclasses import dataclass


@dataclass
class Evaluation:
    """Higher Evaluation.value indicates better scenarios and vice versa."""
    structures_damaged = []
    points_destroyed = 0  # enemy structure points destroyed
    damage_dealt = 0  # damage dealt, ignoring overflow!
    points_scored = 0

    def __str__(self):
        return f'<{__class__} {self.value}>'

    @property
    def value(self):
        return self.points_scored


class ScoutEvaluation(Evaluation):
    @property
    def value(self):
        return super().value + self.damage_dealt/60 + self.points_destroyed/2


class DemolisherEvaluation(Evaluation):
    @property
    def value(self):
        return super().value/2 + self.damage_dealt/60 + self.points_destroyed/2


class InterceptorEvaluation(Evaluation):
    @property
    def value(self):
        return super().value
