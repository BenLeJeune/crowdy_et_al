"""Optimisations that could be done:
Unit stacks! make custom unit with stacked HP
fast pathfinding"""
import dev_helper
import gamelib
import evaluate
import copy
import itertools


class MobileUnitWrapper:
    by_unit = {}
    update_paths = False

    def __init__(self, state: gamelib.GameState, map: gamelib.GameMap, structures, unit=None):
        self.unit = unit
        self.target_edge = state.get_target_edge((unit.x, unit.y))
        self.target_path = state.find_path_to_edge((unit.x, unit.y), self.target_edge)
        self.predict_shielding(map, structures)
        self.steps_on_path = 0
        self.lifetime = 0
        MobileUnitWrapper.by_unit[unit] = self

    def predict_shielding(self, map, structures, shield_name='EF'):
        """Predict shielding attained by the current path.
        Assume path does not change and shielding granted immediately.
        Assume also shields checked every 3 frames for first 3/4 of initial path.
        """
        # This is horrifying but efficient. Although needed not be optimised.
        self.unit.health += sum(
            unit.shieldPerUnit for unit in structures
            if unit.unit_type == shield_name
            and any(map.distance_between_locations(self.target_path[i], (unit.x, unit.y)) < unit.shieldRange
                    for i in range(0, len(self.target_path) * 3 // 4, 3))
        )
        #print(f'shield!!! {self.unit.health} / {self.unit.max_health}')

def initialise(_globals):
    global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR
    WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR = _globals


def unit_self_destruct(state: gamelib.GameState, map: gamelib.GameMap, evaluation: evaluate.Evaluation,
                       destruct_unit: gamelib.GameUnit):
    for i in range(9):
        if map.in_arena_bounds(position := (destruct_unit.x + i//3 - 1, destruct_unit.y + i%3 - 1)):
            units = map[position]
            for unit in units:
                unit.health -= destruct_unit.max_health


def run_step(state: gamelib.GameState, map: gamelib.GameMap, evaluation: evaluate.Evaluation,
             ordered_units, structures, player_index=0):
    unit: gamelib.GameUnit
    # Each Support grants a Shield to any new friendly Mobile units that have entered its range.

    # Each unit attempts to move. Units which have nowhere to move deal self destruct damage,
    # and their health becomes 0. See ‘Patching‘ in advanced info
    for unit in ordered_units:
        if not unit.stationary:
            wrapper = MobileUnitWrapper.by_unit[unit]
            # Only update paths if unit was destroyed
            if MobileUnitWrapper.update_paths:
                wrapper.steps_on_path = 0
                wrapper.target_path = state.find_path_to_edge((unit.x, unit.y), wrapper.target_edge)
            wrapper.steps_on_path += 1
            wrapper.lifetime += 1
            if wrapper.steps_on_path >= len(wrapper.target_path):
                if [unit.x, unit.y] in map.get_edge_locations(wrapper.target_edge):
                    evaluation.points_scored += 1
                else:
                    unit_self_destruct(state, map, evaluation, unit)
                unit.health = 0
            else:
                map[(unit.x, unit.y)].remove(unit)
                unit.x, unit.y = wrapper.target_path[wrapper.steps_on_path]
                map[(unit.x, unit.y)].append(unit)

    # All units attack. See ‘Targeting’ in advanced info
    for unit in ordered_units:
        # TODO: This can be vastly optimised by reusing the attackers list
        # or just by using a list of turret structures
        opposing = state.get_attackers((unit.x, unit.y), unit.player_index)
        for attacker in opposing:
            if state.get_target(attacker) == unit:
                unit.health -= attacker.damage_i

    for unit in ordered_units:
        target = state.get_target(unit)
        if target is None:
            continue

        if target.stationary:
            evaluation.damage_dealt += min(target.health, unit.damage_f)
            target.health -= unit.damage_f
            if target.health <= 0:
                points = target.cost[0]
                if target.upgraded:
                    if target.unit_type == TURRET:
                        points *= 2.5
                    else:
                        points *= 2
                evaluation.points_destroyed += points
        else:
            evaluation.damage_dealt += min(target.health, unit.damage_i)
            target.health -= unit.damage_i


    # Units that were reduced below 0 health by self destructing or taking damage are removed.
    for unit in ordered_units:
        if unit.health <= 0:
            unit.pending_removal = True

    for unit in structures:
        if unit.health <= 0:
            MobileUnitWrapper.update_paths = True

    r = 0
    for i, unit in enumerate(structures):
        if unit.health <= 0:
            structures.pop(i - r)
            map[(unit.x, unit.y)].remove(unit)
            r += 1

    r = 0
    for i, unit in enumerate(ordered_units):
        if unit.pending_removal:
            ordered_units.pop(i - r)
            r += 1


def make_simulation(game_state: gamelib.GameState, unit_type, location, player_index, count):
    # Verify possible locations
    if game_state.contains_stationary_unit(location):
        return None

    # Copy the game state and simulate on the copy
    state = copy.deepcopy(game_state)
    map = state.game_map

    tile_units = [map[(x, y)] for x in range(map.ARENA_SIZE) for y in range(map.ARENA_SIZE) if
                  map.in_arena_bounds((x, y))]
    structures = list(itertools.chain.from_iterable(tile_units))

    # Create units
    MobileUnitWrapper.by_unit = {}
    for _ in range(count):
        ##new_unit = gamelib.GameUnit(unit_type, map.config, player_index, None, location[0], location[1])
        ##MobileUnitWrapper(state, map, structures, new_unit)
        map.add_unit(unit_type, location, player_index)
        MobileUnitWrapper(state, map, structures, map[location][-1])

    # Create evaluation
    unit_type_match = {
        SCOUT: evaluate.ScoutEvaluation,
        DEMOLISHER: evaluate.DemolisherEvaluation,
        INTERCEPTOR: evaluate.InterceptorEvaluation
    }
    evaluation = unit_type_match.get(unit_type, evaluate.Evaluation)()
    return state, map, evaluation, structures


def simulate(state: gamelib.GameState, map: gamelib.GameMap, evaluation, structures):
    # pass simulated states!

    ordered_units = list(MobileUnitWrapper.by_unit.keys())
    #gamelib.debug_write(f'{ordered_units=}')
    i = 0
    while ordered_units:
        # if i == 0 or ((i < 5 or i % 5 == 0) and evaluation.value > 0):
        #     gamelib.debug_write(f'Eval {evaluation.value} {i = }')
        #     dev_helper.print_map(map, gamelib.debug_write)
        MobileUnitWrapper.update_paths = False
        run_step(state, map, evaluation, ordered_units, structures)
        i += 1

    return evaluation
