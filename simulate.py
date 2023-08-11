"""Optimisations that could be done:

fast pathfinding
fast heatmaps"""

import math

import numpy as np

import dev_helper
import gamelib
import evaluate
import copy
import itertools


NONE, HORIZ, VERT, BOTH = [0b00, 0b01, 0b10, 0b11]

class MobileUnitWrapper:
    by_unit = {}
    update_paths = False

    def __init__(self, state: gamelib.GameState, live_map: gamelib.GameMap, structures, unit=None, count=1):
        self.unit = unit
        self.target_edge = state.get_target_edge((unit.x, unit.y))
        self.target_path = state.find_path_to_edge((unit.x, unit.y), self.target_edge)
        self.steps_on_path = 0
        self.frames_until_move = 0
        self.lifetime = 0
        self.shield = self.predict_shielding(live_map, structures)
        unit.health += self.shield
        MobileUnitWrapper.by_unit[unit] = self
        # Simulating multiple units
        self.count = count

    def predict_shielding(self, live_map, structures, shield_name='EF'):
        """Predict shielding attained by the current path.
        Assume path does not change and shielding granted immediately.
        Assume also shields are all activated by first 2/3 of initial path.
        """
        # This is horrifying but efficient. Although needed not be optimised.
        return sum(
            unit.shieldPerUnit for unit in structures
            if unit.unit_type == shield_name
            and any(live_map.distance_between_locations(self.target_path[i], (unit.x, unit.y)) < unit.shieldRange
                    for i in range(0, len(self.target_path) * 2 // 3))
        )

    def on_damage(self):
        if self.unit.health <= 0 and self.count > 1:
            self.count -= 1
            self.unit.health = self.unit.max_health + self.shield

def initialise(_globals):
    global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, UNIT_TYPE_EVALUATIONS
    WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR = _globals

    UNIT_TYPE_EVALUATIONS = {
        SCOUT: evaluate.ScoutEvaluation,
        DEMOLISHER: evaluate.DemolisherEvaluation,
        INTERCEPTOR: evaluate.InterceptorEvaluation
    }

def unit_self_destruct(state: gamelib.GameState, live_map: gamelib.GameMap, evaluation: evaluate.Evaluation,
                       destruct_unit: gamelib.GameUnit, player_index=0):
    damage = destruct_unit.max_health * MobileUnitWrapper.by_unit[destruct_unit].count
    for i in range(9):
        if live_map.in_arena_bounds(position := (destruct_unit.x + i//3 - 1, destruct_unit.y + i%3 - 1)):
            units = live_map[position]
            for target in units:
                if target.player_index != player_index:
                    evaluation.damage_dealt += min(target.health, damage)
                    target.health -= damage


def run_step(state: gamelib.GameState, live_map: gamelib.GameMap, evaluation: evaluate.Evaluation, mobile_units,
             structures, scout_calculation_matrix=None, turret_arrays=None, turret_heatmaps=None, player_index=0,
             frame_number=0):
    unit: gamelib.GameUnit
    # ======== Each Support grants a Shield to any new friendly Mobile units that have entered its range. ========
    # this was approximated in the wrapper.

    # ======== Each unit attempts to move. Units which have nowhere to move deal self destruct damage, ========
    #          and their health becomes 0. See ‘Patching‘ in advanced info
    for unit in mobile_units:
        wrapper = MobileUnitWrapper.by_unit[unit]
        # we will only recalculate unit paths if a structure unit was destroyed
        # which sets the update_paths flag
        if MobileUnitWrapper.update_paths:
            wrapper.steps_on_path = 0
            wrapper.target_path = state.find_path_to_edge((unit.x, unit.y), wrapper.target_edge)
        # advance unit state by one frame
        wrapper.lifetime += 1
        if frame_number % round(1/unit.speed) == 0:
            # take one step along the path (list of locations)
            wrapper.steps_on_path += 1

            # if we reached the end of path we are either at the scoring edge or planned self destruct location
            if wrapper.steps_on_path >= len(wrapper.target_path):
                # Score units
                if [unit.x, unit.y] in live_map.get_edge_locations(wrapper.target_edge):
                    evaluation.points_scored += MobileUnitWrapper.by_unit[unit].count
                    MobileUnitWrapper.by_unit[unit].count = 0
                elif wrapper.lifetime >= 5:  # Self-destruct if moved at least 5 spaces
                    unit_self_destruct(state, live_map, evaluation, unit)

                # Not directly stated in the docs but units that have scored or self destructed do not attack
                # or receive attacks. We do this by killing the unit and setting its damage to 0.
                unit.health = unit.damage_i = unit.damage_f = 0
            else:
                # we move the unit from the past tile to the new tile, updating both the map and the unit object
                live_map[(unit.x, unit.y)].remove(unit)
                unit.x, unit.y = wrapper.target_path[wrapper.steps_on_path]
                live_map[(unit.x, unit.y)].append(unit)

            # we compare to the scoutmap
            if scout_calculation_matrix is not None:
                # we've been supplied a scoutmap. This is an array of 0, 1, 2, 3s
                new_x, new_y = wrapper.target_path[wrapper.steps_on_path]
                dx, dy = abs(new_x - unit.x), abs(new_y - unit.y)
                possible_vals = [0, 1]
                if dx in possible_vals and dy == 1 - dx:
                    # all is going well - now we know what the direction is!
                    favoured_action = (dy << 1) | dx # using bitwise ops to get favoured action
                    previously_calculated = scout_calculation_matrix[unit.x, unit.y] & favoured_action
                    if previously_calculated != 0:
                        # we've calculated this before - we can stop!
                        evaluation.truncated = True
                        break
                    else:
                        # we haven't calculated this before. let's update the scoutmap.
                        scout_calculation_matrix[unit.x, unit.y] = scout_calculation_matrix[unit.x, unit.y] | favoured_action
                else:
                    # something's gone wrong. hmmm.
                    gamelib.debug_write(f'{dx=}, {dy=}')
                    raise Exception('Impossible whoopsie with discontinuous simulated scout path:\n'
                                    + str(scout_calculation_matrix))


    # ======== All units attack. See ‘Targeting’ in advanced info ========
    # the mobile units tell the enemy structures which mobile units they have in range

    for turret in (*turret_arrays[0], *turret_arrays[1]):
        turret.current_target = None

    for unit in mobile_units:
        # we get the turrets in range from the heatmap as a binary index
        # recall heatmap is turrets of the other player attacking the tile
        turret_binary = turret_heatmaps[unit.player_index][unit.x, unit.y]

        # from the heatmap get the list of turrets that are attacking the unit
        for i, turret in enumerate(turret_arrays[1 - unit.player_index]):
            # for each turret, check if the bit is set
            if turret_binary & ( 1 << i ) != 0:
                # the bit isn't 0 (i.e is 1)
                # do the attack logic here
                # TODO: decide between the last unit and current unit. this code just picks the last unit sp.
                turret.current_target = unit

    # the structures decide which units to damage
    for turret in (*turret_arrays[0], *turret_arrays[1]):
        target = turret.current_target
        if target:
            target.health -= turret.damage_i
            MobileUnitWrapper.by_unit[target].on_damage()

    all_units = [*mobile_units, *structures]

    # the mobile units deal damage to structures and units
    for unit in mobile_units:
        target = None
        # Repeat attacks for each unit in the stack, re-targeting if initial target destroyed
        for i in range(MobileUnitWrapper.by_unit[unit].count):
            if target is None or target.health <= 0:
                target = state.get_target_alive_(unit, all_units)
            if target is None:
                continue

            if target.stationary:
                evaluation.damage_dealt += min(target.health, unit.damage_f)
                target.health -= unit.damage_f
            else:
                evaluation.damage_dealt += min(target.health, unit.damage_i)
                target.health -= unit.damage_i
                MobileUnitWrapper.by_unit[target].on_damage()

    # ======== Units that were reduced below 0 health by self destructing or taking damage are removed. ========
    # mark units for deletion: loop through all mobile units and structures
    for unit in itertools.chain(mobile_units, structures):
        if unit.health <= 0:
            unit.pending_removal = True
            # also add pending removal in the live map

    # delete mobile units - we need to use len
    num_deleted = 0
    for i in range(len(mobile_units)):
        if mobile_units[i - num_deleted].pending_removal:
            unit = mobile_units.pop(i - num_deleted)
            if unit in (units := live_map[(unit.x, unit.y)]):
                live_map[(unit.x, unit.y)].remove(unit)
            else:
                raise ValueError(f'Mobile units list has become desynchronised from game map!\n{(unit, units)}')
            num_deleted += 1

    # delete structures
    num_deleted = 0
    for i in range(len(structures)):
        if structures[i - num_deleted].pending_removal:
            unit = structures.pop(i - num_deleted)

            units = live_map[(unit.x, unit.y)]
            if unit in units:
                live_map[(unit.x, unit.y)].remove(unit)
                if unit.player_index != player_index:
                    evaluation.points_destroyed += unit.cost[unit.upgraded]
            else:
                gamelib.debug_write(f'{unit is units[0]}')
                raise ValueError(f'Structures list has become desynchronised from game map!\n{(unit, units)}')
            MobileUnitWrapper.update_paths = True
            num_deleted += 1


def get_heatmap(live_map, turrets, player_index):
    turret_heatmap = np.zeros(shape=(28, 28), dtype=np.int8)

    for turret_index, unit in enumerate(turrets):
        # only consider enemy turrets
        if player_index == unit.player_index:
            continue

        turret_value = 1 << turret_index
        attack_range_ceil = int(math.ceil(unit.attackRange))
        # we first limit ourselves to a square around the turret
        for i in range(-attack_range_ceil, attack_range_ceil + 1):
            for j in range(-attack_range_ceil, attack_range_ceil + 1):
                # then, we check that the distance is less than the attack range
                if (i ** 2) + (j ** 2) <= (unit.attackRange ** 2):
                    # we check it's in bounds
                    if live_map.in_arena_bounds((unit.x + i, unit.y + j)):
                        # if it is, we set the nth bit of the tile
                        turret_heatmap[unit.x + i, unit.y + j] += turret_value
    return turret_heatmap


def copy_internal_map(game_map: gamelib.GameMap):
    """Return a copy of the map array so that the map can be used for simulations.
    This is only necessary if copy_safe=False. Usage:

        __map = simulate.copy_internal_map(game_map)
        game_map.set_map_(__map)"""
    return copy.deepcopy(game_map.get_map_())


def make_simulation_map(state: gamelib.GameState, unit_types, locations, player_indexes=0, copy_safe=True):
    """Set up a hypothetical map and structure units for simulation.
    Warning: do not modify the state attributes or reassign the game map constants!

    Parameters:
        unit_types - list or single unit type
        locations - list or single location
        player_indexes - (optional) list or integer. If unset or after this list stops short, default to 0 (player 1).
        copy_safe - (optional) boolean. Set to True to deepcopy.

    Returns None if any location is occupied.
    """
    if not isinstance(unit_types, list):
        unit_types, locations, player_indexes = [unit_types], [locations], [player_indexes]
    if len(locations) != len(unit_types):
        raise ValueError(f'Length of locations and unit_types is not equal: {len(locations)=}, {len(unit_types)=}')

    for location in locations:
        # Verify possible locations
        if state.contains_stationary_unit(location):
            return None

    if copy_safe:
        initial_map = copy.deepcopy(state.game_map)
    else:
        initial_map = state.game_map

    # Create additional structures
    for unit_type, location, player_index in itertools.zip_longest(unit_types, locations, player_indexes, fillvalue=0):
        initial_map.add_unit(unit_type, location, player_index)

    tile_units = [initial_map[(x, y)] for x in range(initial_map.ARENA_SIZE) for y in range(initial_map.ARENA_SIZE)
                  if initial_map.in_arena_bounds((x, y))]
    structures = list(itertools.chain.from_iterable(tile_units))

    return state, initial_map, structures

def make_simulation(state: gamelib.GameState, initial_map: gamelib.GameMap, structures,
                    unit_types, locations, player_indexes=0, counts=1, copy_safe=True):
    """Set up a hypothetical map and mobile units for simulation.
    Warning: do not modify the state attributes or reassign the game map constants!

    Parameters:
    EITHER *map_parameters - returned by make_simulation_map(...)
        OR state, initial_map, structures - with structures set to None

        unit_types - list or single unit type
        locations - list or single location
        player_indexes - (optional) list or integer. If unset or after this list stops short, default to 0 (player 1).
        counts - (optional) list or integer. If unset or after this list stops short, default to 1. Minimum 1.
        copy_safe - (optional) boolean. Set to True to deepcopy.

    Returns None if any location is occupied.
    """
    if not isinstance(unit_types, list):
        unit_types, locations, player_indexes, counts = [unit_types], [locations], [player_indexes], [counts]
    if not isinstance(player_indexes, list):
        player_indexes = [player_indexes]
    if not isinstance(counts, list):
        counts = [counts]
    if len(locations) != len(unit_types):
        raise ValueError(f'Length of locations and unit_types is not equal: {len(locations)=}, {len(unit_types)=}')
    if not isinstance(initial_map, gamelib.GameMap):
        raise DeprecationWarning('''make_simulation(...): this function's arguments have been changed. Please insert:
simulate.make_simulation(game_state, game_map, None, unit_type, location, player_index, count)
                                     ^^^^^^^^^^^^^^^
Following this change, make_simulation will work as normal, but will also accept multiple units. You may also run:
    parameters_map = make_simulation_map(...) 
    parameters = make_simulation(*parameters_map, ...)
to make structure+unit simulations. Deep-copying the game_state is not required.''')

    # Reset competitions
    MobileUnitWrapper.by_unit = {}

    for location in locations:
        # Verify possible locations
        if state.contains_stationary_unit(location):
            return None

    if copy_safe is True:
        live_map = copy.deepcopy(initial_map)
    else:
        live_map = initial_map

    if structures is None or copy_safe:
        tile_units = [live_map[(x, y)] for x in range(live_map.ARENA_SIZE) for y in range(live_map.ARENA_SIZE)
                      if live_map.in_arena_bounds((x, y))]
        structures = list(itertools.chain.from_iterable(tile_units))

    for unit_type, location, player_index, counts in itertools.zip_longest(
            unit_types, locations, player_indexes, counts, fillvalue=0):  # we fill 0's and then set the 0'd counts to 1
        # Create mobile units
        live_map.add_unit(unit_type, location, player_index)
        # Create a corresponding entry in the MobileUnitWrapper class attribute which tracks additional data
        MobileUnitWrapper(state, live_map, structures, live_map[location][-1], max(1, counts))

    return state, live_map, structures, list(MobileUnitWrapper.by_unit.keys())


def simulate(state: gamelib.GameState, live_map: gamelib.GameMap, structures, mobile_units,
             evaluation_class=evaluate.Evaluation, scout_calculation_matrix=None, player_index=0):
    """Simulate the given state and map.
    Warning: do not modify the state attributes or reassign the game map constants!

    Please note if opposing units are simulated (at the same time as friendlies) they will not cause structure damage.
    We can simulate enemy attacks by setting player_index=1.
    """
    # if player_index != 0:
    #     raise NotImplementedError('Simulation from player 1 perspective not implemented. Set player_index=0.')

    # By default, evaluation is set to the appropriate evaluator of the first unit spawned in the simulation.
    if evaluation_class is evaluate.Evaluation:
        evaluation_class = UNIT_TYPE_EVALUATIONS.get(mobile_units[0].unit_type, evaluate.Evaluation)

    # Create a blank new instance of whatever evaluator we chose
    evaluation = evaluation_class()

    # create an array of the starting turrets for each player.
    # these will be used to map heatmap values to the turrets using binary indexing.
    turret_binary_access_array_0 = tuple(unit for unit in structures if unit.unit_type == TURRET and unit.player_index == 0)
    # these heatmaps display the turrets attacking any tile from the given player's perspective

    turret_binary_access_array_1 = tuple(unit for unit in structures if unit.unit_type == TURRET and unit.player_index == 1)

    heatmap_0 = get_heatmap(live_map, turret_binary_access_array_1, 0)
    heatmap_1 = get_heatmap(live_map, turret_binary_access_array_0, 1)

    i = 0

    while mobile_units and not evaluation.truncated:
        gamelib.debug_write(f'Eval {evaluation.value} ({evaluation.__class__}) {i = } {mobile_units=}')
        if i == 0 or i % 5 == 0: # ((i < 5 or i % 5 == 0) and evaluation.value > 0):
            dev_helper.print_map(live_map, gamelib.debug_write)

        MobileUnitWrapper.update_paths = False
        run_step(state, live_map, evaluation, mobile_units, structures, scout_calculation_matrix,
                 (turret_binary_access_array_0, turret_binary_access_array_1), (heatmap_0, heatmap_1), player_index, i)
        i += 1

    return evaluation


