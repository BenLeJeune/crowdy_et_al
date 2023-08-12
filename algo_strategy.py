import copy

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import itertools

import simulate
import simulate as sim
import dev_helper

IS_DEV_ENABLED = True
IS_PROFILER_ENABLED = False

if IS_PROFILER_ENABLED:
    import cProfile as profile  # if not available, replace with 'import profile'
    import timing_helper
if not IS_DEV_ENABLED:
    dev_helper.print_map = dev_helper.print_state = lambda *args: None  # overwrite test functions
    print = lambda *args, **kwargs: None  # overwrite print as this crashes the engine

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map states
"""

# Some tiles only manipulate enemy direction so are excluded from repairs


class Preset:
    # Ben's asymmetric updated defense - NOT USED
    # initial_walls = [[27, 13], [0,13], [1, 12], [2, 12], [3, 13], [4,12],[8, 12], [26, 12], [7, 11], [25, 11], [7, 10], [24, 10], [8, 9], [23, 9], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8], [19, 8], [20, 8], [21, 8], [22, 8]]
    # initial_turret = [[8, 11]]

    right_cannon_plug = [26, 13]

    shared_walls = [[27, 13], [0, 13], [4, 13], [8, 12], [1, 12], [2, 12], [3, 12], [7, 11], [7, 10], [8, 9], [9, 8], [10, 8],
                    [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8], [19, 8], [20, 8]]
    shared_turret = [[8, 11]]
    shared_upgraded_wall = [[8,12]]

    quaternary_turrets = [[2, 11], [8, 10], [9, 10]]
    # walls_to_upgrade = [[8, 12], [2, 4]]
    # walls_to_upgrade_less_important = [[8, 12]]

    right_walls_forward = [[25, 13], [26, 13], [24, 12], [23, 11], [22, 10], [21, 9]]
    right_walls_backward = [[26, 12], [25, 11], [24, 10], [23, 9], [21, 8], [22, 8]]

    right_turret_forward = [24, 12]
    right_turret_backward = [24, 11]

    @staticmethod
    def get_right_walls(strategy):
        return Preset.right_walls_forward
        # if strategy is None or not strategy[FUNNEL_RIGHT]:
        #     return Preset.right_walls_forward
        # else:
        #     return Preset.right_walls_backward

    # High risk walls will be repaired with the given weights, preferentially but not necessarily in order
    # Will be repaired if health is between 1/n and 1 - 1/n of original.
    # You can repeat walls in this dictionary - it'll take the first occurrence.
    walls_to_repair_weights = {
        30: [[0, 13], [27, 13]],
        15: [[4, 13]],
        8: right_cannon_plug,
        7: right_walls_forward,
        6: right_walls_backward,
        5: shared_walls
    }

# How sensitive we are to triggering attacks and intercepts
# Minimum evaluation value - lower is more sensitive
ATTACK_THRESHOLD = 7
DEFEND_WITH_INTERCEPTORS_THRESHOLD = 6

FUNNEL_LEFT = "funnel_left"
FUNNEL_RIGHT = "funnel_right"
SCOUT_GUN_LEFT = "scout_gun_left"
SCOUT_GUN_RIGHT = "scout_gun_right"
UNKNOWN = "unknown"

LEFT_FUNNEL_COUNTER = "funnel_counter"
FRAGILE_FUNNEL_BLOCKADE = "https://terminal.c1games.com/watchLive/12616305"

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """
        Read in config and perform any initial setup here
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0

        # This is a good place to do initial setup
        self.scored_on_locations = []
        self.score_locations = []
        self.repair_locations = []

        ###############################################################
        # used for predicting enemy scout rushes
        self.enemy_scout_attack_threshold = None
        self.enemy_mp = 0
        self.enemy_previous_mp = 0

        self.always_strong_scouts = False  # returns true if the enemy is likely to upgrade the wall mid-turn

        self.enemy_deletions = []

        ###############################################################

        sim.initialise((WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR))

        # stuff after sim.initialise is used in the new version
        self.support_due = None

        # this is where we detect the opponent's enemy_strategy
        self.enemy_strategy = None

        # whether or not we're attacking (so we know not to plug the wall)
        self.prepared_to_attack = False
        # None = not built, False = back, True = forward
        self.right_layout_forward = None
        self.left_layout_forward = None

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  # Comment or remove this line to enable warnings.

        dev_helper.print_state(game_state, gamelib.debug_write)

        # we want to track the enemy's MP
        self.enemy_previous_mp = self.enemy_mp
        self.enemy_mp = game_state.get_resource(MP, 1)

        gamelib.debug_write(f"Enemy has been observed to attack on {self.enemy_scout_attack_threshold}")

        if IS_PROFILER_ENABLED:
            profile.runctx('self.strategy(game_state)', globals(), locals(),
                           filename=timing_helper.PROFILING_DIR)
            breakpoint()

        self.strategy(game_state)

        # if vulnerable_to_scout_gun
        # scout_gun_strategy(game_state)
        gamelib.debug_write(str(self.detect_enemy_strategies(game_state)))

        game_state.submit_turn()

    """
    --- TURN-BASED STRATEGY ---
    
    on turn 1
        build updated initial defence
        send 5 scouts
        
    on turn 2
        upgrade second funnel turret
        if we see a left funnel we can skip a wall placement and use it for an interceptor (cool gambit)
    
    if turn between 3-5 and opponent can send 15 scouts
        (simulate)
        send interceptor
    
    RUSH IDEA:
    
    use demo to break open thin walling around scout gun and then divert scouts from enemy funnel to opening.
    
    """

    """
    --- GENERAL STRATEGY ---
    
    replace damaged upgraded turrets
        
    if no rebuilding required
        destroys wall, replaces with upgraded support next turn
        
    if spending all on scouts will hit
        scout rush!
    
    NOTE: Scout cannon mechanism a bit weird with new defence formation, i don't understand it but someone who does pls fix.
    
    """

    """
    --- OPTIMISATION ---
    
    scout rush:
        1. find best starting location for a scout rush
    
    """

    def sp_current(self, game_state):
        return game_state.get_resource(SP) - max(0, self.sp_locked)

    def strategy(self, game_state):
        """
        Taking inspiration from some effective defences we've seen.
        """

        # at the start of each turn, predict the strategy.
        self.enemy_strategy = strategy = self.detect_enemy_strategies(game_state)
        gamelib.debug_write(strategy)

        # on the first turn
        if game_state.turn_number == 0:
            # turn 1 we just send interceptors
            self.send_inital_interceptor_covering(game_state)

        # on the second turn
        if game_state.turn_number == 1:

            # on turn 2, we build defences subtly shifted to counter our opponent
            # rebuild anything that's been destroyed

            # or we would, but there's no changes
            #self.build_initial_defences(game_state)
            self.build_core_defences(game_state)
            self.small_scout_attack(game_state)

        # on turns past the second turn
        if game_state.turn_number >= 2:
            # amount of sp to save for next turn
            # we start at -5 because we gain 5 sp per turn
            self.sp_locked = -5

            # first we build the due support
            # since we don't want holes in our walls
            self.build_due_support(game_state)

            # THEN rebuild walls that we deleted last turn for repair
            self.execute_repair_damage(game_state)

            # rebuild the initial defences that have been destroyed
            #self.build_initial_defences(game_state)
            self.build_core_defences(game_state)

            # unset prepared_to_attack

            # if we aren't preparing to attack, we're fine to plug the cannons
            if not self.prepared_to_attack:
                game_state.attempt_spawn(WALL, Preset.right_cannon_plug)

            # build secondary defences
            self.sp_locked += self.build_secondary_defences(game_state)

            # build tertiary defences
            self.sp_locked += self.build_tertiary_defences(game_state)

            # starts replacing walls with supports
            self.sp_locked += self.mark_walls_for_support_deletion(game_state)

            # schedule repairs for next turn

            sp_available = game_state.get_resource(SP)
            sp_proportion_allocated = 0.4
            if self.support_due is not None:
                sp_available -= 6
            self.sp_locked += self.schedule_repair_damage(game_state, sp_available * sp_proportion_allocated)

            # todo: maybe intelligently decide whether we need more support or more defense

            # if we have sufficient MP then walls are marked for removal, to be replaced next turn
            self.mark_walls_for_support_deletion(game_state)

            # the lower priority defences
            # for all lower priority defenses like quaternary we consider saving the SP instead of building them
            sp_available = game_state.get_resource(SP) - self.sp_locked
            self.build_quaternary_defences(game_state, sp_available)

            # predict an enemy attack
            enemy_attack_value = self.predict_enemy_attack(game_state)
            if enemy_attack_value is not None and enemy_attack_value >= DEFEND_WITH_INTERCEPTORS_THRESHOLD:
                self.counter_attack_with_interceptors(game_state)

            # determine if we should send a scout attack
            scout_rush_success_predicted, best_location = self.predict_scout_rush_success(game_state)
            if scout_rush_success_predicted:
                self.scout_rush(game_state, best_location)
                self.prepared_to_attack = False
            else:
                # rebuild the plug
                game_state.attempt_spawn(WALL, Preset.right_cannon_plug)

            # we predict the success of a scout gun rush next turn
            scout_gun_success_predicted = self.predict_future_scout_gun_success(game_state)
            if scout_gun_success_predicted:
                self.prepared_to_attack = True
                self.prepare_for_scout_gun(game_state, left=False)


    """
    --- DEFENCES ---
    """
    """
    def build_initial_defences(self, game_state):
        # This is our initial setup
        # We're not using this

        # spawns
        game_state.attempt_spawn(WALL, Preset.initial_walls)
        game_state.attempt_spawn(TURRET, Preset.initial_turret)
        game_state.attempt_upgrade(Preset.initial_turret)
    """
    def build_core_defences(self, game_state):
        # do something based on where we detect the funnel?

        game_state.attempt_spawn(WALL, Preset.shared_walls)
        game_state.attempt_spawn(TURRET, Preset.shared_turret)
        game_state.attempt_upgrade(Preset.shared_turret)
        game_state.attempt_upgrade(Preset.shared_upgraded_wall)

        # if they don't have a funnel on the right, so we build the right more forward to turn into a scout cannon
        new_right_layout = self.enemy_strategy is None or not self.enemy_strategy[FUNNEL_RIGHT] and (
            game_state.turn_number > 8 and self.sp_current(game_state) > 6)

        # todo: potentially stop having the scout cannon if it gets wrecked a lot
        # = Preset.get_right_walls(self.strategy)
        if new_right_layout or self.right_layout_forward:
            if self.right_layout_forward is None:
                # we haven't built any walls but a right cannon would be effective
                game_state.attempt_spawn(WALL, Preset.right_walls_forward)
                self.right_layout_forward = True
            elif self.right_layout_forward is False:
                # we have built the layout at the back but a right cannon would be effective
                # so we destroy the back layout
                game_state.attempt_remove(Preset.right_walls_backward)
                game_state.attempt_remove(Preset.right_turret_backward)
                game_state.attempt_spawn(WALL, Preset.right_walls_forward)
                self.right_layout_forward = True
            else:
                # building the back
                game_state.attempt_spawn(WALL, Preset.right_walls_forward)
                game_state.attempt_spawn(TURRET, Preset.right_turret_forward)
        else:
            # building the back walls is preferred
            game_state.attempt_spawn(WALL, Preset.right_walls_backward)
            game_state.attempt_spawn(TURRET, Preset.right_turret_backward)
            self.right_layout_forward = False

        if new_right_layout:
            game_state.attempt_spawn(WALL, Preset.right_cannon_plug)

    def build_secondary_defences(self, game_state):
        """
        Builds secondary defence: Upgrades stick out wall, stacks turrets
        """
        turret_location0 = [24,11]
        reinforce_walls = [[0,13],[3,13],[27,13]]
        turret_location1 = [4, 11]
        # this turret should be upgraded if the funnel is on the left

        if(game_state.attempt_spawn(TURRET,turret_location1) and game_state.get_resource(0,0) <= 6):
            return 6

        game_state.attempt_upgrade(turret_location1)
        # todo: move to Preset

        game_state.attempt_upgrade(reinforce_walls)
        game_state.attempt_spawn(TURRET, turret_location0)

        return 0

    def build_tertiary_defences(self, game_state):
        """
        Builds tertiary defence: upgrade walls near funnel turret & wall on left side
        """
        wall_upgrade_locations = [[8, 12], [9, 11]]
        game_state.attempt_spawn(WALL, wall_upgrade_locations)

        if self.right_layout_forward:
            game_state.attempt_spawn(TURRET, Preset.right_turret_forward)
        else:
            game_state.attempt_spawn(TURRET, Preset.right_turret_backward)
            game_state.attempt_spawn(WALL, Preset.right_turret_wall_backward)

        return 0

    def build_quaternary_defences(self, game_state, sp_locked=0):
        """
        Builds 3 more turrets near the funnel, then supports behind the existing supports.
        Then, builds a bunch more turrets near the back
        """

        turret_locations = Preset.quaternary_turrets
        for turret_location in turret_locations:
            if self.free_sp(game_state, 4): return
            game_state.attempt_spawn(TURRET, turret_location)
            if self.free_sp(game_state, 6): return
            game_state.attempt_upgrade(turret_location)
        support_locations = [[13, 7], [14, 7]]
        for support_location in support_locations:
            if self.free_sp(game_state, 3): return
            game_state.attempt_spawn(SUPPORT, support_location)
            if self.free_sp(game_state, 3): return
            game_state.attempt_upgrade(support_location)
        """extra_turret_locations = [[24, 17], [23, 18], [20, 18], [22, 18], [19, 19], [22, 19]]
        for extra_turret_location in extra_turret_locations:
            if self.free_sp(game_state, 4): return
            game_state.attempt_spawn(TURRET, extra_turret_location)
            if self.free_sp(game_state, 6): return
            game_state.attempt_upgrade(extra_turret_location)"""
        

    def free_sp(self, game_state, n):
        return game_state.get_resource(SP) < n + self.sp_locked

    def attempt_function_but_take_into_consideration_the_usage_of_structure_points(self, game_state, function, *args):
        if self.free_sp(game_state, 4):
            function(*args)
            return True
        else:
            return False

    def schedule_repair_damage(self, game_state: gamelib.GameState, sp_available=0):
        """Call this function with the SP to spare for repairs on damaged existing units next turn.
        This will consider the 75% refund!"""
        game_map = game_state.game_map
        if sp_available < 0.99: return 0
        if sp_available > game_state.get_resource(SP):
            gamelib.debug_write('Attempting to use excessive SP to repair damage! Limiting call.')
            sp_available = math.floor(game_state.get_resource(SP))

        friendly_tile_units = [game_map[(x, y)] for x in range(game_map.ARENA_SIZE) for y in range(game_map.HALF_ARENA)
                               if game_map.in_arena_bounds((x, y))]
        friendly_structures = list(itertools.chain.from_iterable(friendly_tile_units))

        sp_locked = 0  # sp that we must assign to repairs next turn
        units_requiring_repairs = []

        # loop through structures and decide which order they should be repaired in
        for unit in friendly_structures:
            for weight, subset in Preset.walls_to_repair_weights.items():
                # We don't want to destroy walls that are too high health (wasteful)
                # or too low health (they might tank a hit)
                if unit in subset and unit.max_health < unit.health * weight < unit.max_health * weight - unit.max_health:
                    units_requiring_repairs.append((unit, weight))

        ordered_units = sorted(units_requiring_repairs, key=lambda unit: (unit.health - unit.y) / weight)
        for unit, weight in ordered_units:
            unit_cost = unit.cost[unit.upgraded]
            if sp_available < unit_cost:
                return sp_locked # ran out of SP
            sp_available += unit_cost * 0.75 * unit.health / unit.max_health
            sp_available -= unit_cost
            self.repair_locations.append((unit.unit_type, [unit.x, unit.y]))
            sp_locked += unit_cost

        return sp_locked

    def execute_repair_damage(self, game_state):
        """Call this function with relatively high priority."""
        for unit_type, location in self.repair_locations:
            game_state.game_map.attempt_spawn(unit_type, location)

        self.repair_locations = []

    def mark_walls_for_support_deletion(self, game_state):
        """
        Figures out which walls can be marked for deletion to be replaced by supports
        """

        support_locations = [[15, 8], [14, 8], [13, 8], [12, 8]]
        # this seems to be a threshold that other algo uses
        if game_state.get_resource(SP) >= 6:
            # find the next support
            wall_remove_location = None
            for location in support_locations:
                wall = game_state.contains_stationary_unit(location)
                if not wall:
                    continue
                # if it's a wall, this is the next one to replace
                if wall.unit_type == WALL:
                    wall_remove_location = location
            # we mark this location for deletion
            if wall_remove_location is not None:
                game_state.attempt_remove(wall_remove_location)
                self.support_due = wall_remove_location
                return 6
            else:
                return 0
        else:
            return 0

    """
    todo: do the same for turrets (when loaded, make sure to allocate turrets vs shields carefully)
    """

    def build_due_support(self, game_state):
        """
        Builds supports at the location in self.support_due
        """
        # we expect there to be only one, but looping just in case
        if self.support_due:
            support_location = self.support_due
            game_state.attempt_spawn(SUPPORT, support_location)
            game_state.attempt_upgrade(support_location)
            self.support_due = None


    """
    --- PREDICTION & COUNTER-ATTACKS ---
    """

    def predict_enemy_attack(self, game_state):
        """
        We will predict if the enemy is going to attack.
        todo: later on, remember when the enemy attacks and anticipate?
        """
        # we simulate the enemy sending a bunch of scouts

        # todo: get an edge location
        right_edge = game_state.game_map.get_edge_locations(game_state.game_map.TOP_RIGHT)
        left_edge = game_state.game_map.get_edge_locations(game_state.game_map.TOP_LEFT)
        edge_locations = [*right_edge, *left_edge]

        no_of_scouts = int(game_state.get_resource(MP, 1) // game_state.type_cost(SCOUT)[MP])
        best_attack = (0, None)
        for scout_location in edge_locations:
            # if the location is full, skip
            if game_state.contains_stationary_unit(scout_location):
                continue
            # if the opponent could spawn here, simulate it
            params = sim.make_simulation(game_state, game_state.game_map, None, SCOUT, scout_location, 1, no_of_scouts)
            if not params is None:
                evaluation = sim.simulate(*params)
                # if it exceeds a (slightly arbitrary) threshold, send an interceptor
                if evaluation.value >= 4 and evaluation.value > best_attack[0]:
                    best_attack = (evaluation.value, scout_location)
                    # spawns an interceptor at [7, 6]
        if best_attack[1] is not None:
            return no_of_scouts
        else:
            return None

    def send_inital_interceptor_covering(self, game_state):
        game_state.attempt_spawn(INTERCEPTOR, [4, 9], 1)
        game_state.attempt_spawn(INTERCEPTOR, [23, 9], 1)

    def counter_attack_with_interceptors(self, game_state, count=None):
        """
        We think the enemy is going to attack, so we send out an interceptor to stop it.
        """
        # todo: make count and location smart
        if count is None:
            count = 1

        interceptor_location = [7, 6]
        game_state.attempt_spawn(INTERCEPTOR, interceptor_location, count)

    """
    --- POSITION ANALYSIS ---
    """

    def get_enemy_contiguous_defence(self, game_state, starting_location):
        """
        Returns contiguous stationary units surrounding an initial location
        """
        contiguous_defence_queue = [starting_location]
        queue_index = 0
        # we use a queue to find every connected defence
        while queue_index < len(contiguous_defence_queue):
            location_x, location_y = contiguous_defence_queue[queue_index]
            for i in range(-1, 1):
                for j in range(-1, 1):
                    if i == j == 0:
                        continue
                    checking_location = [location_x + i, location_y + j]
                    # if this unit hasn't been added already
                    unit = game_state.contains_stationary_unit(checking_location)
                    if unit is None:
                        continue
                    if unit.player_index == 1:
                        # if there's a defence here, and it isn't in the queue, add it
                        if not checking_location in contiguous_defence_queue:
                            contiguous_defence_queue.append(checking_location)
            queue_index += 1
        return contiguous_defence_queue

    # def get_enemy_defence_openings(self, game_state):
    #     """
    #     Returns a list of x-values of gaps in the enemy's structures
    #     """
    #     # self start on the left corner
    #     left_corner = [0, 14]
    #     x_reached = 0
    #     # we start at 0 and traverse across the board
    #     while x_reached

    def get_crossing_x_val(self, given_path):
        if len(given_path) < 2:  # nonsense send that self-destructs for no damage
            return None
        for location in given_path[2:]:
            # if it's crossed over to our side (or close enough)
            if location[1] <= 14:
                return location[0]
        return None

    def detect_enemy_strategies(self, game_state):
        """
        Detects enemy funnel
        """

        strategies = {
            FUNNEL_LEFT: False,
            FUNNEL_RIGHT: False,
            SCOUT_GUN_LEFT: False,
            SCOUT_GUN_RIGHT: False,
            UNKNOWN: False
        }

        # Copy the map to do wall removal tests on it
        __map = simulate.copy_internal_map(game_state.game_map)
        predicted_game_map = game_state.game_map

        walls_to_remove = [[1, 14], [26, 14]]
        for wall in walls_to_remove:
            predicted_game_map.remove_unit(wall)

        game_state.game_map.set_map_(__map)

        # first we simulate a path from the furthest forward left left location
        top_left, top_right = game_state.game_map.TOP_LEFT, game_state.game_map.TOP_RIGHT
        bottom_left, bottom_right = game_state.game_map.BOTTOM_LEFT, game_state.game_map.BOTTOM_RIGHT

        # left edges
        left_edges = game_state.game_map.get_edge_locations(top_left)
        right_edges = game_state.game_map.get_edge_locations(top_right)

        edges = [*left_edges, *right_edges]
        crossing_x_vals = []

        for edge in edges:
            if not game_state.contains_stationary_unit(edge):
                destination = bottom_left
                # if this was placed on the left, the destination will be bottom right
                if edge[0] <= 13:
                    destination = bottom_right

                unit_path = game_state.find_path_to_edge(edge, destination)
                crossing_x_val = self.get_crossing_x_val(unit_path)
                if crossing_x_val is not None:
                    crossing_x_vals.append(crossing_x_val)

        gamelib.debug_write(str(crossing_x_vals))
        for crossing_x_val in crossing_x_vals:
            if 0 < crossing_x_val <= 3:
                strategies[SCOUT_GUN_LEFT] = True
            elif 3 < crossing_x_val <= 13:
                strategies[FUNNEL_LEFT] = True
            elif 13 < crossing_x_val <= 23:
                strategies[FUNNEL_RIGHT] = True
            elif 23 < crossing_x_val <= 27:
                strategies[SCOUT_GUN_RIGHT] = True
            else:
                gamelib.debug_write("strange crossing_x_val value")
                strategies[UNKNOWN] = True

        return strategies

    def predict_enemy_build(self, game_state):
        live_map = game_state.game_map
        tile_units = [live_map[(x, y)] for x in range(live_map.ARENA_SIZE) for y in range(live_map.ARENA_SIZE)
                      if live_map.in_arena_bounds((x, y))]
        structures = list(itertools.chain.from_iterable(tile_units))



    """
    --- ATTACKING ---
    """

    def small_scout_attack(self, game_state):
        """
        This is called on the first turn to send 5 scouts.
        """
        scout_location = [20, 6]
        game_state.attempt_spawn(SCOUT, scout_location, 26)


    def predict_scout_rush_success(self, game_state):
        """
        Returns true if we think a scout rush would be successful, false if it wouldn't be
        """
        # simulates the scout attacks

        game_map = game_state.game_map
        BOTTOM_LEFT, BOTTOM_RIGHT = game_map.BOTTOM_LEFT, game_map.BOTTOM_RIGHT

        bottom_left_locations = game_map.get_edge_locations(BOTTOM_LEFT)
        bottom_right_locations = game_map.get_edge_locations(BOTTOM_RIGHT)
        
        scout_spawn_locations = [*bottom_left_locations, *bottom_right_locations]
        scout_spawn_locations = [s for s in scout_spawn_locations if not game_state.contains_stationary_unit(s)]

        no_of_scouts = game_state.number_affordable(SCOUT)
        best_effort = (0, None)
        # todo: predict enemy counterattack?
        for scout_location in scout_spawn_locations:
            # params = sim.make_simulation(game_state, game_state.game_map, None, [SCOUT, INTERCEPTOR], [scout_location, interceptor_location], [0, 1], no_of_scouts)
            params = sim.make_simulation(game_state, game_state.game_map, None, SCOUT, scout_location, 0, no_of_scouts)
            if not params is None:
                # gamelib.debug_write(str(params))
                evaluation = sim.simulate(*params)
                if evaluation.value >= best_effort[0]:
                    best_effort = (evaluation.value, scout_location)
            # if the damage of a scout rush would exceed a relatively arbitrary threshold, then we fire one.
        if best_effort[0] >= ATTACK_THRESHOLD:
            return True, best_effort[1]
        else:
            gamelib.debug_write(f"Scout rush doesn't look worth it. {evaluation.value}")

            return False, None

    def predict_future_scout_gun_success(self, game_state: gamelib.GameState):
        """
        We will simulate a scout rush next turn with the walls destroyed.
        """

        next_turn_mp = game_state.project_future_MP(1)
        no_of_scouts = int(next_turn_mp // game_state.type_cost(SCOUT)[MP])

        game_map = game_state.game_map
        BOTTOM_LEFT, BOTTOM_RIGHT = game_map.BOTTOM_LEFT, game_map.BOTTOM_RIGHT

        bottom_left_locations = game_map.get_edge_locations(BOTTOM_LEFT)
        bottom_right_locations = game_map.get_edge_locations(BOTTOM_RIGHT)

        scout_spawn_locations = [*bottom_left_locations, *bottom_right_locations]
        scout_spawn_locations = [s for s in scout_spawn_locations if not game_state.contains_stationary_unit(s)]

        best_run = (0, None)

        for scout_location in scout_spawn_locations:
            map_parameters = simulate.make_simulation_map(game_state, [], [], [],  # don't add any new stuff
                                                          remove_locations=[Preset.right_cannon_plug])
            params = simulate.make_simulation(*map_parameters, SCOUT, scout_location, 0, no_of_scouts)
            if not params is None:
                evaluation = sim.simulate(*params)
                if evaluation.value >= best_run[0]:
                    best_run = (evaluation.value, scout_location)

        if best_run[0] >= 10:
            return True
        else:
            return False

    def scout_rush(self, game_state, scout_location):
        """
        Performs a scout rush using as many scouts as possible
        """

        # bottom_left, bottom_right = game_state.game_map.BOTTOM_LEFT, game_state.game_map.BOTTOM_RIGHT
        # scout_spawn_locations = [*bottom_left, bottom_right]
        # scout_spawn_locations = [s for s in scout_spawn_locations if game_state.contains_stationary_unit(s)]

        no_of_scouts = game_state.number_affordable(SCOUT)

        # params = sim.make_simulation(game_state, game_state.game_map, None, [SCOUT, INTERCEPTOR], [scout_location, interceptor_location], [0, 1], no_of_scouts)
        params = sim.make_simulation(game_state, game_state.game_map, None, SCOUT, scout_location, 0, no_of_scouts)
        if not params is None:
            evaluation = sim.simulate(*params)
            if evaluation.value >= 7:
                gamelib.debug_write(f"Spawed f{no_of_scouts} scouts")
                game_state.attempt_spawn(SCOUT, scout_location, no_of_scouts)
            else:
                gamelib.debug_write(f"Scout rush wasn't good enough, weird. Only scored {evaluation.value}")

    def prepare_for_scout_gun(self, game_state, left):
        entrance_location = [26, 13]
        if left:
            # we're preparing for a scout gun on the left
            entrance_location = [3, 11]
        game_state.attempt_remove(entrance_location)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy turrets that can attack each location and multiply by turret damage
                damage += len(game_state.get_attackers(path_location, 0)) * gamelib.GameUnit(TURRET,
                                                                                             game_state.config).damage_i
            damages.append(damage)

        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]

    def detect_enemy_unit(self, game_state, unit_type=None, valid_x=None, valid_y=None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (
                            valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units

    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered
    # OUTDATED
    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)

        if state["turnInfo"][2] == 0:
            # reset the locations marking scores for and against
            self.score_locations = []
            self.scored_on_locations = []

            # keep track of which structures the enemy is deleting
            # used for predicting an attack on the left side, to proactively reinforce
            enemy_deletions = state["p2Units"][6]
            self.enemy_deletions = [[x, y] for x, y, *_ in enemy_deletions]

        events = state["events"]
        breaches = events["breach"]
        spawns = events["spawn"]
        if len(spawns) >= 0:
            # a unit was spawned!
            # something was spawned!
            # we filter to see if the enemy spawned anything
            enemy_spawns = [spawn for spawn in spawns if spawn[3] == 2]
            # 3 is the integer unit type for scouts
            enemy_scout_spawns = [spawn for spawn in enemy_spawns if spawn[1] == 3]
            enemy_scouts_num = len(enemy_scout_spawns)

            # we set the threshold as the minimum amount for them to not send
            # e.g if they have 13mp last turn, we assume they waited for 14
            enemy_threshold_prediction = int(math.floor(self.enemy_previous_mp)) + 1

            # 8 is a relatively arbitrary threshold we've chosen
            if enemy_scouts_num >= 8:
                # the enemy spawned a bunch of scouts
                # we update our attack threshold
                # if this is the first attack they've made this game, we set the attack threshold
                if self.enemy_scout_attack_threshold is None:
                    # we record the number they used
                    self.enemy_scout_attack_threshold = enemy_threshold_prediction
                # if they attacked before our previous attack threshold, this becomes our new threshold
                elif self.enemy_scout_attack_threshold > enemy_threshold_prediction:
                    self.enemy_scout_attack_threshold = enemy_threshold_prediction
                # we raise our threshold if they could have attacked last turn, but didn't.
                elif self.enemy_scout_attack_threshold < enemy_threshold_prediction and self.enemy_previous_mp > self.enemy_scout_attack_threshold:
                    self.enemy_scout_attack_threshold = enemy_threshold_prediction

        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly,
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                # gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                # gamelib.debug_write("All locations: {}".format(self.scored_on_locations))
            else:
                self.score_locations.append(location)


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
