import copy

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import simulate as sim
import itertools

import dev_helper

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
    initial_walls = [[0, 13], [27, 13], [1, 12], [2, 12], [3, 12], [4, 12], [8, 12], [26, 12], [7, 11], [25, 11], [7, 10], [24, 10], [8, 9], [23, 9], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [18, 8], [19, 8], [20, 8], [21, 8], [22, 8]]
    initial_turret = [[8, 11]]

    # High risk walls will be repaired with the given weights
    # Will be repaired if health is between 1/n and 1 - 1/n of original.
    walls_to_repair_weights = {
        30: [[8, 12], [7, 11]],
        5: initial_walls
    }


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

        self.attacked_locations = {  # locations the enemy has attacked
            "right_corner": 0,
            "funnel": 0,
            "left_corner": 0
        }

        self.enemy_deletions = []

        ###############################################################

        sim.initialise((WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR))

        # stuff after sim.initialise is used in the new version
        self.support_due = None

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
    
    """

    """
    --- OPTIMISATION ---
    
    scout rush:
        1. find best starting location for a scout rush
    
    """

    def strategy(self, game_state):
        """
        Taking inspiration from some effective defences we've seen.
        """

        # at the start of each turn, predict the strategy.
        strategy = self.detect_enemy_strategies(game_state)
        gamelib.debug_write(strategy)

        # on the first turn
        if game_state.turn_number == 0:
            # setup the initial defences
            self.build_initial_defences(game_state)
            self.small_scout_attack(game_state)

        # on the second turn
        if game_state.turn_number == 1:
            # rebuild anything that's been destroyed
            self.build_initial_defences(game_state)
            # we build our second wave of defences
            self.build_secondary_defences(game_state)
            self.small_scout_attack(game_state)

        # on turns past the second turn
        if game_state.turn_number >= 2:
            # amount of sp to save for next turn
            self.sp_locked = 0

            # first we build the due support
            # since we don't want holes in our walls
            self.build_due_support(game_state)
            # if we have sufficient MP then walls are marked for removal, to be replaced next turn
            self.mark_walls_for_support_deletion(game_state)
            # THEN rebuild walls that we deleted last turn for repair
            self.execute_repair_damage(game_state)

            # rebuild the initial defences that have been destroyed
            self.build_initial_defences(game_state)


            # build second & tertiary defences
            self.build_secondary_defences(game_state)
            self.build_tertiary_defences(game_state)

            # starts replacing walls with supports
            self.sp_locked += self.mark_walls_for_support_deletion(game_state)

            # schedule repairs for next turn

            sp_available = game_state.get_resource(SP)
            sp_proportion_allocated = 0.4
            if not self.support_due is None:
                sp_available -= 6
            self.sp_locked += self.schedule_repair_damage(game_state, sp_available * sp_proportion_allocated)

            # the lower priority defences
            sp_available = game_state.get_resource(SP) - self.sp_locked
            self.build_quaternary_defences(game_state, sp_available)

            # predict an enemy attack
            enemy_attack_predicted = self.predict_enemy_attack(game_state)
            if enemy_attack_predicted:
                self.counter_attack_with_interceptors(game_state)

            # determine if we should send a scout attack
            scout_rush_success_predicted = self.predict_scout_rush_success(game_state)
            if scout_rush_success_predicted:
                self.scout_rush(game_state)




    """
    --- DEFENCES ---
    """

    def build_initial_defences(self, game_state):
        """
        This is our initial setup
        """
        # initial setup turrets

        # spawns
        game_state.attempt_spawn(WALL, Preset.initial_walls)
        game_state.attempt_spawn(TURRET, Preset.initial_turret)
        game_state.attempt_upgrade(Preset.initial_turret)

    def build_secondary_defences(self, game_state):
        """
        Builds secondary defence: 2 walls and a turret near the funnel
        """
        wall_locations = [[5, 11], [9, 11]]
        # todo: this turret should be upgraded if the funnel is on the left
        turret_location = [4, 11]
        game_state.attempt_spawn(WALL, wall_locations)
        game_state.attempt_spawn(TURRET, turret_location)

    def build_tertiary_defences(self, game_state):
        """
        Builds tertiary defence: upgrade walls near funnel turret & wall on left side
        """
        wall_upgrade_locations = [[8, 12], [9, 11]]
        game_state.attempt_spawn(WALL, wall_upgrade_locations)

        turret_location = [24, 11]
        wall_location = [24, 12]
        game_state.attempt_spawn(WALL, wall_location)
        game_state.attempt_spawn(TURRET, turret_location)

    def build_quaternary_defences(self, game_state, sp_locked=0):
        """
        Builds 3 more turrets near the funnel, then supports behind the existing supports.
        Then, builds a bunch more turrets near the back
        """

        turret_locations = [[8, 10], [3, 11], [4, 10]]
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
        extra_turret_locations = [[24, 17], [23, 18], [20, 18], [22, 18], [19, 19], [22, 19]]
        for extra_turret_location in extra_turret_locations:
            if self.free_sp(game_state, 4): return
            game_state.attempt_spawn(TURRET, extra_turret_location)
            if self.free_sp(game_state, 6): return
            game_state.attempt_upgrade(extra_turret_location)


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
        # we simulate the enemy sending 15 scouts

        # todo: get an edge location
        right_edge = game_state.game_map.get_edge_locations(game_state.game_map.TOP_RIGHT)
        left_edge = game_state.game_map.get_edge_locations(game_state.game_map.TOP_LEFT)
        edge_locations = [*right_edge, *left_edge]

        no_of_scouts = int(game_state.get_resource(MP, 1) // game_state.type_cost(SCOUT)[MP])
        for scout_location in edge_locations:
            # if the location is full, skip
            if game_state.contains_stationary_unit(scout_location):
                continue
            # if the opponent could spawn here, simulate it
            params = sim.make_simulation(game_state, game_state.game_map, None, SCOUT, scout_location, 1, no_of_scouts)
            if not params is None:
                evaluation = sim.simulate(*params)
                # if it exceeds a (slightly arbitrary) damage threshold, send an interceptor
                if evaluation.damage_dealt >= 4:
                    return True
                    # spawns an interceptor at [7, 6]
        return False

    def counter_attack_with_interceptors(self, game_state):
        """
        We think the enemy is going to attack, so we send out an interceptor to stop it.
        """
        interceptor_location = [7, 6]
        no_of_interceptors = 1 # todo: make this not always 1
        game_state.attempt_spawn(INTERCEPTOR, interceptor_location, no_of_interceptors)

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

    def detect_enemy_strategies(self, game_state):
        """
        Detects enemy funnel
        """

        strategies = {
            "funnel_left": False,
            "funnel_right": False,
            "scout_gun_left": False,
            "scout_gun_right": False,
            "unknown": False
        }

        # first we simulate a path from the furthest forward left left location
        top_left, top_right = game_state.game_map.TOP_LEFT, game_state.game_map.TOP_RIGHT
        bottom_left, bottom_right = game_state.game_map.BOTTOM_LEFT, game_state.game_map.BOTTOM_RIGHT

        def get_crossing_x_val(given_path):
            if len(given_path) < 2:  # nonsense send that self-destructs for no damage
                return None
            for location in given_path[2:]:
                # if it's crossed over to our side (or close enough)
                if location[1] <= 14:
                    return location[1]
            return None

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
                crossing_x_val = get_crossing_x_val(unit_path)
                if crossing_x_val is not None:
                    crossing_x_vals.append(crossing_x_val)

        for crossing_x_val in crossing_x_vals:
            if 0 < crossing_x_val <= 3:
                strategies["scout_gun_left"] = True
            elif 3 < crossing_x_val <= 13:
                strategies["funnel_left"] = True
            elif 13 < crossing_x_val <= 23:
                strategies["funnel_right"] = True
            elif 23 < crossing_x_val <= 27:
                strategies["scout_gun_right"] = True
            else:
                gamelib.debug_write("strange crossing_x_val value")
                strategies["unknown"] = True

        return strategies

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
        # simulates the scout attack
        scout_location = [20, 6]
        no_of_scouts = game_state.number_affordable(SCOUT)
        # todo: predict enemy counterattack?
        # params = sim.make_simulation(game_state, game_state.game_map, None, [SCOUT, INTERCEPTOR], [scout_location, interceptor_location], [0, 1], no_of_scouts)
        params = sim.make_simulation(game_state, game_state.game_map, None, SCOUT, scout_location, 0, no_of_scouts)
        if not params is None:
            evaluation = sim.simulate(*params)
            # if the damage of a scout rush would exceed a relatively arbitrary threshold, then we fire one.
            if evaluation.value >= 10:
                return True
        return False

    def scout_rush(self, game_state):
        """
        Performs a scout rush using as many scouts as possible
        """
        scout_location = [20, 6]
        no_of_scouts = game_state.number_affordable(SCOUT)
        game_state.attempt_spawn(SCOUT, scout_location, no_of_scouts)

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
