import copy

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import itertools
import random
import simulate
import simulate as sim
import dev_helper

IS_DEV_ENABLED = True
IS_PROFILER_ENABLED = False
sim.PRINT_MAP_STEPS = False

if IS_PROFILER_ENABLED:
    import cProfile as profile  # if not available, replace with 'import profile'
    import timing_helper
    import io
    import pstats
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
    right_cannon_funnel_block = [[2, 12], [3, 12], [5, 12], [6, 11], [7, 10]]
    right_cannon_funnel_unblock = [[6, 11]]
    right_cannon_plug = [26, 13]

    shared_walls = [[27, 13], [0, 13], [4, 13], [8, 12], [1, 12], [2, 12], [3, 12], [7, 11], [7, 10], [8, 9], [9, 8], [10, 7],
                    [11, 7], [12, 7], [13, 7], [14, 7], [15, 7], [16, 7], [17, 7], [18, 7], [19, 7], [20, 8]]
    shared_turret = [[8, 11]]
    shared_upgraded_wall = [[8,12]]

    quaternary_turrets = [[2, 11], [8, 10], [9, 10]]
    quaternary_supports = [[13, 6], [14, 6]]

    secondary_turret = [[4, 12]]
    reinforce_walls = [[0, 13], [4, 13], [27, 13]]
    # walls_to_upgrade = [[8, 12], [2, 4]]
    # walls_to_upgrade_less_important = [[8, 12]]

    right_walls_forward = [[25, 13], [24, 13], [23, 11], [22, 10], [21, 9]]  # not the [26, 13] plug
    right_walls_backward = [[26, 12], [25, 11], [24, 10], [23, 9], [21, 8], [22, 8]]

    right_turret_forward = [[24, 12]]
    right_turret_backward = [[24, 11]]

    right_turret_wall_forward = [24, 13]
    right_turret_wall_backward = [24, 12]

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
        31: [[8,12]],
        30: reinforce_walls,
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
FUNNEL_CENTRE = "funnel_center"
UNKNOWN = "unknown"

LEFT_FUNNEL_COUNTER = "left_funnel_counter"
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
        self.repair_actions = []
        self.sp_locked = 0
        self.flip = False

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

        if self.flip:
            game_state.game_map.flip_map_()  # flip horizontally

        dev_helper.print_state(game_state, gamelib.debug_write)

        # we want to track the enemy's MP
        self.enemy_previous_mp = self.enemy_mp
        self.enemy_mp = game_state.get_resource(MP, 1)

        gamelib.debug_write(f"Enemy has been observed to attack on {self.enemy_scout_attack_threshold}")

        # if IS_PROFILER_ENABLED:
        #     profile.runctx('self.strategy(game_state)', globals(), locals(),
        #                    filename=timing_helper.PROFILING_DIR)
        #     breakpoint()

        if IS_PROFILER_ENABLED:
            pr = profile.Profile()
            pr.enable()

        # Simulation test
        """game_map = game_state.game_map
        horiz_wall = [[i, 13] for i in range(28)]
        unit_types = [WALL for i in range(28)]
        map_params = simulate.make_simulation_map(game_state, unit_types, horiz_wall)
        evaluation = simulate.make_simulation(*map_params, SCOUT, [14, 0])
        gamelib.debug_write(evaluation)
        breakpoint()"""

        self.strategy(game_state)

        """if IS_PROFILER_ENABLED:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()

            with open('_timing_profile_game_r' + str(game_state.turn_number), 'w+') as f:
                f.write(s.getvalue())"""

        # if vulnerable_to_scout_gun
        # scout_gun_strategy(game_state)
        gamelib.debug_write(str(self.detect_enemy_strategies(game_state)))

        if self.flip:
            game_state.game_map.flip_map_()  # flip horizontally

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
        if(self.flip):
            game_state.game_map.flip_map_()
        # at the start of each turn, predict the strategy.
        self.enemy_strategy = strategy = self.detect_enemy_strategies(game_state)
        gamelib.debug_write(strategy)

        # on the first turn
        if game_state.turn_number == 0:
            # turn 1 we just send interceptors
            self.send_inital_interceptor_covering(game_state)

        # on the second turn
        if game_state.turn_number == 1:

            self.flip_detect(game_state)
            # on turn 2, we build defences subtly shifted to counter our opponent
            # rebuild anything that's been destroyed
            if(self.flip):
                game_state.game_map.flip_map_()
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

            # if we aren't preparing to attack, we're fine to plug the cannons
            if not self.prepared_to_attack:
                game_state.attempt_spawn(WALL, Preset.right_cannon_plug)

            # unset prepared_to_attack

            # build secondary defences
            self.sp_locked += self.build_secondary_defences(game_state)

            # build tertiary defences
            self.sp_locked += self.build_tertiary_defences(game_state)

            # schedule repairs for next turn
            sp_available = game_state.get_resource(SP)
            sp_proportion_allocated = 0.4
            if self.support_due is not None:
                sp_available -= 6
            self.sp_locked += self.schedule_repair_damage(game_state, sp_available * sp_proportion_allocated)

            # if we have sufficient SP then walls are marked for removal, to be replaced next turn
            self.sp_locked += self.mark_walls_for_support_deletion(game_state)


            # the lower priority defences
            # for all lower priority defenses like quaternary we consider saving the SP instead of building them
            sp_available = game_state.get_resource(SP) - self.sp_locked
            self.build_quaternary_defences(game_state, sp_available)

            #When nothing else to do, start reinforcing existing defences with more turrets/shields
            if game_state.get_resource(SP) - self.sp_locked >= 3:
                self.auto_reinforce(game_state)

            # ATTACKS / MOBILE UNITS =================================================

            # gamelib.debug_write('This is map we are running with for attack simulations:')
            # dev_helper.print_map(game_state.game_map)

            # predict an enemy attack
            # enemy_attack_value = self.predict_enemy_attack(game_state)
            # if enemy_attack_value is not None and enemy_attack_value >= DEFEND_WITH_INTERCEPTORS_THRESHOLD:
            #     self.counter_attack_with_interceptors(game_state)

            # determine if we should send a scout attack
            scout_rush_success_value, best_distribution, block_funnel = self.predict_scout_rush_success(game_state)

            # determine if we should send a demolisher attack
            demolisher_rush_success_value, d_best_distribution, block_funnel = self.predict_demolisher_rush_success(game_state)

            if demolisher_rush_success_value and demolisher_rush_success_value > max(scout_rush_success_value, ATTACK_THRESHOLD):
                gamelib.debug_write(f"Doing demolisher rush: {demolisher_rush_success_value}")
                self.optimise_demolisher_rush(game_state, demolisher_rush_success_value, d_best_distribution)
            else:
                gamelib.debug_write(f"Scout rush prediction this turn: {(scout_rush_success_value, best_distribution, block_funnel)}")
                if scout_rush_success_value and scout_rush_success_value > ATTACK_THRESHOLD:
                    if block_funnel:
                        gamelib.debug_write("Attempting scout gun...")
                        funnel_successfully_blocked = game_state.attempt_spawn(WALL, Preset.right_cannon_funnel_block)
                        game_state.attempt_remove(Preset.right_cannon_funnel_unblock)
                        if funnel_successfully_blocked:
                            gamelib.debug_write("Expect scout rush through the gun :)")
                            self.scout_rush(game_state, best_distribution)
                        else:
                            gamelib.debug_write('Not properly blocked the funnel on the left so not scout rushing.')
                    else:
                        gamelib.debug_write("Attempting scout rush through funnel")
                        self.scout_rush(game_state, best_distribution)
                    self.prepared_to_attack = False
                else:
                    # rebuild the plug
                    game_state.attempt_spawn(WALL, Preset.right_cannon_plug)

            # we predict the success of a scout gun rush next turn
            scout_gun_predicted_value, _ = self.predict_scout_gun_next_turn_success(game_state)
            if scout_gun_predicted_value >= ATTACK_THRESHOLD:
                self.prepared_to_attack = True
                self.prepare_for_scout_gun(game_state, left=False)
                gamelib.debug_write(f"Scout gun worth it {scout_gun_predicted_value}")
            else:
                gamelib.debug_write(f"Scout gun not worth it, only predicted {scout_gun_predicted_value}")


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
    
    """
    funnel position detection:
    check for imbalance of turrets. If more on right, then flip.
    """
    def flip_detect(self, game_state : gamelib.GameState):
        right_open = self.enemy_strategy[FUNNEL_RIGHT] and self.enemy_strategy[SCOUT_GUN_RIGHT]
        left_open = self.enemy_strategy[FUNNEL_LEFT] and self.enemy_strategy[SCOUT_GUN_LEFT]
        sparse = left_open and right_open and self.enemy_strategy[FUNNEL_CENTRE]

        if not sparse and left_open and self.enemy_strategy[FUNNEL_RIGHT]:
            self.flip = True
            return
        if not sparse and right_open and self.enemy_strategy[FUNNEL_LEFT]:
            self.flip = False
            return
        if (not self.enemy_strategy[FUNNEL_LEFT] and not self.enemy_strategy[SCOUT_GUN_LEFT] and not self.enemy_strategy[FUNNEL_CENTRE]
            and not self.enemy_strategy[SCOUT_GUN_RIGHT] and self.enemy_strategy[FUNNEL_RIGHT]):
            self.flip = True
            return
        if (not self.enemy_strategy[FUNNEL_RIGHT] and not self.enemy_strategy[SCOUT_GUN_LEFT] and not self.enemy_strategy[FUNNEL_CENTRE]
            and not self.enemy_strategy[SCOUT_GUN_RIGHT] and self.enemy_strategy[FUNNEL_LEFT]):
            self.flip = False
            return

        map = game_state.game_map
        imbalance = 0
        for i in range(28):
            for y in range(14,28):
                instance = map[i,y]
                if(instance and instance[0].unit_type == TURRET):
                    imbalance += (1 if i >= 14 else -1)
        if imbalance > 0:
            self.flip = True


    def auto_reinforce(self, game_state):
        pooledTurretLocations = []
        for i in [Preset.shared_turret,Preset.secondary_turret,Preset.right_turret_backward,Preset.right_turret_forward,Preset.quaternary_turrets]:
            pooledTurretLocations.extend(i)
        pooledSupportLocations = Preset.quaternary_supports
        game_state.attempt_upgrade(pooledTurretLocations)
        game_state.attempt_upgrade(pooledSupportLocations)
        if game_state.get_resource(0, 0) < 3: return
        while(len(pooledTurretLocations) + len(pooledSupportLocations) < 75):
            r = random.random()
            x, y = random.randint(-1, 1), random.randint(-1, 1)
            if x == 0 and y == 0: continue
            if r < len(pooledSupportLocations)/(len(pooledTurretLocations)+len(pooledSupportLocations)):
                spawnLoc = [pooledTurretLocations[random.randrange(len(pooledTurretLocations))][0] + x,pooledTurretLocations[random.randrange(len(pooledTurretLocations))][1] + y]
                if((spawnLoc[0] <= 13 and spawnLoc[0] + spawnLoc[1] <= 14) or (spawnLoc[0] > 13 and (27-spawnLoc[0]) + spawnLoc[1] <= 14)):
                    continue
                if game_state.attempt_spawn(TURRET,spawnLoc):
                    pooledTurretLocations.append(spawnLoc)
            else:
                y -= 1 if y == 1 else 0
                spawnLoc = [pooledSupportLocations[random.randrange(len(pooledSupportLocations))][0] + x,
                            pooledSupportLocations[random.randrange(len(pooledSupportLocations))][1] + y]
                if ((spawnLoc[0] <= 13 and spawnLoc[0] + spawnLoc[1] <= 14) or (
                        spawnLoc[0] > 13 and (27 - spawnLoc[0]) + spawnLoc[1] <= 14)):
                    continue
                if game_state.attempt_spawn(SUPPORT, spawnLoc):
                    pooledSupportLocations.append(spawnLoc)
            if game_state.get_resource(0, 0) < 3: return

    def build_core_defences(self, game_state):
        # do something based on where we detect the funnel?

        game_state.attempt_spawn(WALL, Preset.shared_walls)
        game_state.attempt_spawn(TURRET, Preset.shared_turret)
        game_state.attempt_upgrade(Preset.shared_turret)
        game_state.attempt_upgrade(Preset.shared_upgraded_wall)

        # if they don't have a funnel on the right, so we build the right more forward to turn into a scout cannon
        # new_right_layout = self.enemy_strategy is None or not self.enemy_strategy[FUNNEL_RIGHT] and (
        #                    game_state.turn_number > 8 and self.sp_current(game_state) > 6)

        new_right_layout = self.enemy_strategy is None or not self.enemy_strategy[FUNNEL_RIGHT] or (
            self.sp_current(game_state) >= 3 and game_state.turn_number >= 3)

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
                game_state.attempt_remove(Preset.right_turret_wall_backward) # <-- added this
                game_state.attempt_spawn(WALL, Preset.right_walls_forward)
                self.right_layout_forward = True
            else:
                # building the back
                game_state.attempt_spawn(WALL, Preset.right_walls_forward)
        else:
            # building the back walls is preferred
            game_state.attempt_spawn(WALL, Preset.right_walls_backward)
            game_state.attempt_spawn(TURRET, Preset.right_turret_backward)
            self.right_layout_forward = False

        # if new_right_layout:
        #     game_state.attempt_spawn(WALL, Preset.right_cannon_plug)

    def build_secondary_defences(self, game_state):
        """
        Builds secondary defence: Upgrades stick out wall, stacks turrets
        """
        # this turret should be upgraded if the funnel is on the left

        if(game_state.attempt_spawn(TURRET,Preset.secondary_turret) and game_state.get_resource(0,0) <= 6):
            return 6


        game_state.attempt_upgrade(Preset.secondary_turret)
        # todo: move to Preset

        game_state.attempt_upgrade(Preset.reinforce_walls)

        if not self.right_layout_forward and (self.enemy_strategy[SCOUT_GUN_RIGHT] or self.enemy_strategy[FUNNEL_RIGHT]):
            game_state.attempt_spawn(TURRET, Preset.right_turret_backward)
        else:
            game_state.attempt_spawn(TURRET, Preset.right_turret_forward)

        return 0

    def build_tertiary_defences(self, game_state):
        """
        Builds tertiary defence: upgrade walls near funnel turret & wall on right side`
        """

        if self.right_layout_forward:
            game_state.attempt_upgrade(Preset.right_turret_forward)
            game_state.attempt_spawn(WALL, Preset.right_turret_wall_forward)
            game_state.attempt_upgrade(Preset.right_turret_wall_forward)
        else:
            game_state.attempt_upgrade(Preset.right_turret_backward)
            game_state.attempt_spawn(WALL, Preset.right_turret_wall_backward)
            game_state.attempt_upgrade(Preset.right_turret_wall_backward)



        return 0

    def build_quaternary_defences(self, game_state, sp_locked=0):
        """
        Builds 3 more turrets near the funnel, then supports behind the existing supports.
        Then, builds a bunch more turrets near the back
        """

        turret_locations = Preset.quaternary_turrets
        for turret_location in Preset.quaternary_turrets:
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
        if sp_available < 0.99:
            gamelib.debug_write('Insufficient SP supplied to repair damage! Limiting call.')
            return 0
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

        gamelib.debug_write('Scheduling walls to be repaired next turn!')
        ordered_units = sorted(units_requiring_repairs, key=lambda unit: (unit.health - unit.y*0.125) / weight)
        for unit, weight in ordered_units:
            unit_cost = unit.cost[unit.upgraded]
            if sp_available < unit_cost:
                return sp_locked # ran out of SP
            game_state.attempt_remove(unit[1])
            sp_available += unit_cost * 0.75 * unit.health / unit.max_health
            sp_available -= unit_cost
            self.repair_actions.append((unit.unit_type, [unit.x, unit.y]))
            sp_locked += unit_cost

        return sp_locked

    def execute_repair_damage(self, game_state):
        """Call this function with relatively high priority."""
        for unit_type, location in self.repair_actions:
            game_state.game_map.attempt_spawn(unit_type, location)

        self.repair_actions = []

    def mark_walls_for_support_deletion(self, game_state):
        """
        Figures out which walls can be marked for deletion to be replaced by supports
        """
        support_locations = [[15, 7], [14, 7], [13, 7], [12, 7]]
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
            gamelib.debug_write("Making simulation map under predict_enemy_attack")
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

    def simulate_friendly_scouts(self, game_state):
        """
        Returns the best position to attack from, the value of that attack,
        and whether or not the funnel should be closed for it
        """


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
            FUNNEL_CENTRE: False,
            UNKNOWN: False
        }

        """# Copy the map to do wall removal tests on it
        __map = simulate.copy_internal_map(game_state.game_map)
        predicted_game_map = game_state.game_map

        walls_to_remove = [[1, 14], [26, 14]]
        for wall in walls_to_remove:
            predicted_game_map.remove_unit(wall)

        game_state.game_map.set_map_(__map)"""

        # first we simulate a path from the furthest forward left left location
        top_left, top_right = game_state.game_map.TOP_LEFT, game_state.game_map.TOP_RIGHT
        bottom_left, bottom_right = game_state.game_map.BOTTOM_LEFT, game_state.game_map.BOTTOM_RIGHT

        # left edges
        left_edges = game_state.game_map.get_edge_locations(top_left)
        right_edges = game_state.game_map.get_edge_locations(top_right)

        edges = [*left_edges, *right_edges]
        crossing_x_vals = []

        for edge_loc in edges:
            if not game_state.contains_stationary_unit(edge_loc):

                unit_path = game_state.find_path_to_exit_half(edge_loc)
                if unit_path:
                    crossing_x_vals.append(unit_path[-1][0])

        gamelib.debug_write(str(crossing_x_vals))
        for crossing_x_val in crossing_x_vals:
            if 0 < crossing_x_val <= 3:
                strategies[SCOUT_GUN_LEFT] = True
            elif 3 < crossing_x_val <= 10:
                strategies[FUNNEL_LEFT] = True
            elif 16 < crossing_x_val <= 23:
                strategies[FUNNEL_RIGHT] = True
            elif 23 < crossing_x_val <= 27:
                strategies[SCOUT_GUN_RIGHT] = True
            else:
                strategies[FUNNEL_CENTRE] = True

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
        Rewritten to be more clear
        """
        scout_locations = [[13, 0], [14, 0]]
        no_of_scouts = game_state.number_affordable(SCOUT)

        # these are the possible permutations
        possible_permutations = []

        if no_of_scouts > 6:
            possible_permutations.append([no_of_scouts - 6, 6])
            possible_permutations.append([6, no_of_scouts - 6])
        elif 6 >= no_of_scouts > 2:
            possible_permutations.append([no_of_scouts - 2, 2])
            possible_permutations.append([2, no_of_scouts - 2])
        else:
            gamelib.debug_write('not enough scouts')
            return

        best_effort = (
            0,  # evaluation.value
            None,   # the distribution of the 2 scouts
            None    # true if we should block the funnel, false if not
        )

        # we simulate a rush without blocking the funnel - i.e, the board as is
        for permutation in possible_permutations:
            __map = simulate.copy_internal_map(game_state.game_map)
            gamelib.debug_write("Making simulation under predict scout rush success w/o blocking funnel")
            params = sim.make_simulation(game_state, game_state.game_map, None, [SCOUT, SCOUT], scout_locations, 0, permutation, copy_safe=False)
            if not params is None:
                evaluation = sim.simulate(*params)
                if evaluation.value >= best_effort[0]:
                    best_effort = (evaluation.value, permutation, False)

            # reset the map after the simulation back to the real map
            game_state.game_map.set_map_(__map)

        __map = simulate.copy_internal_map(game_state.game_map)

        gamelib.debug_write(f"If there isn't a message after this, then self.right_layout_forward is false: {self.right_layout_forward}")
        # now we simulate the map with the funnel blocked
        if self.right_layout_forward:
            gamelib.debug_write("Making simulation map under predict_scout_rush_success")
            map_params = sim.make_simulation_map(game_state, [WALL for _ in Preset.right_cannon_funnel_block],
                                                 Preset.right_cannon_funnel_block, copy_safe=False)
            __map_modified = simulate.copy_internal_map(map_params[1])

            gamelib.debug_write(f"Testing the scout gun with a blocked funnel")
            for permutation in possible_permutations:
                gamelib.debug_write("Making simulation under predict_scout_rush_success w/ funnel block")
                params = sim.make_simulation(*map_params, [SCOUT, SCOUT], scout_locations, 0, permutation, copy_safe=False)
                if not params is None:
                    evaluation = sim.simulate(*params)
                    if evaluation.value >= best_effort[0]:
                        gamelib.debug_write(f"A new best attack found - with the funnel blocked! {evaluation.value}")
                        best_effort = (evaluation.value, permutation, True)
                    else:
                        gamelib.debug_write(f"Previous eval without funnel was better: {best_effort[0]} vs {evaluation.value}")

                # reset the map after the simulation back to the real map + our changes
                map_params[1].set_map_(__map_modified)
                __map_modified = simulate.copy_internal_map(map_params[1])

            # reset the map after the simulation back to the real map
            game_state.game_map.set_map_(__map)

        # we return the best effort
        return best_effort

    def predict_scout_gun_next_turn_success(self, game_state):
        
        if not self.right_layout_forward:
            return (0, None)
        
        scout_locations = [[13, 0], [14, 0]]
        next_turn_mp = game_state.project_future_MP(1)
        no_of_scouts = int(next_turn_mp // game_state.type_cost(SCOUT)[MP])

        # possible scout placement permutations
        possible_permutations = []

        if no_of_scouts > 6:
            possible_permutations.append([no_of_scouts - 6, 6])
            possible_permutations.append([6, no_of_scouts - 6])
        elif 6 >= no_of_scouts > 2:
            possible_permutations.append([no_of_scouts - 2, 2])
            possible_permutations.append([2, no_of_scouts - 2])
        else:
            gamelib.debug_write('not enough scouts')
            return (-1, None)

        best_effort = (
            0,  # evaluation.value
            None,  # the distribution of the 2 scouts
        )

        __map = simulate.copy_internal_map(game_state.game_map)

        # now we simulate the map with the funnel blocked
        if self.right_layout_forward:
            gamelib.debug_write('making simulation map - predict_scout_gun_next_turn_success - trying to remove plug and block funnel')
            map_params = sim.make_simulation_map(game_state, [WALL for _ in Preset.right_cannon_funnel_block],
                                                 Preset.right_cannon_funnel_block, copy_safe=False, remove_locations=[[26, 13]])
            __map_modified = simulate.copy_internal_map(map_params[1])

            for permutation in possible_permutations:
                gamelib.debug_write("Making simulation under predict_scout_gun_next_turn_success")
                params = sim.make_simulation(*map_params, [SCOUT, SCOUT], scout_locations, 0, permutation,
                                             copy_safe=False)
                if not params is None:
                    evaluation = sim.simulate(*params)
                    if evaluation.value >= best_effort[0]:
                        best_effort = (evaluation.value, permutation)

                # reset the map after the simulation back to the real map + our changes
                map_params[1].set_map_(__map_modified)
                __map_modified = simulate.copy_internal_map(map_params[1])

            # reset the map after the simulation back to the real map
            game_state.game_map.set_map_(__map)

        return best_effort

    def scout_rush(self, game_state, best_distribution):
        """
        Performs a scout rush using as many scouts as possible
        """
        scout_locations = [[13, 0], [14, 0]]
        no_of_scouts = game_state.number_affordable(SCOUT)
        #
        # # params = sim.make_simulation(game_state, game_state.game_map, None, [SCOUT, INTERCEPTOR], [scout_location, interceptor_location], [0, 1], no_of_scouts)
        # # params = sim.make_simulation(game_state, game_state.game_map, None, SCOUT, scout_location, 0, no_of_scouts)
        # # blocking the funnel
        # map_parameters = list(simulate.make_simulation_map(game_state, [WALL for _ in range(len(Preset.right_cannon_funnel_block))],
        #                                                    Preset.right_cannon_funnel_block))
        # __map = simulate.copy_internal_map(map_parameters[1])
        # params = simulate.make_simulation(*map_parameters, [SCOUT, SCOUT], [[13, 0], [14, 0]], [0, 0], [no_of_scouts - 6, 6], copy_safe=False)
        # if not params is None:
        #     evaluation = sim.simulate(*params)
        #     if evaluation.value >= 7:
        #         gamelib.debug_write(f"Spawed f{no_of_scouts} scouts")
        #         game_state.attempt_spawn(WALL, Preset.right_cannon_funnel_block)
        #         game_state.attempt_spawn(SCOUT, [13, 0], no_of_scouts - 6)
        #         game_state.attempt_spawn(SCOUT, [14 ,0], 6)
        #     else:
        #         gamelib.debug_write(f"Scout rush wasn't good enough, weird. Only scored {evaluation.value}")
        for i, location in enumerate(scout_locations):
            num_spawned = game_state.attempt_spawn(SCOUT, location, best_distribution[i])

    def prepare_for_scout_gun(self, game_state, left):
        entrance_location = [26, 13]
        if left:
            # we're preparing for a scout gun on the left
            entrance_location = [3, 11]
        game_state.attempt_remove(entrance_location)
        gamelib.debug_write("Preparing for a scout gun")

    def predict_demolisher_rush_success(self, game_state: gamelib.GameState):
        """Pure demolisher rushes - Test both long and short paths:
        Funnel
        Left Cannon
        Right Cannon
        """
        ATK_FUNNEL, ATK_LEFT_CANNON, ATK_RIGHT_CANNON = range(3)

        heatmaps, _ = sim.get_heatmaps_and_structures(game_state.game_map)
        # we supply heatmaps to get safe spawn locations only
        safe_spawn_locations = self.spawn_locations(game_state, heatmaps=heatmaps)
        number_affordable = game_state.number_affordable(DEMOLISHER)

        results = []

        # we simulate a rush without blocking the funnel - i.e, the board as is
        for location in safe_spawn_locations:
            __map = simulate.copy_internal_map(game_state.game_map)
            params = sim.make_simulation(game_state, game_state.game_map, None, DEMOLISHER, location, 0, number_affordable, copy_safe=False)
            if not params is None:
                evaluation = sim.simulate(*params)
                results.append((evaluation, location, ATK_FUNNEL))

            # reset the map after the simulation back to the real map
            game_state.game_map.set_map_(__map)

        best = max(result[0].value + result[0].length/50 for result in results)

        if best > ATTACK_THRESHOLD:
            # Prefer longer rushes because enemy interceptors may be destroyed before they get near our demolishers
            # todo: if their interceptors are far back then demolish forwards
            ideal_rushes = [result for result in results if result[0].value + result[0].length/50 >= best]
            evaluation, location, ATK_FUNNEL = random.choice(ideal_rushes)
            unit_type = DEMOLISHER
            # Play strategy randomly weighted towards best strategy
            # choice, evaluation = random.choice([item for item in results.items() if item[1].value > best * 0.9])
            # unit_type, *location = choice
            gamelib.debug_write(f'Demolisher Simulator: attack found ({unit_type}) ({location=}) ({evaluation.value=})')
            return evaluation.value, (location, ATK_FUNNEL), False
        else:
            gamelib.debug_write(f'Demolisher Simulator: no rewarding attacks ({best=})')
            return None, None, None

    def optimise_demolisher_rush(self, game_state, demolisher_rush_success_value, d_best_distribution):
        number_affordable = game_state.number_affordable(DEMOLISHER)
        location = d_best_distribution[0]
        count = number_affordable

        if number_affordable > 2:
            cheaper_number_affordable = (number_affordable + 1) // 2

            # run a simulation to see if a reduced rush has a similar effect
            __map = simulate.copy_internal_map(game_state.game_map)
            params = sim.make_simulation(game_state, game_state.game_map, None, DEMOLISHER, location, 0,
                                         cheaper_number_affordable, copy_safe=False)
            if not params is None:
                evaluation = sim.simulate(*params)
                if evaluation.value >= demolisher_rush_success_value * 0.75:
                    count = cheaper_number_affordable
                    gamelib.debug_write('Choosing to run a cheaper demolisher rush.')

            # reset the map after the simulation back to the real map
            game_state.game_map.set_map_(__map)

        game_state.attempt_spawn(DEMOLISHER, location, count)

    def test_demolisher_escort_rush_success(self):
        """What if we use scouts to block damage taken by demolishers?
        Funnel
        Left Cannon
        Right Cannon
        """
        pass

    def spawn_locations(self, game_state, heatmaps=None, reduce=True):
        """All bottom edge locations that aren't occupied. If heatmaps pair provided, only spawn in safe locations."""
        game_map = game_state.game_map

        # get bottom left and bottom right edges. NOTE - DO NOT GENERALISE - THIS ONLY WORKS BECAUSE BL is before BR

        edges = game_map.get_edges()
        spawn_locations = [*edges[game_map.BOTTOM_LEFT], *edges[game_map.BOTTOM_RIGHT]]
        spawn_locations = [s for s in spawn_locations if not game_state.contains_stationary_unit(s)]

        if heatmaps:
            heatmap_0 = heatmaps[0]
            spawn_locations = [s for s in spawn_locations if heatmap_0[s[0]][s[1]] == 0]

        if reduce:
            if reduced_spawn_locations := [s for s in spawn_locations if s[1] == 2 or s[1] == 7]:
                spawn_locations = reduced_spawn_locations

        return spawn_locations


    #OUTDATED
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
