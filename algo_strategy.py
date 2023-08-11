import copy

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import simulate as sim

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""


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

        # these are the "types of turns" that the algorithm can make
        global ATTACK_SCOUTS_R, ATTACK_DEMOLISHERS_R, ATTACK_WAIT, ATTACK_SCOUTS_R_PREP, ATTACK_DEMOLISHERS_R_PREP, ATTACK_DEMOLISHERS_L, ATTACK_DEMOLISHERS_L_PREP, ATTACK_SCOUTS_L, ATTACK_SCOUTS_L_PREP
        ATTACK_WAIT = 0
        ATTACK_SCOUTS_R = 1
        ATTACK_DEMOLISHERS_R = 2
        ATTACK_SCOUTS_R_PREP = 3
        ATTACK_DEMOLISHERS_R_PREP = 4
        ATTACK_DEMOLISHERS_L = 5
        ATTACK_DEMOLISHERS_L_PREP = 6
        ATTACK_SCOUTS_L = 7
        ATTACK_SCOUTS_L_PREP = 8

        self.attack_type = ATTACK_WAIT

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

        sim.initialise((WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR))

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

        gamelib.debug_write(self.score_locations)
        gamelib.debug_write(self.scored_on_locations)

        # if our scout rush failed, in the future make them stronger.
        if self.attack_type == ATTACK_SCOUTS_R and len(self.score_locations) < 4:
            self.always_strong_scouts = True

        # we want to track the enemy's MP
        self.enemy_previous_mp = self.enemy_mp
        self.enemy_mp = game_state.get_resource(MP, 1)

        gamelib.debug_write(f"Enemy has been observed to attack on {self.enemy_scout_attack_threshold}")

        self.strategy(game_state)

        game_state.submit_turn()

    """
    --- STRATEGY ---
    """

    def strategy(self, game_state):
        """
        Taking inspiration from some effective defences we've seen.
        """

        """ -: hardcoded turns 1 and 2 :-"""

        # if it's the first turn, we stall with interceptors.
        # if game_state.turn_number == 0:
        #     self.turn_1_interceptors(game_state)

        # if it's the second turn, we build our initial defences
        if game_state.turn_number == 1:
            gamelib.debug_write("turn2, building initial defences")
            self.build_initial_defences(game_state)

            # on the second turn, we test the waters with a demolisher
            self.attack_type = ATTACK_DEMOLISHERS_R
            self.demolisher_attack_right(game_state)

        # from the third turn, we start our regular gameplay loop
        if game_state.turn_number > 1:

            """ -: anticipating enemy attacks :- """

            # first, we anticipate any enemy attacks
            # this occurs first so our attack type can take into account the spent MP
            enemy_will_attack = self.predict_enemy_attack(game_state)
            # if we think the enemy will attack, we send interceptors
            if enemy_will_attack:
                # check to see if we think the enemy will attack on the left side
                enemy_will_attack_left = self.predict_enemy_left_attack(game_state)
                if enemy_will_attack_left:
                    # on the left side, we reinforce our defences
                    self.reinforce_left_side(game_state)
                else:
                    # on the right side, we send out interceptors
                    self.counter_attack_with_interceptors(game_state, right_side=True)

            """ -: determining our action this turn :- """

            # now we decide what type of attack we're going to perform
            self.attack_type = self.determine_attack_type(game_state)
            gamelib.debug_write(f"Attack type: {self.attack_type}")

            # before our regular defence maintaining, we adapt to where the enemy has struck
            self.build_immediate_reactive_defences(game_state)

            # we maintain our defences
            # occurs after the attack since the attack type determines which walls we won't build
            # this handles removing defences as part of the attack / attack prep
            self.maintain_defences(game_state)

            """ -: performing the attack :- """

            # now we perform the attack
            if self.attack_type == ATTACK_SCOUTS_R:
                # make a scout rush on the right
                self.scout_rush_right(game_state)
            elif self.attack_type == ATTACK_DEMOLISHERS_R:
                # make a demolisher attack on the right
                self.demolisher_attack_right(game_state)

    """
    --- DEFENCES ---
    """

    def build_initial_defences(self, game_state):
        """
        This is our initial setup
        """
        # build turrets & walls

        # old defensive layout
        # wall_locations = [[0, 13], [1, 13], [23, 13], [25, 13], [26, 13], [27, 13], [20, 12], [22, 12], [2, 11], [20, 11], [23, 11], [3, 10], [19, 10], [4, 9], [18, 9], [5, 8], [6, 8], [7, 8], [8, 8], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [24, 13]]
        # turret_locations = [[1, 12], [23, 12]]

        # new defensive layout
        wall_locations = [[0, 13], [1, 13], [2, 13], [3, 13], [23, 13], [24, 13], [25, 13], [26, 13], [27, 13],
                          [20, 12], [22, 12], [4, 11], [20, 11], [23, 11], [5, 10], [19, 10], [6, 9], [18, 9], [7, 8],
                          [8, 8], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8]]
        turret_locations = [[3, 12], [23, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        game_state.attempt_spawn(TURRET, turret_locations)
        # upgraded locations
        wall_upgrade_location = [24, 13]
        turret_upgrade_location = [23, 12]
        game_state.attempt_upgrade([wall_upgrade_location, turret_upgrade_location])

    def build_immediate_reactive_defences(self, game_state):
        """
        Several reactive defences for the algorithm to build based on where they were hit in the previous round.
        This also updates the self.attacked_locations dictionary
        """
        # these are the possible locations to detect
        right_corner_locations = [[27, 13], [26, 12], [25, 11], [24, 10]]
        right_corner = "right_corner"

        left_corner_locations = [[0, 13], [1, 12], [2, 11], [3, 10]]
        left_corner = "left_corner"

        funnel_locations = [[23, 9], [22, 8], [21, 7]]
        funnel = "funnel"

        scout_wall_entrance = [26, 13]

        # a reactive defence to the right corner being attacked
        for location in right_corner_locations:
            if location in self.scored_on_locations:
                gamelib.debug_write("Attacked in right corner!")
                attacked_in_right_corner = True
                self.attacked_locations[right_corner] += 1
        if self.attacked_locations[right_corner] > 0:
            gamelib.debug_write("Building right corner reactive defences")
            # places additional turrets
            extra_turret_locations = [[24, 12], [25, 13]]
            game_state.attempt_spawn(TURRET, extra_turret_locations)

            wall_locations = [[24, 13], [25, 13]]
            if not self.attack_type == ATTACK_SCOUTS_R:
                wall_locations.append(scout_wall_entrance)
            game_state.attempt_spawn(WALL, wall_locations)

            # attempts upgrades
            game_state.attempt_upgrade(extra_turret_locations)
            game_state.attempt_upgrade(wall_locations)

        # a reactive defence to the funnel being attacked
        for location in funnel_locations:
            if location in self.scored_on_locations:
                self.attacked_locations[right_corner] += 1
                gamelib.debug_write("Attacked in funnel location!")
        if self.attacked_locations[funnel] > 0:
            gamelib.debug_write("Building funnel reactive defences")
            turret_locations = [[23, 12], [20, 10], [24, 12], [22, 11]]
            wall_locations = [[23, 13], [22, 12], [20, 12]]
            game_state.attempt_spawn(TURRET, turret_locations)
            game_state.attempt_spawn(WALL, wall_locations)
            game_state.attempt_upgrade(TURRET, turret_locations)
            game_state.attempt_upgrade(WALL, wall_locations)

        # reactive defences to left side being
        if self.attacked_locations[left_corner] > 0:
            self.reinforce_left_side(game_state)

    def maintain_defences(self, game_state):
        gamelib.debug_write("maintaining defences")
        # the type of attack we'll be performing
        # todo: implement 'repairing' damaged structures, where we delete and instantly re-place next turn.
        attack_type = self.attack_type

        scout_wall_entrance_left = [1, 13]
        scout_wall_entrance_right = [26, 13]
        funnel_entrance = [21, 11]
        demolisher_left_entrance = [4, 11]

        # old defensive layout
        # wall_locations = [[0, 13], [1, 13], [23, 13], [25, 13], [27, 13], [20, 12], [22, 12], [2, 11], [20, 11], [23, 11], [3, 10], [19, 10], [4, 9], [18, 9], [5, 8], [6, 8], [7, 8], [8, 8], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8], [24, 13]]
        # turret_locations = [[1, 12], [23, 12]]

        """ -: basic defensive structure :- """

        # new defensive layout
        wall_locations = [[0, 13], [2, 13], [3, 13], [23, 13], [24, 13], [25, 13], [27, 13], [20, 12],
                          [22, 12], [20, 11], [23, 11], [5, 10], [19, 10], [6, 9], [18, 9], [7, 8], [8, 8],
                          [9, 8], [10, 8], [11, 8], [12, 8], [13, 8], [14, 8], [15, 8], [16, 8], [17, 8]]
        turret_locations = [[3, 12], [23, 12]]

        # we build certain structures if we are making a certain attack this turn
        if attack_type == ATTACK_SCOUTS_R:
            # we're making a scout attack!
            # we want to block off the funnel
            wall_locations.insert(0, funnel_entrance)
        else:
            # if we aren't making a scout attack, we can reinforce the wall of the 'gun'
            wall_locations.insert(0, scout_wall_entrance_right)

        if attack_type == ATTACK_SCOUTS_L:
            wall_locations.insert(0, funnel_entrance)
        else:
            wall_locations.insert(0, scout_wall_entrance_left)

        if attack_type == ATTACK_DEMOLISHERS_L:
            # for now, we'll always send demolishers through their entrance
            gamelib.debug_write("demolishers on the left!")
        else:
            wall_locations.insert(0, demolisher_left_entrance)

        # we want to re-build our basic structures
        game_state.attempt_spawn(WALL, wall_locations)
        game_state.attempt_spawn(TURRET, turret_locations)

        # upgrade the 'core' locations
        wall_upgrade_location = [24, 13]
        turret_upgrade_location = [[23, 12], [3, 12]]
        game_state.attempt_upgrade([wall_upgrade_location, *turret_upgrade_location])

        secondary_turret_location = [[20, 10], [3, 12]]
        game_state.attempt_spawn(TURRET, secondary_turret_location)
        game_state.attempt_upgrade(secondary_turret_location)

        # support_locations = [[22, 9], [19, 9], [20, 8], [19, 8], [18, 8], [19, 7], [18, 7], [17, 7], [16, 7], [15, 7], [18, 6], [17, 6], [16, 6], [15, 6], [17, 5], [16, 5], [14, 7], [13, 7], [14, 6]]
        support_locations = [[13, 7], [14, 7], [12, 7], [15, 7], [13, 6], [14, 6]]
        # they prioritise upgraded supports over more supports
        for location in support_locations:
            game_state.attempt_spawn(SUPPORT, location)
            game_state.attempt_upgrade(location)

        # turret at [2, 12]

        secondary_support_locations = [[12, 6], [15, 6], [16, 7], [17, 7], [18, 7]]
        for location in secondary_support_locations:
            game_state.attempt_spawn(SUPPORT, location)
            game_state.attempt_upgrade(location)

        """ -: removing the necessary walls for the attack during the prep phase :- """

        # we remove the following locations depending on the attack type
        if attack_type == ATTACK_SCOUTS_R_PREP:
            # if we're attacking with scouts next turn, open up the wall
            game_state.attempt_remove(scout_wall_entrance_right)
        else:
            # if the funnel has been closed, re-open it, since we aren't attacking with scouts next turn
            game_state.attempt_remove(funnel_entrance)

    def reinforce_left_side(self, game_state):
        """
        This reinforces the left side by building more walls and turrets
        """
        turret_locations = [[3, 12], [2, 12], [0, 13]]
        wall_locations = [[0, 13], [1, 13], [2, 13], [3, 13]]

        game_state.attempt_spawn(TURRET, turret_locations)
        game_state.attempt_upgrade(turret_locations)
        game_state.attempt_spawn(WALL, wall_locations)

    """
    --- PREDICTION & COUNTER-ATTACKS ---
    """

    def predict_enemy_attack(self, game_state):
        """
        Returns true if we think the enemy is going to attack this turn, so we can counter it.
        """
        # the number of scouts the enemy can summon this turn
        enemy_scouts_num = self.enemy_mp // game_state.type_cost(SCOUT)[MP]

        # compare to last attack, or 8 if there was no prior attack
        # todo: also predict enemy demolisher attacks by introducing enemy_demolisher_attack_threshold
        if self.enemy_scout_attack_threshold is None:
            # we haven't seen them attack yet
            # by default we wait until 8 scouts can be spawned
            if enemy_scouts_num >= 8:
                return True
        else:
            # we have seen them attack
            # the minimum number they're willing to attack with is in self.enemy_attack_threshold
            if enemy_scouts_num >= self.enemy_scout_attack_threshold:
                return True
        # if neither of these occur, we return false
        return False

    def predict_enemy_left_attack(self, game_state):
        """
        This predicts if the enemy will launch an attack on the left side.
        Check to see if any walls are being deleted to make way for attacking units.
        """
        # these are the locations we will check
        enemy_left_wall_locations = [[0, 14], [1, 14], [2, 14], [3, 14]]
        enemy_wall_marked_for_removal = False
        for location in enemy_left_wall_locations:
            if location in self.enemy_deletions:
                enemy_wall_marked_for_removal = True

        # if they're removing a wall, we assume they are going to attack here next turn.
        return enemy_wall_marked_for_removal

    def counter_attack_with_interceptors(self, game_state, right_side=True):
        """
        We think the enemy is going to attack, so we send out an interceptor to stop it.
        """
        # todo: make this do something when called on the left side
        # if we're being attacked on the right side
        if right_side:
            # the location to spawn them at
            interceptor_location = [[18, 4]]
            # the number to spawn
            num_interceptors = 0

            # if we haven't seen the enemy attack yet, we don't send any
            # if we have seen them attack, then we send the number needed to defeat the attack
            if self.enemy_scout_attack_threshold is None:
                num_interceptors = 0
                return
            else:
                enemy_scouts_incoming = self.enemy_mp
                num_interceptors = int(enemy_scouts_incoming // 5)

            # spawning the interceptors
            game_state.attempt_spawn(INTERCEPTOR, interceptor_location, num_interceptors)

    def turn_1_interceptors(self, game_state):
        interceptor_locations = [[5, 8], [22, 8]]
        game_state.attempt_spawn(INTERCEPTOR, interceptor_locations)

    """
    --- POSITION ANALYSIS ---
    """

    def enemy_side_is_open(self, game_state, left=False):
        """
        Returns true if the enemy's left side is open
        """
        wall_locations = []
        if left:
            wall_locations = [[1, 14], [2, 14]]
        else:
            wall_locations = [[25, 14], [26, 14]]

        for location in wall_locations:
            wall = game_state.contains_stationary_unit(location)
            # if there isn't a wall, then the enemy has opened up their left
            if not wall:
                return True
        # if both the walls are there, then it isn't open
        return False

    def strong_scout_attack_needed(self, game_state):
        """"
        Returns whether we need a 'strong' scout attack (6 suicides) or a 'weak' one (2 suicides).
        This is true if the enemy's defensive wall is upgraded, false otherwise.
        """
        if self.always_strong_scouts:
            return True
        enemy_near_wall_locations = [[25, 14], [26, 14], [27, 14]]
        upgraded_walls = False
        for location in enemy_near_wall_locations:
            enemy_wall = game_state.contains_stationary_unit(location)
            if enemy_wall:
                if enemy_wall.unit_type == WALL or enemy_wall:
                    if enemy_wall.upgraded:
                        upgraded_walls = True
        return upgraded_walls

    def num_demolishers_needed(self, game_state):
        """
        Returns the number of demolishers that should be used in a scout rush
        """
        breach_location = [26, 13]
        attackers = game_state.get_attackers(breach_location, 0)
        num_turrets = 0
        num_upgraded_turrets = 0
        for unit in attackers:
            if unit.unit_type == TURRET:
                num_turrets += 1
                if unit.upgraded:
                    num_upgraded_turrets += 1
        return num_upgraded_turrets * 2 + ( (num_turrets - num_upgraded_turrets) // 2 )

    """
    --- ATTACKING ---
    """

    def determine_attack_type(self, game_state):
        """
        Returns either ATTACK_WAITING, ATTACK_SCOUTS, OR ATTACK_DEMOLISHERS
        """
        last_turn_action = self.attack_type

        """ -: resolving a previously prepared attack :- """

        # first, we look at our prepared actions.
        # if we prepared to send demolishers last turn, then we'll fire them off this turn
        if last_turn_action == ATTACK_DEMOLISHERS_R_PREP:
            return ATTACK_DEMOLISHERS_R

        # if we prepared to send scouts last turn, we'll fire them off this turn.
        if last_turn_action == ATTACK_SCOUTS_R_PREP:
            # we make sure all our walls have been re-built
            # if not, we wait another turn
            # necessary_walls = [[21, 11], [19, 10], [18, 9], [7, 8], [8, 8], [9, 8], [10, 8], [11, 8], [12, 8], [13, 8],
            #                    [14, 8], [15, 8], [16, 8], [17, 8]]
            # walls_present = True
            # for location in necessary_walls:
            #     wall = game_state.contains_stationary_unit(location)
            #     if not wall:
            #         return ATTACK_SCOUTS_R_PREP

            # TODO: We should check to see if the scouts are sufficient. Check these scouts will actually deal
            #  damage, otherwise we should wait another turn.
            return ATTACK_SCOUTS_R

        """ -: conditions for demolisher attacks :- """

        # next, we check to see if we want to send demolishers.
        # if we sent demolishers last turn and it worked, let's do that again!
        if last_turn_action == ATTACK_DEMOLISHERS_R and len(self.score_locations) > 0:
            gamelib.debug_write("Our demolishers worked last turn, so let's do them again!")
            return ATTACK_DEMOLISHERS_R

        # we simulate a demolisher attack to see how much damage it would deal
        # if it would deal a lot, then we fire it.
        demolisher_location = [25, 11]
        demolisher_count = game_state.number_affordable(DEMOLISHER)
        params = sim.make_simulation(game_state, game_state.game_map, None, DEMOLISHER, demolisher_location, 0, demolisher_count)
        if params is not None:
            # we've made a simulation, let's see how much damage it dealt
            # todo: un-break the simulation so we can un-comment this
            # evaluation = sim.simulate(*params)
            structure_points = 0 #evaluation.points_destroyed
            # if we destroyed enough structure points worth, then we want to fire this demolisher next turn
            gamelib.debug_write(f"Demolisher would deal {structure_points}")
            if structure_points >= 8:
                if last_turn_action == ATTACK_DEMOLISHERS_R or last_turn_action == ATTACK_DEMOLISHERS_R_PREP:
                    return ATTACK_DEMOLISHERS_R
                else:
                    return ATTACK_DEMOLISHERS_R_PREP

        """ -: conditions for scout attacks :- """

        # now, we don't want to send demolishers.
        # let's see if we want to make a scout attack.
        # we want a minimum payload of 6 damaging scouts
        no_of_scouts = 6
        if self.strong_scout_attack_needed(game_state):
            # if we need a strong attack we want 6 suicides
            no_of_scouts += 6
        else:
            # if we only need a weak one, we'll use 2 scouts
            no_of_scouts += 2
        no_of_demolishers = self.num_demolishers_needed(game_state)
        if game_state.project_future_MP(1) >= (game_state.type_cost(SCOUT)[MP] * no_of_scouts) + (game_state.type_cost(DEMOLISHER)[MP] * 0):
            # if we can, we'll launch a scout attack next turn.
            return ATTACK_SCOUTS_R_PREP

        # if we've made it this far, none of these seem like particularly good options.
        # we'll just wait this turn out and build up some more MP
        return ATTACK_WAIT

    def scout_rush_right(self, game_state):
        """
        We're performing a scout rush
        """
        # the number of scouts we can afford
        scouts_affordable = game_state.number_affordable(SCOUT)
        strong_attack_needed = self.strong_scout_attack_needed(game_state)

        scout_suicide_location = [[14, 0]]
        scout_payload_location = [[13, 0]]

        no_of_suicide_scouts = 2

        # we send the required number of suicide scouts
        if strong_attack_needed:
            # if we need a strong attack, we send 6 suicide scouts
            no_of_suicide_scouts = 6

        demolisher_location = [[22, 8]]
        no_of_demolishers = self.num_demolishers_needed(game_state)

        # spawn demolishers
       #  game_state.attempt_spawn(DEMOLISHER, demolisher_location, no_of_demolishers)

        # spawn suicide scouts
        game_state.attempt_spawn(SCOUT, scout_suicide_location, no_of_suicide_scouts)

        # now that we've prepared the suicide scouts, we send the payload scouts
        game_state.attempt_spawn(SCOUT, scout_payload_location, scouts_affordable - no_of_suicide_scouts)

    def demolisher_attack_right(self, game_state):
        """
        We're performing a demolisher attack
        """
        demolisher_location = [[25, 11]]
        demolishers_affordable = game_state.number_affordable(DEMOLISHER)

        # we simply spawn as many demolishers as we can
        game_state.attempt_spawn(DEMOLISHER, demolisher_location, demolishers_affordable)

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
