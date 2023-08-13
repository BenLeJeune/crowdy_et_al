import dev_helper
import simulate as sim
import json
from gamelib.game_state import GameState
import gamelib

print("hi")


serialized_string = '{"p2Units":[[],[],[],[],[],[],[],[]],"turnInfo":[0,0,-1,0],"p1Stats":[30.0,40.0,5.0,0],"p1Units":[[],[],[],[],[],[],[],[]],"p2Stats":[30.0,40.0,5.0,0],"events":{"selfDestruct":[],"breach":[],"damage":[],"shield":[],"move":[],"spawn":[],"death":[],"attack":[],"melee":[]}}'
config = '{"debug":{"printMapString":false,"printTStrings":false,"printActStrings":false,"printHitStrings":false,"printPlayerInputStrings":false,"printBotErrors":true,"printPlayerGetHitStrings":false},"unitInformation":[{"iconxScale":0.4,"turnsRequiredToRemove":1,"refundPercentage":0.75,"cost1":1.0,"getHitRadius":0.01,"upgrade":{"iconxScale":0.4,"cost1":2.0,"icon":"S3_filter","iconyScale":0.5,"startHealth":120.0},"unitCategory":0,"display":"Filter","icon":"S3_filter","iconyScale":0.5,"startHealth":45.0,"shorthand":"FF"},{"iconxScale":0.5,"refundPercentage":0.75,"cost1":3.0,"upgrade":{"iconxScale":0.5,"shieldBonusPerY":0.3,"shieldRange":10.0,"shieldPerUnit":6.0,"icon":"S3_encryptor","iconyScale":0.5},"shieldRange":4.5,"shieldPerUnit":3.0,"display":"Encryptor","icon":"S3_encryptor","iconyScale":0.5,"shorthand":"EF","turnsRequiredToRemove":1,"shieldBonusPerY":0.0,"getHitRadius":0.01,"unitCategory":0,"startHealth":30.0,"shieldDecay":0.0},{"iconxScale":0.5,"attackRange":2.5,"refundPercentage":0.75,"cost1":4.0,"upgrade":{"iconxScale":0.5,"attackDamageWalker":20.0,"attackRange":4.5,"cost1":6.0,"icon":"S3_destructor","iconyScale":0.5},"display":"Destructor","icon":"S3_destructor","iconyScale":0.5,"shorthand":"DF","attackDamageWalker":8.0,"turnsRequiredToRemove":1,"getHitRadius":0.01,"unitCategory":0,"startHealth":75.0,"attackDamageTower":0.0},{"iconxScale":0.7,"attackRange":2.5,"selfDestructDamageTower":15.0,"cost2":1.0,"metalForBreach":1.0,"display":"Ping","icon":"S3_ping","selfDestructStepsRequired":5,"iconyScale":0.5,"shorthand":"PI","playerBreachDamage":1.0,"speed":1.0,"attackDamageWalker":2.0,"getHitRadius":0.01,"unitCategory":1,"selfDestructDamageWalker":15.0,"startHealth":20.0,"selfDestructRange":1.5,"attackDamageTower":2.0},{"iconxScale":0.47,"attackRange":4.5,"selfDestructDamageTower":5.0,"cost2":3.0,"metalForBreach":1.0,"display":"EMP","icon":"S3_emp","selfDestructStepsRequired":5,"iconyScale":0.5,"shorthand":"EI","playerBreachDamage":2.0,"speed":0.5,"attackDamageWalker":8.0,"getHitRadius":0.01,"unitCategory":1,"selfDestructDamageWalker":5.0,"startHealth":5.0,"selfDestructRange":1.5,"attackDamageTower":8.0},{"iconxScale":0.5,"attackRange":5.5,"selfDestructDamageTower":50.0,"cost2":2.0,"metalForBreach":1.0,"display":"Scrambler","icon":"S3_scrambler","selfDestructStepsRequired":5,"iconyScale":0.5,"shorthand":"SI","playerBreachDamage":1.0,"speed":0.25,"attackDamageWalker":20.0,"getHitRadius":0.01,"unitCategory":1,"selfDestructDamageWalker":50.0,"startHealth":50.0,"selfDestructRange":1.5,"attackDamageTower":0.0},{"iconxScale":0.4,"display":"Remove","icon":"S3_removal","iconyScale":0.5,"shorthand":"RM"},{"iconxScale":0.4,"display":"Upgrade","icon":"S3_upgrade","iconyScale":0.5,"shorthand":"UP"}],"timingAndReplay":{"playReplaySave":1,"waitTimeBotMax":35000,"waitTimeManual":1820000,"waitForever":false,"playWaitTimeBotSoft":5000,"waitTimeEndGame":3000,"waitTimeBotSoft":5000,"playWaitTimeBotMax":35000,"replaySave":1,"storeBotTimes":true,"waitTimeStartGame":3000},"resources":{"bitsPerRound":5.0,"coresPerRound":5.0,"startingBits":5.0,"turnIntervalForBitCapSchedule":10,"turnIntervalForBitSchedule":10,"bitRampBitCapGrowthRate":5.0,"bitDecayPerRound":0.25,"roundStartBitRamp":10,"bitGrowthRate":1.0,"startingHP":30.0,"startingCores":40.0,"maxBits":150.0},"seasonCompatibilityModeP2":5,"seasonCompatibilityModeP1":5}'


config = json.loads(config)

global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
WALL = config["unitInformation"][0]["shorthand"]
SUPPORT = config["unitInformation"][1]["shorthand"]
TURRET = config["unitInformation"][2]["shorthand"]
SCOUT = config["unitInformation"][3]["shorthand"]
DEMOLISHER = config["unitInformation"][4]["shorthand"]
INTERCEPTOR = config["unitInformation"][5]["shorthand"]
MP = 1
SP = 0

game_state = GameState(config, serialized_string)

game_map = game_state.game_map
horiz_wall = [[i, 14] for i in range(28)]
unit_types = [WALL for i in range(28)]
player_indexes = [1 for i in range(28)]
sim.initialise((WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR))
sim.PRINT_MAP_STEPS = True

__map = sim.copy_internal_map(game_map)
map_params = sim.make_simulation_map(game_state, unit_types, horiz_wall, player_indexes,
                                     copy_safe=False,
                                     remove_locations=[[26, 13]])
params = sim.make_simulation(*map_params, [SCOUT, SCOUT], [[14, 0], [15, 1]], 0, [15, 6], copy_safe=False)

evaluation = sim.simulate(*params)

game_state.game_map.set_map_(__map) # resets it?

dev_helper.print_state(game_state, print)
dev_helper.print_map(game_map, print)

print(evaluation.value)

print("hi")