import gamelib

"""
Helper functions for Terminal development.

Usage: insert print_state before submit_turn
        dev_helper.print_state(_globals, gamelib.debug_write)
        _globals.submit_turn()

Prints a map from the perspective of the player.

Upgrades are indicated by ! to the right of the static unit.

When multiple units are stacked,
- the 1st character is unit types, with ? representing a mix of types,
- the 2nd character is unit count, with + representing 10+ units.

By Kevin Gao.
"""

# You can customise these codes
# _SHORTER_CODES = {'FF': 'O ', 'DF': 'X ', 'EF': '# ',
#                   'PI': 'PI', 'EI': 'EI', 'SI': 'SI'}
_SHORTER_CODES = {'FF': 'O ', 'DF': 'X ', 'EF': '# ',
                  'PI': 'Sc', 'EI': 'De', 'SI': 'In'}


def print_state(game_state: gamelib.GameState, _print_function=gamelib.debug_write) -> None:
    gamelib.debug_write(f'My SP/MP {game_state.get_resources(0)} | Enemy SP/MP {game_state.get_resources(1)}')
    print_map(game_state.game_map, _print_function)


def print_map(game_map: gamelib.GameMap, _print_function=gamelib.debug_write) -> None:
    for y in reversed(range(game_map.ARENA_SIZE)):
        # s = [str(y if y < game_map.HALF_ARENA else game_map.ARENA_SIZE - y - 1).ljust(3, ' ')]
        s = [str(y).ljust(3, ' ')]
        for x in range(game_map.ARENA_SIZE):
            if game_map.in_arena_bounds((x, y)):
                s.append(_get_tile_string(x, y, game_map))
            # elif y in (0, game_map.ARENA_SIZE - 1) and x not in (12, 15):
            #     s.append(str(x).ljust(2, ' '))
            else:
                s.append('  ')
        _print_function(''.join(s))
    _print_function('** ' + ''.join(s.ljust(2, ' ') for s in map(str, range(game_map.ARENA_SIZE))))


def _get_tile_string(x, y, game_map: gamelib.GameMap):
    units = game_map[x, y]
    if units:
        code = _SHORTER_CODES.get(units[0].unit_type, units[0].unit_type)
        if units[0].upgraded:
            return code[0] + '!'
        elif len(units) > 1:
            all_same_type = all(unit.unit_type == units[0].unit_type for unit in units)
            types = code[0] if all_same_type else '?'
            count = str(len(units)) if len(units) < 9 else '+'
            return types + count
        else:
            return code
    else:
        return '. '

