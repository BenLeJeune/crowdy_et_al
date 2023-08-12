"""Running this script will print the results of the profiling."""

import pstats
from pstats import SortKey
# PROFILING_DIR = 'crowdy_et_al/_timing_profile'

PROFILING_DIR = 'crowdy_et_al/game_profile/_timing_profile_game_r0'

if __name__ == "__main__":
    p = pstats.Stats(''.join(PROFILING_DIR.split('/')[1:]))
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(35)
