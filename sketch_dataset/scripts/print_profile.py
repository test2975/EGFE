import argparse
from pstats import Stats, SortKey


def print_profile(profile_name: str):
    ps = Stats(profile_name)
    ps.strip_dirs().sort_stats(SortKey.TIME).print_stats(.3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print profile stats')
    parser.add_argument('profile_name', help='Profile name')
    args = parser.parse_args()
    print_profile(args.profile_name)
