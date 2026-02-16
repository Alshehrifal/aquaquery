"""Pre-cache Argo data locally for faster queries.

Downloads data for known ocean basins so subsequent queries
hit the local argopy cache instead of remote GDAC servers.

Usage:
    # Cache last 90 days for all basins (default)
    python -m backend.scripts.precache_argo

    # Specific basins only
    python -m backend.scripts.precache_argo --basins north_atlantic mediterranean

    # Custom time window
    python -m backend.scripts.precache_argo --recent-days 30
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

from backend.config import get_settings
from backend.data.loader import OCEAN_BASINS, ArgoDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-cache Argo data for known ocean basins.",
    )
    parser.add_argument(
        "--basins",
        nargs="+",
        choices=list(OCEAN_BASINS.keys()),
        default=None,
        help="Basins to cache (default: all)",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=90,
        help="Number of recent days to cache (default: 90)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    basins = args.basins or list(OCEAN_BASINS.keys())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.recent_days)

    time_range = (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    settings = get_settings()
    loader = ArgoDataLoader(settings=settings)

    logger.info(
        "Pre-caching %d basins for %s to %s",
        len(basins),
        time_range[0],
        time_range[1],
    )

    succeeded = 0
    failed = 0

    for basin_name in basins:
        bounds = OCEAN_BASINS[basin_name]
        logger.info("Caching basin: %s %s", basin_name, bounds)
        try:
            ds = loader.fetch_region(
                lat_bounds=(bounds["lat_min"], bounds["lat_max"]),
                lon_bounds=(bounds["lon_min"], bounds["lon_max"]),
                time_range=time_range,
            )
            n_profiles = ds.sizes.get("N_PROF", 0)
            logger.info(
                "Cached %s: %d profiles",
                basin_name,
                n_profiles,
            )
            succeeded += 1
        except Exception as e:
            logger.error("Failed to cache %s: %s", basin_name, e)
            failed += 1

    logger.info(
        "Pre-cache complete: %d succeeded, %d failed out of %d basins",
        succeeded,
        failed,
        len(basins),
    )

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
