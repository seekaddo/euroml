"""
Copyright (c) Dennis Kwame Addo

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_URL = "https://www.euro-millions.com"
ARCHIVE_URL = f"{BASE_URL}/de-at/{{year}}-zahlen-archiv"
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
FIRST_ARCHIVE_YEAR = 2004
DRAW_HREF_RE = re.compile(r"/de-at/zahlen/(?P<draw_date>\d{2}-\d{2}-\d{4})")
REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "de-AT,de;q=0.9,en;q=0.8",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)
    return session


def dataset_file_for_year(year: int, dataset_dir: Path = DATASET_DIR) -> Path:
    return dataset_dir / f"{year}_eml.json"


def fetch_page(url: str, session: requests.Session | None = None) -> str:
    owned_session = session is None
    session = session or build_session()
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    finally:
        if owned_session:
            session.close()


def format_draw_date(draw_href: str) -> str:
    match = DRAW_HREF_RE.search(draw_href)
    if not match:
        raise ValueError(f"Could not parse draw date from href: {draw_href}")

    parsed = datetime.strptime(match.group("draw_date"), "%d-%m-%Y")
    return parsed.strftime("%d.%m.%Y")


def extract_balls(container: Tag) -> list[list[int]]:
    main_numbers: list[int] = []
    lucky_stars: list[int] = []

    for item in container.find_all("li", class_="resultBall"):
        classes = item.get("class", [])
        try:
            number = int(item.get_text(strip=True))
        except ValueError as exc:
            raise ValueError(f"Unexpected ball value: {item!r}") from exc

        if "lucky-star" in classes:
            lucky_stars.append(number)
        else:
            main_numbers.append(number)

    if len(main_numbers) != 5 or len(lucky_stars) != 2:
        raise ValueError(
            "Unexpected draw shape: "
            f"{len(main_numbers)} main numbers and {len(lucky_stars)} lucky stars"
        )

    return [main_numbers, lucky_stars]


def parse_result_row(result_row: Tag) -> tuple[str, list[list[int]]]:
    link = result_row.find("a", href=DRAW_HREF_RE)
    balls_container = result_row.find("ul", class_="balls")

    if link is None or balls_container is None:
        raise ValueError("Archive row is missing the expected link or balls container")

    draw_date = format_draw_date(link["href"])
    return draw_date, extract_balls(balls_container)


def parse_archive(html: str) -> dict[str, list[list[int]]]:
    soup = BeautifulSoup(html, "html.parser")
    results_table = soup.find("table", id="resultsTable")
    if results_table is None:
        raise ValueError("Could not find resultsTable on the archive page")

    draw_map: dict[str, list[list[int]]] = {}
    for result_row in results_table.find_all("tr", class_="resultRow"):
        draw_date, balls = parse_result_row(result_row)
        draw_map[draw_date] = balls

    if not draw_map:
        raise ValueError("No draw rows were parsed from the archive page")

    return sort_draw_map(draw_map)


def sort_draw_map(draw_map: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    return dict(
        sorted(
            draw_map.items(),
            key=lambda item: datetime.strptime(item[0], "%d.%m.%Y"),
        )
    )


def fetch_archive_results(year: int, session: requests.Session | None = None) -> dict[str, list[list[int]]]:
    html = fetch_page(ARCHIVE_URL.format(year=year), session=session)
    draw_map = parse_archive(html)

    year_values = {datetime.strptime(draw_date, "%d.%m.%Y").year for draw_date in draw_map}
    if year_values != {year}:
        raise ValueError(f"Archive {year} returned unexpected years: {sorted(year_values)}")

    return draw_map


def write_draw_map(output_path: Path, draw_map: dict[str, list[list[int]]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(draw_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def refresh_year(year: int, dataset_dir: Path = DATASET_DIR, session: requests.Session | None = None) -> Path:
    draw_map = fetch_archive_results(year, session=session)
    output_path = dataset_file_for_year(year, dataset_dir)
    write_draw_map(output_path, draw_map)
    logging.info("Saved %s draws to %s", len(draw_map), output_path)
    return output_path


def refresh_range(
    start_year: int,
    end_year: int,
    dataset_dir: Path = DATASET_DIR,
    session: requests.Session | None = None,
) -> list[Path]:
    if start_year > end_year:
        raise ValueError("start_year cannot be greater than end_year")

    owned_session = session is None
    session = session or build_session()
    try:
        return [refresh_year(year, dataset_dir=dataset_dir, session=session) for year in range(start_year, end_year + 1)]
    finally:
        if owned_session:
            session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EuroMillions yearly archive data into the local dataset folder.",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to download. Overrides --start-year and --end-year.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="First year in the archive range to download.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last year in the archive range to download.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Directory where *_eml.json files are written. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--all-history",
        action="store_true",
        help=f"Download the full archive from {FIRST_ARCHIVE_YEAR} through the current year.",
    )
    return parser.parse_args()


def resolve_year_range(args: argparse.Namespace) -> tuple[int, int]:
    current_year = date.today().year
    if args.year is not None:
        return args.year, args.year

    if args.all_history:
        return FIRST_ARCHIVE_YEAR, current_year

    start_year = args.start_year if args.start_year is not None else current_year
    end_year = args.end_year if args.end_year is not None else start_year
    return start_year, end_year


def main() -> None:
    args = parse_args()
    start_year, end_year = resolve_year_range(args)
    refresh_range(start_year, end_year, dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
