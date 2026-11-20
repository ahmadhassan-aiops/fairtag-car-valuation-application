"""FairTag colours and fonts: a dealer's window sticker. Warm paper, black header bars, condensed type, an orange price
tag. Chart series use a colour-blind-checked order; verdict colours always come with an icon and a label."""

BG = "#f3f1ec"          # warm paper
PANEL = "#ffffff"
LINE = "#dcd7cc"
INK = "#16181d"
MUTED = "#4f5560"
FAINT = "#6b717c"
BAR = "#16181d"         # panel header bar
TAG = "#d9480f"         # price-tag orange (4.6:1 on white)
TAG_DARK = "#a8380b"
TAG_SOFT = "#fff1e8"
GRID = "#ebe7de"

# Where an asking price sits against the fair range
BELOW, BELOW_SOFT = "#2b8a3e", "#ebfbee"
WITHIN, WITHIN_SOFT = "#495057", "#f1f3f5"
ABOVE, ABOVE_SOFT = "#c92a2a", "#fff5f5"
VERDICT_STYLE = {
    "Below the range": (BELOW, BELOW_SOFT, "trending_down"),
    "Within the range": (WITHIN, WITHIN_SOFT, "check_circle"),
    "Above the range": (ABOVE, ABOVE_SOFT, "trending_up"),
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated order (adjacent pairs); at most four series
UP, DOWN = "#2b8a3e", "#c92a2a"                          # euros added / taken off in the price build-up

HEAD = "'Barlow Condensed', 'Arial Narrow', sans-serif"
BODY = "'Barlow', 'Segoe UI', sans-serif"
