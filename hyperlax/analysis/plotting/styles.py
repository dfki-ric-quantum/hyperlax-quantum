"""Color and marker schemes for plotting (family/FA aware)."""

import seaborn as sns
from matplotlib.colors import to_hex, to_rgb
import colorsys


# Stable preferred orders (you can extend/override from callers if needed)
FAMILY_ORDER = ["ppo", "dqn", "sac"]
FA_ORDER     = ["mlp", "tmlp", "drpqc"]

# FA marker shapes (semantic + consistent)
FA_MARKERS_MPL = {"mlp": "o", "tmlp": "s", "drpqc": "^"}
FA_MARKERS_PLY = {"mlp": "circle", "tmlp": "square", "drpqc": "triangle-up"}

# Lightness multipliers aligned to FA_ORDER indices
_FA_LIGHTNESS = [1.00, 0.85, 1.15]  # mlp, tmlp, drpqc


def _hex_to_hls(hex_color: str):
    r, g, b = to_rgb(hex_color)
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h, l, s) -> str:
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return to_hex((r, g, b))


def _vary_lightness(hex_color: str, factor: float) -> str:
    h, l, s = _hex_to_hls(hex_color)
    l = max(0.0, min(1.0, l * factor))
    return _hls_to_hex(h, l, s)


class PlotStyle:
    """
    Family/FA aware style:
      - Family = base hue from colorblind palette
      - FA     = lightness variant of family hue
    """

    def __init__(self, family_order=None, fa_order=None):
        self.family_order = family_order or FAMILY_ORDER
        self.fa_order     = fa_order or FA_ORDER

        # Base palette for families
        base = sns.color_palette("colorblind", n_colors=max(3, len(self.family_order)))
        self._family_base = {fam: to_hex(base[i % len(base)])
                             for i, fam in enumerate(self.family_order)}

        # Global cache for "family-fa" -> hex
        self._cache = {}

    # ----- Colors -----

    def color_for(self, family: str, fa: str | None) -> str:
        key = f"{family}-{fa}" if fa else family
        if key in self._cache:
            return self._cache[key]

        base = self._family_base.get(family, to_hex(sns.color_palette("colorblind", 3)[0]))
        if fa and fa in self.fa_order:
            idx = self.fa_order.index(fa)
            color = _vary_lightness(base, _FA_LIGHTNESS[idx])
        else:
            color = base
        self._cache[key] = color
        return color

    def colors_for_series(self, series_keys: list[str]) -> dict[str, str]:
        """
        keys are algo labels (like 'ppo-mlp (x)').
        """
        from .core import split_family_fa
        out = {}
        for k in series_keys:
            fam, fa = split_family_fa(k)
            out[k] = self.color_for(fam, fa)
        return out

    # ----- Markers -----

    def mpl_marker_for_fa(self, fa: str | None) -> str:
        return FA_MARKERS_MPL.get(fa or "", "o")

    def plotly_marker_for_fa(self, fa: str | None) -> str:
        return FA_MARKERS_PLY.get(fa or "", "circle")


# Global instance
DEFAULT_STYLE = PlotStyle()
