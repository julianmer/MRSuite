"""MRSuite logo — path-traced in Cycles, transparent film.

Run:  /Applications/Blender.app/Contents/MacOS/Blender -b -P render_logo.py

Geometry rules are identical to the vector build: 27 identical unit cubes, modules are
pure axis-aligned integer-cell translations, nothing sliced or resized. What changes is
that the glass is *actually* glass — Principled transmission with a real IOR — lit by a
studio dome that is hidden from camera rays, so the film stays transparent while the
glass still has an environment to refract.

Grid convention: +X right, +Y away from viewer, +Z up. The camera sees +X (right face),
-Y (front face) and +Z (top).
"""
import bpy, bmesh, math, json, os
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("LOGO_OUTDIR", HERE), "render-" + os.environ.get(
    "LOGO_NAME", os.environ.get("LOGO_VARIANT", "crystal")) + ".png")
VARIANT = os.environ.get("LOGO_VARIANT", "crystal")
CRYSTAL = VARIANT.startswith("crystal")             # modules built from their contents
SEAMS = VARIANT != "clean"                          # engraved cube divisions on glass
SIMPLE = VARIANT == "simple"                        # stripped-back mark
TURN = float(os.environ.get("LOGO_TURN", 0.0))      # top-layer twist, degrees
YAW = float(os.environ.get("LOGO_YAW", 45.0))       # 45 is dead-on isometric

NEAT = VARIANT.startswith(("neat", "mark"))   # fewer, larger features
MARK = VARIANT.startswith(("mark", "solid"))  # logo build: restrained palette
SOLID = VARIANT.startswith(("solid", "jax", "cube"))  # solid coloured modules
BARE = VARIANT.startswith("cube")             # geometry only by default
# none | spectrum | all. "spectrum" keeps the one mark that actually says magnetic
# resonance and drops the other two, so the mark means something without getting busy.
MARKS = os.environ.get("LOGO_MARKS", "none" if BARE else "all")
TEX = os.environ.get("LOGO_TEX", "flat")      # flat | lit | rubik
PANELS = os.environ.get("LOGO_PANELS", "uniform")   # uniform|body|modules|all
VARY_BODY = PANELS in ("body", "all")
VARY_MOD = PANELS in ("modules", "all")
FLAT = VARIANT.startswith("jax") or (BARE and TEX != "lit")
CRYSTAL = CRYSTAL or SIMPLE or NEAT

# Crystal family: no glass at all. A wire cage states the cube geometry and the module's
# own contents fill it, so the spectrum and the network *are* the cubes. A scheme sets
# the six sticker colours, the cube's frame, the cage, and the module accent ramp.
RUBIK = {"R": "#C41E3A", "O": "#FF5800", "Y": "#FFD500",
         "G": "#009E60", "B": "#0051BA", "W": "#FFFFFF"}
STYLES = {
    "crystal":      dict(cage=(0.045, 0.050, 0.062, 1), metal=0.55, glow=0.35,
                         accent=("#12BBD6", "#5560E0"), crest="#CFF1F7"),
    "crystal-gold": dict(cage=(0.42, 0.26, 0.07, 1), metal=0.90, glow=0.30,
                         accent=("#F2B23C", "#D9622A"), crest="#FFE9BC"),
    "crystal-neon": dict(cage=(0.02, 0.03, 0.05, 1), metal=0.35, glow=2.20,
                         accent=("#1BE8C4", "#A855F7"), crest="#B8FFF0"),
    "crystal-ice":  dict(cage=(0.62, 0.70, 0.80, 1), metal=0.25, glow=0.25,
                         accent=("#3FA9E0", "#6C7BE8"), crest="#EAF9FF"),

}

# Colour schemes. Each is offered on both the full-detail crystal build (`crystal-<x>`)
# and the pared-back neat build (`neat-<x>`), so palette and detail level are independent.
SCHEMES = {
    "classic": dict(cage=(0.05, 0.055, 0.07, 1), metal=0.5, glow=0.3,
                    accent=("#12BBD6", "#5560E0"), crest="#CFF1F7"),
    "mono":    dict(pal={"R": "#4A515C", "O": "#6C7481", "Y": "#9AA3AF",
                         "G": "#333A44", "B": "#7E8792", "W": "#EDEFF2"},
                    plastic=(0.030, 0.033, 0.040, 1),
                    cage=(0.09, 0.10, 0.12, 1), metal=0.5, glow=0.5,
                    accent=("#00C2CB", "#00C2CB"), crest="#D6FAFC"),
    "duo":     dict(pal={"R": "#E4572E", "O": "#F49F0A", "Y": "#F4E3C1",
                         "G": "#14556E", "B": "#1E7F9E", "W": "#F2F6F7"},
                    plastic=(0.012, 0.030, 0.042, 1),
                    cage=(0.05, 0.13, 0.17, 1), metal=0.45, glow=0.35,
                    accent=("#F49F0A", "#E4572E"), crest="#FFE9C7"),
    "pastel":  dict(pal={"R": "#F4A6A6", "O": "#F8C99B", "Y": "#F6EDA8",
                         "G": "#A9D8BE", "B": "#A6C6E8", "W": "#FAFAF6"},
                    plastic=(0.115, 0.120, 0.135, 1),
                    cage=(0.24, 0.25, 0.28, 1), metal=0.3, glow=0.25,
                    accent=("#7FC8D8", "#A79BE0"), crest="#F0FBFF"),
    "dark":    dict(pal={"R": "#FF4D6D", "O": "#FF9F1C", "Y": "#FFE066",
                         "G": "#2EC4B6", "B": "#4CC9F0", "W": "#E9ECEF"},
                    plastic=(0.008, 0.009, 0.012, 1),
                    cage=(0.03, 0.05, 0.07, 1), metal=0.6, glow=1.6,
                    accent=("#4CC9F0", "#B14CF0"), crest="#CFF6FF"),
    "earth":   dict(pal={"R": "#B5523C", "O": "#D98E4A", "Y": "#E3C567",
                         "G": "#6B8F5E", "B": "#3E6B78", "W": "#EDE6D6"},
                    plastic=(0.045, 0.038, 0.028, 1),
                    cage=(0.16, 0.13, 0.09, 1), metal=0.6, glow=0.3,
                    accent=("#D98E4A", "#6B8F5E"), crest="#F6EBD4"),
}
# Logo builds. Same composition as crystal — cube plus the three extracted modules —
# but the six Rubik hues collapse to three tones of one colour so the 3x3 grid still
# reads, plus a single accent for the modules. Four colours total instead of seven.
def _tones(a, b, c):
    return {"R": c, "O": b, "Y": a, "G": b, "B": c, "W": a}


SCHEMES.update({
    "ink":  dict(pal=_tones("#EDEFF2", "#98A0AB", "#454C57"),
                 plastic=(0.016, 0.018, 0.022, 1), cage=(0.06, 0.07, 0.085, 1),
                 metal=0.4, glow=0.5, accent=("#1E90E8", "#1E90E8"), crest="#CFEAFF"),
    "slate": dict(pal=_tones("#DCE6F2", "#7C9BC0", "#2E4666"),
                  plastic=(0.010, 0.014, 0.024, 1), cage=(0.05, 0.07, 0.10, 1),
                  metal=0.4, glow=0.5, accent=("#FFAE33", "#FFAE33"), crest="#FFE7BE"),
    "moss": dict(pal=_tones("#E6EDE2", "#8FAD8A", "#3B5540"),
                 plastic=(0.014, 0.020, 0.016, 1), cage=(0.06, 0.085, 0.07, 1),
                 metal=0.4, glow=0.5, accent=("#E8703A", "#E8703A"), crest="#FFDCC9"),
    "clay": dict(pal=_tones("#F2E7DC", "#C79A7B", "#6B4634"),
                 plastic=(0.024, 0.018, 0.014, 1), cage=(0.10, 0.075, 0.06, 1),
                 metal=0.4, glow=0.5, accent=("#2F9C8F", "#2F9C8F"), crest="#CFF2ED"),
})

# Grey body, solid coloured modules. The body recedes to neutral so all the colour
# sits on the three extracted pieces, and each piece is a clean solid cube marked with
# what it is rather than a wire cage.
_GREY = {k: "#A6B0BC" for k in "ROYGBW"}
SOLIDS = {
    "signal": ("#12B3B8", "#4C6FE0", "#F2B33D"),
    "rubik":  ("#0051BA", "#C41E3A", "#FFD500"),
    "ocean":  ("#1E9CD8", "#2E4E9E", "#38C4A8"),
    "berry":  ("#E0407A", "#7B54C4", "#38B6C4"),
    "citrus": ("#F2762E", "#F2B33D", "#5FB733"),
}
# A wide set of trios for the bare mark, where the three blocks carry all the meaning.
SOLIDS.update({
    "sunset": ("#E8563F", "#C2447A", "#F0A93B"),
    "forest": ("#2E8B57", "#6B8F3A", "#1E7A6E"),
    "neon":   ("#FF2E88", "#00E5D0", "#B6FF3B"),
    "pastel": ("#F4A6C0", "#B7A6E8", "#8FD9C4"),
    "slate":  ("#4E7CA8", "#7A8AA3", "#D8B25E"),
    "mono":   ("#2E6FA8", "#4E90C8", "#8FBEDC"),
    "ember":  ("#D93A2B", "#F2762E", "#F2B33D"),
    "grape":  ("#6C3FA8", "#A54FC0", "#E06BB0"),
    "arctic": ("#2AA6C4", "#5FC8D8", "#BFE9F2"),
    "clay":   ("#B5523C", "#C79A5B", "#6B8F5E"),
})

def _hx(h, f):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(min(255, max(0, round(v * f))) for v in (r, g, b))


def _lift(h, t):
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(round(v + (255 - v) * t) for v in (r, g, b))


def _body(h):
    """Every panel the same colour. The three visible faces still separate, because the
    flat shader tones each by orientation — but within a face all nine panels match."""
    return {k: h for k in ("R", "O", "Y", "G", "B", "W")}


# Bodies other than grey. The body sets the mood, the three extracted blocks contrast
# against it, so each pairing has to be picked as a whole rather than reusing one trio.
BODIES = {
    # neutral bodies
    "grey-teal":    ("#A6B0BC", ("#12B3B8", "#4C6FE0", "#F2B33D")),
    "grey-rubik":   ("#A6B0BC", ("#0051BA", "#C41E3A", "#FFD500")),
    "grey-berry":   ("#A6B0BC", ("#E0407A", "#7B54C4", "#38B6C4")),
    "grey-citrus":  ("#A6B0BC", ("#F2762E", "#F2B33D", "#5FB733")),
    "grey-neon":    ("#A6B0BC", ("#FF2E88", "#00E5D0", "#B6FF3B")),
    "paper-rubik":  ("#E8EAEC", ("#C41E3A", "#0051BA", "#FFD500")),
    "paper-cool":   ("#E8EAEC", ("#1E7A8C", "#4C6FE0", "#E8563F")),
    "steel-acid":   ("#78859A", ("#B6FF3B", "#FF3B9E", "#22E0D0")),
    # dark bodies
    "ink-teal":     ("#2B303A", ("#12B3B8", "#F2B33D", "#E8563F")),
    "ink-neon":     ("#2B303A", ("#00E5D0", "#FF2E88", "#B6FF3B")),
    "char-cyan":    ("#23262B", ("#22C8E0", "#7A8CF0", "#F0F2F5")),
    "navy-warm":    ("#1E3A5F", ("#3FC8E0", "#F2B33D", "#E8608A")),
    "pine-lime":    ("#24483A", ("#9ED14E", "#F2B33D", "#3FBFA8")),
    "plum-pop":     ("#3E2A4A", ("#E86BB0", "#B07BE0", "#F2C94C")),
    "wine-gold":    ("#5A2436", ("#F2C94C", "#5FD9B0", "#E8608A")),
    "cocoa-warm":   ("#4A382C", ("#E8A33D", "#7FB069", "#4E8FC4")),
    "violet-pop":   ("#4632A0", ("#22E0D0", "#FF9F3B", "#FF6B9E")),
    "rust-amber":   ("#5C3324", ("#F2872E", "#F2C94C", "#3FA89A")),
    # light and mid bodies
    "sand-teal":    ("#C9BBA4", ("#1E7A8C", "#C4502E", "#33456B")),
    "sky-warm":     ("#B6D2E8", ("#22406B", "#E8563F", "#F2B33D")),
    "mint-plum":    ("#BCE3D2", ("#5A3A78", "#E8703A", "#2E5E8C")),
    "olive-cream":  ("#6E7444", ("#F2E3C0", "#C4502E", "#2E8B7A")),
    "slate-gold":   ("#4E5C6E", ("#F2B33D", "#4FC3D9", "#E8608A")),
    # Variations on slate-gold: the module trio keeps its character (warm gold, cool
    # cyan, rose accent) while the body hue shifts around mid-dark desaturated.
    "graphite-gold":("#4A4F58", ("#F2B33D", "#4FC3D9", "#E8608A")),
    "denim-gold":   ("#41556E", ("#F5C158", "#5AD0E0", "#F06B95")),
    "teal-gold":    ("#3E5C5C", ("#F2B33D", "#6FD4E0", "#E85E7E")),
    "plumslate":    ("#544A64", ("#F5C158", "#4FC3D9", "#F06B95")),
    "mossslate":    ("#4C5A48", ("#F2C14C", "#5AC8D8", "#E8608A")),
    "warmslate":    ("#5A4E44", ("#F2B33D", "#4FC3D9", "#E8788A")),
    "indigo-gold":  ("#3B4468", ("#F5C158", "#5AD0E0", "#F06B95")),
    "steel-gold":   ("#5A6673", ("#EFA92E", "#3FB6CE", "#E05480")),
    "char-gold":    ("#3A3F47", ("#F5C158", "#5AD0E0", "#F06B95")),
    "ocean-gold":   ("#3A5566", ("#F2B33D", "#6FD4E0", "#E8608A")),
    "clay-gold":    ("#63544C", ("#F2C14C", "#4FC3D9", "#E85E7E")),
    "forest-gold":  ("#3E5245", ("#F2B33D", "#5AC8D8", "#E8608A")),
    "mauve-gold":   ("#5E4F5C", ("#F5C158", "#4FC3D9", "#F06B95")),
    "midnight-gold":("#2E3646", ("#F2B33D", "#4FC3D9", "#E8608A")),

    # Favourites family: neutral grey or slate body, one cool module and one warm,
    # no pink or magenta anywhere. Built off grey-teal and slate-gold.
    "grey-gold":     ("#A6B0BC", ("#F2B33D", "#4FC3D9", "#2E8B7A")),
    "grey-cyan":     ("#A6B0BC", ("#F2B33D", "#4FC3D9", "#2E5E8C")),
    "grey-orange":   ("#A6B0BC", ("#E8703A", "#1E7A8C", "#F2D06B")),
    "grey-lime":     ("#A6B0BC", ("#5FB733", "#2E8B7A", "#F2B33D")),
    "grey-navy":     ("#A6B0BC", ("#1FA8A0", "#2E5E8C", "#F2C14C")),
    "mist-teal":     ("#C3CAD2", ("#12B3B8", "#4C6FE0", "#F2B33D")),
    "mist-gold":     ("#C3CAD2", ("#F2B33D", "#3FB6CE", "#2E5E8C")),
    "slate-green":   ("#4E5C6E", ("#F2B33D", "#4FC3D9", "#3FA88A")),
    "slate-orange":  ("#4E5C6E", ("#F2B33D", "#4FC3D9", "#E8703A")),
    "slate-blue":    ("#4E5C6E", ("#F2C14C", "#4FC3D9", "#2E5E8C")),
    "slate-cream":   ("#4E5C6E", ("#4C6FE0", "#12B3B8", "#F5F2E8")),
    "slate-white":   ("#4E5C6E", ("#4C6FE0", "#12B3B8", "#FFFFFF")),
    "steel-teal":    ("#5A6673", ("#1FA8A0", "#2E5E8C", "#F2C14C")),
    "steel-lime":    ("#5A6673", ("#8CC63F", "#2E8B7A", "#F2B33D")),
    "denim-amber":   ("#41556E", ("#3FB6CE", "#7FB069", "#F2B33D")),
    "graphite-teal": ("#4A4F58", ("#1FA8A0", "#3E7BC4", "#F2C14C")),
    "midnight-cyan": ("#2E3646", ("#F2C14C", "#4FC3D9", "#3FA88A")),
    "moss-coral":   ("#4A6B4F", ("#F27A5E", "#F2D06B", "#7FD9C4")),
}
for _n, (_b, _m) in BODIES.items():
    SOLIDS[_n] = _m

for _n, _m in SOLIDS.items():
    STYLES[f"solid-{_n}"] = STYLES[f"jax-{_n}"] = STYLES[f"cube-{_n}"] = dict(
        pal=(RUBIK if TEX == "rubik" else
             _body(BODIES[_n][0]) if _n in BODIES else _GREY), mods=_m,
                                 plastic=(0.020, 0.022, 0.027, 1),
                                 cage=(0.06, 0.07, 0.085, 1), metal=0.4, glow=0.0,
                                 accent=(_m[0], _m[1]), crest="#FFFFFF")

for _pfx in ("crystal", "neat", "mark"):
    for _name, _cfg in SCHEMES.items():
        STYLES[f"{_pfx}-{_name}"] = _cfg
STYLE = STYLES.get(VARIANT, STYLES["crystal"])
RES = int(os.environ.get("LOGO_RES", 1800))
SAMPLES = int(os.environ.get("LOGO_SAMPLES", 900))

CUBIE = 0.995         # unit cube body; the residual gap is the seam
BEVEL = 0.028
STICKER = 0.86
PAL = STYLE.get("pal", RUBIK)       # official Rubik colours unless a scheme overrides
ACCENT = STYLE["accent"]


def lin(hexcol):
    def c(v):
        v /= 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return tuple(c(int(hexcol[i:i + 2], 16)) for i in (1, 3, 5)) + (1.0,)


# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def set_in(node, names, value):
    """Principled socket names drift between Blender versions; try each."""
    for n in names if isinstance(names, (list, tuple)) else [names]:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def flat_material(name, base, alpha=1.0):
    """One constant colour per face orientation, emitted rather than lit.

    Dotting the geometric normal with (0.5, -0.75, 1.0) gives 0.5 on the +X face, 0.75
    on the -Y face and 1.0 on +Z — three separated values the ramp snaps to three tones
    with CONSTANT interpolation. The result is flat colour fields over real geometry,
    so the cut-outs still occlude correctly.
    """
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    out = nt.nodes["Material Output"]
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.5, -0.75, 1.0)
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.0, tuple(c * 0.62 for c in base[:3]) + (1,)
    e[1].position, e[1].color = 0.62, tuple(c * 0.86 for c in base[:3]) + (1,)
    e.new(0.87).color = tuple(min(1.0, c * 1.18) for c in base[:3]) + (1,)
    nt.links.new(dot.outputs["Value"], ramp.inputs["Fac"])
    emis = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])
    if alpha >= 1.0:
        nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    else:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx = nt.nodes.new("ShaderNodeMixShader")
        mx.inputs["Fac"].default_value = alpha
        nt.links.new(tr.outputs["BSDF"], mx.inputs[1])
        nt.links.new(emis.outputs["Emission"], mx.inputs[2])
        nt.links.new(mx.outputs["Shader"], out.inputs["Surface"])
    return m


def material(name, base, rough=0.4, metallic=0.0, transmission=0.0, ior=1.45,
             coat=0.0, emit=None, emit_strength=0.0, alpha=1.0, spec=0.5):
    if FLAT:
        return flat_material(name, base, alpha)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    set_in(bsdf, "Base Color", base)
    set_in(bsdf, "Roughness", rough)
    set_in(bsdf, "Metallic", metallic)
    set_in(bsdf, ["Transmission Weight", "Transmission"], transmission)
    set_in(bsdf, "IOR", ior)
    set_in(bsdf, ["Coat Weight", "Clearcoat"], coat)
    set_in(bsdf, "Alpha", alpha)
    set_in(bsdf, ["Specular IOR Level", "Specular"], spec)
    if emit is not None:
        set_in(bsdf, ["Emission Color", "Emission"], emit)
        set_in(bsdf, "Emission Strength", emit_strength)
    return m


MAT_PLASTIC = material("plastic", STYLE.get("plastic", (0.020, 0.022, 0.027, 1)),
                       rough=0.62, spec=0.22)
MAT_SEAM = material("seam", (0.13, 0.16, 0.21, 1), rough=0.55, spec=0.25)
MAT_CAGE = material("cage", STYLE["cage"], rough=0.34,
                    metallic=STYLE["metal"], spec=0.5)
MAT_SHELL = material("shell", (0.021, 0.023, 0.028, 1), rough=0.62)
MAT_STICKER = {k: material(f"stk_{k}", lin(v), rough=0.34, coat=0.10, spec=0.30)
               for k, v in PAL.items()}
MAT_INNER = {k: material(f"in_{k}", tuple(c * 0.42 for c in lin(v)[:3]) + (1.0,),
                         rough=0.55, spec=0.20) for k, v in PAL.items()}
def absorb(m, color, density):
    """Give a glass block real body. Without absorption a clear block shows every
    internal reflection at full strength and the module reads as visual noise."""
    if FLAT:
        return m
    nt = m.node_tree
    vol = nt.nodes.new("ShaderNodeVolumeAbsorption")
    vol.inputs["Color"].default_value = color
    vol.inputs["Density"].default_value = density
    nt.links.new(vol.outputs["Volume"], nt.nodes["Material Output"].inputs["Volume"])
    return m


DBG_OPAQUE = os.environ.get("LOGO_DEBUG_OPAQUE")
def faceted_glass(name, top_col, side_col, top_a, side_a, rough=0.18, spec=0.55):
    """Translucent glass whose tone and opacity follow face orientation.

    With every pane the same tone a block reads as a flat grey plate, not a volume —
    single-sided alpha leaves only one surface per sightline, so there is no second
    edge to give it depth. Keying colour and alpha off |normal.z| makes the top pane
    light and open and the vertical panes darker and denser, which is what actually
    reads as a 3D block. Backfaces stay at zero alpha so contents are veiled once.
    """
    if FLAT:
        return flat_material(name, top_col)
    m = material(name, top_col, rough=rough, spec=spec)
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    up = nt.nodes.new("ShaderNodeMath"); up.operation = "ABSOLUTE"
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], up.inputs[0])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = side_col
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = top_col
    nt.links.new(up.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    rng = nt.nodes.new("ShaderNodeMapRange")
    rng.inputs["To Min"].default_value = side_a
    rng.inputs["To Max"].default_value = top_a
    nt.links.new(up.outputs[0], rng.inputs["Value"])
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    nt.links.new(geo.outputs["Backfacing"], inv.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    nt.links.new(rng.outputs["Result"], mul.inputs[0])
    nt.links.new(inv.outputs[0], mul.inputs[1])
    nt.links.new(mul.outputs[0], bsdf.inputs["Alpha"])
    return m


def single_sided(m, a):
    """Draw only the faces pointing at the camera.

    A closed box has two translucent surfaces in every sightline, so a plain alpha
    material veils its own contents twice and the block washes out to near-white.
    Killing the back faces leaves exactly one layer: the block still reads as a solid
    translucent volume, but whatever sits inside it stays legible.
    """
    if FLAT:
        return m
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    inv = nt.nodes.new("ShaderNodeMath"); inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = a
    nt.links.new(geo.outputs["Backfacing"], inv.inputs[1])
    nt.links.new(inv.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], bsdf.inputs["Alpha"])
    return m


MAT_GLASS = faceted_glass("glass", (0.74, 0.82, 0.93, 1), (0.30, 0.38, 0.52, 1),
                          0.38, 0.38)
MAT_GLASS_T = faceted_glass("glass_t", (0.76, 0.84, 0.94, 1), (0.32, 0.40, 0.54, 1),
                            0.34, 0.34)
if DBG_OPAQUE:
    MAT_GLASS = material("glass_dbg", (0.8, 0.0, 0.6, 1), rough=0.4)
    MAT_GLASS_T = MAT_GLASS
MAT_NODE_A = material("node_a", lin(ACCENT[0]), rough=0.38, spec=0.3,
                      emit=lin(ACCENT[0]), emit_strength=STYLE["glow"])
MAT_NODE_B = material("node_b", lin(ACCENT[1]), rough=0.38, spec=0.3,
                      emit=lin(ACCENT[1]), emit_strength=STYLE["glow"])
MAT_WIRE = material("wire", (0.03, 0.05, 0.12, 1), rough=0.55, spec=0.25)
MAT_RING_R = MAT_INK if False else material("ring_r", lin(PAL["R"]), rough=0.34, coat=0.08, spec=0.3,
                      emit=lin(PAL["R"]), emit_strength=0.0)
MAT_RING_B = MAT_INK if False else material("ring_b", lin(PAL["B"]), rough=0.34, coat=0.08, spec=0.3,
                      emit=lin(PAL["B"]), emit_strength=0.0)
MAT_RING_Y = MAT_INK if False else material("ring_y", lin(PAL["Y"]), rough=0.34, coat=0.08, spec=0.3,
                      emit=lin(PAL["Y"]), emit_strength=0.0)


def ridge_material():
    if FLAT:
        return flat_material("ridge", (0.2, 0.7, 0.9, 1))

    m = bpy.data.materials.new("ridge")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    set_in(bsdf, "Roughness", 0.38)
    set_in(bsdf, ["Coat Weight", "Clearcoat"], 0.0)
    set_in(bsdf, "Emission Strength", 0.0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    grad = nt.nodes.new("ShaderNodeTexGradient")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.02
    ramp.color_ramp.elements[0].color = lin(ACCENT[0])
    ramp.color_ramp.elements[1].position = 0.98
    ramp.color_ramp.elements[1].color = lin(ACCENT[1])
    nt.links.new(tc.outputs["Generated"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    for n in ("Emission Color", "Emission"):
        if n in bsdf.inputs:
            nt.links.new(ramp.outputs["Color"], bsdf.inputs[n])
            break
    return m


MAT_RIDGE = ridge_material()
MAT_CREST = material("crest", lin(STYLE["crest"]), rough=0.34, spec=0.30)


# ---------------------------------------------------------------- primitives
def rounded_box(name, dims, loc, mat, bevel=BEVEL, segments=6):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = dims
    bpy.ops.object.transform_apply(scale=True)
    b = o.modifiers.new("bev", "BEVEL")
    b.width = min(bevel, min(dims) * 0.45)
    b.segments = segments
    b.limit_method = "ANGLE"
    o.data.materials.append(mat)
    return o


def ring(name, center, axis, radius, thickness, mat):
    rot = {"x": (0, math.radians(90), 0), "y": (math.radians(90), 0, 0),
           "z": (0, 0, 0)}[axis]
    bpy.ops.mesh.primitive_torus_add(location=center, rotation=rot,
                                     major_radius=radius, minor_radius=thickness,
                                     major_segments=72, minor_segments=16)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def sphere(name, center, r, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(location=center, radius=r, segments=40,
                                         ring_count=20)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def cylinder(name, p0, p1, r, mat):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length,
                                        location=(p0 + p1) / 2, vertices=16)
    o = bpy.context.active_object
    o.name = name
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


# ---------------------------------------------------------------- cell layout
SPEC = {(i, 0, 1) for i in range(3)}                                  # 3x1x1 front row
NN = {(i, j, k) for i in (1, 2) for j in (1, 2) for k in (0, 1)}      # 2x2x2 corner
TOPC = {(1, 1, 2)}
REMOVED = SPEC | NN | TOPC
BODY = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)
        if (i, j, k) not in REMOVED]

COMPACT = bool(os.environ.get("LOGO_COMPACT"))
SPEC_OFF = (0.0, -1.12 if COMPACT else (-1.70 if SOLID else -1.85), 0.0)
NN_OFF = ((0.85, 0.0, 0.0) if COMPACT else
          (1.35, 0.0, 0.0) if SOLID  # most of the 2x2x2 out, rear layer still in
          else (1.75, 0.0, 0.0))     # straight out of the right face
TOP_OFF = (0.0, 0.0, 0.55 if COMPACT else 1.20)

STK_TOP = {(0, 0): "Y", (1, 0): "Y", (2, 0): "B", (0, 1): "Y", (2, 1): "Y",
           (0, 2): "G", (1, 2): "Y", (2, 2): "Y"}
STK_FRONT = {(0, 2): "R", (1, 2): "R", (2, 2): "O",
             (0, 0): "W", (1, 0): "R", (2, 0): "R"}
STK_RIGHT = {(0, 2): "B", (1, 2): "G", (2, 2): "B", (0, 0): "B"}

# Cavity walls get their own palette rather than borrowing the outer grids. Reusing the
# outer grid dropped a white sticker onto the cavity floor, which reads as a hole in the
# mark on any light background. One warm colour per cavity keeps the interior legible.
INNER = {"top": "Y", "front": "O", "right": "O"}

def variants(hexcol, n=6, spread=0.42):
    """A ladder of tints of one colour, for panel-by-panel variation within a block.
    Deterministic: no randomness, so a scheme always renders identically."""
    return [material(f"v_{hexcol}_{i}",
                     lin(_hx(hexcol, 1 - spread + 2 * spread * i / (n - 1))),
                     rough=0.34, coat=0.10, spec=0.30) for i in range(n)]


def pick(lst, c, f):
    h = c[0] * 7 + c[1] * 13 + c[2] * 23 + {"top": 1, "front": 2, "right": 3}[f] * 5
    return lst[h % len(lst)]


_BODY_HEX = (PAL if isinstance(PAL, dict) else RUBIK)["R"]
MAT_BODY_V = variants(_BODY_HEX) if VARY_BODY else None
MAT_INNER_V = variants(_hx(_BODY_HEX, 0.55)) if VARY_BODY else None


def body_mat(out, col, c, f):
    if VARY_BODY:
        return pick(MAT_BODY_V if out else MAT_INNER_V, c, f)
    return (MAT_STICKER if out else MAT_INNER)[col]



# ---------------------------------------------------------------- the cube body
# A face gets a sticker whenever nothing occupies the cell in front of it. That covers
# the outer surface AND the walls revealed inside each cavity, so the interior reads as
# coloured cube material rather than a dead black void.
BODYSET = set(BODY)
T = STICKER
CELL_OBJS = {}          # cell -> its objects, so any slice can be twisted as one


def keep(o, c):
    CELL_OBJS.setdefault(tuple(c), []).append(o)
    return o


for (i, j, k) in BODY:
    c = (i, j, k)
    keep(rounded_box(f"cubie_{i}{j}{k}", (CUBIE, CUBIE, CUBIE), c, MAT_PLASTIC), c)
    off = CUBIE / 2 - 0.004
    if TURN or (i, j, k + 1) not in BODYSET:                      # +Z exposed
        f_ = "top"
        out = k == 2
        col = STK_TOP[(i, j)] if out else INNER["top"]
        keep(rounded_box(f"s_t_{i}{j}{k}", (T, T, 0.030), (i, j, k + off),
                         body_mat(out, col, c, f_),
                         bevel=0.075), c)
    if TURN or (i, j - 1, k) not in BODYSET:                      # -Y exposed
        f_ = "front"
        out = j == 0
        col = STK_FRONT[(i, k)] if out else INNER["front"]
        keep(rounded_box(f"s_f_{i}{j}{k}", (T, 0.030, T), (i, j - off, k),
                         body_mat(out, col, c, f_),
                         bevel=0.075), c)
    if TURN or (i + 1, j, k) not in BODYSET:                      # +X exposed
        f_ = "right"
        out = i == 2
        col = STK_RIGHT[(j, k)] if out else INNER["right"]
        keep(rounded_box(f"s_r_{i}{j}{k}", (0.030, T, T), (i + off, j, k),
                         body_mat(out, col, c, f_),
                         bevel=0.075), c)

# The far inner corner of the cube. Several cavities line up into through-tunnels;
# these three plates close them off so a hole reads as depth, not as a hole in the mark.
if os.environ.get("LOGO_SHELL"):
    LO, HI, TH = -0.5, 2.5, 0.014
    rounded_box("shell_back", (3.0, TH, 3.0), (1, HI - TH / 2, 1), MAT_SHELL, bevel=0.002)
    rounded_box("shell_left", (TH, 3.0, 3.0), (LO + TH / 2, 1, 1), MAT_SHELL, bevel=0.002)
    rounded_box("shell_floor", (3.0, 3.0, TH), (1, 1, LO + TH / 2), MAT_SHELL, bevel=0.002)

# ---------------------------------------------------------------- glass modules
def glass_module(name, cells, offset, mat=None):
    """One glass volume per module, not one per cube.

    Eight separate glass shells put 16 refracting surfaces in the light path and the
    module turns into visual noise. A single solid block refracts cleanly; the cube
    divisions come back as thin inlaid seams on the three visible faces.
    """
    lo = [min(c[a] for c in cells) + offset[a] - CUBIE / 2 for a in range(3)]
    hi = [max(c[a] for c in cells) + offset[a] + CUBIE / 2 for a in range(3)]
    ctr = [(lo[a] + hi[a]) / 2 for a in range(3)]
    dims = [hi[a] - lo[a] for a in range(3)]
    rounded_box(name, dims, ctr, mat or MAT_GLASS, bevel=0.045)

    w, d = 0.032, 0.014                       # seam width, and how proud it sits
    for a in range(3):                        # internal division planes on each axis
        vals = sorted({c[a] for c in cells})
        for v in vals[:-1]:
            p = v + 0.5 + offset[a]
            if a == 0:                        # cuts the top and front faces
                seam(f"{name}_sx_t", (w, dims[1], d), (p, ctr[1], hi[2]))
                seam(f"{name}_sx_f", (w, d, dims[2]), (p, lo[1], ctr[2]))
            elif a == 1:                      # cuts the top and right faces
                seam(f"{name}_sy_t", (dims[0], w, d), (ctr[0], p, hi[2]))
                seam(f"{name}_sy_r", (d, w, dims[2]), (hi[0], p, ctr[2]))
            else:                             # cuts the front and right faces
                seam(f"{name}_sz_f", (dims[0], d, w), (ctr[0], lo[1], p))
                seam(f"{name}_sz_r", (d, dims[1], w), (hi[0], ctr[1], p))


def seam(name, dims, loc):
    if SEAMS:
        rounded_box(name, dims, loc, MAT_SEAM, bevel=0.006, segments=2)


def wire_cage(cells, offset, r=0.020, outline=False):
    """Every unit-cube edge of a module as a thin rod, deduplicated at shared faces.

    The crystal variant drops the glass entirely: the cage states the cube geometry and
    the module's own contents fill it, so the spectrum and the network *are* the cubes
    rather than objects parked inside a block.
    """
    h, edges, span = CUBIE / 2, set(), None
    if outline:
        lo = [min(c[a] for c in cells) + offset[a] for a in range(3)]
        hi = [max(c[a] for c in cells) + offset[a] for a in range(3)]
        span = [(hi[a] - lo[a]) / 2 + CUBIE / 2 for a in range(3)]
        cells, offset = [[(lo[a] + hi[a]) / 2 for a in range(3)]], (0, 0, 0)
    for c in cells:
        ctr = [c[a] + offset[a] for a in range(3)]
        ext = span or [h, h, h]
        for a in range(3):
            b, d = (a + 1) % 3, (a + 2) % 3
            for sb in (-ext[b], ext[b]):
                for sd in (-ext[d], ext[d]):
                    p0, p1 = [0, 0, 0], [0, 0, 0]
                    p0[a], p1[a] = ctr[a] - ext[a], ctr[a] + ext[a]
                    p0[b] = p1[b] = ctr[b] + sb
                    p0[d] = p1[d] = ctr[d] + sd
                    edges.add((tuple(round(v, 4) for v in p0),
                               tuple(round(v, 4) for v in p1)))
    for p0, p1 in sorted(edges):
        cylinder("cage", p0, p1, r, MAT_CAGE)


MAT_MOD = [material(f"mod{i}", lin(c), rough=0.34, coat=0.10, spec=0.30)
           for i, c in enumerate(STYLE.get("mods", ("#12B3B8", "#4C6FE0", "#F2B33D")))]
MAT_INK = material("ink", (1.0, 1.0, 1.0, 1), rough=0.30, spec=0.25)


def relief(hexcol, name):
    """A light tint of the module's own colour, for marks moulded into the block.

    Pure white shapes sitting on the surface read as decals stuck on. Keeping the mark
    in the same hue, and half-sinking it so the orientation shading catches its raised
    edges, makes it read as relief in the same material instead.
    """
    c = lin(hexcol)
    return material(name, tuple(v + (1.0 - v) * 0.38 for v in c[:3]) + (1.0,),
                    rough=0.34, spec=0.28)


if SOLID:
    _M = STYLE.get("mods", ("#12B3B8", "#4C6FE0", "#F2B33D"))
    MAT_REL = [relief(c, f"rel{i}") for i, c in enumerate(_M)]
    MAT_NODE_A = MAT_NODE_B = MAT_WIRE = MAT_REL[1]
    MAT_RING_R = MAT_RING_B = MAT_RING_Y = MAT_REL[2]


SPEC_FACES = []          # front panels of the spectroscopy row, for engraving


def solid_module(cells, off, mat, tag, vlist=None):
    """The module as clean solid cubes, built exactly like the body's cubies so the two
    read as the same object taken apart — not as a cage or a glass block."""
    cs = set(cells)
    o = CUBIE / 2 - 0.004
    for c in sorted(cells):
        loc = tuple(c[a] + off[a] for a in range(3))
        rounded_box(f"{tag}_{c[0]}{c[1]}{c[2]}", (CUBIE,) * 3, loc, MAT_PLASTIC)
        for f, d in (("top", (0, 0, 1)), ("front", (0, -1, 0)), ("right", (1, 0, 0))):
            if (c[0] + d[0], c[1] + d[1], c[2] + d[2]) in cs:
                continue
            dims = {"top": (T, T, 0.030), "front": (T, 0.030, T),
                    "right": (0.030, T, T)}[f]
            at = {"top": (loc[0], loc[1], loc[2] + o),
                  "front": (loc[0], loc[1] - o, loc[2]),
                  "right": (loc[0] + o, loc[1], loc[2])}[f]
            _o = rounded_box(f"{tag}s_{c}{f}", dims, at,
                             pick(vlist, c, f) if vlist else mat, bevel=0.075)
            if tag == "ms" and f == "front":
                SPEC_FACES.append(_o)


if SOLID:
    _MV = ([variants(c) for c in STYLE.get("mods", ("#12B3B8", "#4C6FE0", "#F2B33D"))]
           if VARY_MOD else [None, None, None])
    solid_module(sorted(SPEC), SPEC_OFF, MAT_MOD[0], "ms", _MV[0])
    solid_module(sorted(NN), NN_OFF, MAT_MOD[1], "mn", _MV[1])
    solid_module(sorted(TOPC), TOP_OFF, MAT_MOD[2], "mt", _MV[2])
elif CRYSTAL:
    for _cells, _off in ((sorted(SPEC), SPEC_OFF), (sorted(NN), NN_OFF),
                         (sorted(TOPC), TOP_OFF)):
        wire_cage(_cells, _off, outline=SIMPLE or NEAT)
elif not os.environ.get("LOGO_NOGLASS"):        # debug aid: see the inlays bare
    glass_module("g_spec", sorted(SPEC), SPEC_OFF)
    glass_module("g_nn", sorted(NN), NN_OFF)
    glass_module("g_top", sorted(TOPC), TOP_OFF, mat=MAT_GLASS_T)

# The trace is only built when a variant actually draws one. The chosen mark has
# no spectrum, so it must not depend on spectrum.json existing.
if MARKS != "none":
    # ---------------------------------------------------------------- spectrum ridge
    # Inlaid into the front pane rather than suspended inside the block. Anything held deep
    # in clear glass gets smeared by refraction into unreadable colour noise; sitting proud
    # of the surface keeps the peak pattern crisp while the block still reads as glass.
    data = json.load(open(os.path.join(HERE, "spectrum.json")))
    data = data[::2]
    if SIMPLE or NEAT:
        # Keep every sample and smooth gently instead of decimating. Max-pooling to a
        # quarter of the points turned the metabolite trace into a coarse angular
        # silhouette and biased the baseline upward; a 3-tap kernel removes jitter while
        # leaving the peak heights and the overall pattern essentially untouched.
        h = [v for _, v in data]
        sm = [h[0]] + [0.25 * h[i - 1] + 0.5 * h[i] + 0.25 * h[i + 1]
                       for i in range(1, len(h) - 1)] + [h[-1]]
        data = [(data[i][0], sm[i]) for i in range(len(data))]
    sx0, sx1 = ((-0.5 + 0.22, 2.5 - 0.22) if SOLID
                else (-0.5 + 0.12, 2.5 - 0.12))              # inside the row
    sy = (0 + SPEC_OFF[1]) - CUBIE / 2                       # the module's front pane
    yF, yB = ((sy - 0.085, sy + 0.085) if SOLID else
              (sy + 0.10, sy + 0.88) if CRYSTAL else (sy - 0.045, sy + 0.42))
    zb, amp = (1 - 0.30, 0.58) if SOLID else (1 - 0.46, 0.86)
    n = len(data)
    XS = [sx0 + (sx1 - sx0) * idx / (n - 1) for idx in range(n)]
    # A real MRS trace is ~90% baseline with a few tall peaks, which read as a thin squiggle
    # at logo size. The exponent lifts mid-size features; being monotonic it leaves the peak
    # pattern and their ordering untouched.
    ZS = [zb + amp * (h ** 0.50) for _, h in data]


    def sweep(name, lo, hi, y0, y1, mat):
        """Extrude a profile along X into a closed solid."""
        verts, faces = [], []
        for idx in range(n):
            verts += [(XS[idx], y0, lo[idx]), (XS[idx], y0, hi[idx]),
                      (XS[idx], y1, hi[idx]), (XS[idx], y1, lo[idx])]
        for idx in range(n - 1):
            a, b = idx * 4, (idx + 1) * 4
            faces += [(a, a + 1, b + 1, b), (a + 1, a + 2, b + 2, b + 1),
                      (a + 2, a + 3, b + 3, b + 2), (a + 3, a, b, b + 3)]
        e = (n - 1) * 4
        faces += [(0, 3, 2, 1), (e, e + 1, e + 2, e + 3)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.validate()
        o = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(o)
        o.data.materials.append(mat)
        return o


    if MARKS == "engrave":
        # Cut the trace out of the coloured panels instead of laying it on top. What shows
        # through is the block's own dark body, so the spectrum is a groove in the object
        # rather than a decal — the complaint every painted-on version drew.
        cut = sweep("cutter", [zb] * n, ZS, sy - 0.12, sy + 0.12, MAT_INK)
        for _f in SPEC_FACES:
            m = _f.modifiers.new("engrave", "BOOLEAN")
            m.operation = "DIFFERENCE"
            m.object = cut
            m.solver = "EXACT"
        cut.hide_render = True
        cut.hide_viewport = True
    elif MARKS in ("spectrum", "all"):
        sweep("ridge", [zb] * n, ZS, yF, yB,
              MAT_REL[0] if SOLID else MAT_RIDGE)
        if not SOLID:
            # a bright crest riding the curve — what makes the peak pattern legible small
            sweep("crest", ZS, [z + 0.030 for z in ZS], yF - 0.02, yB + 0.02, MAT_CREST)

# ---------------------------------------------------------------- MLP on the 2x2x2
# Same reasoning as the ridge: laid onto the right and top panes, half-sunk so it reads
# as inlay. Suspended at mid-depth it refracted into a tangled web from every angle.
NX = 2 + NN_OFF[0] + CUBIE / 2 - 0.16                    # just inside the right pane
NZ = 1 + CUBIE / 2                                       # the module's top pane
YC, ZC = 1.5, 0.5                                        # centre of the 2x2 cross-section


def net(nodes_by_layer, place, wire_r=0.013, node_r=0.078):
    cols = [[place(a, b) for b in row] for a, row in nodes_by_layer]
    for a in range(len(cols) - 1):
        for p in cols[a]:
            for q in cols[a + 1]:
                cylinder("w", p, q, wire_r, MAT_WIRE)
    for li, col in enumerate(cols):
        for p in col:
            sphere("n", p, node_r, MAT_NODE_A if li % 2 == 0 else MAT_NODE_B)


if MARKS != "all":
    pass
elif SOLID:
    # on the surface: inside a solid block nothing is visible
    NXs = 2 + NN_OFF[0] + CUBIE / 2          # on the face: half-sunk studs
    net([(YC + 0.52, [ZC - 0.48, ZC, ZC + 0.48]),
         (YC, [ZC - 0.66, ZC - 0.22, ZC + 0.22, ZC + 0.66]),
         (YC - 0.52, [ZC - 0.48, ZC, ZC + 0.48])],
        lambda y, z: (NXs, y, z), wire_r=0.013, node_r=0.072)
elif CRYSTAL:
    # one node at the centre of each of the eight unit cubes, wired as a 4 -> 4 layer:
    # the network and the 2x2x2 are literally the same object
    layers = [[(i + NN_OFF[0], j + NN_OFF[1], k + NN_OFF[2])
               for j in (1, 2) for k in (0, 1)] for i in (1, 2)]
    if NEAT:
        # cube-graph wiring: each node joins only its axis neighbours. Twelve edges
        # instead of sixteen crossing ones, and the wires trace the 2x2x2 itself.
        pts = layers[0] + layers[1]
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                if sum(abs(pts[a][t] - pts[b][t]) > 1e-6 for t in range(3)) == 1:
                    cylinder("w", pts[a], pts[b], 0.022, MAT_WIRE)
    else:
        for a in layers[0]:
            for b in layers[1]:
                cylinder("w", a, b, 0.020, MAT_WIRE)
    for li, col in enumerate(layers):
        for pt in col:
            sphere("n", pt, 0.185, MAT_NODE_A if li == 0 else MAT_NODE_B)
else:
    net([(YC + 0.52, [ZC - 0.48, ZC, ZC + 0.48]),
         (YC, [ZC - 0.66, ZC - 0.22, ZC + 0.22, ZC + 0.66]),
         (YC - 0.52, [ZC - 0.48, ZC, ZC + 0.48])],
        lambda y, z: (NX, y, z))


# ---------------------------------------------------------------- target inlays
tc = tuple(1 + TOP_OFF[a] if a < 2 else 2 + TOP_OFF[2] for a in range(3))
face = (CUBIE / 2) if SOLID else \
       (CUBIE / 2 - 0.05) if (SIMPLE or NEAT) else \
       (CUBIE / 2 - 0.30) if CRYSTAL else (CUBIE / 2 - 0.13)
FACES = (("x", (tc[0] + face, tc[1], tc[2])),
         ("y", (tc[0], tc[1] - face, tc[2])),
         ("z", (tc[0], tc[1], tc[2] + face)))
for axis, ctr in ([] if MARKS != "all" else
                  FACES[2:] if (SIMPLE or NEAT or SOLID) else FACES):
    plain = SIMPLE or NEAT or SOLID
    ring(f"tr_{axis}", ctr, axis, 0.255 if plain else 0.250,
         0.032 if plain else 0.026, MAT_RING_R)
    if not plain:
        ring(f"tb_{axis}", ctr, axis, 0.148, 0.022, MAT_RING_B)
    sphere(f"td_{axis}", ctr, 0.105 if SOLID else 0.056, MAT_RING_Y)

# ---------------------------------------------------------------- top-layer twist
if TURN:
    # An actual cube move rather than a camera move: one slice rotates about the axis
    # through the cube centre and is caught mid-turn.
    #   z/2 top layer   y/0 the face pointing lower-left   x/2 the face pointing right
    ax_name = os.environ.get("LOGO_TURN_AXIS", "z").lower()
    ax = {"x": 0, "y": 1, "z": 2}[ax_name]
    idx = int(os.environ.get("LOGO_TURN_SLICE", 2))
    sel = {o for cell, objs in CELL_OBJS.items() if cell[ax] == idx for o in objs}
    if ax_name == "z" and idx == 2:
        # the lifted cube came out of the top layer, so it travels with it
        sel |= {o for o in bpy.context.scene.objects
                if o.type == "MESH" and o.name != "studio"
                and sum((o.matrix_world @ Vector(c)).z for c in o.bound_box) / 8.0 > 2.7}
    pivot = Vector((1.0, 1.0, 1.0))
    M = (Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(TURN), 4, ax_name.upper())
         @ Matrix.Translation(-pivot))
    for o in sel:
        o.matrix_world = M @ o.matrix_world

# ---------------------------------------------------------------- lighting
def area(name, loc, rot, size, power, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.size, d.energy, d.color = size, power, color
    o = bpy.data.objects.new(name, d)
    o.location, o.rotation_euler = loc, rot
    bpy.context.collection.objects.link(o)
    return o


area("key", (-6, -9, 11), (math.radians(38), 0, math.radians(-28)), 11, 2050)
area("fill", (11, -6, 3), (math.radians(78), 0, math.radians(58)), 14,
     1800 if MARK else 1150,
     (0.85, 0.91, 1.0))
area("cavity", (9, 4, 1), (math.radians(88), 0, math.radians(112)), 6, 420,
     (1.0, 0.94, 0.86))
area("rim", (2, 9, 8), (math.radians(-42), 0, math.radians(190)), 9, 1400,
     (0.92, 0.96, 1.0))

# studio dome: gives the glass something to refract and reflect, but is hidden from
# camera rays so the rendered film stays fully transparent.
bpy.ops.mesh.primitive_uv_sphere_add(radius=60, location=(1, 1, 1), segments=48,
                                     ring_count=24)
dome = bpy.context.active_object
dome.name = "studio"
dm = bpy.data.materials.new("studio")
dm.use_nodes = True
nt = dm.node_tree
nt.nodes.remove(nt.nodes["Principled BSDF"])
emis = nt.nodes.new("ShaderNodeEmission")
grad = nt.nodes.new("ShaderNodeTexGradient")
gmap = nt.nodes.new("ShaderNodeMapping")
gtc = nt.nodes.new("ShaderNodeTexCoord")
ramp = nt.nodes.new("ShaderNodeValToRGB")
gmap.inputs["Rotation"].default_value = (math.radians(90), 0, 0)
ramp.color_ramp.elements[0].color = (0.10, 0.12, 0.15, 1)
ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)
nt.links.new(gtc.outputs["Generated"], gmap.inputs["Vector"])
nt.links.new(gmap.outputs["Vector"], grad.inputs["Vector"])
nt.links.new(grad.outputs["Color"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], emis.inputs["Color"])
emis.inputs["Strength"].default_value = 1.5
nt.links.new(emis.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
dome.data.materials.append(dm)
dome.visible_camera = False
dome.visible_shadow = False

# ---------------------------------------------------------------- camera
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam
ELEV = 63.0        # 54.74 is true isometric; higher tips the view down less
cam.rotation_euler = (math.radians(ELEV), 0.0, math.radians(YAW))

R = cam.rotation_euler.to_matrix()
f = R @ Vector((0, 0, -1))                           # camera forward
r = R @ Vector((1, 0, 0))                            # screen right
u = R @ Vector((0, 1, 0))                            # screen up

corners = []
for o in bpy.context.scene.objects:
    if o.type != "MESH" or o.name == "studio":
        continue
    for c in o.bound_box:
        corners.append(o.matrix_world @ Vector(c))
rs = [c.dot(r) for c in corners]
us = [c.dot(u) for c in corners]
cr, cu = (min(rs) + max(rs)) / 2, (min(us) + max(us)) / 2
cam_data.ortho_scale = max(max(rs) - min(rs), max(us) - min(us)) * 1.06
cam.location = r * cr + u * cu - f * 40

# ---------------------------------------------------------------- render
scene.render.engine = "CYCLES"
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception as e:
    print("GPU unavailable, using CPU:", e)

scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 32
scene.cycles.transmission_bounces = 24
scene.cycles.transparent_max_bounces = 32
scene.cycles.caustics_refractive = True
scene.render.film_transparent = True
try:
    scene.cycles.film_transparent_glass = False
except Exception:
    pass
scene.render.resolution_x = scene.render.resolution_y = RES
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = OUT
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

bpy.ops.render.render(write_still=True)
print("wrote", OUT)
