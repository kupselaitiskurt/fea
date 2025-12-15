# app.py
# ---------------------------------------------------------------------------
# Streamlit interface for PyNite – editable nodes, supports & nodal loads
# METRIC UNITS: millimetres (mm), kilonewtons (kN), kilonewton‑metres (kN·m)
# * Blank/None rows in sidebar tables are ignored during analysis
# * 3‑D Plotly view now shows loads + deformed shape (scaled) with annotations
# ---------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from Pynite import FEModel3D
from pynite_plotly import Renderer as PlotlyRenderer

VALID_DIRS = {"FX", "FY", "FZ", "MX", "MY", "MZ"}

# ─────────────────────────────────────────────────────────────
# Page settings + white background
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PyNite frame analysis (metric)", layout="wide")
st.title("PyNite frame analysis – editable nodes, supports & loads (metric)")

# ─────────────────────────────────────────────────────────────
# Default tables (metric units)
# ─────────────────────────────────────────────────────────────
DEFAULT_NODES = pd.DataFrame(
    {
        "Name": ["N1", "N2", "N3", "N4"],
        "X":    [   0, -2500,    0,    0],  # mm
        "Y":    [   0,     0,    0, -2500],
        "Z":    [   0,     0, -2500,    0],
    }
)

DEFAULT_SUPPORTS = pd.DataFrame(
    {
        "Node": ["N2", "N3", "N4"],
        "UX":   [True, True, True],
        "UY":   [True, True, True],
        "UZ":   [True, True, True],
        "RX":   [True, True, True],
        "RY":   [True, True, True],
        "RZ":   [True, True, True],
    }
)

DEFAULT_LOADS = pd.DataFrame(
    {
        "Node": ["N1",],
        "Dir":  ["FY",],
        "Mag":  [-25000000.0,],  # kN or kN·m
    }
)

# ─────────────────────────────────────────────────────────────
# Utility cleaners – safely drop blank rows
# ─────────────────────────────────────────────────────────────

def _clean_nodes(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Name", "X", "Y", "Z"])

    out = df.copy()
    out = out.dropna(subset=["Name"])
    out["Name"] = out["Name"].astype(str).str.strip()
    out = out[out["Name"] != ""]

    for col in ("X", "Y", "Z"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["X", "Y", "Z"])

    return out.reset_index(drop=True)


def _clean_supports(df: pd.DataFrame, valid_nodes: set) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns)
    return df[df["Node"].isin(valid_nodes)].reset_index(drop=True)


def _clean_loads(df: pd.DataFrame, valid_nodes: set) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns)

    out = df.copy()
    out = out[out["Node"].isin(valid_nodes)]
    out["Dir"] = out["Dir"].astype(str).str.upper().str.strip()
    out = out[out["Dir"].isin(VALID_DIRS)]
    out["Mag"] = pd.to_numeric(out["Mag"], errors="coerce")
    out = out.dropna(subset=["Mag"])
    return out.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# Build FE model helper
# ─────────────────────────────────────────────────────────────

def build_model(nodes: pd.DataFrame, supports: pd.DataFrame, loads: pd.DataFrame) -> FEModel3D:
    nodes = _clean_nodes(nodes)
    if nodes.empty:
        raise ValueError("No valid nodes defined – please add at least one node.")

    valid_nodes = set(nodes["Name"])
    supports = _clean_supports(supports, valid_nodes)
    loads = _clean_loads(loads, valid_nodes)

    m = FEModel3D()

    # Nodes
    for _, r in nodes.iterrows():
        m.add_node(r["Name"], float(r["X"]), float(r["Y"]), float(r["Z"]))

    # Supports
    for _, r in supports.iterrows():
        m.def_support(r["Node"],
                      bool(r.get("UX", False)), bool(r.get("UY", False)), bool(r.get("UZ", False)),
                      bool(r.get("RX", False)), bool(r.get("RY", False)), bool(r.get("RZ", False)),)

    # Material & section (demo)
    m.add_material("Steel", 210_000, 81_000, 0.30, 7.85e-6)
    m.add_section("Sect", A=15_000, Iy=8.5e8, Iz=8.5e8, J=1.7e9)

    # Members – star‑shaped, root at first node
    root = nodes.iloc[0]["Name"]
    for target in nodes["Name"][1:]:
        m.add_member(f"M_{root}_{target}", root, target, "Steel", "Sect")

    # Loads
    for _, r in loads.iterrows():
        m.add_node_load(r["Node"], r["Dir"], float(r["Mag"]))

    return m

# ─────────────────────────────────────────────────────────────
# Sidebar – inputs
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model inputs")

    st.subheader("Node coordinates (mm)")
    nodes_df = st.data_editor(DEFAULT_NODES, num_rows="dynamic", use_container_width=True, key="nodes_editor")

    st.subheader("Support fixities")
    supports_df = st.data_editor(
        DEFAULT_SUPPORTS,
        column_config={col: st.column_config.CheckboxColumn() for col in ["UX", "UY", "UZ", "RX", "RY", "RZ"]},
        num_rows="dynamic",
        use_container_width=True,
        key="supports_editor",
    )

    st.subheader("Nodal loads (kN / kN·m)")
    loads_df = st.data_editor(DEFAULT_LOADS, num_rows="dynamic", use_container_width=True, key="loads_editor")

    analyse_clicked = st.button("🔎 Analyse", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Main page – analysis & plotting
# ─────────────────────────────────────────────────────────────
if analyse_clicked:
    try:
        with st.spinner("Solving FE model…"):
            model = build_model(nodes_df, supports_df, loads_df)
            model.analyze(check_statics=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
    else:
        st.success("Analysis complete!")

        # Deflections table
        disp_df = pd.DataFrame([
            {
                "Node": n,
                "DX (mm)": model.nodes[n].DX['Combo 1'],
                "DY (mm)": model.nodes[n].DY['Combo 1'],
                "DZ (mm)": model.nodes[n].DZ['Combo 1'],
            } for n in model.nodes]).set_index("Node").round(3)
        st.subheader("Node deflections (DX / DY / DZ) – mm")
        st.dataframe(disp_df, use_container_width=True)

        # 3‑D Plotly view with loads + deformed shape ------------------------
        with st.spinner("Rendering 3‑D view…"):
            try:
                rnd = PlotlyRenderer(model)
            except TypeError:
                rnd = PlotlyRenderer(model, "Combo 1")
            if getattr(rnd, "combo_name", None) in (None, ""):
                rnd.combo_name = "Combo 1"

            # ✨ NEW renderer options
            try:
                rnd.annotation_size = 5      # smaller text
                rnd.render_loads = True       # show force arrows
                rnd.deformed_shape = True     # overlay deformed shape
                rnd.deformed_scale = 40       # exaggeration factor
            except AttributeError:
                # Older pynite_plotly versions might not expose these attrs
                pass

            fig = rnd.render_model()
            fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", scene=dict(bgcolor="#ffffff"))

        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Edit the tables in the sidebar and hit **Analyse** to run the model.")
