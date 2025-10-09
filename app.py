# Fast Flood + Hurricane Viewer (optimized) - final as of now
# - stable map key for interactions, but remount on flood param changes
# - cached / simplified hover polygons
# - disabled draw/measure controls
# - same UI/features you had

import os, requests, hashlib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import ee, folium
from streamlit_folium import st_folium
import geemap.foliumap as geemap

# ---------------- Earth Engine init ----------------
# def init_ee():
#     try:
#         ee.Initialize(project='sea-level-analysis')
#     except Exception as e:
#         st.error("Earth Engine failed to initialize. Run `earthengine authenticate` or use a service account.")
#         st.exception(e); raise

def init_ee():
    import ee, streamlit as st
    try:
        sa_email = st.secrets["gee"]["email"]
        proj     = st.secrets["gee"]["project"]
        pem      = st.secrets["gee"]["private_key"]

        creds = ee.ServiceAccountCredentials(sa_email, key_data=pem)
        ee.Initialize(credentials=creds, project=proj)
    except Exception as e:
        st.error("Earth Engine failed to initialize. Check service account and secrets.")
        st.exception(e); raise

init_ee()



# ---------- [PATCH-0 IMPORTS+UTILS START] ----------
import json, math
import streamlit as st

def ui_apply_slider(label, vmin, vmax, vstep, qkey, default=0.0, help_txt=None):
    """
    Pattern: a smooth slider + an Apply button that commits to query params and session state.
    Returns the committed/binned value. Only recomputes when Apply is clicked.
    """
    live_key = f"{qkey}__live"
    bin_key  = f"{qkey}__bin"

    # init committed value from query params or default
    if bin_key not in st.session_state:
        qp = st.query_params
        st.session_state[bin_key] = float(qp.get(qkey, default))

    # live control
    live_val = st.slider(label, min_value=float(vmin), max_value=float(vmax),
                         value=float(st.session_state[bin_key]),
                         step=float(vstep), key=live_key, help=help_txt)

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Apply", key=f"apply_{qkey}"):
            # optional binning to slider step to reduce churn
            binned = round(live_val / vstep) * vstep
            st.session_state[bin_key] = float(binned)
            # reflect into URL for sharable permalinks
            st.query_params[qkey] = f"{binned:.3f}"
            st.rerun()

    with col2:
        st.caption(f"Committed: {st.session_state[bin_key]:.3f}")

    return float(st.session_state[bin_key])

def init_state_from_query(defaults: dict):
    """Call once on load. Seeds session_state from query params using provided defaults."""
    if st.session_state.get("_bootstrapped"):  # idempotent
        return
    qp = st.query_params
    for k, v in defaults.items():
        st.session_state.setdefault(k, type(v)(qp.get(k, v)))
    st.session_state["_bootstrapped"] = True

def sync_query_params(keys):
    """Reflect selected state keys into URL query params."""
    for k in keys:
        val = st.session_state.get(k)
        if val is None: 
            continue
        st.query_params[k] = f"{val}"

def get_dynamic_scale(zoom: int):
    """Coarser EE scale when zoomed out (big speedup without visible loss)."""
    if zoom >= 11: return 150
    if zoom >= 9:  return 300
    if zoom >= 7:  return 600
    if zoom >= 5:  return 1200
    return 2400

@st.cache_data(show_spinner=False)
def _cache_small_json(tag: str, payload: dict):
    """Generic small-cache helper for tables/metadata you build."""
    return json.dumps(payload)

# ---------- [PATCH-0 IMPORTS+UTILS END] ----------





# ---------------- Constants ----------------
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
DEFAULT_ANALYSIS_SCALE_M = 300
IBTRACS = ee.FeatureCollection('NOAA/IBTrACS/v4')
# Geometry simplification (meters) for hover polygons (bigger = faster/smaller output)
SIMPLIFY_M = 3000  # good balance for performance; raise to 5000+ for even faster

# ---------------- Basemaps ----------------
os.environ["HYBRID"]    = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
os.environ["SATELLITE"] = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

# ---------------- Streamlit page config ----------------
st.set_page_config(layout="wide", page_title="Flood + Hurricane Viewer")

# ---------------- Caches ----------------
@st.cache_resource(show_spinner=False)
def get_dem():
    return ee.ImageCollection("users/amanaryya1/coastal-dem-files").mosaic().rename("elev").toFloat()

@st.cache_resource(show_spinner=False)
def get_sla_image_m(year: str, month_str: str) -> ee.Image:
    img_id = f"projects/sea-level-analysis/assets/Jiayou/sla_{year}-{month_str}-15"
    try:
        sla_mm = ee.Image(img_id).toFloat().unmask(0)  # mm
        return sla_mm.divide(1000.0)  # -> meters
    except Exception:
        return ee.Image.constant(0.0).toFloat()

@st.cache_resource(show_spinner=False)
def get_month_collection(month_str: str):
    years = list(range(1993, 2023))
    ims = []
    for y in years:
        img_id = f"projects/sea-level-analysis/assets/Jiayou/sla_{y}-{month_str}-15"
        try:
            im = ee.Image(img_id).toFloat().unmask(0).divide(1000.0).set('year', y)  # meters
        except Exception:
            im = ee.Image.constant(0.0).toFloat().set('year', y)
        ims.append(im)
    return ee.ImageCollection(ims), years

@st.cache_data(show_spinner=False, ttl=3600)
def geocode(address: str):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={address}"
    headers = {"User-Agent": "streamlit-app"}
    return requests.get(url, headers=headers).json()

# ---- Summaries (two modes) ----
@st.cache_data(show_spinner=False, ttl=900)
def storms_summary_year_local(lat, lon, season_year: int, radius_km: int):
    pt = ee.Geometry.Point([float(lon), float(lat)])
    fc = (IBTRACS
          .filter(ee.Filter.eq('SEASON', int(season_year)))
          .filterBounds(pt.buffer(radius_km * 1000)))
    sids = ee.List(fc.aggregate_array('SID').distinct())

    def summarize_one(sid):
        track   = fc.filter(ee.Filter.eq('SID', sid)).sort('ISO_TIME')
        track_d = track.map(lambda f: f.set('d_m', f.geometry().distance(pt, 1)))
        closest = track_d.sort('d_m').first()
        dmin    = ee.Number(track_d.aggregate_min('d_m'))
        props   = ee.Feature(closest).toDictionary(['NAME','SEASON','ISO_TIME','USA_SSHS','WMO_WIND','BASIN'])
        return ee.Feature(None, props.combine({'SID': sid, 'MIN_DIST_M': dmin}))

    summary = ee.FeatureCollection(sids.map(summarize_one))
    feats = summary.limit(200).getInfo().get('features', [])
    rows = []
    for f in feats:
        p = f.get('properties', {})
        rows.append({
            'Name':            p.get('NAME'),
            'Season':          p.get('SEASON'),
            'USA_SSHS SCALE':  p.get('USA_SSHS'),
            'Wind (kt)':       p.get('WMO_WIND'),
            'DATE Time':       p.get('ISO_TIME'),
            'MinDist (km)':    round((p.get('MIN_DIST_M') or 0)/1000, 1),
            'SID':             p.get('SID'),
            'Basin':           p.get('BASIN'),
        })
    return rows

@st.cache_data(show_spinner=False, ttl=600)
def point_elev_and_sea_m(_DEM_img: ee.Image, _sla_img_m: ee.Image, adj_m: float,
                         _pt: ee.Geometry, scale_m: int, cache_key: str):
    # --- Elevation EXACTLY at the point ---
    elev_stats = _DEM_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=_pt,
        scale=scale_m, maxPixels=1e13, bestEffort=True, tileScale=4
    )
    # --- Sea surface near the point ---
    ocean = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock').lt(0)
    bn = _sla_img_m.bandNames().get(0)
    sea_surface_m = _sla_img_m.select([bn]).add(adj_m).rename('sea').updateMask(ocean)
    sea_stats = sea_surface_m.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=_pt.buffer(25_000),
        scale=5_000, maxPixels=1e13, bestEffort=True, tileScale=4
    )
    try:
        elev_m = float(elev_stats.getInfo().get('elev'))
    except Exception:
        elev_m = None
    try:
        sea_m = float(sea_stats.getInfo().get('sea'))
    except Exception:
        sea_m = None
    return elev_m, sea_m

@st.cache_data(show_spinner=False, ttl=900)
def storms_summary_by_name_year(lat, lon, name_upper: str, season_year: int):
    pt = ee.Geometry.Point([float(lon), float(lat)])
    fc = (IBTRACS
          .filter(ee.Filter.eq('NAME', name_upper))
          .filter(ee.Filter.eq('SEASON', int(season_year))))
    sids = ee.List(fc.aggregate_array('SID').distinct())

    def summarize_one(sid):
        track   = IBTRACS.filter(ee.Filter.eq('SID', sid)).sort('ISO_TIME')
        track_d = track.map(lambda f: f.set('d_m', f.geometry().distance(pt, 1)))
        closest = track_d.sort('d_m').first()
        dmin    = ee.Number(track_d.aggregate_min('d_m'))
        props   = ee.Feature(closest).toDictionary(['NAME','SEASON','ISO_TIME','USA_SSHS','WMO_WIND','BASIN'])
        return ee.Feature(None, props.combine({'SID': sid, 'MIN_DIST_M': dmin}))

    summary = ee.FeatureCollection(sids.map(summarize_one))
    feats = summary.limit(300).getInfo().get('features', [])
    rows = []
    for f in feats:
        p = f.get('properties', {})
        rows.append({
            'Name':            p.get('NAME'),
            'Season':          p.get('SEASON'),
            'USA_SSHS SCALE':  p.get('USA_SSHS'),
            'Wind (kt)':       p.get('WMO_WIND'),
            'DATE Time':       p.get('ISO_TIME'),
            'MinDist (km)':    round((p.get('MIN_DIST_M') or 0)/1000, 1),
            'SID':             p.get('SID'),
            'Basin':           p.get('BASIN'),
        })
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(['MinDist (km)'], ascending=[True])
        return df.to_dict(orient='records')
    return rows

# ---------------- Helpers (server-side) ----------------
def fc_year_local(point: ee.Geometry, season_year: int, radius_km: int):
    return (IBTRACS
            .filter(ee.Filter.eq('SEASON', int(season_year)))
            .filterBounds(point.buffer(radius_km * 1000)))

def fc_by_name_year(name_upper: str, season_year: int):
    return (IBTRACS
            .filter(ee.Filter.eq('NAME', name_upper))
            .filter(ee.Filter.eq('SEASON', int(season_year))))

def track_geoms_for_sids_from_fc(sids: ee.List):
    def _to_feature(sid):
        t = (IBTRACS
             .filter(ee.Filter.eq('SID', sid))
             .sort('ISO_TIME'))
        def _iter(f, acc):
            acc = ee.List(acc)
            coords = ee.Feature(f).geometry().coordinates()
            return acc.add(coords)
        coords = ee.List(t.iterate(_iter, ee.List([])))
        n = coords.size()
        name   = ee.String(t.first().get('NAME'))
        season = ee.String(t.first().get('SEASON'))
        props  = {'SID': sid, 'NAME': name, 'SEASON': season}
        feat = ee.Algorithms.If(
            n.gte(2),
            ee.Feature(ee.Geometry.LineString(coords, None, True), props),
            ee.Feature(ee.Geometry.Point(ee.List(coords.get(0))), props)
        )
        return ee.Feature(feat)
    return ee.FeatureCollection(sids.map(_to_feature))

def label_points_for_sids(sids: ee.List, ref_point: ee.Geometry):
    def _mk(sid):
        tr  = IBTRACS.filter(ee.Filter.eq('SID', sid)).sort('ISO_TIME')
        trd = tr.map(lambda f: f.set('d_m', f.geometry().distance(ref_point, 1)))
        close = ee.Feature(trd.sort('d_m').first())
        return ee.Feature(close.geometry(), {
            'SID': sid, 'NAME': close.get('NAME'), 'SEASON': close.get('SEASON')
        })
    return ee.FeatureCollection(sids.map(_mk))

def time_series_for_point(point: ee.Geometry, month_str: str, scale_m: int):
    col, _ = get_month_collection(month_str)
    def per_im(im):
        bn = im.bandNames().get(0)
        val_m = ee.Number(im.sample(point, scale=scale_m).first().get(bn))
        return ee.Feature(None, {'year': im.get('year'), 'value_m': val_m})
    fc = col.map(per_im)
    years  = fc.aggregate_array('year').getInfo()
    values = fc.aggregate_array('value_m').getInfo()
    return years, values

def add_circles_by_sshs_overlay(m, fc_filtered, region, scale, circles_alpha=0.3):
    try:
        if fc_filtered is None or fc_filtered.size().getInfo() == 0:
            return None
    except Exception:
        return None

    nm_to_m = ee.Number(1852)
    def _num_or0(f, key): return ee.Number(ee.Algorithms.If(f.get(key), f.get(key), 0))

    def _with_radii(f):
        f = ee.Feature(f)
        r34 = ee.Number(ee.List([_num_or0(f,'USA_R34_NE'),_num_or0(f,'USA_R34_SE'),_num_or0(f,'USA_R34_SW'),_num_or0(f,'USA_R34_NW')]).reduce(ee.Reducer.max()))
        r50 = ee.Number(ee.List([_num_or0(f,'USA_R50_NE'),_num_or0(f,'USA_R50_SE'),_num_or0(f,'USA_R50_SW'),_num_or0(f,'USA_R50_NW')]).reduce(ee.Reducer.max()))
        r64 = ee.Number(ee.List([_num_or0(f,'USA_R64_NE'),_num_or0(f,'USA_R64_SE'),_num_or0(f,'USA_R64_SW'),_num_or0(f,'USA_R64_NW')]).reduce(ee.Reducer.max()))
        return f.set({'radius_m_r34': r34.multiply(nm_to_m),
                      'radius_m_r50': r50.multiply(nm_to_m),
                      'radius_m_r64': r64.multiply(nm_to_m)})

    fc_with_r = fc_filtered.map(_with_radii)

    def _make_buffers(prop_key):
        fc_pos = fc_with_r.filter(ee.Filter.gt(prop_key, 0))
        return fc_pos.map(lambda f: ee.Feature(ee.Feature(f).geometry().buffer(ee.Feature(f).getNumber(prop_key), SIMPLIFY_M))
                          .copyProperties(f))

    buf_fc_r34 = _make_buffers('radius_m_r34')
    buf_fc_r50 = _make_buffers('radius_m_r50')
    buf_fc_r64 = _make_buffers('radius_m_r64')

    base_alpha = float(circles_alpha or 0.30)
    ring_color = ['#ff8800']

    img_r34 = ee.Image().paint(buf_fc_r34, 1).selfMask().visualize(palette=ring_color, min=1, max=1)
    img_r50 = ee.Image().paint(buf_fc_r50, 1).selfMask().visualize(palette=ring_color, min=1, max=1)
    img_r64 = ee.Image().paint(buf_fc_r64, 1).selfMask().visualize(palette=ring_color, min=1, max=1)

    m.addLayer(img_r34, {}, "34 kt wind (USA_R34)", opacity=base_alpha * 0.55)
    m.addLayer(img_r50, {}, "50 kt wind (USA_R50)", opacity=base_alpha * 0.75)
    m.addLayer(img_r64, {}, "64 kt wind (USA_R64)", opacity=base_alpha * 0.95)

    try:
        outline_fc  = buf_fc_r34.map(lambda f: ee.Feature(ee.Feature(f).geometry().boundaries(1)))
        outline_img = ee.Image().paint(outline_fc, 1, 1).visualize(palette=['#ff8800'])
        m.addLayer(outline_img, {}, "Wind extent outline (34 kt)", opacity=0.9)
    except Exception:
        pass

    circle_mask_outer = ee.Image().paint(buf_fc_r34, 1).selfMask()
    circ_area_img     = circle_mask_outer.multiply(ee.Image.pixelArea())
    circ_stats = circ_area_img.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=region, scale=scale,
        maxPixels=1e13, bestEffort=True, tileScale=4
    )
    try:
        impact_area_km2 = (circ_stats.getInfo().get('constant') or 0.0) / 1e6
    except Exception:
        impact_area_km2 = 0.0
    return impact_area_km2

# ---------- FAST HOVER: cached & simplified GeoJSON ----------
def _hover_geojson_cache_key(mode: str, year: str, storm_name: str, rad_km, region_bbox):
    key_str = f"{mode}|{year}|{storm_name or 'ALL'}|R{int(rad_km or 0)}|{region_bbox}"
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()

def _build_hover_geojson(fc_filtered, region):
    nm_to_m = ee.Number(1852)
    def _num_or0(f, key):
        val = ee.Number(ee.Algorithms.If(f.get(key), f.get(key), 0))
        return ee.Number(ee.Algorithms.If(val, val, 0))

    def _with_radii_and_cat(f):
        f = ee.Feature(f)
        r34 = ee.Number(ee.List([_num_or0(f,'USA_R34_NE'),_num_or0(f,'USA_R34_SE'),_num_or0(f,'USA_R34_SW'),_num_or0(f,'USA_R34_NW')]).reduce(ee.Reducer.max())).multiply(nm_to_m)
        r50 = ee.Number(ee.List([_num_or0(f,'USA_R50_NE'),_num_or0(f,'USA_R50_SE'),_num_or0(f,'USA_R50_SW'),_num_or0(f,'USA_R50_NW')]).reduce(ee.Reducer.max())).multiply(nm_to_m)
        r64 = ee.Number(ee.List([_num_or0(f,'USA_R64_NE'),_num_or0(f,'USA_R64_SE'),_num_or0(f,'USA_R64_SW'),_num_or0(f,'USA_R64_NW')]).reduce(ee.Reducer.max())).multiply(nm_to_m)
        wind = ee.Number(ee.Algorithms.If(f.get('USA_WIND'), f.get('USA_WIND'), -1))
        cat = ee.String(ee.Algorithms.If(
            wind.lt(0), 'NA',
            ee.Algorithms.If(wind.lt(34), 'TD',
            ee.Algorithms.If(wind.lt(64), 'TS',
            ee.Algorithms.If(wind.lt(83), '1',
            ee.Algorithms.If(wind.lt(96), '2',
            ee.Algorithms.If(wind.lt(113),'3',
            ee.Algorithms.If(wind.lt(137),'4','5')))))))
        )
        return f.set({'r34_m': r34, 'r50_m': r50, 'r64_m': r64,
                      'CAT': cat, 'ISO_TIME_keep': f.get('ISO_TIME'), 'USA_WIND_keep': wind, 'NAME_keep': f.get('NAME')})

    base = fc_filtered.map(_with_radii_and_cat)

    def _mk_ring_fc(prop_key, label):
        fc_pos = base.filter(ee.Filter.gt(prop_key, 0))
        def _buf(f):
            f = ee.Feature(f)
            r = ee.Number(f.get(prop_key))
            geom = f.geometry().buffer(r, SIMPLIFY_M).simplify(SIMPLIFY_M)
            return ee.Feature(geom).set({
                'NAME':     f.get('NAME_keep'),
                'ISO_TIME': f.get('ISO_TIME_keep'),
                'USA_WIND': f.get('USA_WIND_keep'),
                'CAT':      f.get('CAT'),
                'RING':     label,
                'RADIUS_KM': r.divide(1000)
            })
        return fc_pos.map(_buf).filterBounds(region)

    fc_tips = ee.FeatureCollection([_mk_ring_fc('r34_m','R34'),
                                    _mk_ring_fc('r50_m','R50'),
                                    _mk_ring_fc('r64_m','R64')]).flatten()

    gj = geemap.ee_to_geojson(fc_tips)
    return gj

def add_wind_extent_hover_tooltips_cached(m, fc_filtered, region, cache_key, layer_name="Wind extent (hover info)"):
    try:
        if fc_filtered is None or fc_filtered.size().getInfo() == 0:
            return
    except Exception:
        return

    if 'hover_cache' not in st.session_state:
        st.session_state['hover_cache'] = {}

    cache = st.session_state['hover_cache']
    if cache_key in cache:
        gj = cache[cache_key]
    else:
        try:
            gj = _build_hover_geojson(fc_filtered, region)
        except Exception:
            return
        if not gj or not gj.get("features"):
            return
        cache[cache_key] = gj

    try:
        folium.map.Pane('hoverpane', m)
        m.get_root().html.add_child(folium.Element(
            "<style>.leaflet-pane.hoverpane{z-index:650;pointer-events:auto;}</style>"
        ))
        pane_kw = {"pane": "hoverpane"}
    except Exception:
        pane_kw = {}

    def _style(_):
        return {"weight": 0, "color": "#000000", "opacity": 0,
                "fill": True, "fillColor": "#000000", "fillOpacity": 0}

    folium.GeoJson(
        gj,
        name=layer_name,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "ISO_TIME", "USA_WIND", "CAT"],
            aliases=["Name", "Time", "Wind (kt)", "Category"],
            sticky=True, labels=True, localize=True
        ),
        **pane_kw
    ).add_to(m)

# ---------------- UI state ----------------
if "selected_month" not in st.session_state: st.session_state["selected_month"] = "01"
if "center_lat"   not in st.session_state: st.session_state["center_lat"] = 30.0
if "center_lon"   not in st.session_state: st.session_state["center_lon"] = -90.0
if "zoom"         not in st.session_state: st.session_state["zoom"] = 5
if "clicked_lat"  not in st.session_state: st.session_state["clicked_lat"] = st.session_state["center_lat"]
if "clicked_lon"  not in st.session_state: st.session_state["clicked_lon"] = st.session_state["center_lon"]
if "storm_name"   not in st.session_state: st.session_state["storm_name"] = ""

st.title("Flood + Hurricane Viewer")

with st.sidebar:
    st.markdown('<h2 style="margin-top:0rem;margin-bottom:0.5rem">Sea Level Rise</h2>', unsafe_allow_html=True)
    years = [str(y) for y in range(1993, 2023)]
    year = st.selectbox("SLA Year", years, index=years.index("2020"))

    st.markdown("### Select Month")
    month_map = {label: f"{i+1:02d}" for i, label in enumerate(MONTH_LABELS)}
    for row in [MONTH_LABELS[i:i+4] for i in range(0, 12, 4)]:
        cols = st.columns(4, gap="small")
        for i, label in enumerate(row):
            if cols[i].button(label, key=f"month_{label}"):
                st.session_state["selected_month"] = month_map[label]
                # 🔁 Ensure immediate refresh of flood layer on month change
                st.rerun()

    st.markdown("### Flood Controls")
    adj_val = st.slider("Extra Sea Level Rise (m)", 0.0, 5.0, 0.0, 0.05, key="sld_adj")
    adj_bin = round(adj_val / 0.25) * 0.25

    st.markdown("### Hurricanes (NOAA IBTrACS v4)")
    show_hist_hurr = st.checkbox("Show storms near clicked point", value=True, key="chk_hist_hurr")
    rad_km = st.slider("Search radius (km)", 50, 600, 250, 25, key="sld_radius") if show_hist_hurr else None

    impact_mode = "None"
    if show_hist_hurr:
        impact_mode = st.selectbox(
            "Choose overlay", ["None", "Circles by SSHS"], index=0,
            help="Draw approximate affected area via category-based concentric circles."
        )

    st.session_state["storm_name"] = st.text_input("Search by storm NAME",
        value=st.session_state.get("storm_name", "")).strip()

    st.markdown("### Location Search")
    c1, c2 = st.columns(2)
    with c1: st.text_input("Lat", value=str(st.session_state["center_lat"]), key="lat_in")
    with c2: st.text_input("Lon", value=str(st.session_state["center_lon"]), key="lon_in")
    address = st.text_input("Type a location (e.g., Miami Beach)", key="txt_addr")
    if st.button("Search", key="btn_search") and address.strip():
        try:
            resp = geocode(address)
            if resp:
                st.session_state["center_lat"] = float(resp[0]["lat"])
                st.session_state["center_lon"] = float(resp[0]["lon"])
                st.session_state["clicked_lat"] = st.session_state["center_lat"]
                st.session_state["clicked_lon"] = st.session_state["center_lon"]
                st.session_state["zoom"] = 5
                st.success(f"📍 Found: {resp[0]['display_name']}")
                st.rerun()  # refresh map right away after search
            else:
                st.warning("❌ No location found.")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

selected_month = st.session_state["selected_month"]
center_lat = float(st.session_state["center_lat"])
center_lon = float(st.session_state["center_lon"])
zoom       = int(st.session_state["zoom"])
storm_name_filter = st.session_state.get("storm_name", "").upper()

# ---------------- Data layers ----------------
DEM = get_dem()
sla_m = get_sla_image_m(year, selected_month).divide(1000.0)
adj_img = ee.Image.constant(float(adj_bin)).toFloat()
water_level = sla_m.add(adj_img)
flood_depth = water_level.subtract(DEM).max(0)
flooded = flood_depth.updateMask(flood_depth.gt(0).And(DEM.gt(0)))

# ---------------- Map build ----------------
m = geemap.Map(center=(center_lat, center_lon), zoom=zoom)

# Disable interactive controls that steal pointer events
for fn in ("remove_draw_control", "remove_measure_control", "remove_inspector"):
    try: getattr(m, fn)()
    except Exception: pass
m.get_root().html.add_child(folium.Element("""
<style>
  .leaflet-control-container .leaflet-draw { display:none !important; }
  .leaflet-control-container .leaflet-measure-toggle { display:none !important; }
</style>
"""))

m.add_basemap("HYBRID"); #m.add_basemap("Google Labels")

flood_vis = {
    'min': 0, 'max': 5,
    'palette': ['#e0ffff','#d2fdff','#c3fbff','#b5f8ff','#a6f2ff','#97ecff','#88e6ff',
                '#79dfff','#69d7ff','#59cfff','#49c5ff','#38bbff','#28afff','#1aa2ff',
                '#0f95ff','#0a86f2','#0878e0','#066bcc','#045db8','#034ea3']
}
# Flood layer is always added, independent of hurricane UI
m.addLayer(flooded, flood_vis, f"Flooded Depth - {year}-{selected_month} (+{adj_bin:.2f} m)", opacity=0.7)

# Clicked point geometry
clicked_lat = float(st.session_state["clicked_lat"])
clicked_lon = float(st.session_state["clicked_lon"])
point = ee.Geometry.Point([clicked_lon, clicked_lat])

# ---- Choose storm mode ----
use_name_mode = len(storm_name_filter) > 0

# Storms: table rows + server-styled tracks/points
storm_rows, fc_filtered, sids = [], None, None

if show_hist_hurr:
    if use_name_mode:
        season_year = int(year)
        storm_rows = storms_summary_by_name_year(clicked_lat, clicked_lon, storm_name_filter, season_year)
        fc_filtered = fc_by_name_year(storm_name_filter, season_year)
        sids = ee.List(fc_filtered.aggregate_array('SID').distinct())
        try:
            tracks_fc = track_geoms_for_sids_from_fc(sids)
            tracks_img = tracks_fc.style(color='yellow', width=2, pointSize=3)
            m.addLayer(tracks_img, {}, f"IBTrACS tracks (NAME={storm_name_filter}, {season_year})")
            label_fc = label_points_for_sids(sids, point)
            label_img = label_fc.style(pointSize=3, color='yellow')
            m.addLayer(label_img, {}, "Closest fixes to clicked point")
        except Exception:
            pass
    else:
        storm_rows = storms_summary_year_local(clicked_lat, clicked_lon, int(year), int(rad_km or 250))
        fc_filtered = fc_year_local(point, int(year), int(rad_km or 250))
        sids = ee.List(fc_filtered.aggregate_array('SID').distinct())
        try:
            tracks_fc = track_geoms_for_sids_from_fc(sids)
            tracks_img = tracks_fc.style(color='yellow', width=2, pointSize=3)
            m.addLayer(tracks_img, {}, f"IBTrACS tracks ({year}, within {int(rad_km or 250)} km)")
            label_fc = label_points_for_sids(sids, point)
            label_img = label_fc.style(pointSize=3, color='yellow')
            m.addLayer(label_img, {}, "Closest storm fixes")
        except Exception:
            pass

        if rad_km:
            folium.Circle([clicked_lat, clicked_lon], radius=int(rad_km)*1000,
                          color="#ffff00", weight=1, fill=False).add_to(m)

# Clicked point marker
folium.CircleMarker([clicked_lat, clicked_lon], radius=3, color="red",
                    fill=True, fill_opacity=1.0).add_to(m)

# Colorbar
m.add_colorbar(flood_vis, label="Flood Depth (m)",
               layer_name=f"Flooded Depth - {year}-{selected_month}")

# ---- Analytics region (used by overlays + metrics) ----
scale = DEFAULT_ANALYSIS_SCALE_M

# Slider value in meters (scalar)
adj_val_m = float(adj_bin)

# Build a small cache key so results refresh when these change
pt_key = f"{year}-{selected_month}:{clicked_lat:.4f},{clicked_lon:.4f}:{adj_val_m:.2f}:{scale}"

elev_m_at_point, sea_m_nearby = point_elev_and_sea_m(
    DEM, sla_m, adj_val_m, point, scale, pt_key
)

cur_zoom = int(st.session_state["zoom"])
size_deg = max(0.1, 8.0 / max(1, cur_zoom))
half = size_deg / 2.0
region = ee.Geometry.Rectangle(
    [st.session_state["clicked_lon"] - half,
     st.session_state["clicked_lat"] - half,
     st.session_state["clicked_lon"] + half,
     st.session_state["clicked_lat"] + half],
    proj='EPSG:4326', geodesic=False
)

# 🔴 Add storm impact overlay BEFORE rendering the map
impact_area_km2 = None
if show_hist_hurr and fc_filtered is not None and impact_mode == "Circles by SSHS":
    impact_area_km2 = add_circles_by_sshs_overlay(
        m, fc_filtered, region, scale, circles_alpha=0.30
    )

# Hover layer (cached & simplified)
if show_hist_hurr and fc_filtered is not None:
    region_bbox = tuple([round(st.session_state["clicked_lon"] - half, 3),
                         round(st.session_state["clicked_lat"] - half, 3),
                         round(st.session_state["clicked_lon"] + half, 3),
                         round(st.session_state["clicked_lat"] + half, 3)])
    mode = "NAME" if use_name_mode else "LOCAL"
    cache_key = _hover_geojson_cache_key(mode, year, storm_name_filter, rad_km, region_bbox)
    add_wind_extent_hover_tooltips_cached(m, fc_filtered, region, cache_key)

# --------- Layout: Map (left) & Right panel with storms + analytics ----------
map_col, right_col = st.columns([3, 2], gap="large")

with map_col:
    # 🔑 Remount the map whenever flood inputs change, so the flood layer always refreshes
    map_key = f"main_map_{year}_{selected_month}_{adj_bin:.2f}"
    st_data = st_folium(m, width=None, height=720, key=map_key)

# -------- Instant analyze on map click (no pending state / no button) --------
click_info = st_data.get("last_clicked") if st_data else None
if click_info:
    new_lat = float(click_info["lat"])
    new_lon = float(click_info["lng"])
    if (abs(new_lat - float(st.session_state["clicked_lat"])) > 1e-6 or
        abs(new_lon - float(st.session_state["clicked_lon"])) > 1e-6):
        st.session_state["clicked_lat"] = new_lat
        st.session_state["clicked_lon"] = new_lon
        st.session_state["center_lat"]  = new_lat
        st.session_state["center_lon"]  = new_lon
        st.session_state["zoom"]        = max(st.session_state.get("zoom", 5), 5)
        st.rerun()

with right_col:
    if show_hist_hurr:
        if use_name_mode:
            st.markdown(f"## Storms (NAME = {storm_name_filter or '—'}, Year = {year})")
        else:
            st.markdown(f"## Storms near point ({year}, within {int(rad_km or 250)} km)")
        if storm_rows:
            cols = ["Name", "Season", "USA_SSHS SCALE", "Wind (kt)", "DATE Time", "MinDist (km)", "SID", "Basin"]
            df = pd.DataFrame(storm_rows, columns=cols)
            header_h = 38; row_h = 32
            height = header_h + row_h * max(1, len(df))
            st.dataframe(df, width='stretch', height=height, hide_index=True)
        else:
            st.info("No storms found for the current selection.")

    st.markdown("## Point Analytics & Flood Stats")
    a_left, a_right = st.columns([3, 2])

    with a_left:
        years_list, values_list = time_series_for_point(point, selected_month, scale)
        label = MONTH_LABELS[int(selected_month)-1]
        st.markdown(f"**SLA Time Series at ({st.session_state['clicked_lat']:.4f}, "
                    f"{st.session_state['clicked_lon']:.4f}) — {label} 1993–2022**")
        fig, ax = plt.subplots()
        ax.plot(years_list, values_list, marker='o')
        ax.set_xlabel("Year"); ax.set_ylabel("Sea Level Anomaly (m)")
        ax.set_title(f"SLA at ({st.session_state['clicked_lat']:.3f}, {st.session_state['clicked_lon']:.3f}) — {label}")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    with a_right:
        flood_area_img = flooded.gt(0).selfMask().multiply(ee.Image.pixelArea()).rename('area')
        area_stats = flood_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale,
            maxPixels=1e13,
            bestEffort=True,
            tileScale=4
        )
        try:
            area_val = area_stats.getInfo()
            area_km2 = (area_val.get('area') or 0.0) / 1e6
        except Exception:
            area_km2 = 0.0

        st.metric("Elevation at Clicked Point", f"{elev_m_at_point:,.3f} m" if elev_m_at_point is not None else "n/a")
        st.metric("Sea level at Clicked Point", f"{sea_m_nearby:,.3f} m" if sea_m_nearby is not None else "n/a")

        if impact_area_km2 is not None:
            title = "Storm Impact Area (selected cyclone)" if use_name_mode else "Storm Impact Area (overlay)"
            st.markdown(f"**{title}**")
            st.metric(label="Affected Area", value=f"{impact_area_km2:,.2f} km²")
