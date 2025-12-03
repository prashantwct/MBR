import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import pydeck as pdk
import altair as alt
import time
import subprocess
import sys
from io import StringIO
from datetime import datetime, timedelta
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from scipy.interpolate import splprep, splev
import joblib

# --- R PACKAGE INSTALLATION (Cloud Compatibility) ---
# This ensures 'ctmm' and 'sf' are installed in the cloud environment
@st.cache_resource
def install_r_packages():
    # Check if R is available
    try:
        import rpy2
    except ImportError:
        return False

    # Check if packages are installed
    check_script = """
    if (!require("sf")) quit(status=1)
    if (!require("ctmm")) quit(status=1)
    """
    check_res = subprocess.run(["R", "-e", check_script], capture_output=True)
    
    if check_res.returncode != 0:
        with st.spinner("Installing R packages (sf, ctmm)... this takes 5-10 mins on first load."):
            repo = "https://cloud.r-project.org"
            install_script = f"""
            install.packages("sf", repos="{repo}")
            install.packages("ctmm", repos="{repo}")
            """
            process = subprocess.run(["R", "-e", install_script], capture_output=True, text=True)
            if process.returncode != 0:
                st.error(f"R Installation Failed: {process.stderr}")
                return False
    return True

# Trigger installation check
r_ready = install_r_packages()

# Try importing RPY2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False

# Page Config
st.set_page_config(page_title="Movebank Explorer", layout="wide")
st.title("Movebank Explorer 🗺️")

# --- Helper Functions ---
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

def normalize_cols(df):
    df.columns = [c.lower().replace('-', '_').replace('.', '_') for c in df.columns]
    return df

def get_random_color(seed_str):
    hash_val = hash(seed_str)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = (hash_val & 0x0000FF)
    return [r, g, b]

def r_list_to_dict(r_list):
    try:
        keys = list(r_list.names)
        return {k: r_list.rx2(k) for k in keys}
    except:
        return {}

def get_tile_layer(style):
    if style == "Satellite":
        url = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    elif style == "Light":
        url = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    else: 
        url = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
        
    return pdk.Layer(
        "TileLayer",
        data=url,
        id=f"basemap-layer-{style}",
        min_zoom=0,
        max_zoom=19,
        opacity=1.0
    )

def run_st_dbscan(df, spatial_eps_m, temporal_eps_min, min_samples):
    mean_lat = df['lat'].mean()
    mean_lon = df['lon'].mean()
    lat_scale = 111320
    lon_scale = 111320 * np.cos(np.radians(mean_lat))
    X = (df['lon'] - mean_lon) * lon_scale
    Y = (df['lat'] - mean_lat) * lat_scale
    coords = np.column_stack([X, Y])
    nbrs = NearestNeighbors(radius=spatial_eps_m, algorithm='ball_tree').fit(coords)
    adj_matrix = nbrs.radius_neighbors_graph(coords, mode='connectivity')
    adj_coo = adj_matrix.tocoo()
    rows, cols = adj_coo.row, adj_coo.col
    times = df['timestamp'].values.astype(np.int64) // 10**9 // 60 
    time_diffs = np.abs(times[rows] - times[cols])
    mask = time_diffs <= temporal_eps_min
    new_data = np.ones(np.sum(mask), dtype=int)
    new_row = rows[mask]
    new_col = cols[mask]
    filtered_adj = csr_matrix((new_data, (new_row, new_col)), shape=adj_matrix.shape)
    n_components, labels = connected_components(csgraph=filtered_adj, directed=False, return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique[counts >= min_samples]
    final_labels = np.array([l if l in valid_clusters else -1 for l in labels])
    return final_labels

# --- ML Functions ---
MODEL_FILE = "cluster_model.pkl"
DATA_FILE = "labeled_clusters.csv"

def load_model():
    if os.path.exists(MODEL_FILE): return joblib.load(MODEL_FILE)
    return None

def train_and_save(new_data_df):
    if os.path.exists(DATA_FILE):
        old_df = pd.read_csv(DATA_FILE)
        full_df = pd.concat([old_df, new_data_df], ignore_index=True)
    else:
        full_df = new_data_df
    full_df.to_csv(DATA_FILE, index=False)
    train_df = full_df[full_df['Label'] != 'Unclassified']
    if len(train_df) > 5:
        X = train_df[['Duration (hrs)', 'Points']]
        y = train_df['Label']
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        return True, len(train_df)
    return False, len(train_df)

# --- Sidebar: Connection ---
with st.sidebar:
    st.header("1. Connection")
    default_user = os.getenv("MOVEBANK_USER", "")
    default_pass = os.getenv("MOVEBANK_PASS", "")
    username = st.text_input("Movebank Username", value=default_user)
    password = st.text_input("Movebank Password", value=default_pass, type="password")
    study_id = st.text_input("Study ID", placeholder="e.g., 2911040")
    fetch_btn = st.button("Fetch Data", type="primary")
    
    st.header("2. Map Settings")
    map_style = st.selectbox("Basemap Style", ["Satellite", "Light", "Dark"], index=0)

# --- Main Logic ---
if 'data' not in st.session_state:
    st.session_state.data = None

if fetch_btn:
    if not username or not password or not study_id:
        st.error("⚠️ Please provide a Study ID and Movebank credentials.")
    else:
        url_events = f"https://www.movebank.org/movebank/service/direct-read?entity_type=event&study_id={study_id}"
        url_ref = f"https://www.movebank.org/movebank/service/direct-read?entity_type=individual&study_id={study_id}"
        try:
            with st.spinner("Downloading Data..."):
                r_ev = requests.get(url_events, auth=(username, password))
                if r_ev.status_code != 200:
                    st.error(f"Event Fetch Failed: {r_ev.status_code}")
                    st.stop()
                df_ev = pd.read_csv(StringIO(r_ev.text))
                df_ev = normalize_cols(df_ev)
                r_ref = requests.get(url_ref, auth=(username, password))
                if r_ref.status_code == 200:
                    df_ref = pd.read_csv(StringIO(r_ref.text))
                    df_ref = normalize_cols(df_ref)
                    if 'individual_id' in df_ev.columns and 'id' in df_ref.columns:
                        df_ref = df_ref.add_prefix('ref_')
                        df_final = pd.merge(df_ev, df_ref, left_on='individual_id', right_on='ref_id', how='left')
                    else:
                        df_final = df_ev
                else:
                    df_final = df_ev
                if df_final.empty:
                    st.warning("No data found.")
                else:
                    map_cols = {'location_lat': 'lat', 'location_long': 'lon'}
                    if set(map_cols.keys()).issubset(df_final.columns):
                        df_final = df_final.rename(columns=map_cols)
                    if 'timestamp' in df_final.columns:
                        df_final['timestamp'] = pd.to_datetime(df_final['timestamp'])
                        if df_final['timestamp'].dt.tz is None:
                            df_final['timestamp'] = df_final['timestamp'].dt.tz_localize('UTC')
                        df_final['timestamp'] = df_final['timestamp'].dt.tz_convert('Asia/Kolkata')
                        df_final = df_final.dropna(subset=['lat', 'lon', 'timestamp'])
                        df_final = df_final.sort_values(['individual_id', 'timestamp'])
                        grouped = df_final.groupby('individual_id')
                        df_final['step_dist'] = haversine_np(df_final['lon'], df_final['lat'], grouped['lon'].shift(1), grouped['lat'].shift(1)).fillna(0)
                        df_final['time_diff'] = (df_final['timestamp'] - grouped['timestamp'].shift(1)).dt.total_seconds().fillna(1)
                        df_final['speed'] = (df_final['step_dist'] * 1000) / df_final['time_diff']
                        df_final = df_final[(df_final['speed'] <= 5) | (df_final['speed'].isna())]
                    st.session_state.data = df_final
                    st.success(f"✅ Loaded {len(df_final)} rows.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Dashboard View ---
if st.session_state.data is not None:
    df = st.session_state.data.copy()
    
    all_cols = sorted(list(df.columns))
    preferred = ['ref_local_identifier', 'individual_local_identifier', 'local_identifier', 'individual_id']
    default_index = 0
    for p in preferred:
        if p in all_cols:
            default_index = all_cols.index(p)
            break
    id_col = st.sidebar.selectbox("Select Animal ID Column", all_cols, index=default_index)
    df['individual_id'] = df[id_col].astype(str)
    
    st.sidebar.header("3. Filters")
    all_ids = sorted(df['individual_id'].unique())
    now = datetime.now(pd.Timestamp.now(tz='Asia/Kolkata').tz)
    cutoff_15d = now - timedelta(days=15)
    
    active_ids = []
    if 'timestamp' in df.columns:
        active_ids = df[df['timestamp'] >= cutoff_15d]['individual_id'].unique().tolist()
    
    default_selection = active_ids if active_ids else all_ids[:5]
    selected_ids = st.sidebar.multiselect("Select Individuals", all_ids, default=default_selection)
    
    if 'timestamp' in df.columns:
        subset = df[df['individual_id'].isin(selected_ids)] if selected_ids else df
        if not subset.empty:
            min_date, max_date = subset['timestamp'].min().date(), subset['timestamp'].max().date()
        else:
            min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
        default_start = (now - timedelta(days=15)).date()
        slider_start = max(min_date, min(default_start, max_date))
        if min_date >= max_date:
            date_range = (min_date, max_date)
        else:
            date_range = st.sidebar.slider("Date Range", min_date, max_date, (slider_start, max_date))
        if selected_ids:
            df = df[df['individual_id'].isin(selected_ids)]
        mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
        filtered_df = df.loc[mask].copy()
    else:
        filtered_df = pd.DataFrame()

    if filtered_df.empty:
        st.warning("No data found.")
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Map 🗺️", "Analysis 📈", "Deployments 📡", "R-Home Range 🏠", "Kill Clusters 🍖", "Exploration 🐾"])
        
        with tab1:
            if 'lat' in filtered_df.columns:
                layers = [get_tile_layer(map_style)]
                for ind in filtered_df['individual_id'].unique():
                    d = filtered_df[filtered_df['individual_id'] == ind]
                    if len(d) < 2: continue
                    path = d[['lon','lat']].values.tolist()
                    color = get_random_color(ind)
                    layers.append(pdk.Layer("PathLayer", data=[{"path": path, "name": ind}], pickable=True, get_color=color, width_scale=20, width_min_pixels=2, get_path="path"))
                view_state = pdk.data_utils.compute_view(filtered_df[['lon', 'lat']], view_type=pdk.ViewState)
                st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers))

        with tab2:
            st.caption("Timezone: Asia/Kolkata")
            plot_data = filtered_df.copy().sort_values(['individual_id', 'timestamp'])
            grouped = plot_data.groupby('individual_id')
            plot_data['step_dist'] = haversine_np(plot_data['lon'], plot_data['lat'], plot_data['lon'].shift(1), plot_data['lat'].shift(1)).fillna(0)
            plot_data.loc[plot_data['individual_id'] != plot_data['individual_id'].shift(1), 'step_dist'] = 0
            plot_data['cum_dist'] = plot_data.groupby('individual_id')['step_dist'].cumsum()
            c4 = alt.Chart(plot_data).mark_line().encode(x='timestamp:T', y='cum_dist:Q', color='individual_id:N')
            st.altair_chart(c4, use_container_width=True)

        with tab3:
            full = st.session_state.data.copy()
            full['individual_id'] = full[id_col].astype(str)
            stats = full.groupby('individual_id').agg(Start=('timestamp','min'), End=('timestamp','max'), Pings=('timestamp','count')).reset_index()
            st.dataframe(stats, use_container_width=True)

        with tab4:
            st.subheader("R-Powered Home Range")
            if not R_AVAILABLE: st.error("R/rpy2 not installed.")
            else:
                hr_id = st.selectbox("Select Animal", filtered_df['individual_id'].unique())
                if hr_id and st.button("Run AKDE"):
                    try:
                        animal_data = filtered_df[filtered_df['individual_id'] == hr_id].copy()
                        animal_data['timestamp'] = animal_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        r_df_input = pd.DataFrame({
                            'timestamp': animal_data['timestamp'],
                            'long': animal_data['lon'],
                            'lat': animal_data['lat'],
                            'individual.local.identifier': animal_data['individual_id']
                        })
                        with localconverter(ro.default_converter + pandas2ri.converter):
                            r_df_r = ro.conversion.py2rpy(r_df_input)
                            r_code = """
                            library(ctmm)
                            library(sf)
                            function(df) {
                                out_status <- "ok"
                                out_msg <- ""
                                out_stats <- NULL
                                out_coords <- NULL
                                tryCatch({
                                    tel <- as.telemetry(df, time.format="%Y-%m-%d %H:%M:%S")
                                    if (class(tel)[1] == "list") { tel <- tel[[1]] }
                                    GUESS <- ctmm.guess(tel, interactive=FALSE)
                                    if (class(GUESS)[1] == "list") { GUESS <- GUESS[[1]] }
                                    FIT <- ctmm.fit(tel, GUESS)
                                    if (class(FIT)[1] == "list") { FIT <- FIT[[1]] }
                                    UD <- akde(tel, FIT)
                                    if (class(UD)[1] == "list") { UD <- UD[[1]] }
                                    summ <- summary(UD, units=FALSE)
                                    out_stats <- as.numeric(summ$CI)
                                    sp_obj <- SpatialPolygonsDataFrame.UD(UD, level.UD=0.95)
                                    sf_obj <- st_as_sf(sp_obj)
                                    sf_obj <- st_transform(sf_obj, crs=4326)
                                    out_coords <- st_coordinates(sf_obj)
                                }, error = function(e) {
                                    out_status <<- "error"
                                    out_msg <<- as.character(e$message)
                                })
                                return(list(status=out_status, msg=out_msg, stats=out_stats, coords=out_coords))
                            }
                            """
                            r_func = ro.r(r_code)
                            r_result = r_func(r_df_r)
                            res_dict = r_list_to_dict(r_result)
                            if str(res_dict['status'][0]) == "error":
                                st.error(f"R Logic Error: {res_dict['msg'][0]}")
                            else:
                                stats_vec = np.array(res_dict['stats'])
                                est_km2 = float(stats_vec[1]) / 1_000_000
                                coords_np = np.array(res_dict['coords'])
                                df_coords = pd.DataFrame(coords_np, columns=['lon', 'lat', 'L1', 'L2'][:coords_np.shape[1]])
                                polygons = [group[['lon', 'lat']].values.tolist() for _, group in df_coords.groupby('L1')] if 'L1' in df_coords.columns else [df_coords[['lon', 'lat']].values.tolist()]
                                st.metric("AKDE Area (95%)", f"{est_km2:.2f} km²")
                                layer_poly = pdk.Layer("PolygonLayer", data=[{"polygon": p} for p in polygons], get_polygon="polygon", filled=True, get_fill_color=[0,0,255,50], get_line_color=[0,0,255,200], get_line_width=2, stroked=True)
                                st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=pdk.data_utils.compute_view(df_coords[['lon', 'lat']]), layers=[get_tile_layer(map_style), layer_poly]))
                    except Exception as e:
                        st.error(f"R Execution Failed: {e}")

        with tab5:
            st.subheader("Kill Cluster Identification 🍖")
            c1, c2, c3 = st.columns(3)
            spat_thresh = c1.number_input("Spatial Radius (m)", value=50, step=10)
            temp_thresh = c2.number_input("Temporal Radius (min)", value=120, step=30)
            min_pts = c3.number_input("Min Points", value=5, min_value=3)
            kc_id = st.selectbox("Select Animal for Clustering", filtered_df['individual_id'].unique())
            if kc_id and st.button("Find Clusters"):
                kc_data = filtered_df[filtered_df['individual_id'] == kc_id].copy().sort_values('timestamp')
                with st.spinner("Running ST-DBSCAN..."):
                    labels = run_st_dbscan(kc_data, spat_thresh, temp_thresh, min_pts)
                    kc_data['cluster'] = labels
                    clusters = kc_data[kc_data['cluster'] != -1]
                    if clusters.empty: st.warning("No clusters found.")
                    else:
                        stats = clusters.groupby('cluster').agg(Start=('timestamp', 'min'), End=('timestamp', 'max'), Points=('timestamp', 'count'), Lat=('lat', 'mean'), Lon=('lon', 'mean')).reset_index()
                        stats['Duration (hrs)'] = (stats['End'] - stats['Start']).dt.total_seconds() / 3600
                        clf = load_model()
                        if clf: stats['Label'] = clf.predict(stats[['Duration (hrs)', 'Points']])
                        else: stats['Label'] = "Unclassified"
                        layers = [get_tile_layer(map_style)]
                        layers.append(pdk.Layer("ScatterplotLayer", data=stats, get_position='[Lon, Lat]', get_radius=spat_thresh*2, get_fill_color=[255,0,0,150], pickable=True))
                        st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=pdk.data_utils.compute_view(stats[['Lon', 'Lat']]), layers=layers, tooltip={"text": "{Label}\n{Duration (hrs)} hrs"}))
                        edited_df = st.data_editor(stats[['cluster', 'Duration (hrs)', 'Points', 'Label']], column_config={"Label": st.column_config.SelectboxColumn(options=["Unclassified", "Kill", "Resting", "Natal", "Other"], required=True)}, use_container_width=True)
                        if st.button("Save & Retrain Model"):
                            valid = edited_df[edited_df['Label'] != 'Unclassified']
                            if not valid.empty:
                                success, count = train_and_save(valid[['Duration (hrs)', 'Points', 'Label']])
                                if success: st.success(f"Model Retrained! Total: {count}")
                                else: st.warning(f"Saved. Need >5 examples (Current: {count})")

        with tab6:
            st.subheader("Exploration Trajectories 🐾")
            c1, c2 = st.columns([2, 2])
            anim_speed = c1.slider("Anim Speed (sec)", 0.01, 1.0, 0.1)
            num_steps = c2.slider("Steps", 1, 500, 50)
            start_anim = st.button("Play Animation ▶️")
            
            full_data = st.session_state.data
            full_data['individual_id'] = full_data[id_col].astype(str)
            exp_data = full_data[full_data['individual_id'].isin(selected_ids)].copy().sort_values(['individual_id', 'timestamp'])
            
            if exp_data.empty:
                st.warning("No animals selected.")
            else:
                map_placeholder = st.empty()
                if start_anim: step_range = range(1, num_steps + 1, 5)
                else: step_range = [num_steps]
                
                for current_step in step_range:
                    layers = [get_tile_layer(map_style)] # Base Layer
                    all_coords = []
                    for ind in selected_ids:
                        ind_steps = exp_data[exp_data['individual_id'] == ind].head(current_step + 1).reset_index(drop=True)
                        if len(ind_steps) < 1: continue
                        
                        raw_points = ind_steps[['lon', 'lat']].values.tolist()
                        color = get_random_color(ind)
                        
                        # 2. Path (No smoothing)
                        layers.append(pdk.Layer("PathLayer", data=[{"path": raw_points, "name": ind}], get_color=color, width_scale=20, width_min_pixels=3, get_path="path", pickable=True))
                        
                        # 3. Release Marker (Red)
                        layers.append(pdk.Layer("ScatterplotLayer", data=ind_steps.iloc[[0]], get_position='[lon, lat]', get_fill_color=[255, 0, 0, 255], get_line_color=[255,255,255,255], get_line_width=300, get_radius=300, radius_min_pixels=6, stroked=True))
                        
                        all_coords.extend(raw_points)

                    if all_coords:
                        df_bounds = pd.DataFrame(all_coords, columns=['lon', 'lat'])
                        view_state = pdk.data_utils.compute_view(df_bounds[['lon', 'lat']], view_type=pdk.ViewState)
                        view_state.zoom = view_state.zoom - 0.5 
                        
                        deck = pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers, tooltip={"text": "{name}"})
                        map_placeholder.pydeck_chart(deck)
                        if start_anim: time.sleep(anim_speed)
