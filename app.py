import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import pydeck as pdk
import altair as alt
import time
from io import StringIO
from datetime import datetime, timedelta
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.interpolate import splprep, splev
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import joblib

# Page Config
st.set_page_config(
    page_title="Animal Movement Explorer",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
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

def get_tile_layer(style):
    if style == "Satellite":
        url = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    elif style == "Light":
        url = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"
    else: 
        url = "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
    return pdk.Layer("TileLayer", data=url, id=f"basemap-layer-{style}", min_zoom=0, max_zoom=19, opacity=1.0)

def format_duration(td):
    """Formats a timedelta object into a human-readable string."""
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0: return f"{days}d {hours}h ago"
    if hours > 0: return f"{hours}h {minutes}m ago"
    if minutes > 0: return f"{minutes}m ago"
    return "Just now"

def r_list_to_dict(r_list):
    try:
        keys = list(r_list.names)
        return {k: r_list.rx2(k) for k in keys}
    except:
        return {}

# --- ALGORITHMS ---
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

def calculate_mcp(df):
    if len(df) < 3: return 0, []
    mean_lat = df['lat'].mean()
    mean_lon = df['lon'].mean()
    x = (df['lon'].values - mean_lon) * 111320 * np.cos(np.radians(mean_lat))
    y = (df['lat'].values - mean_lat) * 111320
    points_proj = np.column_stack([x, y])
    try:
        hull = ConvexHull(points_proj)
        area_km2 = hull.volume / 1e6
        vertices_idx = hull.vertices
        vertices_lon = df['lon'].values[vertices_idx]
        vertices_lat = df['lat'].values[vertices_idx]
        final_coords = []
        for i in range(len(vertices_idx)):
            final_coords.append([vertices_lon[i], vertices_lat[i]])
        final_coords.append(final_coords[0])
        return area_km2, final_coords
    except:
        return 0, []

def calculate_kde_95(df, grid_size=100, bw_adjust=1.0):
    mean_lat = df['lat'].mean()
    mean_lon = df['lon'].mean()
    x = (df['lon'] - mean_lon) * 111320 * np.cos(np.radians(mean_lat))
    y = (df['lat'] - mean_lat) * 111320
    values = np.vstack([x, y])
    try:
        kernel = gaussian_kde(values)
        kernel.set_bandwidth(bw_method=kernel.factor * bw_adjust)
        pad_x = (x.max() - x.min()) * 0.2
        pad_y = (y.max() - y.min()) * 0.2
        xmin, xmax = x.min() - pad_x, x.max() + pad_x
        ymin, ymax = y.min() - pad_y, y.max() + pad_y
        X, Y = np.mgrid[xmin:xmax:complex(0, grid_size), ymin:ymax:complex(0, grid_size)]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kernel(positions).T, X.shape)
        sorted_z = np.sort(Z.ravel())[::-1]
        cumulative = np.cumsum(sorted_z) / np.sum(sorted_z)
        idx = np.searchsorted(cumulative, 0.95)
        threshold = sorted_z[idx]
        pixel_area = ((xmax - xmin) / grid_size) * ((ymax - ymin) / grid_size)
        area_m2 = np.sum(Z > threshold) * pixel_area
        return area_m2 / 1e6
    except:
        return 0

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

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🐾 Controls")
    
    if st.button("🏠 Reset / Home", use_container_width=True):
        st.session_state.data = None
        st.rerun()

    # 1. Connection (Auto-Collapse)
    is_connected = st.session_state.get('data') is not None
    with st.expander("1. Connection", expanded=not is_connected):
        default_user = os.getenv("MOVEBANK_USER", "")
        default_pass = os.getenv("MOVEBANK_PASS", "")
        username = st.text_input("Username", value=default_user)
        password = st.text_input("Password", value=default_pass, type="password")
        study_id = st.text_input("Study ID", placeholder="e.g., 2911040")
        if st.button("📥 Fetch Data", type="primary", use_container_width=True):
            st.session_state.fetch_trigger = True

    # 2. Map Settings (Collapsible)
    with st.expander("2. Map Settings", expanded=False):
        map_style = st.selectbox("Basemap", ["Satellite", "Light", "Dark"], index=0)

# --- DATA FETCH LOGIC ---
if 'data' not in st.session_state:
    st.session_state.data = None

if st.session_state.get('fetch_trigger', False):
    st.session_state.fetch_trigger = False
    if not username or not password or not study_id:
        st.error("⚠️ Please provide a Study ID and credentials.")
    else:
        url = f"https://www.movebank.org/movebank/service/direct-read?entity_type=event&study_id={study_id}"
        try:
            with st.spinner("Connecting to Movebank..."):
                r = requests.get(url, auth=(username, password))
                if r.status_code != 200:
                    st.error(f"Event Fetch Failed: {r.status_code}")
                else:
                    df_ev = pd.read_csv(StringIO(r.text))
                    df_ev = normalize_cols(df_ev)
                    url_ref = f"https://www.movebank.org/movebank/service/direct-read?entity_type=individual&study_id={study_id}"
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
                            df_final['step_dist'] = haversine_np(
                                df_final['lon'], df_final['lat'], 
                                grouped['lon'].shift(1), grouped['lat'].shift(1)
                            ).fillna(0)
                            df_final['time_diff'] = (df_final['timestamp'] - grouped['timestamp'].shift(1)).dt.total_seconds().fillna(1)
                            df_final['speed'] = (df_final['step_dist'] * 1000) / df_final['time_diff']
                            df_final = df_final[(df_final['speed'] <= 5) | (df_final['speed'].isna())]
                        
                        st.session_state.data = df_final
                        st.toast(f"✅ Loaded {len(df_final)} rows!", icon="🎉")
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN APP ---
if st.session_state.data is None:
    st.title("Animal Movement Explorer 🐾")
    st.markdown("### Welcome! Please connect to Movebank in the sidebar.")
    st.info("Use the sidebar to enter your Movebank credentials and Study ID.")
else:
    df = st.session_state.data.copy()
    
    c_head1, c_head2 = st.columns([3, 1])
    c_head1.title("Animal Movement Explorer 🗺️")
    
    all_cols = sorted(list(df.columns))
    preferred = ['ref_local_identifier', 'individual_local_identifier', 'local_identifier', 'individual_id']
    default_index = 0
    for p in preferred:
        if p in all_cols:
            default_index = all_cols.index(p)
            break
            
    id_col = c_head2.selectbox("ID Column", all_cols, index=default_index)
    df['individual_id'] = df[id_col].astype(str)
    
    # --- Sidebar Filters (Collapsible) ---
    with st.sidebar.expander("3. Data Filters", expanded=True):
        all_ids = sorted(df['individual_id'].unique())
        now = datetime.now(pd.Timestamp.now(tz='Asia/Kolkata').tz)
        cutoff_15d = now - timedelta(days=15)
        
        active_ids = []
        if 'timestamp' in df.columns:
            active_ids = df[df['timestamp'] >= cutoff_15d]['individual_id'].unique().tolist()
        
        default_selection = active_ids if active_ids else all_ids[:5]
        selected_ids = st.multiselect("Select Individuals", all_ids, default=default_selection)
        
        if 'timestamp' in df.columns:
            subset = df[df['individual_id'].isin(selected_ids)] if selected_ids else df
            if not subset.empty:
                min_date, max_date = subset['timestamp'].min().date(), subset['timestamp'].max().date()
            else:
                min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
            
            slider_start = max(min_date, (now - timedelta(days=15)).date()) if min_date < (now - timedelta(days=15)).date() else min_date
            date_range = st.slider("Date Range", min_date, max_date, (slider_start, max_date))
            
            if selected_ids:
                df = df[df['individual_id'].isin(selected_ids)]
            mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
            filtered_df = df.loc[mask].copy()
        else:
            filtered_df = pd.DataFrame()

    if filtered_df.empty:
        st.warning("No data matches your filters.")
    else:
        tab_home, tab_map, tab_anal, tab_dep, tab_hr, tab_kill, tab_exp, tab_help = st.tabs([
            "📊 Dashboard", "🗺️ Map", "📈 Analysis", "📡 Deployment", "📍 Home Range", "🍖 Kill Clusters", "🐾 Exploration", "ℹ️ Help"
        ])
        
        # --- 0. DASHBOARD (Filtered View) ---
        with tab_home:
            st.markdown("### 📡 Monitoring Dashboard (Last 15 Days)")
            
            max_study_time = st.session_state.data['timestamp'].max()
            is_historical = (now - max_study_time).days > 30
            reference_time = max_study_time if is_historical else now
            ref_label = "Study End" if is_historical else "Now"
            
            if is_historical:
                st.caption(f"⚠️ Historical Data: Stats relative to {max_study_time.date()}")

            dashboard_data = []
            full_df = st.session_state.data.copy()
            full_df['individual_id'] = full_df[id_col].astype(str)
            
            for ind in all_ids:
                ind_data = full_df[full_df['individual_id'] == ind]
                if ind_data.empty: continue
                
                last_fix = ind_data['timestamp'].max()
                time_since = reference_time - last_fix
                
                # Filter: Hide if older than 15 days
                if time_since > timedelta(days=15):
                    continue
                
                # Status: Active if seen in last 24h
                is_active = time_since < timedelta(hours=24)
                status_icon = "🟢 Active" if is_active else "🟡 Inactive"
                
                # Dist 24h
                cutoff_24h = reference_time - timedelta(hours=24)
                recent_mov = ind_data[ind_data['timestamp'] >= cutoff_24h]
                dist_24h = recent_mov['step_dist'].sum()
                
                dashboard_data.append({
                    "Individual": ind,
                    "Status": status_icon,
                    "Last Seen": last_fix.strftime('%Y-%m-%d %H:%M'),
                    "Time Since": format_duration(time_since),
                    "Dist (24h)": f"{dist_24h:.2f} km"
                })
            
            if not dashboard_data:
                st.info("No animals seen in the last 15 days relative to reference time.")
            else:
                dash_df = pd.DataFrame(dashboard_data)
                st.dataframe(
                    dash_df.sort_values("Status", ascending=True),
                    use_container_width=True,
                    column_config={
                        "Status": st.column_config.TextColumn("Status"),
                        "Dist (24h)": st.column_config.TextColumn(f"Dist (Last 24h)")
                    },
                    hide_index=True
                )
                
                c1, c2 = st.columns(2)
                c1.metric("Active (24h)", len(dash_df[dash_df['Status'] == "🟢 Active"]))
                c2.metric("Tracked (15d)", len(dash_df))

        # --- 1. MAP ---
        with tab_map:
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

        # --- 2. ANALYSIS ---
        with tab_anal:
            plot_data = filtered_df.copy().sort_values(['individual_id', 'timestamp'])
            grouped = plot_data.groupby('individual_id')
            plot_data['step_dist'] = haversine_np(plot_data['lon'], plot_data['lat'], plot_data['lon'].shift(1), plot_data['lat'].shift(1)).fillna(0)
            plot_data.loc[plot_data['individual_id'] != plot_data['individual_id'].shift(1), 'step_dist'] = 0
            plot_data['cum_dist'] = plot_data.groupby('individual_id')['step_dist'].cumsum()
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Cumulative Distance")
                chart = alt.Chart(plot_data).mark_line().encode(x='timestamp:T', y='cum_dist:Q', color='individual_id:N')
                st.altair_chart(chart, use_container_width=True)
            with c2:
                st.caption("Daily Distance")
                daily = plot_data.set_index('timestamp').groupby(['individual_id', pd.Grouper(freq='D')])['step_dist'].sum().reset_index()
                chart2 = alt.Chart(daily).mark_bar().encode(x='timestamp:T', y='step_dist:Q', color='individual_id:N')
                st.altair_chart(chart2, use_container_width=True)

        # --- 3. DEPLOYMENT ---
        with tab_dep:
            full = st.session_state.data.copy()
            full['individual_id'] = full[id_col].astype(str)
            stats = full.groupby('individual_id').agg(Start=('timestamp','min'), End=('timestamp','max'), Pings=('timestamp','count')).reset_index()
            st.dataframe(stats, use_container_width=True, hide_index=True)

        # --- 4. HOME RANGE ---
        with tab_hr:
            c_hr1, c_hr2 = st.columns([1, 2])
            hr_id = c_hr1.selectbox("Select Animal", filtered_df['individual_id'].unique())
            bw_adjust = c_hr2.slider("KDE Bandwidth", 0.1, 3.0, 1.0, 0.1)
            
            if hr_id:
                hr_data = filtered_df[filtered_df['individual_id'] == hr_id]
                if len(hr_data) > 5:
                    mcp_area, mcp_coords = calculate_mcp(hr_data)
                    kde_area = calculate_kde_95(hr_data, bw_adjust=bw_adjust)
                    col_a, col_b = st.columns(2)
                    show_mcp = col_a.checkbox(f"MCP (100%): {mcp_area:.2f} km²", value=True)
                    show_kde = col_b.checkbox(f"KDE (95%): {kde_area:.2f} km²", value=True)
                    layers = [get_tile_layer(map_style)]
                    if show_kde:
                        layers.append(pdk.Layer("HeatmapLayer", data=hr_data, get_position='[lon, lat]', opacity=0.6, threshold=0.05, radius_pixels=40 * bw_adjust, intensity=1))
                    if show_mcp and mcp_coords:
                        layers.append(pdk.Layer("PolygonLayer", data=[{"polygon": mcp_coords}], get_polygon="polygon", filled=True, get_fill_color=[0, 0, 255, 30], get_line_color=[0, 0, 255, 200], get_line_width=2, stroked=True))
                    layers.append(pdk.Layer("ScatterplotLayer", data=hr_data, get_position='[lon, lat]', get_radius=20, get_color=[0,0,0,150]))
                    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=pdk.data_utils.compute_view(hr_data[['lon', 'lat']]), layers=layers))
                else:
                    st.warning("Not enough points.")

        # --- 5. KILL CLUSTERS ---
        with tab_kill:
            c1, c2, c3 = st.columns(3)
            spat_thresh = c1.number_input("Spatial Radius (m)", value=50, step=10)
            temp_thresh = c2.number_input("Temporal Radius (min)", value=120, step=30)
            min_pts = c3.number_input("Min Points", value=5, min_value=3)
            kc_id = st.selectbox("Select Animal for Clustering", filtered_df['individual_id'].unique())
            if kc_id and st.button("Find Clusters", type="primary", use_container_width=True):
                kc_data = filtered_df[filtered_df['individual_id'] == kc_id].copy().sort_values('timestamp')
                labels = run_st_dbscan(kc_data, spat_thresh, temp_thresh, min_pts)
                kc_data['cluster'] = labels
                clusters = kc_data[kc_data['cluster'] != -1]
                if clusters.empty:
                    st.info("No clusters found.")
                else:
                    stats = clusters.groupby('cluster').agg(Start=('timestamp', 'min'), End=('timestamp', 'max'), Points=('timestamp', 'count'), Lat=('lat', 'mean'), Lon=('lon', 'mean')).reset_index()
                    stats['Duration (hrs)'] = ((stats['End'] - stats['Start']).dt.total_seconds() / 3600).round(2)
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

        # --- 6. EXPLORATION ---
        with tab_exp:
            c1, c2 = st.columns([2, 1])
            anim_speed = c1.slider("Animation Speed", 0.01, 1.0, 0.1)
            num_steps = c2.number_input("Steps", min_value=1, value=50)
            start_anim = st.checkbox("▶️ Play Animation")
            
            if start_anim:
                st.markdown("""<style>.main {width: 100% !important; max-width: 100% !important; padding: 0rem !important;} header {display: none !important;} section[data-testid="stSidebar"] {display: none !important;} .stDeckGlJsonChart {height: 90vh !important;}</style>""", unsafe_allow_html=True)
            
            full_data = st.session_state.data
            full_data['individual_id'] = full_data[id_col].astype(str)
            exp_data = full_data[full_data['individual_id'].isin(selected_ids)].copy().sort_values(['individual_id', 'timestamp'])
            
            if not exp_data.empty:
                map_placeholder = st.empty()
                step_range = range(1, num_steps + 1, 5) if start_anim else [num_steps]
                for current_step in step_range:
                    layers = [get_tile_layer(map_style)]
                    all_coords = []
                    for ind in selected_ids:
                        ind_steps = exp_data[exp_data['individual_id'] == ind].head(current_step + 1).reset_index(drop=True)
                        if len(ind_steps) < 1: continue
                        raw_points = ind_steps[['lon', 'lat']].values.tolist()
                        color = get_random_color(ind)
                        layers.append(pdk.Layer("PathLayer", data=[{"path": raw_points, "name": ind}], get_color=color, width_scale=20, width_min_pixels=3, get_path="path", pickable=True))
                        layers.append(pdk.Layer("ScatterplotLayer", data=ind_steps.iloc[[0]], get_position='[lon, lat]', get_fill_color=[255, 0, 0, 255], get_line_color=[255,255,255,255], get_line_width=300, get_radius=300, radius_min_pixels=6, stroked=True))
                        all_coords.extend(raw_points)
                    if all_coords:
                        df_bounds = pd.DataFrame(all_coords, columns=['lon', 'lat'])
                        view_state = pdk.data_utils.compute_view(df_bounds[['lon', 'lat']], view_type=pdk.ViewState)
                        view_state.zoom = view_state.zoom - 0.5
                        deck = pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers, tooltip={"text": "{name}"})
                        map_placeholder.pydeck_chart(deck)
                        if start_anim: time.sleep(anim_speed)
        
        # --- 7. HELP ---
        with tab_help:
            st.markdown("### 📖 User Guide")
            st.markdown("""
            **Getting Started**
            1.  Enter your **Movebank Username** and **Password** in the sidebar.
            2.  Provide a valid **Study ID**.
            3.  Click **Fetch Data**.
            
            **Tabs Overview**
            * **📊 Dashboard:** Status of active animals (seen in last 24h) and recent movement stats.
            * **🗺️ Map:** Visualizes full trajectories.
            * **📈 Analysis:** Graphs for Daily Distance, Activity Patterns, and Displacement.
            * **📍 Home Range:** Calculates Minimum Convex Polygon (MCP) and Kernel Density Estimation (KDE).
            * **🍖 Kill Clusters:** Identifies potential kill/rest sites using Spatiotemporal DBSCAN.
            * **🐾 Exploration:** Animates the initial movement path from release.
            """)
