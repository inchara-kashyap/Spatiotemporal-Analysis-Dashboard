import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
#import geopandas as gpd
import pickle
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Austin 911 Emergency Response Analysis",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all required data files"""
    try:

        import zipfile
        import os
                
        if os.path.exists('APD_911_Final_Processed.csv.zip') and not os.path.exists('APD_911_Final_Processed.csv'):
            zipfile.ZipFile('APD_911_Final_Processed.csv.zip', 'r').extractall('.')

        # Load main processed data
        df = pd.read_csv('APD_911_Final_Processed.csv')
        
        # Load DBSCAN anomalies
        dbscan_anomalies = pd.read_csv('DBSCAN_Anomalies.csv')
        
        # Load anomaly summary
        f = open('anomaly_summary.pkl', 'rb')
        anomaly_summary = pickle.load(f)
        f.close()
            
        return df, dbscan_anomalies, anomaly_summary
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_austin_districts():
    """Load Austin council district boundaries - simplified for deployment"""
    return None

def create_district_summary(df):
    """Create district-level summary statistics"""
    district_summary = df.groupby('council_district').agg({
        'response_time_sec': ['mean', 'median', 'count'],
        'response_time_min': 'mean',
        'delayed': ['sum', 'mean'],
        'is_hotspot': 'first'
    }).round(2)
    
    district_summary.columns = [
        'avg_response_sec', 'median_response_sec', 'total_calls',
        'avg_response_min', 'total_delayed', 'delay_rate', 'is_hotspot'
    ]
    district_summary = district_summary.reset_index()
    district_summary['delay_percentage'] = district_summary['delay_rate'] * 100
    
    return district_summary

def create_choropleth_map(district_summary, austin_gdf):
    """Create bar chart instead of choropleth for deployment"""
    fig = px.bar(
        district_summary.sort_values('avg_response_min'),
        x='council_district',
        y='avg_response_min',
        color='is_hotspot',
        color_discrete_map={True: '#d62728', False: '#1f77b4'},
        title='Average Response Time by Council District (Hotspots in Red)',
        labels={
            'council_district': 'Council District',
            'avg_response_min': 'Average Response Time (minutes)',
            'is_hotspot': 'Hotspot Status'
        }
    )
    
    fig.update_layout(height=600)
    return fig

def create_district_bar_chart(district_summary):
    """Create bar chart of response times by district"""
    fig = px.bar(
        district_summary.sort_values('avg_response_min'),
        x='council_district',
        y='avg_response_min',
        color='is_hotspot',
        color_discrete_map={True: '#d62728', False: '#1f77b4'},
        title='Average Response Time by Council District',
        labels={
            'council_district': 'Council District',
            'avg_response_min': 'Average Response Time (minutes)',
            'is_hotspot': 'Hotspot Status'
        }
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_title="Council District",
        yaxis_title="Average Response Time (minutes)"
    )
    
    return fig

def create_anomaly_heatmap(dbscan_anomalies):
    """Create heatmap of anomalous hours by district"""
    # Create pivot table for heatmap
    heatmap_data = dbscan_anomalies.pivot_table(
        index='district', 
        columns='hour', 
        values='delay_rate', 
        fill_value=0
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="District", color="Delay Rate"),
        title="Anomalous Response Patterns by District and Hour",
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Austin 911 Emergency Response Analysis</h1>', unsafe_allow_html=True)
    st.markdown("*Spatiotemporal analysis of emergency response times across Austin council districts*")
    st.markdown("---")
    
    # Load data
    df, dbscan_anomalies, anomaly_summary = load_data()
    austin_gdf = load_austin_districts()
    
    if df is None:
        st.error("Unable to load data files. Please ensure all required files are in the correct directory.")
        return
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    st.sidebar.info(f"**Data Summary**\n- Total calls analyzed: {len(df):,}\n- Time period: 2019-2024\n- Council districts: {df['council_district'].nunique()}")
    
    # Create district summary
    district_summary = create_district_summary(df)
    
    # Main dashboard tabs
    tab1, tab2 = st.tabs(["Percentile-Based Hotspots", "ML Anomaly Detection"])
    
    with tab1:
        st.header("Percentile-Based Hotspot Analysis")
        st.markdown("*Identifying consistently problematic districts using 90th percentile threshold*")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        total_calls = len(df)
        avg_response = district_summary['avg_response_min'].mean()
        hotspot_districts = district_summary[district_summary['is_hotspot']]['council_district'].tolist()
        overall_delay_rate = df['delayed'].mean() * 100

        col1.metric("Total Calls", f"{total_calls:,}")
        col2.metric("Avg Response Time", f"{avg_response:.1f} min")
        col3.metric("Hotspot Districts", f"{len(hotspot_districts)}")
        col4.metric("Overall Delay Rate", f"{overall_delay_rate:.1f}%")
        
        st.markdown("---")
        
        # Choropleth map
        st.subheader("Response Time Choropleth Map")
        choropleth_fig = create_choropleth_map(district_summary, austin_gdf)
        
        if choropleth_fig:
            st.plotly_chart(choropleth_fig, use_container_width=True)
        else:
            # Fallback bar chart if map fails
            st.warning("Choropleth map unavailable. Showing bar chart instead.")
            bar_fig = create_district_bar_chart(district_summary)
            st.plotly_chart(bar_fig, use_container_width=True)
        
        # District analysis
        col1, col2 = st.columns([2, 1])
        
        # Left column - District summary table
        col1.subheader("District Performance Summary")
        
        # Format and display district summary table
        display_df = district_summary[['council_district', 'avg_response_min', 'delay_percentage', 'total_calls', 'is_hotspot']].copy()
        display_df = display_df.sort_values('avg_response_min', ascending=False)
        
        col1.dataframe(
            display_df,
            use_container_width=True
        )
        
        # Right column - Key findings
        col2.subheader("Key Findings")
        
        # Identify key insights
        slowest_district = district_summary.loc[district_summary['avg_response_min'].idxmax()]
        fastest_district = district_summary.loc[district_summary['avg_response_min'].idxmin()]
        
        col2.markdown(f"""
        **Slowest District:** {slowest_district['council_district']} 
        - {slowest_district['avg_response_min']:.1f} min average
        - {slowest_district['delay_percentage']:.1f}% delay rate
        
        **Fastest District:** {fastest_district['council_district']}
        - {fastest_district['avg_response_min']:.1f} min average
        - {fastest_district['delay_percentage']:.1f}% delay rate
        
        **Hotspot Districts:** {', '.join(map(str, hotspot_districts))}
        
        **Impact:** Hotspot districts have {slowest_district['avg_response_min'] - fastest_district['avg_response_min']:.1f} min longer response times on average.
        """)
    
    with tab2:
        st.header("ML-Based Anomaly Detection")
        st.markdown("*DBSCAN clustering identifies unusual operational patterns*")
        
        if dbscan_anomalies is None or anomaly_summary is None:
            st.error("ML anomaly data not available.")
            return
        
        # ML metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Anomalous Patterns", f"{len(dbscan_anomalies)}")
        
        affected_districts = dbscan_anomalies['district'].nunique()
        col2.metric("Affected Districts", f"{affected_districts}")
        
        avg_anomaly_response = dbscan_anomalies['avg_response_min'].mean()
        col3.metric("Avg Anomaly Response", f"{avg_anomaly_response:.1f} min")
        
        avg_anomaly_delay = dbscan_anomalies['delay_rate'].mean() * 100
        col4.metric("Avg Anomaly Delay Rate", f"{avg_anomaly_delay:.1f}%")
        
        st.markdown("---")
        
        # District rankings
        col1, col2 = st.columns([1, 1])
        
        # Left column - Rankings table
        col1.subheader("Most Anomalous Districts")
        rankings_df = anomaly_summary['district_rankings'].copy()
        col1.dataframe(
            rankings_df,
            use_container_width=True
        )
        
        # Right column - Heatmap
        col2.subheader("Anomaly Patterns Heatmap")
        heatmap_fig = create_anomaly_heatmap(dbscan_anomalies)
        col2.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Detailed analysis for top districts
        st.subheader("🔍 Detailed District Analysis")
        
        top_3_districts = anomaly_summary['top_3_districts']
        
        for i, district_id in enumerate(top_3_districts):
            expanded = True if i == 0 else False
            expander = st.expander(f"District {district_id} - Detailed Analysis", expanded=expanded)
            
            details = anomaly_summary['detailed_analysis'][district_id]
            
            col1, col2, col3 = expander.columns(3)
            
            col1.metric("Anomalous Hours", details['anomalous_hours'])
            col2.metric("Avg Response Time", f"{details['avg_response']:.1f} min")
            col3.metric("Avg Delay Rate", f"{details['avg_delay_rate']:.1%}")
            
            expander.markdown("**Worst Performing Hours:**")
            worst_hours_df = pd.DataFrame(details['worst_hours'])
            if not worst_hours_df.empty:
                expander.dataframe(
                    worst_hours_df,
                    use_container_width=True
                )
        
        # Key insights for ML approach
        st.subheader("ML Insights vs Percentile Analysis")
        
        st.markdown(f"""
        **Percentile Approach Found:**
        - District {hotspot_districts[0] if hotspot_districts else 'None'} as primary hotspot (consistent slowness)
        
        **ML Approach Found:**
        - District {top_3_districts[0]} has most severe anomalous patterns ({anomaly_summary['detailed_analysis'][top_3_districts[0]]['avg_delay_rate']:.1%} delay rate during problems)
        - Operational issues occur during specific hours, not consistently
        - Different districts have different types of problems (chronic vs episodic)
        
        **Actionable Insights:**
        - Percentile hotspots need systemic resource increases
        - ML anomalies suggest operational improvements during specific time periods
        - Both approaches provide complementary views for emergency services planning
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"*Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    st.markdown("**Data Source:** Austin Police Department Calls for Service (2019-2024)")

if __name__ == "__main__":
    main()