import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# Configuración de la página
st.set_page_config(page_title="SuperFollow - Dashboard", layout="wide", initial_sidebar_state="expanded")

DB_PATH = "superfollow.db"


@st.cache_data(ttl=30)
def load_data():
    """Carga todos los datos de tickets y ítems (cache de 30 segundos)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    query = """
    SELECT 
        r.id as receipt_id,
        r.date, 
        s.name as store, 
        s.cuit,
        p.name as product, 
        ri.quantity, 
        ri.unit_price, 
        ri.total_line_price,
        r.total_amount as ticket_total,
        r.currency
    FROM receipt_items ri
    JOIN receipts r ON ri.receipt_id = r.id
    JOIN stores s ON r.store_id = s.id
    JOIN products p ON ri.product_id = p.id
    ORDER BY r.date DESC, r.id DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Formato limpio YYYY-MM para agrupación mensual
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        df['week'] = df['date'].dt.to_period('W').astype(str)
    
    return df


def load_summary_stats(df):
    """Calcula estadísticas resumidas."""
    if df.empty:
        return {}
    
    tickets = df.drop_duplicates(subset=['receipt_id'])
    return {
        'total_spent': tickets['ticket_total'].sum(),
        'avg_ticket': tickets['ticket_total'].mean(),
        'total_items': len(df),
        'unique_products': df['product'].nunique(),
        'unique_stores': df['store'].nunique(),
        'total_tickets': len(tickets),
        'avg_item_price': df['unit_price'].mean(),
    }


st.title("📊 SuperFollow: Dashboard de Consumo")
st.markdown("---")

try:
    df = load_data()
    
    if df.empty:
        st.warning("⚠️ No hay datos en la base de datos. Procesa tu primer ticket con `python main.py Ticket.pdf`")
        st.stop()
    
    # --- SIDEBAR: Filtros ---
    st.sidebar.header("🔍 Filtros")
    
    stores = st.sidebar.multiselect(
        "Comercios",
        options=sorted(df['store'].unique()),
        default=sorted(df['store'].unique())
    )
    
    if df['date'].notna().any():
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        date_range = st.sidebar.date_input(
            "Rango de Fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date, end_date = None, None
    
    # Filtrar DataFrame
    mask = df['store'].isin(stores)
    if start_date and end_date:
        mask = mask & (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    df_filtered = df[mask].copy()
    
    if df_filtered.empty:
        st.warning("⚠️ No hay datos para los filtros seleccionados.")
        st.stop()
    
    stats = load_summary_stats(df_filtered)
    
    # --- KPIs PRINCIPALES ---
    st.subheader("📈 Métricas Principales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Gasto Total", f"${stats['total_spent']:,.2f}", 
                 delta=f"${stats['avg_ticket']:,.2f} promedio por ticket")
    
    with col2:
        st.metric("🛒 Tickets Procesados", stats['total_tickets'])
    
    with col3:
        st.metric("📦 Productos Únicos", stats['unique_products'])
    
    with col4:
        st.metric("🏪 Comercios", stats['unique_stores'])
    
    st.markdown("---")
    
    # --- TABS PARA ORGANIZAR ANÁLISIS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen General", 
        "🏪 Por Comercio", 
        "🛍️ Productos", 
        "📅 Tendencias", 
        "💾 Datos"
    ])
    
    with tab1:
        st.subheader("Resumen General")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Evolución del gasto diario
            st.markdown("#### 💸 Evolución del Gasto Diario")
            tickets_daily = df_filtered.drop_duplicates(subset=['receipt_id']).groupby('date')['ticket_total'].sum().reset_index()
            if not tickets_daily.empty:
                fig_daily = px.line(
                    tickets_daily, 
                    x='date', 
                    y='ticket_total',
                    markers=True,
                    labels={'ticket_total': 'Total ($)', 'date': 'Fecha'},
                    title="Gasto por día"
                )
                fig_daily.update_traces(line_color='#1f77b4', line_width=2)
                st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Distribución de gastos por comercio
            st.markdown("#### 🏪 Distribución por Comercio")
            store_totals = df_filtered.drop_duplicates(subset=['receipt_id']).groupby('store')['ticket_total'].sum().reset_index()
            store_totals = store_totals.sort_values('ticket_total', ascending=False)
            fig_stores = px.pie(
                store_totals,
                values='ticket_total',
                names='store',
                title="% del gasto total por comercio"
            )
            st.plotly_chart(fig_stores, use_container_width=True)
        
        # Gasto mensual acumulado
        st.markdown("#### 📅 Gasto Mensual Acumulado")
        tickets_monthly = df_filtered.drop_duplicates(subset=['receipt_id']).groupby('year_month')['ticket_total'].sum().reset_index()
        tickets_monthly = tickets_monthly.sort_values('year_month')
        
        # Si solo hay una fecha única, mostrar la fecha del ticket en lugar del mes
        unique_dates = df_filtered.drop_duplicates(subset=['receipt_id'])['date'].dt.date.unique()
        if len(unique_dates) == 1:
            # Un solo ticket: mostrar la fecha de carga
            single_date = unique_dates[0]
            fig_monthly = px.bar(
                tickets_monthly,
                x='year_month',
                y='ticket_total',
                labels={'ticket_total': 'Total ($)', 'year_month': f'Fecha de carga: {single_date.strftime("%d/%m/%Y")}'},
                title=f"Gasto del ticket - Fecha: {single_date.strftime('%d/%m/%Y')}"
            )
        else:
            # Múltiples fechas: mostrar por mes
            tickets_monthly['year_month_formatted'] = pd.to_datetime(tickets_monthly['year_month'] + '-01').dt.strftime('%b %Y')
            fig_monthly = px.bar(
                tickets_monthly,
                x='year_month_formatted',
                y='ticket_total',
                labels={'ticket_total': 'Total ($)', 'year_month_formatted': 'Mes'},
                title="Gasto acumulado por mes"
            )
        
        fig_monthly.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab2:
        st.subheader("Análisis por Comercio")
        
        selected_store = st.selectbox("Seleccionar Comercio", options=sorted(df_filtered['store'].unique()))
        df_store = df_filtered[df_filtered['store'] == selected_store].copy()
        
        if not df_store.empty:
            store_tickets = df_store.drop_duplicates(subset=['receipt_id'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tickets en este comercio", len(store_tickets))
            with col2:
                st.metric("Gasto total", f"${store_tickets['ticket_total'].sum():,.2f}")
            with col3:
                st.metric("Ticket promedio", f"${store_tickets['ticket_total'].mean():,.2f}")
            
            # Evolución del gasto en este comercio
            st.markdown("#### Evolución del Gasto")
            store_daily = store_tickets.groupby('date')['ticket_total'].sum().reset_index()
            fig_store_evol = px.line(
                store_daily,
                x='date',
                y='ticket_total',
                markers=True,
                labels={'ticket_total': 'Total ($)', 'date': 'Fecha'},
                title=f"Gasto diario en {selected_store}"
            )
            st.plotly_chart(fig_store_evol, use_container_width=True)
            
            # Productos más comprados en este comercio
            st.markdown("#### Productos Más Comprados")
            top_products = df_store.groupby('product')['quantity'].sum().reset_index()
            top_products = top_products.sort_values('quantity', ascending=False).head(10)
            fig_top = px.bar(
                top_products,
                x='quantity',
                y='product',
                orientation='h',
                labels={'quantity': 'Cantidad Total', 'product': 'Producto'},
                title=f"Top 10 productos en {selected_store}"
            )
            fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_top, use_container_width=True)
    
    with tab3:
        st.subheader("Análisis de Productos")
        
        selected_product = st.selectbox(
            "Seleccionar Producto para Análisis",
            options=sorted(df_filtered['product'].unique()),
            key="product_selector"
        )
        df_prod = df_filtered[df_filtered['product'] == selected_product].copy()
        
        if not df_prod.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Veces comprado", len(df_prod))
            with col2:
                st.metric("Cantidad total", f"{df_prod['quantity'].sum():.2f}")
            with col3:
                st.metric("Precio promedio", f"${df_prod['unit_price'].mean():,.2f}")
            
            # Historial de precios por comercio
            st.markdown("#### 📈 Historial de Precios")
            df_prod_sorted = df_prod.sort_values('date')
            fig_prod = px.line(
                df_prod_sorted,
                x='date',
                y='unit_price',
                color='store',
                markers=True,
                labels={'unit_price': 'Precio Unitario ($)', 'date': 'Fecha', 'store': 'Comercio'},
                title=f"Evolución de precio: {selected_product}"
            )
            st.plotly_chart(fig_prod, use_container_width=True)
            
            # Comparativa de precios actuales por comercio
            st.markdown("#### 💰 Comparativa de Precios Actuales")
            latest_prices = df_prod_sorted.groupby('store')['unit_price'].last().reset_index()
            latest_prices = latest_prices.sort_values('unit_price', ascending=False)
            fig_comparison = px.bar(
                latest_prices,
                x='store',
                y='unit_price',
                labels={'unit_price': 'Precio Unitario ($)', 'store': 'Comercio'},
                title=f"Último precio registrado por comercio: {selected_product}"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Top productos más comprados (global)
        st.markdown("#### 🏆 Top 15 Productos Más Comprados")
        top_all = df_filtered.groupby('product')['quantity'].sum().reset_index()
        top_all = top_all.sort_values('quantity', ascending=False).head(15)
        fig_top_all = px.bar(
            top_all,
            x='quantity',
            y='product',
            orientation='h',
            labels={'quantity': 'Cantidad Total', 'product': 'Producto'},
            title="Productos más comprados (todas las compras)"
        )
        fig_top_all.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_all, use_container_width=True)
    
    with tab4:
        st.subheader("Tendencias y Patrones")
        
        # Gasto semanal
        st.markdown("#### 📊 Gasto Semanal")
        tickets_weekly = df_filtered.drop_duplicates(subset=['receipt_id']).groupby('week')['ticket_total'].sum().reset_index()
        tickets_weekly = tickets_weekly.sort_values('week')
        
        # Formatear semanas de forma más legible
        if not tickets_weekly.empty:
            tickets_weekly['week_formatted'] = tickets_weekly['week'].apply(
                lambda x: pd.to_datetime(x.split('/')[0] if '/' in str(x) else x).strftime('%d/%m/%Y') if pd.notna(pd.to_datetime(x.split('/')[0] if '/' in str(x) else x, errors='coerce')) else str(x)
            )
            fig_weekly = px.bar(
                tickets_weekly,
                x='week_formatted',
                y='ticket_total',
                labels={'ticket_total': 'Total ($)', 'week_formatted': 'Semana'},
                title="Gasto acumulado por semana"
            )
            fig_weekly.update_xaxes(tickangle=45)
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Heatmap: días de la semana vs gasto
        st.markdown("#### 📅 Patrón de Compras por Día de la Semana")
        df_filtered['day_of_week'] = df_filtered['date'].dt.day_name()
        df_filtered['day_of_week_num'] = df_filtered['date'].dt.dayofweek
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_totals = df_filtered.drop_duplicates(subset=['receipt_id']).groupby('day_of_week')['ticket_total'].sum().reindex(day_order, fill_value=0).reset_index()
        fig_days = px.bar(
            day_totals,
            x='day_of_week',
            y='ticket_total',
            labels={'ticket_total': 'Total ($)', 'day_of_week': 'Día de la Semana'},
            title="Gasto por día de la semana"
        )
        st.plotly_chart(fig_days, use_container_width=True)
        
        # Análisis de frecuencia de compra
        st.markdown("#### 🔄 Frecuencia de Compra")
        tickets_dates = df_filtered.drop_duplicates(subset=['receipt_id'])[['date', 'store']].copy()
        tickets_dates = tickets_dates.sort_values('date')
        if len(tickets_dates) > 1:
            tickets_dates['days_between'] = tickets_dates.groupby('store')['date'].diff().dt.days
            avg_days = tickets_dates['days_between'].mean()
            st.metric("Días promedio entre compras", f"{avg_days:.1f} días")
    
    with tab5:
        st.subheader("Datos Detallados")
        
        # Resumen de tickets
        st.markdown("#### Resumen de Tickets")
        tickets_summary = df_filtered.drop_duplicates(subset=['receipt_id'])[
            ['receipt_id', 'date', 'store', 'ticket_total', 'currency']
        ].copy()
        tickets_summary = tickets_summary.sort_values('date', ascending=False)
        st.dataframe(tickets_summary, use_container_width=True)
        
        # Detalle completo
        with st.expander("Ver detalle completo de transacciones"):
            st.dataframe(
                df_filtered[
                    ['date', 'store', 'product', 'quantity', 'unit_price', 'total_line_price', 'ticket_total']
                ],
                use_container_width=True
            )
        
        # Exportar datos
        st.markdown("#### Exportar Datos")
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"superfollow_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

except sqlite3.OperationalError as e:
    if "no such table" in str(e).lower():
        st.error("⚠️ La base de datos no está inicializada. Ejecuta `python main.py Ticket.pdf` primero para crear las tablas.")
    else:
        st.error(f"Error de base de datos: {e}")
except Exception as e:
    st.error(f"Error al cargar datos: {e}")
    st.info("💡 Asegúrate de haber procesado al menos un ticket con `python main.py Ticket.pdf`")
