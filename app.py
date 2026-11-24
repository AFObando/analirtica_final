from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"

REQUIRED_COLUMNS = {
    "Order ID",
    "Order Date",
    "Ship Date",
    "Ship Mode",
    "Customer ID",
    "Customer Name",
    "Segment",
    "Region",
    "State",
    "Category",
    "Sub-Category",
    "Sales",
}

REGION_NAMES = {
    "Central": "Centro",
    "East": "Este",
    "South": "Sur",
    "West": "Oeste",
}
CATEGORY_NAMES = {
    "Furniture": "Muebles",
    "Office Supplies": "Suministros de oficina",
    "Technology": "Tecnología",
}
SUBCATEGORY_NAMES = {
    "Accessories": "Accesorios",
    "Appliances": "Electrodomésticos",
    "Art": "Arte",
    "Binders": "Archivadores",
    "Bookcases": "Estanterías",
    "Chairs": "Sillas",
    "Copiers": "Fotocopiadoras",
    "Envelopes": "Sobres",
    "Fasteners": "Sujetadores",
    "Furnishings": "Mobiliario",
    "Labels": "Etiquetas",
    "Machines": "Máquinas",
    "Paper": "Papel",
    "Phones": "Teléfonos",
    "Storage": "Almacenamiento",
    "Supplies": "Suministros",
    "Tables": "Mesas",
}
MONTH_ABBREVIATIONS = (
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
)
SEGMENT_NAMES = {
    "Consumer": "Consumo",
    "Corporate": "Corporativo",
    "Home Office": "Oficina en casa",
}
SHIP_MODE_NAMES = {
    "First Class": "Primera clase",
    "Same Day": "Mismo día",
    "Second Class": "Segunda clase",
    "Standard Class": "Estándar",
}
REGION_COLORS = {
    "Oeste": "#526F8A",
    "Este": "#A84F32",
    "Centro": "#687A52",
    "Sur": "#9A7A49",
}
POSITIVE_COLOR = "#687A52"
NEGATIVE_COLOR = "#A84F32"
# El CSV no trae costos: porcentajes de costo de mercancía sobre la venta,
# calibrados con los márgenes típicos del conjunto Superstore.
DEFAULT_COST_RATES = {
    "Furniture": 87,
    "Office Supplies": 73,
    "Technology": 73,
}
DEFAULT_OPERATING_RATE = 10
LOYALTY_SEGMENTS = (
    ("Campeones", "Compran seguido, hace poco y con alto valor."),
    ("Leales", "Compran con frecuencia y recientemente."),
    ("Potenciales", "Compraron hace poco, pero aún con baja frecuencia."),
    ("En riesgo", "Solían comprar seguido, pero hace tiempo que no lo hacen."),
    ("Inactivos", "Pocas compras y ninguna reciente."),
)
SEGMENT_COLORS = {
    "Campeones": "#3F5A35",
    "Leales": "#687A52",
    "Potenciales": "#9A7A49",
    "En riesgo": "#A84F32",
    "Inactivos": "#92958E",
}


st.set_page_config(
    page_title="Desempeño de ventas | Superstore",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e2d9;
        padding: 0.85rem 1rem;
        border-radius: 0.65rem;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.76rem;
        line-height: 1.2;
        white-space: normal !important;
    }
    [data-testid="stMetricValue"] div {
        font-size: 1.35rem;
        line-height: 1.2;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }
    div[data-testid="stSidebar"] {background: #f3f1eb;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Cargando y validando los datos...")
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the project CSV and validate fields used by the dashboard."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path}")

    data = pd.read_csv(path, encoding="utf-8-sig")
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas en train.csv: {missing}")

    for column in ("Order Date", "Ship Date"):
        data[column] = pd.to_datetime(
            data[column],
            format="%d/%m/%Y",
            errors="coerce",
        )
        if data[column].isna().any():
            invalid_count = int(data[column].isna().sum())
            raise ValueError(
                f"Hay {invalid_count} fechas inválidas en la columna «{column}»."
            )

    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    if data["Sales"].isna().any() or not data["Sales"].ge(0).all():
        raise ValueError("La columna Sales contiene valores inválidos o negativos.")

    if data["Order ID"].isna().any():
        raise ValueError("Hay pedidos sin identificador en la columna Order ID.")
    for column in ("Customer ID", "Customer Name", "State"):
        if data[column].isna().any():
            raise ValueError(f"Hay registros sin dato en la columna «{column}».")
    if data["Ship Date"].lt(data["Order Date"]).any():
        raise ValueError("Hay fechas de envío anteriores a las fechas de pedido.")

    data["Month"] = data["Order Date"].dt.to_period("M").dt.to_timestamp()
    data["Year"] = data["Order Date"].dt.year
    return data


def format_number(value: float, decimals: int = 0) -> str:
    """Format numbers with Spanish separators and without assuming a currency."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def comparison_periods(
    dimension_data: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    available_min: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, str] | None:
    """Return the selected date range and its equivalent from the prior year."""
    previous_start = start - pd.DateOffset(years=1)
    previous_end = end - pd.DateOffset(years=1)
    if previous_start < available_min:
        return None

    current = dimension_data.loc[dimension_data["Order Date"].between(start, end)]
    previous = dimension_data.loc[
        dimension_data["Order Date"].between(previous_start, previous_end)
    ]
    if current.empty or previous.empty:
        return None

    current_label = f"{start:%d/%m/%Y}–{end:%d/%m/%Y}"
    previous_label = (
        f"{previous_start:%d/%m/%Y}–{previous_end:%d/%m/%Y}"
    )
    return current, previous, f"{current_label} frente a {previous_label}"


def change_percent(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current / previous - 1) * 100


def summarize_regions(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    summary = (
        frame.groupby("Region")
        .agg(
            Ventas=("Sales", "sum"),
            Pedidos=("Order ID", "nunique"),
        )
        .reset_index()
    )
    summary[f"Venta media {suffix}"] = summary["Ventas"] / summary["Pedidos"]
    return summary.rename(
        columns={
            "Ventas": f"Ventas {suffix}",
            "Pedidos": f"Pedidos {suffix}",
        }
    )


def make_empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": "#70776F"},
    )
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


try:
    df = load_data(str(DATA_PATH))
except (FileNotFoundError, ValueError, UnicodeDecodeError) as error:
    st.error(f"No fue posible cargar el conjunto de datos. {error}")
    st.stop()

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

st.sidebar.title("Filtros")
st.sidebar.caption(
    "El rango de fechas y la región se aplican a todos los indicadores y gráficos."
)
selected_dates = st.sidebar.date_input(
    "Período (desde / hasta)",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY",
    help=(
        "Los indicadores usan este rango. Las variaciones comparan esas mismas "
        "fechas con el año anterior, si están disponibles."
    ),
)

if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.info("Seleccione una fecha de inicio y otra de fin para consultar el tablero.")
    st.stop()

start_date, end_date = selected_dates
if start_date > end_date:
    st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
    st.stop()

region_options = sorted(df["Region"].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Región",
    options=region_options,
    default=region_options,
    format_func=lambda value: REGION_NAMES.get(value, value),
)

with st.sidebar.expander("Filtros adicionales", expanded=False):
    category_options = sorted(df["Category"].dropna().unique())
    selected_categories = st.multiselect(
        "Categoría",
        options=category_options,
        default=category_options,
        format_func=lambda value: CATEGORY_NAMES.get(value, value),
    )
    segment_options = sorted(df["Segment"].dropna().unique())
    selected_segments = st.multiselect(
        "Segmento de cliente",
        options=segment_options,
        default=segment_options,
        format_func=lambda value: SEGMENT_NAMES.get(value, value),
    )
    ship_options = sorted(df["Ship Mode"].dropna().unique())
    selected_ship_modes = st.multiselect(
        "Modalidad de envío",
        options=ship_options,
        default=ship_options,
        format_func=lambda value: SHIP_MODE_NAMES.get(value, value),
    )

with st.sidebar.expander("Supuestos de rentabilidad", expanded=False):
    st.caption(
        "El archivo no contiene costos. Ajuste el costo de mercancía por categoría "
        "y los gastos operativos, ambos como porcentaje de la venta."
    )
    cost_rates = {
        category: st.slider(
            f"Costo de mercancía · {CATEGORY_NAMES.get(category, category)}",
            min_value=0,
            max_value=100,
            value=DEFAULT_COST_RATES.get(category, 75),
            format="%d%%",
        )
        / 100
        for category in sorted(df["Category"].dropna().unique())
    }
    operating_rate = (
        st.slider(
            "Gastos operativos",
            min_value=0,
            max_value=50,
            value=DEFAULT_OPERATING_RATE,
            format="%d%%",
        )
        / 100
    )

if not selected_regions or not selected_categories or not selected_segments or not selected_ship_modes:
    st.warning("Mantenga al menos una opción seleccionada en cada filtro.")
    st.stop()

start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
dimension_data = df.loc[
    df["Region"].isin(selected_regions)
    & df["Category"].isin(selected_categories)
    & df["Segment"].isin(selected_segments)
    & df["Ship Mode"].isin(selected_ship_modes)
].copy()
dimension_data["Cost"] = dimension_data["Sales"] * dimension_data["Category"].map(
    cost_rates
).fillna(0.75)
filtered = dimension_data.loc[dimension_data["Order Date"].between(start, end)].copy()

if filtered.empty:
    st.warning("No hay registros para la combinación de filtros seleccionada.")
    st.stop()

comparison = comparison_periods(
    dimension_data,
    start,
    end,
    pd.Timestamp(df["Order Date"].min()),
)
comparison_current, comparison_previous, comparison_label = (
    comparison if comparison is not None else (None, None, None)
)

comparison_metrics = None
category_change = None
region_summary = None
if comparison is not None:
    current_sales = float(comparison_current["Sales"].sum())
    previous_sales = float(comparison_previous["Sales"].sum())
    current_orders = int(comparison_current["Order ID"].nunique())
    previous_orders = int(comparison_previous["Order ID"].nunique())
    current_average = current_sales / current_orders if current_orders else 0.0
    previous_average = previous_sales / previous_orders if previous_orders else 0.0
    comparison_metrics = {
        "sales": change_percent(current_sales, previous_sales),
        "orders": change_percent(current_orders, previous_orders),
        "average": change_percent(current_average, previous_average),
    }

    current_category_sales = comparison_current.groupby("Category")["Sales"].sum()
    previous_category_sales = comparison_previous.groupby("Category")["Sales"].sum()
    category_change = (
        pd.concat(
            [
                current_category_sales.rename("Actual"),
                previous_category_sales.rename("Anterior"),
            ],
            axis=1,
        )
        .fillna(0)
        .assign(Cambio=lambda table: table["Actual"] - table["Anterior"])
        .reset_index()
    )
    category_change["Categoría"] = category_change["Category"].map(CATEGORY_NAMES)

    current_regions = summarize_regions(comparison_current, "actual")
    previous_regions = summarize_regions(comparison_previous, "anterior")
    region_summary = (
        current_regions.merge(previous_regions, on="Region", how="outer").fillna(0)
    )
    region_summary["Cambio de ventas (%)"] = region_summary.apply(
        lambda row: change_percent(row["Ventas actual"], row["Ventas anterior"]),
        axis=1,
    )
    region_summary["Cambio de pedidos (%)"] = region_summary.apply(
        lambda row: change_percent(row["Pedidos actual"], row["Pedidos anterior"]),
        axis=1,
    )
    region_summary["Cambio de venta media (%)"] = region_summary.apply(
        lambda row: change_percent(
            row["Venta media actual"], row["Venta media anterior"]
        ),
        axis=1,
    )
    region_summary["Región"] = region_summary["Region"].map(REGION_NAMES)

st.title("Desempeño de ventas y pedidos")
st.caption(
    f"Período consultado: {start_date.strftime('%d/%m/%Y')} – "
    f"{end_date.strftime('%d/%m/%Y')} · "
    f"{len(filtered):,} líneas de pedido".replace(",", ".")
)


total_sales = float(filtered["Sales"].sum())
order_count = int(filtered["Order ID"].nunique())
average_order = total_sales / order_count if order_count else 0.0
active_months = int(filtered["Month"].nunique())
average_monthly_sales = total_sales / active_months if active_months else 0.0

goods_cost = float(filtered["Cost"].sum())
operating_expenses = total_sales * operating_rate
total_cost = goods_cost + operating_expenses
gross_profit = total_sales - goods_cost
net_profit = gross_profit - operating_expenses
gross_margin = gross_profit / total_sales * 100 if total_sales else 0.0
net_margin = net_profit / total_sales * 100 if total_sales else 0.0

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Ganancia",
    format_number(net_profit, 2),
    help=(
        "Ventas menos costo de mercancía y gastos operativos estimados "
        "(ver «Supuestos de rentabilidad» en la barra lateral)."
    ),
)
metric_columns[1].metric(
    "Pedidos distintos",
    format_number(order_count),
    help="Cantidad de identificadores Order ID únicos.",
)
metric_columns[2].metric(
    "Venta media por pedido",
    format_number(average_order, 2),
    help="Ventas registradas divididas por pedidos distintos.",
)
metric_columns[3].metric(
    "Ventas promedio por mes",
    format_number(average_monthly_sales, 2),
    help=(
        "Ventas registradas divididas por los meses con al menos un pedido dentro "
        "del período seleccionado."
    ),
)

st.markdown("**Rendimientos**")
performance_columns = st.columns(3)
performance_columns[0].metric(
    "Ingreso bruto",
    format_number(total_sales, 2),
    delta=f"Ganancia bruta {format_number(gross_profit, 2)} ({format_number(gross_margin, 1)} %)",
    delta_color="off",
    help="Total de ventas del período. La ganancia bruta descuenta solo el costo de mercancía.",
)
performance_columns[1].metric(
    "Ingreso neto",
    format_number(net_profit, 2),
    delta=f"Margen neto {format_number(net_margin, 1)} %",
    delta_color="off",
    help="Ingreso bruto menos costo de mercancía y gastos operativos.",
)
performance_columns[2].metric(
    "Costo total",
    format_number(total_cost, 2),
    delta=f"{format_number(total_cost / total_sales * 100 if total_sales else 0, 1)} % de las ventas",
    delta_color="off",
    help=(
        f"Costo de mercancía ({format_number(goods_cost, 2)}) más gastos "
        f"operativos ({format_number(operating_expenses, 2)})."
    ),
)

if comparison is None:
    st.caption(
        "No hay datos disponibles para comparar este rango con las mismas fechas "
        "del año anterior. Las visualizaciones muestran el período seleccionado."
    )


if comparison_metrics is not None:
    st.divider()
    st.subheader("Síntesis para la reunión")
    st.caption(
        f"Variación de {comparison_label}. Estos cambios comparan solo esos dos "
        "períodos; los indicadores principales de arriba cubren todo el rango filtrado."
    )
    comparison_columns = st.columns(3)
    comparison_labels = (
        ("sales", "Variación de ventas"),
        ("orders", "Variación de pedidos"),
        ("average", "Variación de venta media por pedido"),
    )
    for column, (key, label) in zip(comparison_columns, comparison_labels):
        value = comparison_metrics[key]
        formatted_value = (
            "Sin base comparable"
            if value is None
            else f"{value:+.1f} %".replace(".", ",")
        )
        column.metric(
            label,
            formatted_value,
            delta_color="off",
            help=f"Cambio porcentual entre {comparison_label}.",
        )

    insight_columns = st.columns(2)
    positive_categories = category_change.loc[category_change["Cambio"].gt(0)]
    negative_categories = category_change.loc[category_change["Cambio"].lt(0)]
    with insight_columns[0]:
        if not positive_categories.empty:
            largest_increase = positive_categories.loc[
                positive_categories["Cambio"].idxmax()
            ]
            st.markdown(
                "**Mayor aumento por categoría**  \n"
                f"{largest_increase['Categoría']}: "
                f"+{format_number(float(largest_increase['Cambio']), 2)}"
            )
        else:
            st.caption("No se observan aumentos de ventas por categoría.")
    with insight_columns[1]:
        if not negative_categories.empty:
            largest_decrease = negative_categories.loc[
                negative_categories["Cambio"].idxmin()
            ]
            st.markdown(
                "**Mayor disminución por categoría**  \n"
                f"{largest_decrease['Categoría']}: "
                f"{format_number(float(largest_decrease['Cambio']), 2)}"
            )
        else:
            st.caption("No se observan disminuciones de ventas por categoría.")

    opposing_trends = region_summary.loc[
        region_summary["Cambio de ventas (%)"].lt(0)
        & region_summary["Cambio de pedidos (%)"].gt(0)
    ]
    if not opposing_trends.empty:
        signals = [
            f"{row['Región']}: ventas {format_number(abs(row['Cambio de ventas (%)']), 1)} % "
            f"menores y pedidos {format_number(row['Cambio de pedidos (%)'], 1)} % mayores"
            for _, row in opposing_trends.iterrows()
        ]
        

st.divider()
left_chart, right_chart = st.columns([1.7, 1])

with left_chart:
    st.subheader("Evolución mensual de ventas")
    monthly = (
        filtered.groupby(["Month", "Region"], as_index=False)["Sales"]
        .sum()
        .sort_values("Month")
    )
    monthly["Región"] = monthly["Region"].map(REGION_NAMES)
    line_figure = px.line(
        monthly,
        x="Month",
        y="Sales",
        color="Región",
        markers=True,
        color_discrete_map=REGION_COLORS,
        labels={
            "Month": "Mes",
            "Sales": "Ventas registradas (unidad original)",
            "Región": "Región",
        },
    )
    line_figure.update_layout(
        template="plotly_white",
        height=390,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend_title_text="",
        hovermode="x unified",
    )
    tick_values = pd.date_range(
        monthly["Month"].min(),
        monthly["Month"].max(),
        freq="3MS",
    )
    tick_text = [
        f"{MONTH_ABBREVIATIONS[month.month - 1]}<br>{month.year}"
        for month in tick_values
    ]
    line_figure.update_xaxes(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        range=[
            monthly["Month"].min(),
            monthly["Month"].max() + pd.offsets.MonthBegin(1),
        ],
        showgrid=False,
    )
    line_figure.update_yaxes(tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF")
    st.plotly_chart(line_figure, width="stretch", config={"displayModeBar": False})

with right_chart:
    st.subheader("Cambio de ventas por categoría")
    region_order = [REGION_NAMES[region] for region in sorted(REGION_NAMES)]
    if comparison is not None:
        current_cat_region = comparison_current.groupby(["Category", "Region"])["Sales"].sum()
        previous_cat_region = comparison_previous.groupby(["Category", "Region"])["Sales"].sum()
        stacked = (
            pd.concat(
                [
                    current_cat_region.rename("Actual"),
                    previous_cat_region.rename("Anterior"),
                ],
                axis=1,
            )
            .fillna(0)
            .assign(Valor=lambda table: table["Actual"] - table["Anterior"])
            .reset_index()
        )
        x_title = "Cambio en ventas (apilado por región)"
        caption = (
            f"{comparison_label} · cada color es el aporte de una región; "
            "el rombo marca el cambio neto de la categoría."
        )
    else:
        stacked = (
            filtered.groupby(["Category", "Region"], as_index=False)["Sales"]
            .sum()
            .rename(columns={"Sales": "Valor"})
        )
        x_title = "Ventas (apilado por región)"
        caption = "Sin comparación anterior disponible; se muestra el total del período."

    stacked["Categoría"] = stacked["Category"].map(CATEGORY_NAMES)
    stacked["Región"] = stacked["Region"].map(REGION_NAMES)
    category_totals = stacked.groupby("Categoría")["Valor"].sum().sort_values()
    category_figure = px.bar(
        stacked,
        x="Valor",
        y="Categoría",
        color="Región",
        orientation="h",
        color_discrete_map=REGION_COLORS,
        category_orders={"Categoría": list(category_totals.index), "Región": region_order},
        labels={"Valor": x_title, "Categoría": ""},
    )
    category_figure.update_traces(
        hovertemplate="%{y} · %{fullData.name}<br>%{x:,.2f}<extra></extra>"
    )
    if comparison is not None:
        category_figure.add_trace(
            go.Scatter(
                x=category_totals.values,
                y=category_totals.index,
                mode="markers",
                name="Cambio neto",
                marker={"symbol": "diamond", "size": 11, "color": "#2F3430"},
                hovertemplate="%{y}<br>Cambio neto: %{x:,.2f}<extra></extra>",
            )
        )
        category_figure.add_vline(x=0, line_color="#92958E", line_width=1)
    category_figure.update_layout(
        template="plotly_white",
        barmode="relative",
        height=390,
        margin={"l": 10, "r": 15, "t": 10, "b": 10},
        legend={"orientation": "h", "y": -0.2, "title_text": ""},
    )
    category_figure.update_xaxes(tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF")
    st.caption(caption)
    st.plotly_chart(
        category_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

st.divider()
region_column, subcategory_column = st.columns([1.15, 1])

with region_column:
    st.subheader("Comparación por región")
    if comparison is not None:
        display_regions = region_summary[
            [
                "Región",
                "Cambio de ventas (%)",
                "Cambio de pedidos (%)",
                "Cambio de venta media (%)",
            ]
        ].copy()
        for column in display_regions.columns[1:]:
            display_regions[column] = display_regions[column].map(
                lambda value: "—" if pd.isna(value) else f"{value:+.1f} %".replace(".", ",")
            )
        st.caption(f"Variación: {comparison_label}.")
        st.dataframe(display_regions, hide_index=True, width="stretch")
    else:
        st.caption("No hay período anterior disponible para calcular variaciones regionales.")
        sales_by_region = (
            filtered.groupby("Region", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
        )
        sales_by_region["Región"] = sales_by_region["Region"].map(REGION_NAMES)
        st.dataframe(
            sales_by_region[["Región", "Sales"]].rename(
                columns={"Sales": "Ventas registradas"}
            ),
            hide_index=True,
            width="stretch",
        )

with subcategory_column:
    st.subheader("Subcategorías: top 5 mejores y top 5 peores")
    if comparison is not None:
        current_subcategories = comparison_current.groupby("Sub-Category")["Sales"].sum()
        previous_subcategories = comparison_previous.groupby("Sub-Category")["Sales"].sum()
        subcategory_values = (
            pd.concat(
                [
                    current_subcategories.rename("Actual"),
                    previous_subcategories.rename("Anterior"),
                ],
                axis=1,
            )
            .fillna(0)
            .assign(Valor=lambda table: table["Actual"] - table["Anterior"])
            .reset_index()
        )
        value_title = "Cambio en ventas"
        st.caption(f"Ordenadas por cambio de ventas · {comparison_label}.")
    else:
        subcategory_values = (
            filtered.groupby("Sub-Category", as_index=False)["Sales"]
            .sum()
            .rename(columns={"Sales": "Valor"})
        )
        value_title = "Ventas"
        st.caption("Sin comparación anterior disponible; ordenadas por ventas del período.")

    subcategory_values["Subcategoría"] = subcategory_values["Sub-Category"].map(
        lambda value: SUBCATEGORY_NAMES.get(value, value)
    )
    best_subcategories = subcategory_values.nlargest(5, "Valor")
    worst_subcategories = subcategory_values.loc[
        ~subcategory_values["Sub-Category"].isin(best_subcategories["Sub-Category"])
    ].nsmallest(5, "Valor")
    subcategory_figure = go.Figure()
    for group, color, label in (
        (worst_subcategories, NEGATIVE_COLOR, "Top 5 peores"),
        (best_subcategories, POSITIVE_COLOR, "Top 5 mejores"),
    ):
        group = group.sort_values("Valor")
        subcategory_figure.add_trace(
            go.Bar(
                x=group["Valor"],
                y=group["Subcategoría"],
                orientation="h",
                name=label,
                marker_color=color,
                text=group["Valor"].map(lambda value: format_number(value, 0)),
                textposition="auto",
                hovertemplate=f"%{{y}}<br>{value_title}: %{{x:,.2f}}<extra>{label}</extra>",
            )
        )
    if comparison is not None:
        subcategory_figure.add_vline(x=0, line_color="#92958E", line_width=1)
    subcategory_figure.update_layout(
        template="plotly_white",
        height=380,
        margin={"l": 10, "r": 10, "t": 5, "b": 10},
        xaxis_title=value_title,
        yaxis_title="",
        legend={"orientation": "h", "y": -0.2, "traceorder": "reversed"},
    )
    subcategory_figure.update_xaxes(
        tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF"
    )
    st.plotly_chart(
        subcategory_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

st.divider()
st.subheader("Análisis de clientes, mercados y operación")
customer_tab, state_tab, dispatch_tab = st.tabs(
    ["Clientes", "Estados", "Despacho"]
)

with customer_tab:
    customer_summary = (
        filtered.groupby("Customer ID", as_index=False)
        .agg(
            Cliente=("Customer Name", "first"),
            Ventas=("Sales", "sum"),
            Pedidos=("Order ID", "nunique"),
        )
    )
    customer_count = len(customer_summary)
    repeat_customers = int(customer_summary["Pedidos"].gt(1).sum())
    repeat_share = repeat_customers / customer_count * 100
    average_frequency = float(customer_summary["Pedidos"].mean())
    top_customers = customer_summary.nlargest(10, "Ventas").sort_values("Ventas")

    # Retención anual: % de clientes de un año que vuelven a comprar el año siguiente.
    # Usa los años calendario completos que toca el período (con los demás filtros).
    customers_by_year = dimension_data.groupby("Year")["Customer ID"].agg(set)
    retention_rows = []
    for year in sorted(filtered["Year"].unique()):
        if year - 1 in customers_by_year.index:
            previous_customers = customers_by_year[year - 1]
            retained = len(previous_customers & customers_by_year[year])
            retention_rows.append(
                {
                    "Año": str(year),
                    "Retención": retained / len(previous_customers) * 100,
                    "Clientes retenidos": retained,
                    "Clientes año anterior": len(previous_customers),
                }
            )
    retention = pd.DataFrame(retention_rows)

    customer_metrics = st.columns(4)
    customer_metrics[0].metric(
        "Clientes en el período",
        format_number(customer_count),
    )
    customer_metrics[1].metric(
        "Tasa de recompra",
        f"{format_number(repeat_share, 1)} %",
        delta=f"{format_number(repeat_customers)} clientes con 2+ pedidos",
        delta_color="off",
        help="Porcentaje de clientes que compraron más de una vez en el período.",
    )
    customer_metrics[2].metric(
        "Retención anual",
        "—" if retention.empty else f"{format_number(retention['Retención'].iloc[-1], 1)} %",
        delta=None if retention.empty else f"Año {retention['Año'].iloc[-1]} vs. anterior",
        delta_color="off",
        help=(
            "Porcentaje de los clientes del año anterior que volvieron a comprar en el año. "
            "Se calcula con años calendario completos."
        ),
    )
    customer_metrics[3].metric(
        "Frecuencia de compra",
        f"{format_number(average_frequency, 1)} pedidos",
        help="Pedidos promedio por cliente en el período (cuántas veces nos compró).",
    )

    frequency_column, retention_column = st.columns(2)
    with frequency_column:
        st.markdown("**¿Cuántas veces nos compró cada cliente?**")
        frequency_bins = pd.cut(
            customer_summary["Pedidos"],
            bins=[0, 1, 2, 3, 5, 9, float("inf")],
            labels=["1", "2", "3", "4–5", "6–9", "10+"],
        )
        frequency = (
            frequency_bins.value_counts(sort=False)
            .rename_axis("Compras")
            .reset_index(name="Clientes")
        )
        frequency["Porcentaje"] = frequency["Clientes"] / customer_count * 100
        frequency_figure = go.Figure(
            go.Bar(
                x=frequency["Compras"].astype(str),
                y=frequency["Clientes"],
                marker_color=[NEGATIVE_COLOR] + [POSITIVE_COLOR] * (len(frequency) - 1),
                text=frequency["Porcentaje"].map(lambda value: f"{format_number(value, 1)} %"),
                textposition="outside",
                hovertemplate="%{x} compras<br>Clientes: %{y}<extra></extra>",
            )
        )
        frequency_figure.update_layout(
            template="plotly_white",
            height=320,
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
            xaxis_title="Número de pedidos en el período",
            yaxis_title="Clientes",
        )
        frequency_figure.update_yaxes(gridcolor="#E8E6DF")
        st.plotly_chart(frequency_figure, width="stretch", config={"displayModeBar": False})

    with retention_column:
        st.markdown("**Retención de clientes año contra año**")
        if retention.empty:
            st.caption("Se necesita al menos un año previo en los datos para medir retención.")
        else:
            retention_figure = go.Figure(
                go.Bar(
                    x=retention["Año"],
                    y=retention["Retención"],
                    marker_color=POSITIVE_COLOR,
                    text=retention["Retención"].map(lambda value: f"{format_number(value, 1)} %"),
                    textposition="outside",
                    customdata=retention[["Clientes retenidos", "Clientes año anterior"]],
                    hovertemplate=(
                        "%{x}<br>Retención: %{y:.1f} %"
                        "<br>%{customdata[0]} de %{customdata[1]} clientes<extra></extra>"
                    ),
                )
            )
            retention_figure.update_layout(
                template="plotly_white",
                height=320,
                margin={"l": 10, "r": 10, "t": 20, "b": 10},
                xaxis_title="Año",
                yaxis_title="% de clientes retenidos",
                yaxis_range=[0, 100],
            )
            retention_figure.update_yaxes(gridcolor="#E8E6DF")
            st.plotly_chart(retention_figure, width="stretch", config={"displayModeBar": False})

    st.markdown("**Fidelización de clientes (modelo RFM)**")
    st.caption(
        "Cada cliente recibe una puntuación de 1 a 4 en Recencia (días desde su última "
        "compra), Frecuencia (número de pedidos) y Monto (ventas), según su cuartil dentro "
        "del período. Con esas puntuaciones se asigna un segmento de fidelidad."
    )
    if customer_count >= 4:
        rfm = customer_summary.copy()
        last_purchase = filtered.groupby("Customer ID")["Order Date"].max()
        rfm["Recencia"] = (end.normalize() - rfm["Customer ID"].map(last_purchase)).dt.days
        quartile_labels = [1, 2, 3, 4]
        rfm["R"] = pd.qcut(
            rfm["Recencia"].rank(method="first", ascending=False), 4, labels=quartile_labels
        ).astype(int)
        rfm["F"] = pd.qcut(
            rfm["Pedidos"].rank(method="first"), 4, labels=quartile_labels
        ).astype(int)
        rfm["M"] = pd.qcut(
            rfm["Ventas"].rank(method="first"), 4, labels=quartile_labels
        ).astype(int)

        def loyalty_segment(row: pd.Series) -> str:
            if row["R"] >= 3 and row["F"] >= 3:
                return "Campeones" if row["M"] >= 3 and row["R"] == 4 else "Leales"
            if row["R"] >= 3:
                return "Potenciales"
            if row["F"] >= 3:
                return "En riesgo"
            return "Inactivos"

        rfm["Segmento"] = rfm.apply(loyalty_segment, axis=1)
        loyal_share = rfm["Segmento"].isin(["Campeones", "Leales"]).mean() * 100
        segment_table = (
            rfm.groupby("Segmento")
            .agg(
                Clientes=("Customer ID", "count"),
                Ventas=("Ventas", "sum"),
                Pedidos_promedio=("Pedidos", "mean"),
                Recencia_media=("Recencia", "mean"),
            )
            .reindex([name for name, _ in LOYALTY_SEGMENTS])
            .dropna(subset=["Clientes"])
            .reset_index()
        )
        segment_table["% clientes"] = segment_table["Clientes"] / customer_count * 100
        segment_table["% ventas"] = segment_table["Ventas"] / total_sales * 100

        loyalty_metric, loyalty_chart = st.columns([1, 2])
        with loyalty_metric:
            st.metric(
                "Índice de fidelización",
                f"{format_number(loyal_share, 1)} %",
                help="Porcentaje de clientes en los segmentos Campeones o Leales.",
            )
            loyal_sales_share = segment_table.loc[
                segment_table["Segmento"].isin(["Campeones", "Leales"]), "% ventas"
            ].sum()
            st.metric(
                "Ventas de clientes fieles",
                f"{format_number(loyal_sales_share, 1)} %",
                help="Participación en ventas de los segmentos Campeones y Leales.",
            )
        with loyalty_chart:
            loyalty_figure = go.Figure()
            for metric_name, opacity in (("% clientes", 0.55), ("% ventas", 1.0)):
                loyalty_figure.add_trace(
                    go.Bar(
                        x=segment_table["Segmento"],
                        y=segment_table[metric_name],
                        name=metric_name,
                        marker_color=[SEGMENT_COLORS[name] for name in segment_table["Segmento"]],
                        marker_opacity=opacity,
                        text=segment_table[metric_name].map(
                            lambda value: f"{format_number(value, 1)} %"
                        ),
                        textposition="outside",
                        hovertemplate="%{x}<br>" + metric_name + ": %{y:.1f} %<extra></extra>",
                    )
                )
            loyalty_figure.update_layout(
                template="plotly_white",
                barmode="group",
                height=320,
                margin={"l": 10, "r": 10, "t": 20, "b": 10},
                yaxis_title="Porcentaje",
                legend={"orientation": "h", "y": 1.12},
            )
            loyalty_figure.update_yaxes(gridcolor="#E8E6DF")
            st.plotly_chart(loyalty_figure, width="stretch", config={"displayModeBar": False})

        descriptions = dict(LOYALTY_SEGMENTS)
        display_segments = pd.DataFrame(
            {
                "Segmento": segment_table["Segmento"],
                "Descripción": segment_table["Segmento"].map(descriptions),
                "Clientes": segment_table["Clientes"].map(lambda v: format_number(v)),
                "% clientes": segment_table["% clientes"].map(lambda v: f"{format_number(v, 1)} %"),
                "% ventas": segment_table["% ventas"].map(lambda v: f"{format_number(v, 1)} %"),
                "Pedidos promedio": segment_table["Pedidos_promedio"].map(
                    lambda v: format_number(v, 1)
                ),
                "Días desde última compra": segment_table["Recencia_media"].map(
                    lambda v: format_number(v, 0)
                ),
            }
        )
        st.dataframe(display_segments, hide_index=True, width="stretch")

        st.markdown("**¿Quiénes son los clientes más fieles?**")
        rfm["Puntaje RFM"] = rfm["R"] + rfm["F"] + rfm["M"]
        rfm["Última compra"] = rfm["Customer ID"].map(last_purchase)
        available_segments = [
            name for name, _ in LOYALTY_SEGMENTS if name in set(rfm["Segmento"])
        ]
        selector_column, size_column = st.columns([3, 1])
        chosen_segments = selector_column.multiselect(
            "Segmentos a listar",
            options=available_segments,
            default=[s for s in ("Campeones", "Leales") if s in available_segments],
        )
        list_size = size_column.number_input(
            "Clientes a mostrar", min_value=5, max_value=200, value=20, step=5
        )
        ranking = (
            rfm.loc[rfm["Segmento"].isin(chosen_segments)]
            .sort_values(
                ["Puntaje RFM", "Pedidos", "Ventas", "Recencia"],
                ascending=[False, False, False, True],
            )
            .head(int(list_size))
        )
        if ranking.empty:
            st.caption("Seleccione al menos un segmento para ver sus clientes.")
        else:
            ranking_display = pd.DataFrame(
                {
                    "#": range(1, len(ranking) + 1),
                    "Cliente": ranking["Cliente"],
                    "ID": ranking["Customer ID"],
                    "Segmento": ranking["Segmento"],
                    "Puntaje RFM (3–12)": ranking["Puntaje RFM"],
                    "Pedidos": ranking["Pedidos"],
                    "Ventas": ranking["Ventas"].map(lambda v: format_number(v, 2)),
                    "Última compra": ranking["Última compra"].dt.strftime("%d/%m/%Y"),
                    "Días sin comprar": ranking["Recencia"],
                }
            )
            st.dataframe(ranking_display, hide_index=True, width="stretch")
            st.caption(
                "Orden: mayor puntaje RFM (suma de Recencia + Frecuencia + Monto, 1 a 4 "
                "cada uno); en empate, más pedidos, más ventas y compra más reciente."
            )
    else:
        st.caption("Se necesitan al menos 4 clientes en el período para segmentar.")

    st.markdown("**Top 10 clientes por ventas**")

    customer_figure = go.Figure(
        go.Bar(
            x=top_customers["Ventas"],
            y=top_customers["Cliente"],
            orientation="h",
            marker_color=POSITIVE_COLOR,
            customdata=top_customers[["Pedidos"]],
            hovertemplate=(
                "%{y}<br>Ventas: %{x:,.2f}"
                "<br>Pedidos: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    customer_figure.update_layout(
        template="plotly_white",
        height=390,
        margin={"l": 10, "r": 15, "t": 10, "b": 10},
        xaxis_title="Ventas registradas (unidad original)",
        yaxis_title="",
    )
    customer_figure.update_xaxes(
        tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF"
    )
    st.plotly_chart(
        customer_figure,
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "Recompra, frecuencia y RFM se calculan dentro del período y filtros "
        "seleccionados; la retención anual usa años calendario completos."
    )

with state_tab:
    st.caption(
        "El desglose por estado permite ubicar dónde se concentran las ventas; "
        "no mide tamaño de mercado ni potencial de crecimiento."
    )
    if comparison is not None:
        current_state_sales = comparison_current.groupby("State")["Sales"].sum()
        previous_state_sales = comparison_previous.groupby("State")["Sales"].sum()
        state_change = (
            pd.concat(
                [
                    current_state_sales.rename("Actual"),
                    previous_state_sales.rename("Anterior"),
                ],
                axis=1,
            )
            .fillna(0)
            .assign(Cambio=lambda table: table["Actual"] - table["Anterior"])
            .reset_index()
        )
        increases = state_change.loc[state_change["Cambio"].gt(0)].nlargest(
            5, "Cambio"
        )
        decreases = state_change.loc[state_change["Cambio"].lt(0)].nsmallest(
            5, "Cambio"
        )
        state_change = (
            pd.concat([decreases, increases])
            .drop_duplicates("State")
            .sort_values("Cambio")
        )
        state_figure = go.Figure(
            go.Bar(
                x=state_change["Cambio"],
                y=state_change["State"],
                orientation="h",
                marker_color=[
                    NEGATIVE_COLOR if value < 0 else POSITIVE_COLOR
                    for value in state_change["Cambio"]
                ],
                customdata=state_change[["Actual", "Anterior"]],
                hovertemplate=(
                    "%{y}<br>Cambio: %{x:,.2f}"
                    "<br>Período actual: %{customdata[0]:,.2f}"
                    "<br>Período anterior: %{customdata[1]:,.2f}<extra></extra>"
                ),
            )
        )
        state_figure.add_vline(x=0, line_color="#92958E", line_width=1)
        state_figure.update_layout(
            template="plotly_white",
            height=440,
            margin={"l": 10, "r": 15, "t": 10, "b": 10},
            xaxis_title="Cambio en ventas registradas",
            yaxis_title="",
        )
        state_figure.update_xaxes(
            tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF"
        )
        st.caption(f"Mayores variaciones por estado · {comparison_label}.")
    else:
        state_change = (
            filtered.groupby("State", as_index=False)["Sales"]
            .sum()
            .nlargest(10, "Sales")
            .sort_values("Sales")
        )
        state_figure = px.bar(
            state_change,
            x="Sales",
            y="State",
            orientation="h",
            labels={
                "Sales": "Ventas registradas (unidad original)",
                "State": "Estado",
            },
            color_discrete_sequence=[POSITIVE_COLOR],
        )
        state_figure.update_layout(
            template="plotly_white",
            height=440,
            margin={"l": 10, "r": 15, "t": 10, "b": 10},
        )
        state_figure.update_xaxes(
            tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF"
        )
    st.plotly_chart(
        state_figure,
        width="stretch",
        config={"displayModeBar": False},
    )

with dispatch_tab:
    order_dispatch = (
        filtered.groupby("Order ID", as_index=False)
        .agg(
            Fecha=("Order Date", "min"),
            Envío=("Ship Date", "max"),
            Modalidad=("Ship Mode", "first"),
            Región=("Region", "first"),
        )
    )
    order_dispatch["Días hasta el despacho"] = (
        order_dispatch["Envío"] - order_dispatch["Fecha"]
    ).dt.days
    median_dispatch_days = float(order_dispatch["Días hasta el despacho"].median())
    same_day_share = float(
        order_dispatch["Días hasta el despacho"].eq(0).mean() * 100
    )
    one_day_share = float(
        order_dispatch["Días hasta el despacho"].le(1).mean() * 100
    )

    dispatch_metrics = st.columns(3)
    dispatch_metrics[0].metric(
        "Mediana de días hasta despacho",
        format_number(median_dispatch_days, 1),
    )
    dispatch_metrics[1].metric(
        "Pedidos despachados el mismo día",
        f"{format_number(same_day_share, 1)} %",
    )
    dispatch_metrics[2].metric(
        "Pedidos despachados en 1 día o menos",
        f"{format_number(one_day_share, 1)} %",
    )

    mode_summary = (
        order_dispatch.groupby("Modalidad", as_index=False)
        .agg(
            Pedidos=("Order ID", "nunique"),
            Mediana=("Días hasta el despacho", "median"),
            Percentil_90=("Días hasta el despacho", lambda values: values.quantile(0.9)),
        )
        .sort_values("Mediana")
    )
    mode_summary["Modalidad"] = mode_summary["Modalidad"].map(
        lambda value: SHIP_MODE_NAMES.get(value, value)
    )
    mode_figure = go.Figure(
        go.Bar(
            x=mode_summary["Mediana"],
            y=mode_summary["Modalidad"],
            orientation="h",
            marker_color=POSITIVE_COLOR,
            customdata=mode_summary[["Pedidos", "Percentil_90"]],
            hovertemplate=(
                "%{y}<br>Mediana: %{x:.1f} días"
                "<br>Pedidos: %{customdata[0]}"
                "<br>Percentil 90: %{customdata[1]:.1f} días<extra></extra>"
            ),
        )
    )
    mode_figure.update_layout(
        template="plotly_white",
        height=330,
        margin={"l": 10, "r": 15, "t": 10, "b": 10},
        xaxis_title="Días calendario desde el pedido hasta el despacho",
        yaxis_title="",
    )
    mode_figure.update_xaxes(gridcolor="#E8E6DF")
    st.plotly_chart(mode_figure, width="stretch", config={"displayModeBar": False})

    region_dispatch = (
        order_dispatch.groupby("Región", as_index=False)
        .agg(
            Pedidos=("Order ID", "nunique"),
            Mediana=("Días hasta el despacho", "median"),
            Percentil_90=("Días hasta el despacho", lambda values: values.quantile(0.9)),
        )
        .sort_values("Mediana", ascending=False)
    )
    region_dispatch["Región"] = region_dispatch["Región"].map(REGION_NAMES)
    region_dispatch = region_dispatch.rename(
        columns={
            "Mediana": "Mediana de días",
            "Percentil_90": "Percentil 90 de días",
        }
    )
    region_dispatch["Pedidos"] = region_dispatch["Pedidos"].map(format_number)
    for column in ("Mediana de días", "Percentil 90 de días"):
        region_dispatch[column] = region_dispatch[column].map(
            lambda value: format_number(float(value), 1)
        )
    st.dataframe(region_dispatch, hide_index=True, width="stretch")
    st.caption(
        "Mide días calendario entre pedido y despacho, no tiempo de entrega ni "
        "cumplimiento de una fecha prometida: esos datos no están en el archivo."
    )

st.divider()
 

st.caption(
    "Fuente: train.csv (Superstore). El conjunto contiene ventas y fechas de pedido/envío, "
    "pero no costos ni utilidad: la ganancia, el costo y los rendimientos se estiman con "
    "los porcentajes definidos en «Supuestos de rentabilidad»."
)
