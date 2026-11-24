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

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Ventas registradas",
    format_number(total_sales, 2),
    help="Suma de las ventas registradas en el período y los filtros seleccionados.",
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
    if comparison is not None:
        category_change = category_change.sort_values("Cambio")
        colors = [
            NEGATIVE_COLOR if value < 0 else POSITIVE_COLOR
            for value in category_change["Cambio"]
        ]
        category_figure = go.Figure(
            go.Bar(
                x=category_change["Cambio"],
                y=category_change["Categoría"],
                orientation="h",
                marker_color=colors,
                customdata=category_change[["Actual", "Anterior"]],
                hovertemplate=(
                    "%{y}<br>Cambio: %{x:,.2f}"
                    "<br>Período actual: %{customdata[0]:,.2f}"
                    "<br>Período anterior: %{customdata[1]:,.2f}<extra></extra>"
                ),
            )
        )
        category_figure.add_vline(x=0, line_color="#92958E", line_width=1)
        category_figure.update_layout(
            template="plotly_white",
            height=390,
            margin={"l": 10, "r": 15, "t": 10, "b": 10},
            xaxis_title="Cambio en ventas registradas",
            yaxis_title="",
        )
        category_figure.update_xaxes(tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF")
        st.caption(f"{comparison_label} · barras a la derecha indican aumento.")
        st.plotly_chart(
            category_figure,
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        category_sales = (
            filtered.groupby("Category", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales")
        )
        category_sales["Categoría"] = category_sales["Category"].map(CATEGORY_NAMES)
        category_figure = px.bar(
            category_sales,
            x="Sales",
            y="Categoría",
            orientation="h",
            labels={"Sales": "Ventas registradas (unidad original)", "Categoría": ""},
            color_discrete_sequence=[POSITIVE_COLOR],
        )
        category_figure.update_layout(
            template="plotly_white",
            height=390,
            margin={"l": 10, "r": 15, "t": 10, "b": 10},
        )
        category_figure.update_xaxes(tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF")
        st.caption("Sin comparación anterior disponible; se muestra el total del período.")
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
    st.subheader("Cambios por subcategoría")
    if comparison is not None:
        current_subcategories = comparison_current.groupby("Sub-Category")["Sales"].sum()
        previous_subcategories = comparison_previous.groupby("Sub-Category")["Sales"].sum()
        subcategory_change = (
            pd.concat(
                [
                    current_subcategories.rename("Actual"),
                    previous_subcategories.rename("Anterior"),
                ],
                axis=1,
            )
            .fillna(0)
            .assign(Cambio=lambda table: table["Actual"] - table["Anterior"])
            .reset_index()
        )
        subcategory_change["Subcategoría"] = subcategory_change["Sub-Category"].map(
            lambda value: SUBCATEGORY_NAMES.get(value, value)
        )
        subcategory_change = (
            subcategory_change.assign(
                Magnitud=subcategory_change["Cambio"].abs()
            )
            .nlargest(10, "Magnitud")
            .sort_values("Cambio")
        )
        colors = [
            NEGATIVE_COLOR if value < 0 else POSITIVE_COLOR
            for value in subcategory_change["Cambio"]
        ]
        subcategory_figure = go.Figure(
            go.Bar(
                x=subcategory_change["Cambio"],
                y=subcategory_change["Subcategoría"],
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>Cambio: %{x:,.2f}<extra></extra>",
            )
        )
        subcategory_figure.add_vline(x=0, line_color="#92958E", line_width=1)
        subcategory_figure.update_layout(
            template="plotly_white",
            height=330,
            margin={"l": 10, "r": 10, "t": 5, "b": 10},
            xaxis_title="Cambio en ventas registradas",
            yaxis_title="",
        )
        subcategory_figure.update_xaxes(
            tickformat=",.0f", separatethousands=True, gridcolor="#E8E6DF"
        )
        st.plotly_chart(
            subcategory_figure,
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        top_subcategories = (
            filtered.groupby("Sub-Category", as_index=False)["Sales"]
            .sum()
            .nlargest(10, "Sales")
            .sort_values("Sales")
        )
        top_subcategories["Subcategoría"] = top_subcategories["Sub-Category"].map(
            lambda value: SUBCATEGORY_NAMES.get(value, value)
        )
        figure = px.bar(
            top_subcategories,
            x="Sales",
            y="Subcategoría",
            orientation="h",
            labels={"Sales": "Ventas registradas", "Subcategoría": ""},
            color_discrete_sequence=[POSITIVE_COLOR],
        )
        figure.update_layout(template="plotly_white", height=330)
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})

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
    repeat_customers = int(customer_summary["Pedidos"].gt(1).sum())
    repeat_share = repeat_customers / len(customer_summary) * 100
    top_customers = customer_summary.nlargest(10, "Ventas").sort_values("Ventas")
    top_customer_share = (
        float(top_customers["Ventas"].sum()) / total_sales * 100
        if total_sales
        else 0.0
    )

    customer_metrics = st.columns(3)
    customer_metrics[0].metric(
        "Clientes en el período",
        format_number(len(customer_summary)),
    )
    customer_metrics[1].metric(
        "Clientes con varios pedidos",
        format_number(repeat_customers),
        delta=f"{format_number(repeat_share, 1)} % del total",
        delta_color="off",
        help="Cuenta clientes con más de un pedido dentro de los filtros activos.",
    )
    customer_metrics[2].metric(
        "Ventas de los 10 principales clientes",
        f"{format_number(top_customer_share, 1)} %",
        help="Participación de los 10 clientes con mayores ventas en el período.",
    )

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
        "La repetición de pedidos se calcula solo dentro del período y filtros "
        "seleccionados; no representa retención ni lealtad histórica."
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
with st.expander("Consultar y descargar los registros filtrados"):
    visible_columns = [
        "Order Date",
        "Order ID",
        "Region",
        "Category",
        "Sub-Category",
        "Sales",
        "Ship Mode",
        "Ship Date",
    ]
    detail = filtered[visible_columns].copy()
    detail = detail.rename(
        columns={
            "Order Date": "Fecha del pedido",
            "Order ID": "Identificador del pedido",
            "Region": "Región",
            "Category": "Categoría",
            "Sub-Category": "Subcategoría",
            "Sales": "Ventas registradas",
            "Ship Mode": "Modalidad de envío",
            "Ship Date": "Fecha de envío",
        }
    )
    for column in ("Región", "Categoría", "Subcategoría", "Modalidad de envío"):
        mappings = {
            "Región": REGION_NAMES,
            "Categoría": CATEGORY_NAMES,
            "Subcategoría": SUBCATEGORY_NAMES,
            "Modalidad de envío": SHIP_MODE_NAMES,
        }
        detail[column] = detail[column].map(
            lambda value: mappings[column].get(value, value)
        )
    st.dataframe(
        detail.sort_values("Fecha del pedido", ascending=False).head(500),
        hide_index=True,
        width="stretch",
        height=350,
    )
    st.caption(
        f"Vista previa de hasta 500 de {len(detail):,} líneas filtradas. "
        "La descarga incluye todas las filas que cumplen los filtros.".replace(",", ".")
    )
    st.download_button(
        "Descargar datos filtrados (CSV)",
        data=detail.to_csv(index=False).encode("utf-8-sig"),
        file_name="ventas_superstore_filtradas.csv",
        mime="text/csv",
    )

st.caption(
    "Fuente: train.csv (Superstore). El conjunto contiene ventas y fechas de pedido/envío, "
    "pero no costos, utilidad, descuentos, unidades ni existencias. Por ello, este tablero "
    "no calcula márgenes ni atribuye causas."
)
