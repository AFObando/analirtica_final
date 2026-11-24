# Tablero de ventas — Superstore

Aplicación interactiva en Streamlit para explorar las ventas, pedidos, categorías y regiones del archivo `data/train.csv`.

## Ejecutar en Windows

Desde PowerShell:

```powershell
cd "$HOME\Downloads\trabajo 3\trabajo final"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Streamlit abrirá el tablero en el navegador. Para detenerlo, presiona `Ctrl+C` en la terminal.

## Qué incluye

- Rango de fechas ajustable y filtros globales de región.
- Filtros opcionales de categoría, segmento y modalidad de envío.
- Indicadores de ventas registradas, pedidos distintos, venta media por pedido y ventas promedio por mes.
- Tendencia mensual, cambios por categoría y subcategoría, y comparación de ventas, pedidos y venta media.
- Análisis de clientes recurrentes y concentración de ventas en los 10 principales clientes.
- Comparación por estado y análisis de días entre pedido y despacho por modalidad y región.
- Tabla de detalle y descarga del archivo filtrado.

Las comparaciones usan exactamente el rango seleccionado frente a las mismas fechas del año anterior, cuando esos datos están disponibles; las etiquetas muestran ambos rangos. Si el CSV no contiene el período anterior, las visualizaciones muestran el período filtrado sin comparación. La repetición de pedidos se calcula dentro del período filtrado. El indicador operativo mide días hasta el despacho, no tiempo de entrega ni cumplimiento de una fecha prometida.

## Consideraciones sobre los datos

La aplicación conserva la unidad de la columna `Sales`; el CSV no especifica una moneda. El archivo no tiene costo, utilidad, descuento, unidades ni inventario, por lo que las ventas no se presentan como rentabilidad y el tablero no atribuye causas.

## Publicación en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub.
2. En Streamlit Community Cloud, selecciona el repositorio y la rama.
3. Indica `app.py` como archivo principal.
4. La plataforma instalará las dependencias de `requirements.txt`.

El CSV está incluido en `data/`, por lo que la aplicación no depende de una ruta local fuera del proyecto.
