# Tablero de ventas — Superstore

Aplicación interactiva en Streamlit para explorar las ventas, pedidos, categorías y regiones del archivo `data/train.csv`.

## Ejecutar en Windows

Desde PowerShell:

```powershell
cd "$HOME\Downloads\trabajo 3"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Streamlit abrirá el tablero en el navegador. Para detenerlo, presiona `Ctrl+C` en la terminal.

## Qué incluye

- Rango de fechas ajustable y filtros globales de región.
- Filtros opcionales de categoría, segmento y modalidad de envío.
- Indicadores de ventas registradas, pedidos distintos, venta media por pedido y participación del cuarto trimestre.
- Tendencia mensual por región, variación por categoría, comparación regional y cambios por subcategoría.
- Tabla de detalle y descarga del archivo filtrado.

Las comparaciones usan los dos últimos años completos dentro del periodo seleccionado; para rangos menores, usan las mismas fechas del año anterior cuando están disponibles.

## Consideraciones sobre los datos

La aplicación conserva la unidad de la columna `Sales`; el CSV no especifica una moneda. El archivo no tiene costo, utilidad, descuento, unidades ni inventario, por lo que las ventas no se presentan como rentabilidad y el tablero no atribuye causas.

## Publicación en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub.
2. En Streamlit Community Cloud, selecciona el repositorio y la rama.
3. Indica `app.py` como archivo principal.
4. La plataforma instalará las dependencias de `requirements.txt`.

El CSV está incluido en `data/`, por lo que la aplicación no depende de una ruta local fuera del proyecto.
