import os
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Carpeta donde se guardarán los archivos descargados
DOWNLOAD_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

START_YEAR = 2010
END_YEAR = 2025

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n🔄 Procesando año {year}...")
        page.goto("https://paramh2o.aguaquito.gob.ec/reportes/diario/", timeout=60000)
        page.wait_for_selector("form#form_consultas", timeout=15000)

        # Seleccionar "Sistema y Subcuenca"
        page.locator("input#id_filtro_1").click()

        # Seleccionar sistema Noroccidente y el año
        page.wait_for_selector("select#id_sistema", timeout=5000)
        page.select_option("select#id_sistema", value="5")
        page.select_option("select#id_year", value=str(year))

        # Esperar la tabla de resultados
        try:
            page.wait_for_selector("table#table_reporte", timeout=5000)
        except PlaywrightTimeout:
            print(f"⚠️ No hay datos para el año {year}.")
            continue

        # Obtener el número de filas reales
        rows = page.locator("table#table_reporte tbody tr:visible")
        # count = rows.count()
        # print(f"   → {count} botones de descarga encontrados.")
        count = 9  # Asumimos que siempre son 9 estaciones visibles para Noroccidente
        print(f"   → Forzando descarga de {count} estaciones visibles para el sistema Noroccidente.")

        for i in range(count):
            try:
                # Reaplicar filtros porque tras la primera descarga se reinician
                page.select_option("select#id_sistema", value="5")
                page.select_option("select#id_year", value=str(year))
                page.wait_for_selector("table#table_reporte", timeout=5000)
                time.sleep(1)  # darle tiempo al DOM para actualizar

                # Obtener nombre de estación
                nombre = page.locator("table#table_reporte tbody tr td:nth-child(1)").nth(i).inner_text().strip()
                nombre_archivo = nombre.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
                file_path = os.path.join(DOWNLOAD_DIR, f"{nombre_archivo}_{year}.xlsx")

                if os.path.exists(file_path):
                    print(f"     ⚠️ Ya existe: {file_path}, se omite.")
                    continue

                print(f"     ⏳ Descargando {nombre}...")

                with page.expect_download(timeout=1000) as download_info:
                    nombre = rows.nth(i).locator("td").nth(0).inner_text().strip()
                    page.locator('button[name="descargar"]:visible').nth(i).click()


                download = download_info.value
                download.save_as(file_path)
                print(f"     ✅ Guardado: {file_path}")

            except PlaywrightTimeout:
                print(f"     ❌ Timeout (2s) en: {year} - {nombre}")
            except Exception as e:
                print(f"     ❌ Error inesperado en {year} - {nombre}: {e}")

            time.sleep(1)

    print("\n🎉 Proceso completado.")
    browser.close()
