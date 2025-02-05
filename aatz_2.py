import streamlit as st
import os
import numpy as np
import laspy
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import geopandas as gpd
import zipfile
import io
import tempfile

def apstradat_las_failu(las_fails):
    """
    Izpilda visu nepieciešamo loģiku: lasa LAS, interpolē, 
    aprēķina kontūrlīnijas un saglabā rezultātus pagaidu mapē.
    Atgriež pilnos ceļus uz izveidotajiem failiem.
    """

    # Izveidojam pagaidu mapi
    tmp_dir = tempfile.mkdtemp()

    # Sagatavojam izejas failu ceļus
    tiff_output_path = os.path.join(tmp_dir, "surface_model_interpolated.tif")
    contour_shp_path = os.path.join(tmp_dir, "slope_20_contours.shp")

    # Nolasām LAS failu no 'las_fails' (bytesIO objektu saglabājam pagaidām diskā)
    las_local_path = os.path.join(tmp_dir, "uploaded.las")
    with open(las_local_path, "wb") as f:
        f.write(las_fails.getvalue())

    las = laspy.read(las_local_path)

    # Filtrējam tikai zemes punktus (parasti klasifikācijas kods 2 ir zemes punkti)
    if "classification" in las.point_format.dimension_names:
        ground_mask = las.classification == 2
        x = las.x[ground_mask]
        y = las.y[ground_mask]
        z = las.z[ground_mask]
        st.write(f"Atlasīti {ground_mask.sum()} zemes punkti.")
    else:
        st.write("Klasifikācijas dati nav pieejami – izmantojam visus punktus.")
        x = las.x
        y = las.y
        z = las.z

    # Uzstādām EPSG:3059 koordinātu sistēmu
    crs = "EPSG:3059"

    # Definējam režģa šūnas izmēru (0.25, jo LiDAR dati ir augstas izšķirtspējas)
    cell_size = 0.25

    # Noskaidrojam datu robežas
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Izveidojam regulāru režģi
    grid_x = np.arange(xmin, xmax + cell_size, cell_size)
    grid_y = np.arange(ymin, ymax + cell_size, cell_size)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Veicam virsmas interpolāciju ar "cubic" metodi
    dem = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Ja ir tukšas vērtības (nan), aizpildām tās ar "nearest" interpolāciju
    mask = np.isnan(dem)
    if np.any(mask):
        dem[mask] = griddata((x, y), z, (grid_x[mask], grid_y[mask]), method='nearest')

    # Apgriežam DEM matricu vertikāli, lai sakristu ar GeoTIFF transformācijas kārtību
    dem = np.flipud(dem)

    # Saglabājam DEM kā GeoTIFF (oriģinālais DEM)
    transform = from_origin(xmin, ymax, cell_size, cell_size)
    with rasterio.open(
        tiff_output_path,
        'w',
        driver='GTiff',
        height=dem.shape[0],
        width=dem.shape[1],
        count=1,
        dtype=dem.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(dem, 1)

    # DEM līdzināšana (smoothing) pirms kontūrlīniju vilkšanas
    filterx = 7
    filtery = 7
    sig_digits = 2

    dem_smoothed = uniform_filter(dem, size=(filtery, filterx))
    dem_smoothed = np.around(dem_smoothed, sig_digits)

    # Aprēķinām slīpumu un izvelkam kontūrlīnijas, kur slīpums ir 20°
    dy, dx = np.gradient(dem_smoothed, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Izmantojam matplotlib, lai atrastu kontūras pie 20° slīpuma
    cs = plt.contour(slope_deg, levels=[20])
    plt.close()  # aizveram grafisko logu

    lines = []
    # cs.allsegs satur atrastās kontūras ceļus – izmantojam tikai kontūras līmeni 20°
    for seg in cs.allsegs[0]:
        # Katrs 'seg' ir masīvs ar punktiem (rinda, kolonna).
        # Pārvēršam punktus uz pasaules koordinātēm:
        #   world_x = xmin + (kolonna * cell_size)
        #   world_y = ymax - (rinda * cell_size)
        coords = [(xmin + pt[0] * cell_size, ymax - pt[1] * cell_size) for pt in seg]
        if len(coords) >= 2:
            line = LineString(coords)
            lines.append(line)

    # Izveidojam GeoDataFrame ar atrastajām kontūrlīnijām un uzstādām EPSG:3059
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    gdf.to_file(contour_shp_path)

    return tiff_output_path, contour_shp_path, tmp_dir


def izveidot_zip(tiff_path, shp_path, tmp_dir):
    """
    Ieliek GeoTIFF un visus kontūrlīniju failus (SHP, SHX, DBF, PRJ, ...) vienā ZIP arhīvā.
    Atgriež ZIP saturu kā bytes (lai to varētu ērti lejupielādēt).
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Pievienojam GeoTIFF
        zf.write(tiff_path, arcname=os.path.basename(tiff_path))

        # SHP faila komponentes (dbf, shx, prj, cpg, utt.)
        shp_name = os.path.splitext(os.path.basename(shp_path))[0]
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            fpath = os.path.join(tmp_dir, shp_name + ext)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=os.path.basename(fpath))

    zip_buffer.seek(0)
    return zip_buffer


def main():
    st.title("LAS faila apstrāde un kontūrlīniju ģenerēšana")
    st.write("Augšupielādē .las failu un iegūsti interpolētu virsmas modeli (GeoTIFF) un 20° slīpuma kontūrlīnijas (SHP)")

    uploaded_file = st.file_uploader("Augšupielādē LAS failu", type=["las"])

    if uploaded_file is not None:
        st.write("Faila augšupielāde veiksmīga. Sākam apstrādi...")

        # Izpildām apstrādes funkciju
        tiff_path, shp_path, tmp_dir = apstradat_las_failu(uploaded_file)

        st.success("Apstrāde pabeigta! Rezultāti saglabāti pagaidu mapē.")
        st.write("GeoTIFF fails:", tiff_path)
        st.write("Kontūrlīniju SHP fails:", shp_path)

        # Izveidojam ZIP pakotni lejupielādei
        zip_bytes = izveidot_zip(tiff_path, shp_path, tmp_dir)

        st.download_button(
            label="Lejupielādēt rezultātu (ZIP)",
            data=zip_bytes,
            file_name="las_apstrades_rezultats.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
