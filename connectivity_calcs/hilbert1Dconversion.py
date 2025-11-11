import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import pandas as pd

def convertHilbertCoordinates(x_min, x_max, y_min, y_max, z_min, z_max, x_arr, y_arr, z_arr):
    """
    Converts arrays of (x, y, z) coordinates into 1D Hilbert indices.
    Returns a numpy array of Hilbert indices.
    """
    # Compute maximum span across all axes
    max_span = max(x_max - x_min + 1,
                   y_max - y_min + 1,
                   z_max - z_min + 1)

    # Number of bits required per axis
    p = int(np.ceil(np.log2(max_span)))

    # Define the Hilbert curve object
    hilbert_curve = HilbertCurve(p=p, n=3)

    # Offset to positive space
    x_off = (x_arr - x_min).astype(int)
    y_off = (y_arr - y_min).astype(int)
    z_off = (z_arr - z_min).astype(int)

    # Compute Hilbert distance for each coordinate
    hilbert_indices = [
        hilbert_curve.distance_from_point([int(xi), int(yi), int(zi)])
        for xi, yi, zi in zip(x_off, y_off, z_off)
    ]

    return np.array(hilbert_indices)


if __name__ == '__main__':
    # --- 1. Define your ranges ---
    x_min, x_max = -56, 79
    y_min, y_max = -73, -5
    z_min, z_max = -2, 129

    # --- 2. Load your CSV file ---
    filename = r"C:\\Users\\GBalasubramanian\\OneDrive - The Bionics Institute of Australia\\Documents\\FMRI_FNIRS_Work\\coordinates_x_y_z.csv"
    df = pd.read_csv(filename)

    # --- 3. Extract columns as numpy arrays ---
    x = df['CordinateX'].to_numpy()
    y = df['CordinateY'].to_numpy()
    z = df['CordinateZ'].to_numpy()

    # --- 4. Compute Hilbert indices ---
    onedOutput = convertHilbertCoordinates(x_min, x_max, y_min, y_max, z_min, z_max, x, y, z)

    # --- 5. Add to the same DataFrame ---
    df['HilbertIndex'] = onedOutput

    # --- 6. Save to a new CSV file ---
    output_filename = filename.replace('.csv', '_with_HilbertIndex.csv')
    df.to_csv(output_filename, index=False)

    print(f"✅ Hilbert indices computed and saved to:\n{output_filename}")
    print(df.head())
