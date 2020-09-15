import argparse
import pandas as pd
import numpy as np

def clean(data_path, out_path):
    """Do cleaning of missing values in input csv

    Args:
        data_path (str): Path to csv file we will clean
        out_path (str): Path to which cleaned csv will be written

    Returns:
        None

    """
    df = pd.read_csv(data_path, delimiter=',')

    # Remove rows with empty ID
    mask = df.ID.notnull()
    df = df[mask]

    # Remove rows for which follow-up status is NaN
    mask = df['Follow-up Status'].str.lower().notnull()
    df = df[mask]

    # Remove columns columns
    df = df[['Unnamed: 1', 'MYC IHC', 'BCL2 IHC', 'BCL6 IHC', 'CD10 IHC',
                             'MUM1 IHC', 'HANS', 'BCL6 FISH', 'MYC FISH', 'BCL2 FISH', 'Age',
                             'ECOG PS', 'LDH', 'EN', 'Stage', 'IPI Score',
                             'IPI Risk Group (4 Class)', 'RIPI Risk Group',
                             'OS', 'PFS (yr)', 'Follow-up Status']]

    # Name columns Labels
    cols = df.columns.values
    cols[0] = 'Label'
    cols[-2] = 'PFS'
    df.columns = cols

    # Clean MYC IHC (NO DATA --> np.nan)
    df['MYC IHC'] = df['MYC IHC'].str.lower()
    df['MYC IHC'] = df['MYC IHC'].replace(['no data'], np.nan).astype(np.float64)

    # Clean BCL2 IHC (NO DATA --> np.nan)
    df['BCL2 IHC'] = df['BCL2 IHC'].str.lower()
    df['BCL2 IHC'] = df['BCL2 IHC'].replace(['no data'], np.nan).astype(np.float64)

    # Clean BCL2 IHC (NO DATA --> np.nan)
    df['BCL6 IHC'] = df['BCL6 IHC'].str.lower()
    df['BCL6 IHC'] = df['BCL6 IHC'].replace(['no data'], np.nan).astype(np.float64)

    # Binarize Follow-up Status
    outcomes = (df['Follow-up Status'].str.lower() == 'deceased').astype(int)
    df['Follow-up Status'] = outcomes

    # Binarize IHC
    df['CD10 IHC'] = df['CD10 IHC'].str.lower()
    df['CD10 IHC'] = df['CD10 IHC'].replace(['no data'], np.nan)
    df['CD10 IHC'] = df['CD10 IHC'].replace(['pos'], 1)
    df['CD10 IHC'] = df['CD10 IHC'].replace(['neg'], 0)

    df['MUM1 IHC'] = df['MUM1 IHC'].str.lower()
    df['MUM1 IHC'] = df['MUM1 IHC'].replace(['no data'], np.nan)
    df['MUM1 IHC'] = df['MUM1 IHC'].replace(['pos'], 1)
    df['MUM1 IHC'] = df['MUM1 IHC'].replace(['neg'], 0)

    # Binarize HANS (GC: 1, NGC: 0)
    df['HANS'] = df['HANS'].str.lower()
    df['HANS'] = df['HANS'].replace(['no data'], np.nan)
    df['HANS'] = df['HANS'].replace(['gc'], 1)
    df['HANS'] = df['HANS'].replace(['ngc'], 0)

    # Binarize FISH
    df['BCL6 FISH'] = df['BCL6 FISH'].str.lower()
    df['BCL6 FISH'] = df['BCL6 FISH'].replace(['no data'], np.nan)
    df['BCL6 FISH'] = df['BCL6 FISH'].replace(['pos'], 1)
    df['BCL6 FISH'] = df['BCL6 FISH'].replace(['neg'], 0)

    df['MYC FISH'] = df['MYC FISH'].str.lower()
    df['MYC FISH'] = df['MYC FISH'].replace(['no data'], np.nan)
    df['MYC FISH'] = df['MYC FISH'].replace(['pos'], 1)
    df['MYC FISH'] = df['MYC FISH'].replace(['neg'], 0)

    df['BCL2 FISH'] = df['BCL2 FISH'].str.lower()
    df['BCL2 FISH'] = df['BCL2 FISH'].replace(['no data'], np.nan)
    df['BCL2 FISH'] = df['BCL2 FISH'].replace(['pos'], 1)
    df['BCL2 FISH'] = df['BCL2 FISH'].replace(['neg'], 0)

    # Binarize ECOG PS (we choose the worse in not sure e.g. 0-1 --> 1)
    df['ECOG PS'] = df['ECOG PS'].replace(['0-1'], 1)
    df['ECOG PS'] = df['ECOG PS'].replace(['2-3'], 3).astype(int)

    # Binarize LDH (H: 1, N: 0)
    df['LDH'] = df['LDH'].str.lower()
    df['LDH'] = df['LDH'].replace(['no data', 'n/a', 'unk'], np.nan)
    df['LDH'] = df['LDH'].replace(['h'], 1)
    df['LDH'] = df['LDH'].replace(['n'], 0)

    # Clean EN (Remove plusses, e.g. 4+ --> 4)
    df['EN'] = df['EN'].str.lower()
    df['EN'] = df['EN'].replace(['n/a'], np.nan)
    df['EN'] = df['EN'].replace(['4+'], 4)
    df['EN'] = df['EN'].replace(['3+'], 3)
    df['EN'] = df['EN'].replace(['2+'], 2).astype(np.float64)

    # Clean IPI Score (0-1 --> 1)
    df['IPI Score'] = df['IPI Score'].replace(['0-1'], 1).astype(int)

    # Categorize IPI Risk Group (L: 0, LI: 1, HI: 2, H: 3, UNK: np.nan)
    df['IPI Risk Group (4 Class)'] = df['IPI Risk Group (4 Class)'].str.lower()
    df['IPI Risk Group (4 Class)'] = df['IPI Risk Group (4 Class)'].replace(['no data', 'unk'], np.nan)
    df['IPI Risk Group (4 Class)'] = df['IPI Risk Group (4 Class)'].replace(['l'], 0)
    df['IPI Risk Group (4 Class)'] = df['IPI Risk Group (4 Class)'].replace(['li'], 1)
    df['IPI Risk Group (4 Class)'] = df['IPI Risk Group (4 Class)'].replace(['hi'], 2)
    df['IPI Risk Group (4 Class)'] = df['IPI Risk Group (4 Class)'].replace(['h'], 3).astype(np.float64)

    # Categorize RIPI Risk Group (VG: 0, G: 1, P: 2)
    df['RIPI Risk Group'] = df['RIPI Risk Group'].str.lower()
    df['RIPI Risk Group'] = df['RIPI Risk Group'].replace(['no data', 'unk'], np.nan)
    df['RIPI Risk Group'] = df['RIPI Risk Group'].replace(['vg'], 0)
    df['RIPI Risk Group'] = df['RIPI Risk Group'].replace(['g'], 1)
    df['RIPI Risk Group'] = df['RIPI Risk Group'].replace(['p'], 2).astype(int)


    # Drops all rows with nan (left with 124 rows)
    no_nan = df.dropna(subset=df.columns)

    # Saves one file with rows with nan and one without
    df.to_csv(out_path + 'cleaned_data_with_nan.csv', index=False)
    no_nan.to_csv(out_path + 'cleaned_data_without_nan.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean data in csv file')
    parser.add_argument('-d', type=str, nargs='?', required='True',
                        help='absolute path to csv file to be cleaned')
    parser.add_argument('-o', type=str, nargs='?', required='True',
                        help='absolute path to intended output folder')
    args = parser.parse_args()
    data_path = args.d
    out_path = args.o
    clean(data_path, out_path)
