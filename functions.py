import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join


def clean_volumes_database():
    """
    This function cleans the volumes database by reading data from a CSV and an Excel file, merging them, 
    splitting the structure path into depths, renaming columns, and removing unnecessary columns. 
    It also handles missing values and ensures that the 'CCF_volume' column is of float type.
    
    Returns:
    pd.DataFrame: A cleaned pandas DataFrame representing the volumes database.
    """
    
    # Read the query data from a CSV file
    query = pd.read_csv("query.csv")
    
    # Read the table data from an Excel file
    table_s4 = pd.read_excel('table_S4_wang2020.xlsx', header=1)
    
    # Concatenate the 'structure ID' and 'CCF Volume' columns from the table
    temp = pd.concat([table_s4['structure ID'], table_s4['CCF Volume']], axis=1)
    temp.columns = ['id', 'CCF_volume']
    
    # Merge the query and temp DataFrames based on 'id'
    volumes = pd.merge(query, temp, how='left', left_on='id', right_on='id')
    
    # Split the 'structure_id_path' into depths
    structure_path = volumes.structure_id_path.str.split("/", expand = True)
    
    # Remove the first and last columns as they are not needed
    structure_path=structure_path.drop(structure_path.columns[[0,12]], axis=1)
    
    # Append the structure_path DataFrame to the volumes DataFrame
    volumes = pd.concat([volumes, structure_path], axis=1)
    
    # Rename the columns to have the correct depth level
    rename_cols = {i+1:i for i in range(0,11)}
    volumes.rename(columns=rename_cols, inplace=True)
    
    # Convert the depth columns to numeric
    for i in range(0,11):
        volumes[i] = pd.to_numeric(volumes[i])
    
    # Remove unnecessary columns
    columns_to_drop = ['ontology_id', 'hemisphere_id', 'weight', 'graph_id', 'graph_order', 'color_hex_triplet', \
    'neuro_name_structure_id', 'neuro_name_structure_id_path', 'failed', 'sphinx_id', 'structure_name_facet', \
     'failed_facet']
    volumes = volumes.drop(columns_to_drop, axis=1)
    
    # Replace '#N/D' values in the 'CCF_volume' column with np.nan
    volumes['CCF_volume'].replace(to_replace='#N/D', value=np.nan, inplace=True)
    
    # Convert the 'CCF_volume' column to float
    volumes["CCF_volume"] = pd.to_numeric(volumes["CCF_volume"])
    
    return volumes



def create_df_single_animal(path):
    """
    Reads all CSV files from a given directory, concatenates them into a single DataFrame,
    extracts slice information from the 'Image' column, and inserts the slice information as the first column.

    Parameters:
    path (str): The directory path where the CSV files are located.

    Returns:
    pd.DataFrame: A DataFrame containing data from all CSV files with an added 'slice' column.
    """
    # List all files in the specified directory
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # Read and concatenate all CSV files into a single DataFrame
    df_list = []
    for file in onlyfiles:
        # Append each DataFrame to a list
        df_list.append(pd.read_csv(join(path, file), sep=None, engine='python'))

    # Concatenate all DataFrames in the list
    df = pd.concat(df_list, ignore_index=True)

    # Extract the slice information (before the full stop) from the 'Image' column
    df['slice'] = df['Image'].str.extract(r'(coor_\d+_\d+)')

    # Insert the 'slice' column as the first column
    df.insert(0, 'slice', df.pop('slice'))

    return df




def find_children(area_id, l, vol):
    """
    Finds all areas that have a given area ID in their hierarchy up to a specified level.

    Parameters:
    area_id (int): The ID of the area to search for in the hierarchy.
    l (int): The starting level in the hierarchy to search from.
    vol (pd.DataFrame): The volume table containing hierarchical data.

    Returns:
    np.ndarray: An array of safe names of the areas found in the hierarchy.
    """
    # Initialize an empty list to hold the children areas
    children = []

    # Loop through levels from l to 10 (inclusive)
    for i in range(l, 11):
        # Find subareas at the current level that have the given area_id
        found_subareas = vol[vol[i] == area_id]['safe_name'].values
        # If any subareas are found, add them to the children list
        if len(found_subareas):
            children.append(found_subareas)

    # Flatten the list of arrays into a single array
    children = np.concatenate(children) if children else np.array([])

    return children



def aggregate_cells_per_area(df_mouse, vol, area, density):
    """
    Aggregates the number of cells per area, optionally calculating the density of cells.

    Parameters:
    df_mouse (pd.DataFrame): The DataFrame containing mouse data.
    vol (pd.DataFrame): The DataFrame containing volume data.
    area (str): The name of the area.
    density (bool): If True, calculate the density of cells. If False, count the total number of cells.

    Returns:
    pd.DataFrame: A DataFrame with either the total number of cells or the density of cells per area.
    """
    # Find the ID and depth of the specified area
    area_info = vol[vol['safe_name'] == area].iloc[0]
    area_id = area_info['id']
    area_depth = area_info['depth']

    # Find all areas that have the specified area as a parent up to the area level
    children = find_children(area_id=area_id, l=area_depth, vol=vol)

    # Initialize the DataFrame to hold results
    df = pd.DataFrame(index=children, columns=['n_cells' if not density else 'Density'])

    # Loop over each child area to count cells or calculate density
    n_cells = []
    for child in children:
        area_child_info = vol[vol['safe_name'] == child].iloc[0]
        area_child_acronym = area_child_info['acronym']
        
        # Filter rows for the current child area in df_mouse
        child_area_data = df_mouse[df_mouse['Name'] == area_child_acronym]

        if density:
            # Calculate density as the sum of cells over the sum of the area surface
            area_surface = child_area_data['Area µm^2'].sum()
            if area_surface > 0:
                n_cells.append(child_area_data['Num Detections'].sum() / area_surface)
            else:
                n_cells.append(0)
        else:
            # Count the total number of cells
            n_cells.append(child_area_data['Num Detections'].sum())

    # Populate the DataFrame with the calculated values
    df.iloc[:, 0] = n_cells

    return df




def aggregate_cells_per_slice(df_mouse, vol, area, density):
    """
    Aggregates the number of cells per slice for a specified area, optionally calculating the density of cells.

    Parameters:
    df_mouse (pd.DataFrame): The DataFrame containing mouse data.
    vol (pd.DataFrame): The DataFrame containing volume data.
    area (str): The name of the area.
    density (bool): If True, calculate the density of cells. If False, count the total number of cells.

    Returns:
    pd.DataFrame: A DataFrame with either the total number of cells or the density of cells per slice.
    """
    # Initialize the DataFrame with unique slices
    slices = df_mouse['slice'].unique()
    df = pd.DataFrame({'slice': slices})
    
    # Get the acronym of the specified area
    area_acronym = vol.loc[vol['safe_name'] == area, 'acronym'].values[0]

    # Initialize list to hold detection counts or densities
    num_detections = []

    # Iterate over each slice and count detections or calculate density
    for single_slice in slices:
        # Filter data for the current slice
        df_slice = df_mouse[df_mouse['slice'] == single_slice]

        # Count the number of cells in the specified area for the current slice
        n_cells = df_slice[df_slice['Name'] == area_acronym]['Num Detections'].sum()
        
        if density:
            # Calculate density if density flag is True
            area_surface = df_slice[df_slice['Name'] == area_acronym]['Area µm^2'].sum()
            area_density = n_cells / area_surface if area_surface > 0 else 0
            num_detections.append(area_density)
        else:
            # Append the total number of detections
            num_detections.append(n_cells)
    
    # Add the results to the DataFrame
    df['Density' if density else 'Num Detections'] = num_detections

    return df


def create_table_by_slice(df_mouse, areas, volumes, density=False):
    """
    Creates a table with the number of cells or cell density for specified areas and slices.

    Parameters:
    df_mouse (pd.DataFrame): The DataFrame containing mouse data.
    areas (list): A list of area names.
    volumes (pd.DataFrame): The DataFrame containing volume data.
    density (bool): If True, calculate the density of cells. If False, count the total number of cells.

    Returns:
    pd.DataFrame: A DataFrame with areas as rows and slices as columns containing cell counts or densities.
    """
    # Initialize the DataFrame with unique slices as columns
    df = pd.DataFrame(columns=df_mouse['slice'].unique())

    # Find all areas with their children
    areas_with_children = []
    for area in areas:
        # Retrieve area_id and depth for the specified area
        area_info = volumes[volumes['safe_name'] == area].iloc[0]
        area_id = area_info['id']
        area_depth = area_info['depth']
        # Find children areas for the specified area
        areas_with_children.append(find_children(area_id=area_id, l=area_depth, vol=volumes))

    # Flatten the list of children areas
    flattened_list = [item for sublist in areas_with_children for item in sublist]

    # Set 'area' as the index of the DataFrame
    df['area'] = flattened_list
    df = df.set_index('area')

    # Loop over slices and areas to fill the DataFrame
    for sl in df_mouse['slice'].unique():
        for area in flattened_list:
            # Aggregate cell counts or density per slice
            df_aggregated = aggregate_cells_per_slice(df_mouse=df_mouse, vol=volumes, area=area, density=density)
            # Retrieve the value for the current slice and area
            value = df_aggregated[df_aggregated['slice'] == sl]
            if not value.empty:
                df.loc[area, sl] = value.iloc[0, 1]  # Use .iloc to get the first value in the relevant column

    return df



def create_table_by_mouse(mice_list, areas, volumes, density=False):
    """
    Creates a table with the number of cells or cell density for specified areas and mice.

    Parameters:
    mice_list (list): A list of mouse names.
    areas (list): A list of area names.
    volumes (pd.DataFrame): The DataFrame containing volume data.
    density (bool): If True, calculate the density of cells. If False, count the total number of cells.

    Returns:
    pd.DataFrame: A DataFrame with areas as rows and mice as columns containing cell counts or densities.
    """
    # Initialize the DataFrame with mice names as columns
    df = pd.DataFrame(columns=mice_list)

    # Find all areas with their children
    areas_with_children = []
    for area in areas:
        # Retrieve area_id and depth for the specified area
        area_info = volumes[volumes['safe_name'] == area].iloc[0]
        area_id = area_info['id']
        area_depth = area_info['depth']
        # Find children areas for the specified area
        areas_with_children.append(find_children(area_id=area_id, l=area_depth, vol=volumes))

    # Flatten the list of children areas
    flattened_list = [item for sublist in areas_with_children for item in sublist]

    # Set 'area' as the index of the DataFrame
    df['area'] = flattened_list
    df.set_index('area', inplace=True)

    # Loop over mice and areas to fill the DataFrame
    for mouse in mice_list:
        # Load DataFrame for the current mouse
        df_mouse = create_df_single_animal(path='./' + mouse + '/tab_800/')
        for area in flattened_list:
            # Aggregate cell counts or density for the current area and mouse
            cell_info = aggregate_cells_per_area(df_mouse=df_mouse, vol=volumes, area=area, density=density).loc[area].values
            # Assign the value to the DataFrame
            df.loc[area, mouse] = cell_info[0] if len(cell_info) > 0 else 0  # Ensure correct indexing and handle empty data

    return df

