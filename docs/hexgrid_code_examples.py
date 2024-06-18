
#https://python.plainenglish.io/creating-beautiful-hexagon-maps-with-python-25c9291eeeda

import ellipsis as el

pathId = '1a24a1ee-7f39-4d21-b149-88df5a3b633a'
timestampId = '45c47c8a-035e-429a-9ace-2dff1956e8d9'

sh_countries = el.path.vector.timestamp.listFeatures(pathId, timestampId)['result']

sh_usa = sh_countries[sh_countries['NAME'] == 'United States']
sh_usa.plot()


import h3pandas
resolution = 3
hexagons = sh_usa.h3.polyfill_resample(resolution)
hexagons.plot()


pathId = '632aeb3f-ca77-4bdd-a8cd-5f733dbd87ee'
timestampId = 'da3bb83d-8cc0-45e4-a96d-e6db0f83616b'

sh_hur = el.path.vector.timestamp.listFeatures(pathId, timestampId)['result']
sh_hur_relevant = sh_hur[ sh_hur.intersects(sh_usa['geometry'].values[0])]
sh_hur_relevant.plot()


hexagons['count'] = 0
for i in range(sh_hur_relevant.shape[0]):
    earth_quake = sh_hur_relevant['geometry'].values[i]
    update_bools = hexagons.intersects(earth_quake)
    hexagons.loc[update_bools, 'count'] = hexagons.loc[update_bools, 'count'] + 1

hexagons['count'] = hexagons['count'] * 10000

hexagons[['geometry', 'count']].head()


token = el.account.logIn('demo_user', 'demo_user')
pathId = el.path.vector.add('Hexagon covering', token = token)['id']
timestampId = el.path.vector.timestamp.add(pathId, token)['id']

el.path.vector.timestamp.feature.add(pathId, timestampId, hexagons, token)

#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------

#https://geographicdata.science/book/data/h3_grid/build_sd_h3_grid.html

%matplotlib inline

from h3 import h3
import geopandas
from shapely.geometry import Polygon
import contextily as ctx
import cenpy


sd = geopandas.read_file("sd_city_centre.geojson")
ax = sd.plot(alpha=0.25, color="orange", figsize=(9, 9))
ctx.add_basemap(ax, crs=sd.crs.to_string())


%time hexs = h3.polyfill(sd.geometry[0].__geo_interface__, 8, geo_json_conformant = True)


polygonise = lambda hex_id: Polygon(
                                h3.h3_to_geo_boundary(
                                    hex_id, geo_json=True)
                                    )

%time all_polys = geopandas.GeoSeries(list(map(polygonise, hexs)), \
                                      index=hexs, \
                                      crs="EPSG:4326" \
                                     )
									 
									 
ax = all_polys.plot(alpha=0.5, color="xkcd:pale yellow", figsize=(9, 9))
ctx.add_basemap(ax, crs=all_polys.crs.to_string())
ax.set_title(f"{all_polys.shape[0]} Hexagons");


census = cenpy.Decennial2010()
tracts = census.from_msa("San Diego, CA", level="tract")


ax = tracts.plot(alpha=0.5, color="xkcd:pale green")
ctx.add_basemap(ax, crs=tracts.crs.to_string())


tracts.head()


ax = tracts_land_one.to_crs(epsg=4269).plot()
all_polys.to_crs(epsg=4269).plot(color="xkcd:pale yellow", ax=ax)


h3_all = geopandas.GeoDataFrame({"geometry": all_polys,
                                 "hex_id": all_polys.index},
                                crs=all_polys.crs
                               )


ax = h3_land.plot(alpha=0.5, color="xkcd:pale yellow")
ctx.add_basemap(ax, crs=h3_land.crs.to_string())


! rm sd_h3_grid.gpkg
h3_land.drop("id", axis=1).to_file("sd_h3_grid.gpkg", driver="GPKG")


shp = geopandas.GeoDataFrame(
    {"geometry": [h3_land.unary_union],
     "id": ["one"]
    }, crs = h3_land.crs).to_crs(tracts.crs)
sub = geopandas.overlay(tracts, shp, how="intersection")


ax = sub.plot(facecolor="k", edgecolor="xkcd:lilac", figsize=(9, 9))
h3_land.to_crs(sub.crs).plot(facecolor="none", linewidth=0.5, edgecolor="xkcd:light aqua", ax=ax)
ctx.add_basemap(ax, crs=sub.crs.to_string())



#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------

#https://gis.utah.gov/blog/2022-10-26-using-h3-hexes/


# associating address points with a hex id
#def assign_h3(df, resolution):
#    df[f'h3_{resolution}'] = df.apply(lambda row: h3.h3_to_string(h3.geo_to_h3(row['SHAPE']['y'], row['SHAPE']['x'], resolution)), axis=1)
#
#addr_df = pd.DataFrame.spatial.from_featureclass(r'H3\H3.gdb\address_points_20220727_wgs84')
#
#%timeit assign_h3(addr_df, 9)


state_boundary_df = pd.DataFrame.spatial.from_featureclass(r'H3\opensgid.agrc.utah.gov.sde\opensgid.boundaries.state_boundary')

#: Buffer the state boundary by 5km
#: Row index 0 is a mask outside the state boundary, index 1 is the boundary itself.
#: Also, the SHAPE field is the last item in the column index
buffered = state_boundary_df.iloc[1, -1].buffer(5000)
buffered_df = pd.DataFrame.spatial.from_df(
                pd.DataFrame({'SHAPE': [buffered]}),
                geometry_column='SHAPE', sr=26912
                )

#: Project our polygon to WGS84 and convert to geojson for the H3 analysis
buffered_df.spatial.project(4326)
geojson_dict = json.loads(buffered_df.spatial.to_featureset().to_geojson)

#: Get both the hexagon numbers and their hexadecimal representation in string form
#: We have to specify the geometry from the first feature in the geosjon dict, even though we only have
#: one feature in there
resolution = 6
hexes = h3.polyfill(geojson_dict['features'][0]['geometry'], resolution, geo_json_conformant=True)
str_hexes = [h3.h3_to_string(h) for h in hexes]

#: Create a new spatially-enabled data frame by calling h3_to_geo_boundary and creating a Geometry object
#: in one fell swoop
#: And yes, it's generally bad form to assign a lambda function a name, but this helps break the code up
polygoniser = lambda hex_id: arcgis.geometry.Geometry({
                'rings': [h3.h3_to_geo_boundary(hex_id, geo_json=True)],
                'spatialReference': {'wkid': 4326}
                })
hexes_df = pd.DataFrame.spatial.from_df(
                pd.DataFrame({'hex_id': str_hexes, 'SHAPE': list(map(polygoniser, hexes))}),
                geometry_column='SHAPE', sr=4326
                )

#: Finally, write it out
hexes_df.spatial.to_featureclass(r'H3\H3.gdb\state_h3_6_wgs')


#: Load the open source places, project to lat/long, and get hex ids at level 6
osp_points_df = pd.DataFrame.spatial.from_featureclass(r'H3\opensgid.agrc.utah.gov.sde\opensgid.society.open_source_places')
osp_points_df.spatial.project(4326)
assign_h3(osp_points_df, 6)

#: Perform the aggregation for each dataset and call the count() method on a column to get the number of features
#: in each group.
#: Note that we pre-filter the places to the 'supermarket' category.
addrs_per_hex = addr_df.groupby('h3_6')['utaddptid'].count()
supermarkets_per_hex = osp_points_df[osp_points_df['category']=='supermarket'].groupby('h3_6')['osm_id'].count()

#: Merge our two resulting series together into a dataframe, moving the hex id out of the index and renaming columns.
#: We also use dropna to drop any hexes that are missing either address points or supermarkets.
data_hexes = pd.concat([addrs_per_hex, supermarkets_per_hex], axis=1)
                .dropna()
                .reset_index()
                .rename(columns={'utaddptid': 'addrs', 'osm_id': 'supermarkets'})

#: Calculate our new metric
data_hexes['addrs_per_supermarket'] = data_hexes['addrs'] / data_hexes['supermarkets']

#: Join our new data with the hex polygons, keeping only the ones
merged_df = hexes_df.merge(data_hexes, left_on='hex_id', right_on='h3_6', how='inner').sort_values(by='addrs_per_supermarket')


map1 = merged_df.spatial.plot(renderer_type='c',
                              method='esriClassifyQuantile',
                              col='addrs_per_supermarket',
                              class_count=5,
                              cmap='RdYlGn_r',
                              alpha=.5)
map1.center = [40.6, -111.9]
map1.zoom = 9
map1.take_screenshot()



#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------

#https://towardsdatascience.com/fast-geospatial-indexing-with-h3-90e862482585

def query_radius(self,
                    location: np.ndarray,
                    r: float) -> np.ndarray:
    edge_len = h3.edge_length(self.h3res, unit="m")
    idx = h3.geo_to_h3(location[0], location[1], self.h3res)

    ring = h3.k_ring(idx, 1 + int(round(r / edge_len)))

    i0 = np.searchsorted(self.h3arr, ring, side='left', sorter=self.h3idx)
    i1 = np.searchsorted(self.h3arr, ring, side='right', sorter=self.h3idx)

    indices = np.hstack([np.arange(i, j) for i, j in zip(i0, i1) if i != j])

    idx = self.h3idx[indices]
    dist = gm.vec_haversine(self.locations[idx, 0], self.locations[idx, 1],
                            location[0], location[1])
    return self.h3idx[indices[np.argwhere(dist <= r).ravel()]]
	
# parallel
# mp is python multiprocessing
def __init__(self, locations: np.ndarray, resolution=10):
    self.locations = locations
    self.h3res = resolution
    cpus = mp.cpu_count()
    arrays = np.array_split(locations, cpus)
    fn = partial(geo_to_h3_array, resolution=resolution)
    with mp.Pool(processes=cpus) as pool:
        results = pool.map(fn, arrays)
    flattened = [item for sublist in results for item in sublist]
    self.h3arr = np.array(flattened, dtype=np.uint64)
    self.h3idx = np.argsort(self.h3arr)	
	
	
def query_knn(self, location: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    idx = h3.geo_to_h3(location[0], location[1], self.h3res)

    i = 0
    indices = np.zeros(0, dtype=np.uint64)
    ring = np.zeros(0, dtype=np.uint64)
    while indices.shape[0] < k:
        i += 2
        k_ring = h3.k_ring(idx, i)
        ring = np.setdiff1d(k_ring, ring, assume_unique=True)

        i0 = np.searchsorted(self.h3arr, ring, side='left', sorter=self.h3idx)
        i1 = np.searchsorted(self.h3arr, ring, side='right', sorter=self.h3idx)

        indices = np.hstack((indices,
                                np.hstack([np.arange(i, j, dtype=np.uint64)
                                        for i, j in zip(i0, i1) if i != j])))

    idx = self.h3idx[indices]
    dist = gm.vec_haversine(self.locations[idx, 0],
                            self.locations[idx, 1],
                            location[0], location[1])

    dist_idx = np.argsort(dist)
    return idx[dist_idx[:k]], dist[dist_idx[:k]]	
	
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------
#--------------------------------------------