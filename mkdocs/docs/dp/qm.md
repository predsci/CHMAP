# Synchronic & Quality Maps

Synchronic maps are a standard data product from the CHMAP pipeline. These are instantaneous full-sun maps created
by combining co-temporal images from multiple spacecraft (e.g. STEREO-A, STEREO-B, & AIA).

More examples of synchronic maps can be found on our [Coronal Hole Detection](../chd/chd.md) page.

Each synchronic map has an accompanying "quality" map. The goal of a quality maps is to display where data at each pixel came from, and the cosine of the center-to-limb angle (mu) of the origin image at that 
point.  

The code for these maps is found [here](https://github.com/predsci/CHD/blob/master/chmap/maps/synchronic/quality_maps.py). 


## Example Maps
### Synchronic Maps on June 18, 2011 
[EUV Map](../img/dp/qual_map/EUV_Combined_Map_08162011.png) | [Quality EUV Map](../img/dp/qual_map/EUV_Quality_Map_08162011.png) 
:-: | :-: 
![EUV Map](../img/dp/qual_map/EUV_Combined_Map_08162011.png) | ![Quality EUV Map](../img/dp/qual_map/EUV_Quality_Map_08162011.png) 

[CHD Map](../img/dp/qual_map/CHD_Map_08162011.png) | [Quality CHD Map](../img/dp/qual_map/CHD_Quality_Map_08162011.png) 
:-: | :-: 
![CHD Map](../img/dp/qual_map/CHD_Map_08162011.png) | ![Quality CHD Map](../img/dp/qual_map/CHD_Quality_Map_08162011.png) 

[Full CR EUV Map](../img/dp/full_cr/CR_EUV_Map_052011.png) | [Quality CR EUV Map](../img/dp/full_cr/EUV_Quality_Map_052011.png)
:-: | :-: 
![Full CR EUV Map](../img/dp/full_cr/CR_EUV_Map_052011.png) | ![Quality CR EUV Map](../img/dp/full_cr/EUV_Quality_Map_052011.png)

## Code Outline

```python
def quality_map(db_session, map_data_dir, inst_list, query_pd, euv_combined, chd_combined=None, color_list=None):
    euv_origin_image = euv_combined.origin_image
    euv_origins = np.unique(euv_origin_image)
    euv_image = np.empty(euv_origin_image.shape, dtype=object)
    for euv_id in euv_origins:
        query_ind = np.where(query_pd['data_id'] == euv_id)
        instrument = query_pd['instrument'].iloc[query_ind[0]]
        if len(instrument) != 0:
            euv_image = np.where(euv_origin_image != euv_id, euv_image, instrument.iloc[0])
    Plotting.PlotQualityMap(euv_combined, euv_image, inst_list, color_list, nfig='EUV Quality Map ' + str(euv_combined.image_info.date_obs[0]),
                            title='EUV Quality Map: Mu Dependent\n' + str(euv_combined.image_info.date_obs[0]))
    if chd_combined is not None:
        chd_origin_image = chd_combined.origin_image
        chd_origins = np.unique(chd_origin_image)
        chd_image = np.empty(chd_origin_image.shape, dtype=object)
        for chd_id in chd_origins:
            query_ind = np.where(query_pd['data_id'] == chd_id)
            instrument = query_pd['instrument'].iloc[query_ind[0]]
            if len(instrument) != 0:
                chd_image = np.where(euv_origin_image != chd_id, chd_image, instrument.iloc[0])
        Plotting.PlotQualityMap(chd_combined, chd_image, inst_list, color_list, nfig='CHD Quality Map ' + str(chd_combined.image_info.date_obs[0]),
                                title='CHD Quality Map: Mu Dependent\n' + str(chd_combined.image_info.date_obs[0]), map_type='CHD')
    # save these maps to database
    return None
```

* 1.) <code>query_ind = np.where(query_pd['image_id'] == euv_id)</code>
    * loop through the unique list of image ids and determine at what indices
    they are present
* 2.) <code>euv_image = np.where(euv_origin_image != euv_id, euv_image, instrument.iloc[0])</code>
    * add instrument name to array in correct pixel position
* 3.) <code>Plotting.PlotQualityMap</code>
    * plot a quality map based off instrument and mu value of the final map

