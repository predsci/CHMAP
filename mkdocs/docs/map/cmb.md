# Combining Maps
Maps are combined using a minimum intensity merge method. Maps are originally created for each instrument image individually then
merged. There can be some points of overlap between different instrument maps and this is resolved by taking the data with
the minimum intensity from each point of overlap. Coronal Hole data is then chosen based off which data points are used to 
create the original EUV maps. This method ensures that resulting maps are more continuous at seams. We also use a cutoff mu
value to limit limb data distortion. In merging regions of overlap, we use data with a mu value greater than the cutoff value.
In areas without overlap, any data available is used (mu cutoff of 0).  

##Combine Maps Function
The combine maps function can be found [here](https://github.com/predsci/CHD/blob/master/modules/map_manip.py).  

```python
def combine_maps(map_list, chd_map_list=None, mu_cutoff=0.0, mu_cut_over=None, del_mu=None):
    """
    function to combine maps from a list of PsiMap objects based on a mu_cutoff and minimum intensity merge
    Two methods to determine best minimum intensity merge - using mu cutoff values, or using del mu value
    return: combined EUV map, combined CHD map
    """
    map_list[ii].data[map_list[ii].mu < mu_cutoff] = map_list[ii].no_data_val
    if mu_cut_over is not None:
        overlap[:, :, ii] = np.logical_and(data_array[:, :, ii] != map_list[0].no_data_val, data_array[:, :, jj] != map_list[0].no_data_val)
        good_index[:, :, ii] = np.logical_or(np.logical_and(overlap[:, :, ii], mu_array[:, :, ii] >= mu_cut_over), 
                                            np.logical_and(data_array[:, :, ii] != map_list[0].no_data_val, mu_array[:, :, ii] >= mu_cutoff))
        data_array[np.logical_not(good_index)] = float_info.max
    elif del_mu is not None:
        good_index[:, :, ii] = mu_array[:, :, ii] > (max_mu - del_mu)
        data_array[np.logical_not(good_index)] = float_info.max
    data_array[data_array == map_list[0].no_data_val] = float_info.max
    map_index = np.argmin(data_array, axis=2)
    keep_data = data_array[row_index, col_index, map_index] 
    keep_chd = chd_array[row_index, col_index, map_index]
    chd_map = psi_d_types.PsiMap(keep_chd, map_list[0].x, map_list[0].y, mu=keep_mu,
                                 origin_image=keep_image, no_data_val=map_list[0].no_data_val)
    euv_map = psi_d_types.PsiMap(keep_data, map_list[0].x, map_list[0].y, mu=keep_mu,
                                 origin_image=keep_image, no_data_val=map_list[0].no_data_val)
    return euv_map, chd_map
```  

* 1.) <code>map_list[ii].data[map_list[ii].mu < mu_cutoff] = map_list[ii].no_data_val</code>  
    * for all pixels with mu < mu_cutoff, set intensity to no_data_val  
* 2.1) <code>np.logical_or(np.logical_and(overlap[:, :, ii], mu_array[:, :, ii] >= mu_cut_over), np.logical_and(
                                                data_array[:, :, ii] != map_list[0].no_data_val, mu_array[:, :, ii] >= mu_cutoff))</code>  
    * to determine "good indices" based of Caplan et. al. 2016: 
        * check for overlap, in areas of overlap choose data where mu > mu_cut_over  
        * in areas of no overlap, choose data where mu > mu_cutoff  
* 2.2) <code> mu_array[:, :, ii] > (max_mu - del_mu)</code>  
    * determines "good indices" based off the value of del_mu  
* 3.) <code>data_array[np.logical_not(good_index)] = float_info.max, data_array[data_array == map_list[0].no_data_val] = float_info.max</code>  
    * make poor mu pixels unusable to merge, make no_data_vals unusable to merge  
* 4.) <code>map_index = np.argmin(data_array, axis=2)</code>  
    * find minimum intensity of remaining pixels  
* 5.) <code>keep_data = data_array[row_index, col_index, map_index],  
    keep_data = data_array[row_index, col_index, map_index]</code>  
    * choose data to use for the EUV and CHD map
* 6.) <code>psi_d_types.PsiMap</code>  
    * create new PsiMap object for both EUV and CHD combined maps  

