import numpy as np

def create_tiling(feat_range, bins, offset):
    """
    Create 1 tiling spec of 1 dimension(feature)
    feat_range: feature range; example: [-1, 1]
    bins: number of bins for that feature; example: 10
    offset: offset for that feature; example: 0.2
    """
    return np.linspace(feat_range[0], feat_range[1], bins+1)[1:-1] + offset

def create_tilings(tilingsShape=None, dimensionsReduction=True, outOfMiddleDistance=True):
    """
    feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings; example: 3 tilings
    bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
    offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
    """
    if tilingsShape == None:
        number_tilings = 2
        feature_range_pure = [[65, 115], [0, 30], [-150, 150]]
        bins_input = [7, 5, 5]
    else:
        number_tilings = tilingsShape["numberTilings"]
        feature_range_pure = tilingsShape["featureRange"]
        bins_input = tilingsShape["bins"]

    if dimensionsReduction == False:
        feature_range_pure = [feature_range_pure[0], feature_range_pure[0], feature_range_pure[1], feature_range_pure[1], feature_range_pure[2]]
        bins_input = [bins_input[0], bins_input[0], bins_input[1], bins_input[1], bins_input[2]]

    if outOfMiddleDistance == False:
        del bins_input[-1]
        del feature_range_pure[-1]

    bins = []
    for i in range(number_tilings):
        bins.append(bins_input) 
    
    offsets = []
    feature_ranges = []
    for i, feature_range in enumerate(feature_range_pure):
        new_offset = (feature_range[1]-feature_range[0]) / ((bins_input[i] - 1) * number_tilings - 1)
        feature_ranges.append([feature_range[0] - number_tilings * new_offset, feature_range[1] + new_offset])
        for j in range(number_tilings):
            if i == 0:
                offsets.append([])
            offsets[j].append(new_offset*j)
    
    #if tilingsShape == None:
    #    number_tilings = 2 # ToDo: test with 2
    #    feature_ranges = [[55, 125], [-10, 40], [-250, 250]]
    # L/R-Elbw: :65:75:85:95:105:115: ~>50  bei größer 115 macht es kein Unterschied mehr
    # L/R-Shoulder: :0:10:20:30: ~> 30
    # middle :-150:-50:50:150: ~> 300/(2n-1) = offset
    #    bins = [[7, 5, 5], [7, 5, 5]]
    # bins = [[7, 7, 5, 5], [7, 7, 5, 5]]  # 7*7*5*5 = 49*25 = 1250 Bereich, aber stark korreliert, viele leere Bereiche
    #    offsets = [[0, 0, 0], [9, 3.3, 33]]

    tilings = []
    #for each tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]
        
        tiling = []
        # for each feature dimension
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            # tiling for 1 feature
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        tilings.append(tiling)
    return np.array(tilings)

def get_tile_coding(feature, tilings):
    """
    feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
    tilings: tilings with a few layers
    return: the encoding for the feature on each layer
    """
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]  # tiling on that dimension
            coding_i = np.digitize(feat_i, tiling_i)
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    return np.array(feat_codings)