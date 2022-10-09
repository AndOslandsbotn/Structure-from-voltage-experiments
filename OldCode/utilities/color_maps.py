from matplotlib import colors

def color_map():
    cmap = {}
    cmap['color_red'] = '#CD3333'
    cmap['color_yel'] = '#E3CF57'
    cmap['color_lightblue'] = '#3380f2'
    cmap['color_darkgray'] = '#838B8B'
    cmap['color_green'] = '#aaffdd'
    cmap['color_lime'] = '#ddffbb'
    cmap['color_pink'] = '#fbbbbf'
    cmap['color_lightgreen'] = '#c0ff0c'
    cmap['color_orange'] = '#f5a565'
    cmap['color_darkgreen'] = '#40826d'
    return cmap

def color_map_list():
    cmap = color_map()
    cmap_list = []
    for key in cmap.keys():
        cmap_list.append(cmap[key])
    return cmap_list

def color_map_for_mnist():
    color_list = color_map_list()
    cmap_mnist = []
    for color in color_list:
        cmap_mnist.append(colors.ListedColormap(['#FF000000', color]))

    #cmap0 = colors.ListedColormap(['#FF000000', cmap['color_red']])
    #cmap1 = colors.ListedColormap(['#FF000000', cmap['color_yel']])
    #cmap2 = colors.ListedColormap(['#FF000000', cmap['color_darkgreen']])
    #cmap3 = colors.ListedColormap(['#FF000000', cmap['color_pink']])
    #cmap4 = colors.ListedColormap(['#FF000000', cmap['color_blue']])
    #cmap5 = colors.ListedColormap(['#FF000000', cmap['color_gray']])
    #cmap6 = colors.ListedColormap(['#FF000000', cmap['color_lime']])
    #cmap7 = colors.ListedColormap(['#FF000000', cmap['color_darkgray']])
    #cmap8 = colors.ListedColormap(['#FF000000', cmap['color_orange']])
    #cmap9 = colors.ListedColormap(['#FF000000', cmap['color_green']])
    #cmap_mnist = [cmap0, cmap1, cmap2, cmap3, cmap4, cmap5, cmap6, cmap7, cmap8, cmap9]
    return cmap_mnist