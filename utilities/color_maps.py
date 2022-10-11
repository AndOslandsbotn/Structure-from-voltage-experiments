from matplotlib import colors

def color_map():
    """Specify colors to use for visualizing mnist"""
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
    cmap['gray_white'] = '#f2f0e7'
    cmap['transparent'] = '#FFFFFF00'
    return cmap

def color_map_for_mnist():
    """Make a cmap that can be used in matplotlib.offsetbox.OffsetImage"""
    cmap = color_map()  # Get colors that we have specified
    cmap_mnist = []
    for key in cmap.keys():
        #cmap_mnist.append(colors.ListedColormap(['#FF000000', cmap[key]]))
        cmap_mnist.append(colors.ListedColormap([cmap['transparent'], cmap[key]]))
    return cmap_mnist