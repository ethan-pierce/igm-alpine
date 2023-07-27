import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse


def main():
    """Plot results from IGM simulation."""
    parser = argparse.ArgumentParser(description = 'Plot IGM results')
    parser.add_argument('dir', type=str, help='Directory where results are stored')
    parser.add_argument('--vars', '--list', type=str, nargs='+', help='List of variables to plot')
    parser.add_argument('--inputs', action='store_true', help='Plot vars from input file instead of results')
    parser.add_argument('--keys', action='store_true', help='Print list of keys')
    args = parser.parse_args()

    cwd = args.dir
    variables = args.vars
    inputs = args.inputs

    if args.inputs:
        ncfile = 'geology.nc'
        data = xr.open_dataset(cwd + ncfile)
    else:
        ncfile = 'geology-optimized.nc'
        data = xr.open_dataset(cwd + ncfile)

    if args.keys:
        print(data.keys())

    for var in variables:
        if var not in data.keys():
            print('Could not find ' + str(var) + ' in ' + str(ncfile))

        im = plt.imshow(data[var][:])
        plt.colorbar(im)
        plt.show()
    

if __name__ == '__main__':
    main()