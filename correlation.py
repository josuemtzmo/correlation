from IPython.display import clear_output
import collections
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import dask.array as dsa
import xarray as xr

data_dict = collections.defaultdict(list)
        
def signif(t_test,threshold=2):
    sig = np.zeros(len(t_test))*np.nan
    sig[abs(t_test) > threshold]=1
    return sig

class Compute_Correlation():
    """Class to compute correlation between two datasets"""
    
    def __init__(self,y1, y2,nc=None,method="Trim"):
        # De-mean both timeseries 
        self.y1t = y1 - np.mean(y1) 
        self.y2t = y2 - np.mean(y2)   
        self.n = len(y1)
        self.nc = nc
        self.method=method
        self.get_datapoints_2_correlate()
        
    def update_y(self,y1,y2):
        self.y1t = y1 - np.nanmean(y1) 
        self.y2t = y2 - np.nanmean(y2)
        
    def cor_series(self):
        corr = np.zeros(self.nc)
        t = np.zeros(self.nc)
        for j in np.arange(0, self.nc):
            corr[j],t[j] = self.corr_one_roll(j)
        sig = signif(t)
        return corr,sig,t
    
    def get_datapoints_2_correlate(self):
        ## Looking for lagged correlations between two datasets
        ## Assumes y1 is leading
        ## Avoid end effects by using only first 2/3rds of timeseries
        if not self.nc and self.method!="Cyclic":
            self.nc = int(self.n/3.5)
        elif self.method == "Cyclic":
            self.nc = self.n
    
    def corr_one_roll(self,j):
        x1t=np.roll(self.y1t, j)
        x2t = self.y2t
        if self.method!="Cyclic":
            x1t = x1t[j:]
            x2t = x2t[j:]
            
        corr = np.nanmean(x1t*x2t)/np.sqrt(np.nanmean(x1t**2)*np.nanmean(x2t**2))
        try:
            r1, tmp = ss.pearsonr(x1t[1:], np.roll(x1t, 1)[1:])
            r2, tmp = ss.pearsonr(x2t[1:], np.roll(x2t, 1)[1:])
            Neff = self.n*(1-r1*r2)/(1+r1*r2)
            t = corr*Neff**(0.5)*(1-corr**2)**(-0.5)
        except:
            t=0
        return corr,t
    
    def plot(self,data=data_dict, title='',ax=None,**kargs):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        if not data:
            data['x']= range(self.n)
            data['y1']= self.y1t
            data['y2']= self.y2t
            
        if 'y1' in data.keys() and 'y2' in data.keys():
            ax.plot(data['x'],data['y1'],'-k',**kargs)
            ax.plot(data['x'],data['y2'],'-', color='steelblue',**kargs)
        else:
            ax.plot(data['x'],data['y'],**kargs)
        ax.set_title(title)
        ax.grid()
        return ax
    
    def animate(self):
        data_series = collections.defaultdict(list)
        data = collections.defaultdict(list)
        corr = np.zeros(self.nc)*np.nan
        t = np.zeros(self.nc)*np.nan
        min_n = int(self.nc*0.1)
        for j in np.arange(0, self.nc):
            # Clear ouput
            clear_output(wait=True)
            fig, axs  = plt.subplots(2)
            fig.subplots_adjust(hspace=0.4)
            corr[j],t[j] = self.corr_one_roll(j)
            sig = signif(t)
            data_series['x'] = np.arange(len(self.y1t))
            data_series['y1'] = np.roll(self.y1t, j)
            data_series['y2'] = self.y2t
            ax = self.plot(data=data_series,title='De-mean signals'+' Lag:{0}'.format(j),ax=axs[0])
            data['x'] = np.arange(self.nc)
            data['y'] = corr
            ax = self.plot(data=data,title='Correlation',ax=axs[1])
            if j < min_n:
                ax.set_xlim(0,min_n)
            ax.set_ylim(-1.1,1.1)
            plt.show()
            
class xarrayCompute_Correlation():
    
    def __init__(self, dataarray1, dataarray2, axis=-1, nc=None, method="Trim"):
        if dataarray1.dims != dataarray2.dims or dataarray1.shape != dataarray2.shape:
            raise ValueError("Dimensions of both datasets must be identical")
        if len(dataarray1.shape) < 2:
            raise ValueError("Datarrays must be at least 2D (x,y)")
        self.data1 = dataarray1
        self.data2 = dataarray2
        self.axis = axis
        len_sel_axis = len(dataarray1[dataarray1.dims[axis]])
        d_s = self.dummy_series(len_sel_axis)
        self.corr_class = Compute_Correlation(d_s, d_s, nc = nc, method = method)
        
    def dummy_series(self,length):
        return range(length)
    
    def _calc_correlation_1D(self,y1,y2):
        self.corr_class.update_y(y1,y2)
        corr,sig,t = self.corr_class.cor_series()
        return corr,sig,t
    
    def _calc_correlation_along_axis(self,chunks=None):
        if chunks==None:
            chunk_dict = {dim:-1 if dim=='time' else int(len(self.data1[dim])*0.02)+1 for dim in self.data1.dims}
        elif type(chunks)==dict: 
            chunk_dict = chunks
        else:
            raise TypeError("type(chunks) must be a dict")
        arrs = [self.data1.chunk(chunk_dict),self.data2.chunk(chunk_dict)]
        corr = self.multi_apply_along_axis(self._calc_correlation_1D, self.axis, arrs)
        return corr
        
    
    @staticmethod
    def multi_apply_along_axis(func1d, axis, arrs, *args, **kwargs):
        """
        Given a function `func1d(A, B, C, ..., *args, **kwargs)`  that acts on 
        multiple one dimensional arrays, apply that function to the N-dimensional
        arrays listed by `arrs` along axis `axis`

        If `arrs` are one dimensional this is equivalent to::

            func1d(*arrs, *args, **kwargs)

        If there is only one array in `arrs` this is equivalent to::

            numpy.apply_along_axis(func1d, axis, arrs[0], *args, **kwargs)

        All arrays in `arrs` must have compatible dimensions to be able to run
        `numpy.concatenate(arrs, axis)`

        Arguments:
            func1d:   Function that operates on `len(arrs)` 1 dimensional arrays,
                      with signature `f(*arrs, *args, **kwargs)`
            axis:     Axis of all `arrs` to apply the function along
            arrs:     Iterable of numpy arrays
            *args:    Passed to func1d after array arguments
            **kwargs: Passed to func1d as keyword arguments
        Based on function by Scott Wales:
            https://climate-cms.org/2019/07/29/multi-apply-along-axis.html
        """
        # Concatenate the input arrays along the calculation axis to make one big
        # array that can be passed in to `apply_along_axis`
        carrs = xr.concat(arrs, arrs[0].dims[axis])
        # We'll need to split the concatenated arrays up before we apply `func1d`,
        # here's the offsets to split them back into the originals
        offsets=[]
        start=0
        for i in range(len(arrs)-1):
            start += arrs[i].shape[axis]
            offsets.append(start)
        # The helper closure splits up the concatenated array back into the components of `arrs`
        # and then runs `func1d` on them
        def helperfunc(array, *args, **kwargs):
            arrs = np.split(array, offsets)
            return func1d(*[*arrs, *args], **kwargs)

        # Run `apply_along_axis` along the concatenated array
        return dsa.apply_along_axis(helperfunc, axis, carrs, *args, **kwargs)