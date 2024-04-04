# Imports

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.visualization import simple_norm
from astropy.io import fits
import scipy.constants as const

import matplotlib.pyplot as plt
import astropy.stats as stats
from photutils import Background2D, MedianBackground,aperture_photometry,CircularAperture,CircularAnnulus
from photutils.utils import calc_total_error
from photutils import Background2D, SExtractorBackground, MedianBackground
from uncertainties import ufloat
import pyphot

from astropy.io import fits
from astropy.wcs import WCS
from photutils import DAOStarFinder
from astropy import coordinates
from astropy import units as u
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from uncertainties import unumpy as unp

import math
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from photutils.psf import extract_stars
from astropy.visualization import simple_norm
from photutils import EPSFBuilder
import numpy.ma as ma
import scipy 

import os
import pickle
import lmfit
import pandas as pd

import emcee
import corner
from scipy.optimize import curve_fit
from lmfit import Model

from scipy.optimize import curve_fit
from scipy.constants import Stefan_Boltzmann

from astropy.modeling import models, fitting
##from astropy.stats import sigma_clip, SigmaClip, gaussian_fwhm_to_sigma

plt.rcParams.update({
    "text.usetex": False})

import astropy
import dateutil.parser

# Useful functions

def flambda(flux_jy, wavelength_micron):
    # Convert flux in jy to flux_lambda (cgs)
    # F(lambda = (3 x 10^18 F(nu)) / lambda^2), where lambda is in angstroms and F(nu) is in erg/(cm2 sec Hz)
    # from https://www.stsci.edu/itt/APT_help20/P2PI_ENG/ch03_targ_fixed11.html
    return flux_jy * 3e-5 / ((wavelength_micron*1e4)**2)

def fnu(flux_lambda, wavelength_micron):
    # Convert flux in jy to flux_lambda (cgs)
    # F(lambda = (3 x 10^18 F(nu)) / lambda^2), where lambda is in angstroms and F(nu) is in erg/(cm2 sec Hz)
    # from https://www.stsci.edu/itt/APT_help20/P2PI_ENG/ch03_targ_fixed11.html
    return flux_lambda / 3e-5  * ((wavelength_micron*1e4)**2)

def fnu_angstrom(flux_lambda, wavelength_angstrom):
    # Convert flux in jy to flux_lambda (cgs)
    # F(lambda = (3 x 10^18 F(nu)) / lambda^2), where lambda is in angstroms and F(nu) is in erg/(cm2 sec Hz)
    # from https://www.stsci.edu/itt/APT_help20/P2PI_ENG/ch03_targ_fixed11.html
    return flux_lambda / 3e-5  * ((wavelength_angstrom)**2)

def SB_law(T,R):
    SB_const = 5.670374419E-5
    R = R * 1E15
    return 4 * np.pi * R**2 * T**4 * SB_const    



def mjday(day):
    "Convert observation date such as 20171131 into Julian day"
    return [astropy.time.Time(dateutil.parser.parse(i)).jd - 2400000.5 for i in day]

def bin_dataframe(phot_df,filters,step=1):
    """
    Function takes in a df with columns mjd, mag, magerr and filtername
    Will do a nightly binning (optionally whatever bin size)
    Uncertainty will be SEM (care with small numbers of observations)
    """
    bins = np.arange(phot_df.mjd.min()-1, phot_df.mjd.max()+2, step=step)
    groups = phot_df.groupby(pd.cut(phot_df.mjd, bins))
    bin_dict={}
    
    for filtername in filters :
        print(filtername)
        bin_dict["mjd"] = np.array(groups.mean()["mjd"])
        binned_mags = np.array(groups.mean()[filtername])
        bin_dict[filtername] = binned_mags
        # For error, should be photometry error if bin size is one, and SEM if not.
        binned_errs_mean = np.array(groups.mean()[filtername+"_err"])
        binned_errs_count = np.array(groups.count()[filtername])
        binned_mags_std = np.array(groups.std()[filtername])
        err_list = []
        for mean, count, std in zip(binned_errs_mean,binned_errs_count,binned_mags_std):
#            print(mean,count,std)
            if count == 1:
                err_list.append(mean)
            elif count > 1 :
                err_list.append(std / np.sqrt(count))
            else :
                err_list.append(np.nan)
#        print(err_list)
        bin_dict[filtername+"_err"] = err_list
    out_df = pd.DataFrame(bin_dict)

    return out_df

def reshuffle_df(df,fil=["U","B","V","R","I","u","g","r","i","z","J","H","K","W1","W2","G","o","c"]):
    """
    Changes a df to the format where filternames are used as col heads
    Gives a df with a column with all the mags, and a filter column with the filternames
    """

    
    filters=[]; filters_err=[]; filternames = []
    for column_header in df.columns :
        if column_header in fil :
            filters.append(column_header)
            filters_err.append(column_header+"_err")
        
    new_autophot_df = df[["mjd"]+filters+filters_err]

    for i in filters :
        filternames.append("filtername_"+i)
        new_autophot_df["filtername_"+i]= new_autophot_df[i].where(new_autophot_df[i].isnull(),other=i)
        
    final_autophot_df = pd.DataFrame()
    final_autophot_df["mjd"] = df["mjd"]
    final_autophot_df["mag"]=new_autophot_df[filters].sum(1)
    final_autophot_df["mag_err"]=new_autophot_df[filters_err].sum(1)
    new_df = new_autophot_df[filternames[0]]
    for i in filternames[1:] :
        new_df = new_df.combine_first(new_autophot_df[i])
    final_autophot_df["filtername"]=new_df

    return final_autophot_df

class BB:
    """
    A class used to contain photometry, process it, and fit some blackbodies

    Can read in:
    - Reads a phot_df generated from my code that processes autophot results.
    - Use external interpolation and extrapolation before this code
    - uses a provided extinction.tbl (from https://irsa.ipac.caltech.edu/applications/DUST/)

        
        Attributes
    ----------
    TNS_name : str
        Name of source in TNS
    ZTF_name : str
        Name of source in ZTF
    ZTF_phot: datatable
        Input ZTF data taken from alerce (can be something else later), converted into a pandas table
    ATLAS_phot: datatable
        Input ATLAS data taken from ATLAS site, converted into a pandas table
    pos : np array, 1x2
        The ra and dec of the transient, in degrees

    """

    def __init__(self,z,dist,observations_df,interped_df=pd.DataFrame(),
                 TNS_name="No name given",all_AB=None,
                 extinction_file=None,path=None):
        
        if extinction_file :
            # Process the extinction file, taken from https://irsa.ipac.caltech.edu/applications/DUST/
                n_header = 19
                extinction_table = pd.read_fwf(extinction_file,skiprows=n_header, header=None,delimiter=None)
                # Create header froim first line of the file
                with open(extinction_file) as f:
                    lines = f.readlines()
                    for line in lines :
                        if line[1] == "E":
                            E_BV = float(line[23:29])
                        if line[0] == "|" :
                            header_columns = line.replace(' ', '').split('|')[1:-1]
                            break
                extinction_table.columns = header_columns
                extinction_table = extinction_table.set_index('Filter_name')
                self.extinction_table = extinction_table
                # Set E_BV with S&F, i.e with this formula
                self.E_BV = 0.86 *  E_BV
        lib = pyphot.get_library()              

        zp_AB = {'u': 867.6, 'g': 487.6, 'r': 282.9, 'i': 184.9, 'z': 98.6, 'y': 117.8, 'w': 303.8, 'Y': 117.8,
              'U': 847.1, 'B': 569.7, 'V': 362.8, 'R': 257.8, 'I': 169.2, 'G': 278.5, 'E': 278.5,
              'J': 72.2, 'H': 40.5, 'K': 23.5, 'UVW2': 2502.2, 'UVM2': 2158.3, 'UVW1': 1510.9, 'F': 4536.6, 'N': 2049.9,
              'o': 238.9, 'c': 389.3, 'W': 9.9, 'Q': 5.2, # These are taken from SUPERBOL
               "r_ZTF":lib['ZTF_r'].AB_zero_flux.magnitude * 1E11, "g_ZTF":lib['ZTF_g'].AB_zero_flux.magnitude * 1E11,
                 "i_ZTF":lib['ZTF_i'].AB_zero_flux.magnitude * 1E11, 
                 "W1":lib['WISE_RSR_W1'].AB_zero_flux.magnitude * 1E11,
                 "W2":lib['WISE_RSR_W2'].AB_zero_flux.magnitude * 1E11, # multiplicative factors to stay consistent with SUPERBOL
                 }
      
        zp_Vega = {'u': 351.1, 'g': 526.6, 'r': 242.6, 'i': 127.4, 'z': 49.5, 'y': 71.5, 'w': 245.7, 'Y': 71.5,
              'U': 396.5, 'B': 613.3, 'V': 362.7, 'R': 217.0, 'I': 112.6, 'G': 249.8, 'E': 249.8,
              'J': 31.3, 'H': 11.3, 'K': 4.3, 'UVW2': 523.7, 'UVM2': 457.9, 'UVW1': 408.4, 'F': 650.6, 'N': 445.0,
              'o': 193.1, 'c': 400.3, 'W': 0.818, 'Q': 0.242,
                 "W1":lib['WISE_RSR_W1'].Vega_zero_flux.magnitude * 1E11,
                 "W2":lib['WISE_RSR_W2'].Vega_zero_flux.magnitude * 1E11,
                   }

        default_sys = {'u': 'AB',  'g': 'AB', 'r': 'AB', 'i': 'AB',  'z': 'AB', 'y': 'AB', 'w': 'AB', 'Y': 'Vega',
             'U': 'AB',  'B': 'Vega', 'V': 'Vega', 'R': 'Vega', 'G': 'AB', 'E': 'AB', 'I': 'Vega',
             'J': 'Vega', 'H': 'Vega', 'K': 'Vega', 'UVW2': 'AB', 'UVM2': 'AB', 'UVW1': 'AB',  'F': 'AB',
             'N': 'AB', 'o': 'AB', 'c': 'AB', 'W': 'Vega', 'Q': 'Vega',"r_ZTF":"AB", "g_ZTF":"AB", "i_ZTF":"AB",
            "W1":"Vega","W2":"Vega",
                       }

        if all_AB :
            default_sys = {'u': 'AB',  'g': 'AB', 'r': 'AB', 'i': 'AB',  'z': 'AB', 'y': 'AB', 'w': 'AB', 'Y': 'AB',
             'U': 'AB',  'B': 'AB', 'V': 'AB', 'R': 'AB', 'G': 'AB', 'E': 'AB', 'I': 'AB',
             'J': 'AB', 'H': 'AB', 'K': 'AB', 'UVW2': 'AB', 'UVM2': 'AB', 'UVW1': 'AB',  'F': 'AB',
             'N': 'AB', 'o': 'AB', 'c': 'AB', 'W': 'AB', 'Q': 'AB',"r_ZTF":"AB", "g_ZTF":"AB", "i_ZTF":"AB",
            "W1":"AB","W2":"AB",
                       }             
        
        zp_Jy = {'u': lib['SDSS_u'].AB_zero_Jy.magnitude,
                 'g': lib['SDSS_g'].AB_zero_Jy.magnitude,
                 'r': lib['SDSS_g'].AB_zero_Jy.magnitude,
                 'i': lib['SDSS_g'].AB_zero_Jy.magnitude,
                 'z': lib['SDSS_g'].AB_zero_Jy.magnitude,
                 'g_ZTF': lib['SDSS_g'].AB_zero_Jy.magnitude,
                 'r_ZTF': lib['SDSS_g'].AB_zero_Jy.magnitude,
                 'UVW2': 3630.78, 'UVM2': 3630.78, 'UVW1': 3630.78,
                 "U":3630.78,
                 "o": 3630.7805477010024,
                "c": 3630.7805477010024,
                 "B": lib['GROUND_JOHNSON_B'].AB_zero_Jy.magnitude,
                 "V": lib['GROUND_JOHNSON_B'].AB_zero_Jy.magnitude,
                "J": lib['2MASS_J'].AB_zero_Jy.magnitude,
                 'H': lib['2MASS_H'].AB_zero_Jy.magnitude,
                 "K": lib['2MASS_Ks'].AB_zero_Jy.magnitude,
                 'W1': lib['WISE_RSR_W1'].AB_zero_Jy.magnitude,
                 "W2": lib['WISE_RSR_W2'].AB_zero_Jy.magnitude,
                 }

      
        wle = {'UVW2': 2030, 'UVM2': 2231, 'UVW1': 2634, 'u': 3560, 'U': 3600, 'B': 4380,
               'g': 4830, 'c': 5330, 'V': 5450, 'r': 6260,'R': 6410, 'G': 6730,'o': 6790, 
               'i': 7670, 'z': 8890, 'I': 7980, 'J': 12200, 'H': 16300,
               'K': 21900, 'F': 1516, 'N': 2267,  'w':5985,'y': 9600,  'Y': 9600,
               'W': 33526, 'Q': 46028,'W1': 34000, 'W2': 46000, "r_ZTF":6436.92, "g_ZTF":4804.82,"i_ZTF":7670
                }

        extco = {
         'u': 4.786,  'g': 3.587, 'r': 2.471, 'i': 1.798,  'z': 1.403, 'y': 1.228, 'w':2.762, 'Y': 1.228,
         'U': 4.744,  'B': 4.016, 'V': 3.011, 'R': 2.386, 'G': 2.216, 'I': 1.684, 'J': 0.813, 'H': 0.516,
         'K': 0.337, 'UVW2': 8.795, 'UVM2': 9.270, 'UVW1': 6.432,  'F': 8.054,  'N': 8.969, 'o': 2.185, 'c': 3.111,
         'W': 0.190, 'Q': 0.127,'W1': 0.190, 'W2': 0.127, 'g_ZTF': 3.587, 'r_ZTF': 2.471,'i_ZTF': 1.798,
                 }
        
        width = {#,'y': 9600,'w':5985,'Y': 9600,,'I': 7980,
             'g': 1450, 'r': 1480, 'i': 1710, 'z': 2000,'o': 2580, 'c': 2280,
         'B': 1000, 'V': 800, 'R': 1300,  'J': 1630, 'H': 2960,'K': 2800, 'W1':10000 ,'W2': 10000,
             'UVW2': 671, 'UVM2': 446, 'UVW1': 821,'U': 1800,'G': 6730,'u': 3560, 'g_ZTF': 1450, 'r_ZTF': 1480,'i_ZTF': 1710,
#         'S': 2030, 'D': 2231, 'A': 2634, 'F': 1516, 'N': 2267, 
#        'W': 33526, 'Q': 46028,
        }

        cols = {'u': 'dodgerblue', 'g': 'g', 'r': 'r', 'i': 'goldenrod', 'z': 'k', 'y': '0.5', 'w': 'firebrick',
            'Y': '0.5', 'U': 'slateblue', 'B': 'b', 'V': 'yellowgreen', 'R': 'crimson', 'I': 'chocolate',
            'G': 'salmon', 'E': 'salmon', 'J': 'darkred', 'H': 'orangered', 'K': 'saddlebrown',
            'UVW2': 'mediumorchid', 'UVM2': 'purple', 'UVW1': 'midnightblue',
            'F': 'hotpink', 'N': 'magenta', 'o': 'darkorange', 'c': 'cyan',
            'W1': 'forestgreen', 'W2': 'peru'}
        
        bandlist = 'FSDNAuUBgcVwrRoGEiIzyYJHKWQ'

        # Assuming here that the input is a nice interpolated/extrapolated/binned df.
        # Can later build these functions in, for now focusing on the BB part.
        self.autophot_df = observations_df
        self.interpolation_df = interped_df

            
        self.zp_AB = zp_AB
        self.zp_Vega = zp_Vega
        self.wle = wle
        self.default_sys = default_sys
        self.extco = extco 
        self.TNS_name = TNS_name
        self.dist = dist
        self.z = z
        self.zp_Jy = zp_Jy
        self.dist_factor=4*np.pi*((3.086e24*self.dist)**2)
        self.width = width
        self.path = path

    def process_SED(self,epoch,tol=1,E_BV=None,plot=None,logplot=None,expdate=None):
        """
        Process the phot_df to produce the SED that the BB fitting will be performed on.
        WE CORRECT FOR EXTINCTION, either using a given value, or the MW taken from an extinction.tbl
        Input a df generated from measurements, containing actual data (binned)
        Can also input a df generated from PISCOLA, which will only be queried if there are no
        observations for a filter at the requested epoch.
        
        Required:
        epoch: the date where we will take the SED, in MJD
        """
        
        observations_df = self.autophot_df
        interped_df = self.interpolation_df
        default_sys = self.default_sys
        
        if E_BV == None:
            print("Taking extinction of E(B-V) = " +str(self.E_BV)+ " from the MW, using the extinction.tbl file")
            E_BV = self.E_BV
            
        # convert Mpc to cm, since flux in erg/s/cm2/A
        dist = self.dist*3.086e24 
            
        SED_epoch_mask = [all(constraint) for constraint in zip(
                abs(observations_df['mjd']-epoch) < tol,\
                        )]
        

        
        # Check what filters have data given the daterange given
        filters=[]
        for column_header in observations_df.columns :
            if column_header in default_sys :
            	if len(observations_df[SED_epoch_mask][column_header].dropna()) != 0:
                    #print(column_header)
                    filters.append(column_header)
        # Check which filters we have interpolation for
        if not interped_df.empty :
            SED_epoch_mask_interp = [all(constraint) for constraint in zip(
                abs(interped_df['mjd']-epoch) < tol,\
                        )]        
            interped_filters=[]
            for column_header in interped_df.columns :
                if column_header in default_sys :

                    if len(interped_df[SED_epoch_mask_interp][column_header].dropna()) != 0:
                        print(column_header)
                        interped_filters.append(column_header)
        else :
            interped_filters = None
            
        self.filters = filters
        self.interped_filters = interped_filters
        
        observed_SED_df = pd.DataFrame({})
        observed_SED_df['mjd'] = observations_df["mjd"][SED_epoch_mask]
        #print(observed_SED_df)
        for filt in self.filters :
            observed_SED_df[filt] = observations_df[filt][SED_epoch_mask] - E_BV*self.extco[filt] # apply extinction
            observed_SED_df[filt+"_err"] = observations_df[filt+"_err"][SED_epoch_mask] # get error
            if default_sys[filt] == "AB":
                fref = (self.zp_AB[filt]*1e-11) * (1+self.z)
            if default_sys[filt] == "Vega":
                fref = (self.zp_Vega[filt]*1e-11) * (1+self.z)
            observed_SED_df[filt+"_flux"] = fref*10**(-0.4*observed_SED_df[filt]) #* 4*np.pi*dist**2
            observed_SED_df[filt+"_flux_err"] =  2.5/np.log(10) *  observed_SED_df[filt+"_flux"] * observed_SED_df[filt+"_err"]
            # also get the Jy fluxes
            # I am assuming AB for SDSS and Vega for JHK    
            if default_sys[filt] == "Vega":
                print("Vega + Jy is not implemented")
                observed_SED_df[filt+"_flux_Jy"] = fref_Jy*10**(-0.4*observed_SED_df[filt]) #* 4*np.pi*dist**2
                observed_SED_df[filt+"_flux_Jy_err"] =  2.5/np.log(10) *  observed_SED_df[filt+"_flux_Jy"] * observed_SED_df[filt+"_err"]
            else :
                fref_Jy = (self.zp_Jy[filt]) * (1+self.z)
                observed_SED_df[filt+"_flux_Jy"] = fref_Jy*10**(-0.4*observed_SED_df[filt]) #* 4*np.pi*dist**2
                observed_SED_df[filt+"_flux_Jy_err"] =  2.5/np.log(10) *  observed_SED_df[filt+"_flux_Jy"] * observed_SED_df[filt+"_err"]

        if self.interped_filters :
            SED_epoch_mask = [all(constraint) for constraint in zip(
                abs(interped_df['mjd']-epoch) < tol/2,\
                        )]
            interped_SED_df = pd.DataFrame({})
            interped_SED_df['mjd'] = interped_df["mjd"][SED_epoch_mask]
            for filt in self.interped_filters :
                #if filt not in interped_filters :
                #    continue
                interped_SED_df[filt] = interped_df[filt][SED_epoch_mask] - E_BV*self.extco[filt]
                interped_SED_df[filt+"_err"] = interped_df[filt+"_err"][SED_epoch_mask]
                if default_sys[filt] == "AB":
                    fref = (self.zp_AB[filt]*1e-11) * (1+self.z)
                if default_sys[filt] == "Vega":
                    fref = (self.zp_Vega[filt]*1e-11) * (1+self.z)
                interped_SED_df[filt+"_flux"] = fref*10**(-0.4*interped_SED_df[filt]) #* 4*np.pi*dist**2
                interped_SED_df[filt+"_flux_err"] =  2.5/np.log(10) *  interped_SED_df[filt+"_flux"] * interped_SED_df[filt+"_err"]
                # Also do Jy fluxes
                fref_Jy = (self.zp_Jy[filt]*1e-11) * (1+self.z)
                #observed_SED_df[filt+"_flux_Jy"] = fref_Jy*10**(-0.4*observed_SED_df[filt]) * 4*np.pi*dist**2
                #observed_SED_df[filt+"_flux_Jy_err"] =  2.5/np.log(10) *  observed_SED_df[filt+"_flux_Jy"] * observed_SED_df[filt+"_err"]

                observed_SED_df[filt+"_flux_Jy"] = ( 1E-6 * 10**((23.9-interped_SED_df[filt])/2.5) ) #* 4*np.pi*dist**2
                observed_SED_df[filt+"_flux_Jy_err"] =  2.5/np.log(10) *  observed_SED_df[filt+"_flux_Jy"] * observed_SED_df[filt+"_err"]
                
                self.interped_SED_df = interped_SED_df
        else :
            self.interped_SED_df = None

        if plot :
            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(16,12))
            for filt in self.filters :
                if observed_SED_df[filt + "_flux"].dropna(how='all').empty :
                    continue
                else :
                    fl = np.mean(observed_SED_df[filt+"_flux"].dropna())
                    fl_err = np.mean(observed_SED_df[filt+"_flux_err"].dropna())

                    fl_Jy = np.mean(observed_SED_df[filt+"_flux_Jy"].dropna())
                    fl_Jy_err = np.mean(observed_SED_df[filt+"_flux_Jy_err"].dropna())
                    
                if logplot :
                    ax1.errorbar(self.wle[filt]/(1+self.z),np.log10(fl),yerr=fl_err/fl*np.log10(2.5),
			xerr=self.width[filt]/2,capsize=5,capthick=1,
			marker="o",markersize=10,label=filt,ls="",mec="k")
                    ax2.errorbar(self.wle[filt]/(1+self.z),
                                 np.log10(fl_Jy),
                                 yerr=fl_Jy_err/fl_Jy*np.log10(2.5),
			xerr=self.width[filt]/2,capsize=5,capthick=1,
			marker="o",markersize=10,label=filt,ls="",mec="k")
                else :
                    ax1.errorbar(self.wle[filt]/(1+self.z),fl,yerr=fl_err,
                                xerr=self.width[filt]/2,capsize=5,capthick=1,
			marker="o",markersize=10,label=filt,ls="",mec="k")
                    ax2.errorbar(self.wle[filt]/(1+self.z),
                                 fl_Jy,
                                 yerr=fl_Jy_err,
                                xerr=self.width[filt]/2,capsize=5,capthick=1,
			marker="o",markersize=10,label=filt,ls="",mec="k")


                    
                ax1.legend();ax2.legend()

            if self.interped_filters :
                for filt in self.interped_filters:
                    #if filt not in self.filters :
                        if interped_SED_df[filt + "_flux"].dropna(how='all').empty :
                            continue
                        else :
                            fl = np.mean(interped_SED_df[filt+"_flux"].dropna())
                            fl_err = np.mean(interped_SED_df[filt+"_flux_err"].dropna())
                        if logplot :
                            ax1.errorbar(self.wle[filt]/(1+self.z),np.log10(fl),yerr=fl_err/fl*np.log10(2.5),
                                xerr=self.width[filt]/2,capsize=5,capthick=1,
                                marker="D",markersize=10,label=filt,ls="",mec="k")
                            ax2.errorbar(self.wle[filt]/(1+self.z),
                                         np.log10(fl_Jy),
                                         yerr=fl_Jy_err/fl_Jy*np.log10(2.5),
                                xerr=self.width[filt]/2,capsize=5,capthick=1,
                                marker="D",markersize=10,label=filt,ls="",mec="k")
                        else :
                            ax1.errorbar(self.wle[filt]/(1+self.z),fl,yerr=fl_err,
                                    xerr=self.width[filt]/2,capsize=5,capthick=1,
                                    marker="D",markersize=10,label=filt,ls="",mec="k")
                            ax2.errorbar(self.wle[filt]/(1+self.z),
                                         fl_Jy,
                                         yerr=fl_Jy_err,
                                    xerr=self.width[filt]/2,capsize=5,capthick=1,
                                    marker="D",markersize=10,label=filt,ls="",mec="k")
                ax1.legend();ax2.legend()

                

                
            ax1.set_xlabel("Wavelength (Å)"); ax2.set_xlabel("Wavelength (Å)")
            ax1.set_ylabel("Flux (erg/s/cm**2/Å)"); ax2.set_ylabel("Flux (Jy)")

        self.observed_SED_df = observed_SED_df
        self.tol=tol
        self.epoch = epoch
        if expdate :
            self.phase = (epoch - expdate)/(1+self.z)
        else :
            self.phase=epoch



    def fit_BB(self,method="MCMC",scale=[1e-10,1e-3],temp=[1e3,1e5],
               #A=[1E-4,1E-1],
               scale_2 = None, temp_2 = None,
               nwalkers = 300, ndim= 2, nsample = 100,burnin=None,
               error_method = "percentiles",
               filepath="/home/treynolds/temp/",
               exclude_filters = [],
               BB_count=1,
               interp=None,
               save_loc=None,
               log=None,
               modified=None,
               plot_Jy=None,
               power_law =None,
               opacity_table=None
               ):

        if BB_count==2:
                ndim=4
                scale_lower_1,scale_upper_1 = scale
                T_lower_1,T_upper_1 = temp
                if not scale_2 :    
                    scale_lower_2,scale_upper_2 = [0.1,100]
                else :
                    scale_lower_2,scale_upper_2 = scale_2                
                if not temp_2 :    
                    T_lower_2,T_upper_2 = [100,3000]
                else :
                    T_lower_2,T_upper_2 = temp_2


        else :
                scale_lower,scale_upper = scale
                T_lower,T_upper = temp
        
        epoch=self.epoch
        dist= self.dist*3.086e24  # should be cm
        dist_scale = 4*np.pi*dist**2
        # Maybe fiddle with this. The idea is to discard the early samples, so the walkers represent
        # the results properly.
        if not burnin :
            burnin = int(nsample * 0.5)
        filters = []; wl=[];fl=[];fl_err=[]; fl_Jy=[];fl_Jy_err=[]
        # might need to rewrite to deal with the nans
        filters_exclude = []; wl_exclude=[];fl_exclude=[];fl_err_exclude=[]; fl_Jy_exclude=[];fl_Jy_err_exclude=[]
        
        bins = [epoch-self.tol, epoch+self.tol]
        groups = self.observed_SED_df.groupby(pd.cut(self.observed_SED_df.mjd, bins))

        for filtername in self.filters :
            if filtername in exclude_filters :
                wl_val = self.wle[filtername]
                try :
                    fl_val = groups.mean()[filtername+"_flux"].dropna().iloc[0]
                    fl_err_val = groups.mean()[filtername+"_flux_err"].dropna().iloc[0]
                    fl_Jy_val = groups.mean()[filtername+"_flux_Jy"].dropna().iloc[0]
                    fl_Jy_err_val =  groups.mean()[filtername+"_flux_Jy_err"].dropna().iloc[0]
                
                    wl_exclude.append(wl_val)
                    filters_exclude.append(filtername)
                    fl_exclude.append(fl_val)
                    fl_err_exclude.append(fl_err_val)
                    fl_Jy_exclude.append(fl_Jy_val)
                    fl_Jy_err_exclude.append(fl_Jy_err_val)
                    interp_flag = False
                except :
                    print("failed for " + filtername)
                    interp_flag = True                
            else :
                wl_val = self.wle[filtername]
                try :
                    fl_val = groups.mean()[filtername+"_flux"].dropna().iloc[0]
                    fl_err_val = groups.mean()[filtername+"_flux_err"].dropna().iloc[0]
                    fl_Jy_val = groups.mean()[filtername+"_flux_Jy"].dropna().iloc[0]
                    fl_Jy_err_val =  groups.mean()[filtername+"_flux_Jy_err"].dropna().iloc[0]
                    
                    wl.append(wl_val)
                    filters.append(filtername)
                    fl.append(fl_val)
                    fl_err.append(fl_err_val)
                    fl_Jy.append(fl_Jy_val)
                    fl_Jy_err.append(fl_Jy_err_val)
                    interp_flag = False
                except :
                    print("failed for " + filtername)
                    interp_flag = True
                    
                if interp_flag and interp:
                    if filtername in self.interped_filters :
                        fl_val = self.interped_SED_df[filtername+"_flux"][self.interped_SED_df[filtername].dropna().index[0]]
                        fl_err_val = self.interped_SED_df[filtername+"_flux_err"][self.interped_SED_df[filtername+"_err"].dropna().index[0]]
                        fl_Jy_val = self.interped_SED_df[filtername+"_flux_Jy"][self.interped_SED_df[filtername].dropna().index[0]]
                        fl_Jy_err_val =  self.interped_SED_df[filtername+"_flux_Jy_err"][self.interped_SED_df[filtername+"_err"].dropna().index[0]]
                        wl.append(wl_val)
                        fl.append(fl_val)
                        fl_err.append(fl_err_val)
                        fl_Jy.append(fl_Jy_val)
                        fl_Jy_err.append(fl_Jy_err_val)

        if interp:
            for filtername in self.interped_filters :
                print("interpolating..."+filtername)
                wl_val = self.wle[filtername]
                fl_val = self.interped_SED_df[filtername+"_flux"][self.interped_SED_df[filtername].dropna().index[0]]
                fl_err_val = self.interped_SED_df[filtername+"_flux_err"][self.interped_SED_df[filtername+"_err"].dropna().index[0]]
                fl_Jy_val = self.interped_SED_df[filtername+"_flux_Jy"][self.interped_SED_df[filtername].dropna().index[0]]
                fl_Jy_err_val =  self.interped_SED_df[filtername+"_flux_Jy_err"][self.interped_SED_df[filtername+"_err"].dropna().index[0]]
                filters.append(filtername)
                wl.append(wl_val)
                fl.append(fl_val)
                fl_err.append(fl_err_val)
                fl_Jy.append(fl_Jy_val)
                fl_Jy_err.append(fl_Jy_err_val)
                print(wl_val,fl_val,fl_err_val)

        # wl in angstroms, f_l in erg/s/cm**2/ang, f_nu in Jy,  Temp in K

        #convert to rest frame
        wl = np.array(wl) /(1+self.z)
        wl_exclude = np.array(wl_exclude) /(1+self.z)
        c = 2.99792458E8
        freq = c / (wl * 1E-10)
        print(filters, wl, fl, fl_err, fl_Jy, fl_Jy_err, scale,temp,dist)
        self.fl_Jy = fl_Jy
        self.fl_Jy_err = fl_Jy_err


        if opacity_table :
            print("using the opacity file...")
            opacity_files = {
                    "multiple_0.25": self.path + "/opacity_files/multiple_size/kappanu_abs_c025.dat",
                    "multiple_1":self.path + "/opacity_files/multiple_size/kappanu_abs_c1.dat",
                    "multiple_0.5":self.path + "/opacity_files/multiple_size/kappanu_abs_c05.dat",
                    "multiple_0.1":self.path + "/opacity_files/multiple_size/kappanu_abs_c01.dat",
                    "multiple_0.05":self.path + "/opacity_files/multiple_size/kappanu_abs_c005.dat",
                    "multiple_0.01":self.path + "/opacity_files/multiple_size/kappanu_abs_c001.dat",
                    "single_1":self.path + "/opacity_files/single_size/kappanu_abs_c1.dat",
                    "single_0.5":self.path + "/opacity_files/single_size/kappanu_abs_c05.dat",
                    "single_0.1":self.path + "/opacity_files/single_size/kappanu_abs_c01.dat",
                    "single_0.05":self.path + "/opacity_files/single_size/kappanu_abs_c005.dat",
                    "single_0.01":self.path + "/opacity_files/single_size/kappanu_abs_c001.dat",
                    }

            valid_opacity_files = list(opacity_files.keys())
            err_message = f"Invalid file. Choose between:basic,..."
            assert opacity_table in valid_opacity_files, err_message
            opacity_file_df = pd.read_csv(opacity_files[opacity_table],sep=" ",index_col=False,names=["frequency","opacity"])
            opacity_dict = get_opacities(wl,opacity_file_df,doprint=True)
            self.opacity_df = opacity_file_df
            #assert kernel2 in valid_kernels, err_message
            if BB_count == 1:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, opacity_ln_likelihood, 
                                            args=[wl, fl_Jy, fl_Jy_err, scale, temp, dist,opacity_dict])
                pos0 = np.transpose([np.random.uniform(low=scale_lower, high=scale_upper, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower, high=T_upper, size=(nwalkers,)),\
                                     ])            
            else :
                scale_1 = scale; temp_1 = temp
                sampler = emcee.EnsembleSampler(nwalkers, ndim, opacity_ln_likelihood_2, 
                                            args=[wl, fl_Jy, fl_Jy_err,scale_1,scale_2,temp_1,temp_2, dist,opacity_dict])
                pos0 = np.transpose([np.random.uniform(low=scale_lower_1, high=scale_upper_1, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower_1, high=T_upper_1, size=(nwalkers,)),\
                                 np.random.uniform(low=scale_lower_2, high=scale_upper_2, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower_2, high=T_upper_2, size=(nwalkers,))     
                                 ]) 
            

        elif power_law == True :
            scale = [0,1]
            q = [1E-3,1]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood_power_law, 
                                            args=[wl, fl_Jy, fl_Jy_err,scale,q, dist])
            pos0 = np.transpose([np.random.uniform(low=scale[0], high=scale[1], size=(nwalkers,)),\
                                 np.random.uniform(low=q[0], high=q[1], size=(nwalkers,)),\
                                     ])        
        elif BB_count == 1:

            if modified :
                # For the modified BBs, want to fit in frequency space 
                sampler = emcee.EnsembleSampler(nwalkers, ndim, modified_ln_likelihood_wav, 
                                            args=[wl, fl_Jy, fl_Jy_err,scale,temp, dist])
                pos0 = np.transpose([np.random.uniform(low=scale_lower, high=scale_upper, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower, high=T_upper, size=(nwalkers,)),\
                                     ])
            else :
                sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood_wav, 
                                            args=[wl, fl, fl_err,scale,temp, dist])
                pos0 = np.transpose([np.random.uniform(low=scale_lower, high=scale_upper, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower, high=T_upper, size=(nwalkers,))])
             
        elif BB_count == 2:

            if modified :
                scale_1 = scale; temp_1 = temp
                if not scale_2 :
                    scale_2 = [0.1,100];
                if not temp_2 : 
                    temp_2 = [100,3000]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, modified_ln_likelihood_wav_2, 
                                            args=[wl, fl_Jy, fl_Jy_err,scale_1,scale_2,temp_1,temp_2, dist])
                pos0 = np.transpose([np.random.uniform(low=scale_lower_1, high=scale_upper_1, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower_1, high=T_upper_1, size=(nwalkers,)),\
                                 np.random.uniform(low=scale_lower_2, high=scale_upper_2, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower_2, high=T_upper_2, size=(nwalkers,))     
                                 ])
            else:
                scale_1 = scale; temp_1 = temp
                if not scale_2 :
                    scale_2 = [0.1,100];
                if not temp_2 : 
                    temp_2 = [100,3000]
                sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_likelihood_wav_2, 
                                            args=[wl, fl, fl_err,scale_1,scale_2,temp_1,temp_2, dist])
                pos0 = np.transpose([np.random.uniform(low=scale_lower_1, high=scale_upper_1, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower_1, high=T_upper_1, size=(nwalkers,)),\
                                 np.random.uniform(low=scale_lower_2, high=scale_upper_2, size=(nwalkers,)),\
                                 np.random.uniform(low=T_lower_2, high=T_upper_2, size=(nwalkers,))     
                                 ])



            

            # Starting positions predetermined with small perturbations
        #    p0 = np.array([0.1, 1000])                                       # Starting positon for walkers
        #    pos0 = [p0 + 0.0001 * np.random.randn(ndim) for j in range(nwalkers)]     # Perturb starting positions
            
        # Starting positions across flat prior range
        # Starting positions drawn from normal distribution
        #    pos0 = np.transpose([np.random.normal(loc=scale_init, scale=0.2*scale_init, size=nwalkers),\
        #                         np.random.normal(loc=temp_init, scale=0.2*temp_init, size=nwalkers)])
            
            #self.temp = (sampler,pos0,nsample)
        sampler.run_mcmc(pos0, nsample)  ## Run the sampler nsample times
        tau = sampler.get_autocorr_time(quiet=True)
        print("autocorr time: ",tau , "\nShould run chain for: ",
              np.max(tau*50), " samples and burnin should be ", 5*np.max(tau))
        
        ### Make a trace plot
        label_names=[r"$Radius (10^{15}cm)$", r"$T (K)$", "$Radius_{2} (10^{15}cm)$", r"$T_{2}(K)$"]
        # Should re-write to "get_chain"
        data = sampler.chain # *not* .flatchain
        fig, axs = plt.subplots(ndim, 1)
        for i in range(ndim): # free MCMC_run(wl,fl, fl_err, dist, epoch)parameters
            for j in range(int(np.floor(nwalkers/5))): # walkers
                axs[i].plot( np.arange(nsample), data[j,:,i],lw=0.5)
                axs[i].set_ylabel(label_names[i],fontsize=6)
                # x-axis is just the current iteration number
            fig.savefig(filepath + str(np.round(epoch,2)) +  "_trace.pdf",bbox_inches='tight')

        ### Make a corner plot
        figure = corner.corner(
            sampler.get_chain(discard=150, thin=15, flat=True),
                                 labels=label_names,
                                 quantiles=[0.16, 0.5, 0.84])
                                 # Discard first 300 samples (burn in)
        
        figure.savefig(filepath +"corner_" + str(np.round(epoch,2))  + ".pdf",
                           bbox_inches='tight')


        self.sampler =sampler
        samples = sampler.chain[:, burnin:, :].reshape(-1, 2) 
        self.samples = samples
        
        error_calc_methods = [
            "sigma_clipped_sigma",
            "percentiles"
            ]
        err_message = f"Invalid error calculation method. Choose between: {error_calc_methods}"
        assert error_method in error_calc_methods, err_message
        
        if BB_count == 2 :
        
            scale_best_1 = np.median(sampler.chain[:, burnin:, 0])
            T_best_1     = np.median(sampler.chain[:, burnin:, 1])   
            scale_best_2 = np.median(sampler.chain[:, burnin:, 2])
            T_best_2    = np.median(sampler.chain[:, burnin:, 3])   # Averaging over sampler chains and walkers
        
            # One (probably) robust measurement is the 3 sigma clipped uncertainty
            if error_method == "sigma_clipped_sigma" :
                scale_1_err = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=3)),
                               np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=3)))
                T_1_err     = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=3)),
                                np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=3)))
                scale_2_err = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 2],sigma=3)),
                                np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 2],sigma=3)))
                T_2_err     = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 3],sigma=3)),
                                np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 3],sigma=3)))
            #  Another robust measurement are 16th and 84th percentiles
            elif error_method == "percentiles" :
                scale_1_err = abs(np.percentile(sampler.chain[:,burnin:,0],[16, 84]) - scale_best_1)
                T_1_err = abs(np.percentile(sampler.chain[:,burnin:,1],[16, 84])- T_best_1)
                scale_2_err = abs(np.percentile(sampler.chain[:,burnin:,2],[16, 84])- scale_best_2)
                T_2_err = abs(np.percentile(sampler.chain[:,burnin:,3],[16, 84])- T_best_2)


            self.BB_results={
                            "BB_1":{"T":(T_best_1,T_1_err),"scale":(scale_best_1,scale_1_err)},
                            "BB_2":{"T":(T_best_2,T_2_err),"scale":(scale_best_2,scale_2_err)}
                          }
            print("Best Temps: ", (T_best_1,T_1_err), "K and ", (T_best_2,T_2_err))
            print("Best scale: ", (scale_best_1,scale_1_err), "x 10^15 cm and", (scale_best_2,scale_2_err))
            #print("errors:",np.percentile(sampler.chain[:,burnin:,0],[16, 84]))

            if opacity_table :
                if plot_Jy :
                    print(self.BB_results,wl,np.array(fl_Jy)*dist_scale,np.array(fl_Jy_err)*dist_scale)
                    ax = bestplot_opacity_2(
                                self.BB_results,wl,np.array(fl_Jy)*dist_scale,np.array(fl_Jy_err)*dist_scale,
                                phase=self.phase,save_loc=save_loc,dist_factor=self.dist_factor,
            			log=log,opacity_df=opacity_file_df,
                                excluded_wl=wl_exclude,
                                excluded_flux=np.array(fl_Jy_exclude)*dist_scale,
                                excluded_flux_err=np.array(fl_Jy_err_exclude)*dist_scale,
            			)    

            elif not modified :
                if plot_Jy :
                    ax = bestplot_modified_2(
                                self.BB_results,wl,np.array(fl_Jy)*dist_scale,np.array(fl_Jy_err)*dist_scale,
                                phase=self.phase,save_loc=save_loc,dist_factor=self.dist_factor,
            			log=log,q=0,
            			)                    
                else :
                    ax = bestplot_2(
                                self.BB_results,wl,np.array(fl)*dist_scale,np.array(fl_err)*dist_scale,
                                phase=self.phase,save_loc=save_loc,dist_factor=self.dist_factor,
            			log=log,
            			)
            else :
                ax = bestplot_modified_2(
                                self.BB_results,wl,np.array(fl_Jy)*dist_scale,np.array(fl_Jy_err)*dist_scale,
                                phase=self.phase,save_loc=save_loc,dist_factor=self.dist_factor,
            			log=log,
            			)

        elif power_law == True :
            scale_best_1 = np.median(sampler.chain[:, burnin:, 0])
            q_best_1     = np.median(sampler.chain[:, burnin:, 1])
            
            if error_method == "sigma_clipped_sigma" :
                scale_err = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=3)),
                             np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=3)))
                q_err     = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=3)),
                             np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=3)))     
            elif error_method == "percentiles" :
                scale_err    = abs(np.percentile(sampler.chain[:,burnin:,0],[16, 84]) - scale_best_1)  
                q_err        = abs(np.percentile(sampler.chain[:,burnin:,1],[16, 84])- q_best_1)      
            
            print("Power_law_fit_results",scale_best_1,q_best_1)

            self.results={
                            "power_law":{"q":(q_best_1,q_err),"scale":(scale_best_1,scale_err)}
                          }
            
            bestplot_power_law(q_best_1,q_err,scale_best_1,scale_err,wl,np.array(fl_Jy)*dist_scale,np.array(fl_Jy_err)*dist_scale,
                               epoch,save_loc=save_loc,dist_factor=self.dist_factor,
                         log=log,)
        else :
            scale_best_1 = np.median(sampler.chain[:, burnin:, 0])
            T_best_1     = np.median(sampler.chain[:, burnin:, 1])
            
            if error_method == "sigma_clipped_sigma" :
                scale_err = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=3)),
                             np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 0],sigma=3)))
                T_err     = (np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=3)),
                             np.std(astropy.stats.sigma_clip(sampler.chain[:, burnin:, 1],sigma=3)))     
            elif error_method == "percentiles" :
                scale_err    = abs(np.percentile(sampler.chain[:,burnin:,0],[16, 84]) - scale_best_1)  
                T_err        = abs(np.percentile(sampler.chain[:,burnin:,1],[16, 84])- T_best_1)               
            self.results={
                            "BB":{"T":(T_best_1,T_err),"scale":(scale_best_1,scale_err)}
                          }
            
            print("Best Temp: ", T_best_1)
            print("Best scale: ", scale_best_1)

            print(T_best_1,scale_best_1,wl,fl,fl_err,epoch)
            if not modified :
                bestplot(T_best_1,T_err,scale_best_1,scale_err,wl,np.array(fl)*dist_scale,np.array(fl_err)*dist_scale,epoch,save_loc=save_loc,dist_factor=self.dist_factor,
                         log=log,
                         )
            else :
                bestplot_modified(T_best_1,T_err,scale_best_1,scale_err,wl,np.array(fl_Jy)*dist_scale,np.array(fl_Jy_err)*dist_scale,epoch,dist_factor=self.dist_factor,
                                  save_loc=save_loc,log=log,
                                  )

        print("Mean acceptance fraction: {0:.3f}"
                        .format(np.mean(sampler.acceptance_fraction)))
        self.plotting_filepath = filepath
        self.MCMC_inputs = (wl,fl,fl_err,fl_Jy,fl_Jy_err) 
        self.wl = wl

# stuff for the MCMC

def get_opacities(wav,opacity_file_df,doprint=None):
    opacities_dict = {}
    c = 2.99792458E10
    for wl in wav:
        freq = c  / (wl*1E-8) # Hz
        opacities_dict[wl] = opacity_file_df.iloc[(opacity_file_df['frequency'] - freq).abs().argsort()[:1]]["opacity"].iloc[0] #/ dist_factor**2
    if doprint:
        print(opacities_dict)
    return opacities_dict 
        

def bestplot_opacity_2(results,wl,flux,flux_err,phase,save_loc=None,dist_factor=1,log=None,opacity_df=None,
                        excluded_wl=np.array([]),excluded_flux=None,excluded_flux_err=None):      
    plt.clf()
    c  = const.c

    T_1 = results["BB_1"]["T"][0]; T_2 = results["BB_2"]["T"][0]
    T_1_errs = results["BB_1"]["T"][1]; T_2_errs = results["BB_2"]["T"][1]
    R_1 = results["BB_1"]["scale"][0]; R_2 = results["BB_2"]["scale"][0]
    R_1_errs = results["BB_1"]["scale"][1]; R_2_errs = results["BB_2"]["scale"][1]
    
    fig = plt.figure(figsize=(8,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 

    ax1.errorbar(wl,np.array(flux)/dist_factor, np.array(flux_err)/dist_factor,
                 linestyle='',marker = 'o',color = 'xkcd:red', label='Data',mec="k")
    if excluded_wl.any() :
        ax1.errorbar(excluded_wl,np.array(excluded_flux)/dist_factor, np.array(excluded_flux_err)/dist_factor,
                 linestyle='',marker = 'o',color = 'xkcd:blue', label='Excluded data',mec="k")
    
    x = np.linspace(np.min(wl),np.max(wl),1000)
    y = []
    for item in x :
        y.append(c/item)
    opacity_dict = get_opacities(x,opacity_df)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,modified_bbody_absorbed(x,T_1,R_1,q=0)/dist_factor,
             label='BB_1, T= '+str(np.round(T_1,1))+"$_{-"+str(np.round(T_1_errs[0],0))+"}^{+"+str(np.round(T_1_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R_1,2))+"$_{-"+str(np.round(R_1_errs[0],2))+"}^{+"+str(np.round(R_1_errs[1],2))+"}$x10$^{15}$ cm")
    ax1.plot(x,bbody_fit_with_opacity(x,T_2,R_2,opacity_dict)/dist_factor,
             label='BB_2, T= '+str(np.round(T_2,1))+"$_{-"+str(np.round(T_2_errs[0],0))+"}^{+"+str(np.round(T_2_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R_2,2))+"$^{+"+str(np.round(R_2_errs[0],2))+"}_{-"+str(np.round(R_2_errs[1],2))+"}$x10$^{15}$ cm")
    ax1.plot(x,bbody_fit_with_opacity_2(x,T_1,T_2,R_1,R_2,opacity_dict)/dist_factor, label='BB sum')
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    #ax1.axes.set_xlim(wl[0]-1E3,wl[-1]+1E3)    
    #Here I am hacking for a specific plot, redo later!
    #ax1.set_yscale('log')
    #ax1.set_ylim(1E37,1E40)
    #Comment one out
    ax1.set_title("BB fit at:"+str(np.round(phase,0))+"d",fontsize=18,y=0)
    
    if log == True:
        ax1.axes.set_yscale("log")
        print(np.min(flux)/10/dist_factor,np.max(flux)/10/dist_factor)
        ax1.set_ylim(np.min(flux)/10/dist_factor,np.max(flux)*10/dist_factor)

    
    if save_loc != None :
        plt.savefig(save_loc,dpi=300,bbox_inches="tight",facecolor="white")
    plt.show()


def bestplot(T,T_errs,R,R_errs,wl,flux,flux_err,phase,save_loc=None,log=None,dist_factor=1):      
    plt.clf()
    c  = const.c

    fig = plt.figure(figsize=(8,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 

    ax1.errorbar(wl,np.array(flux)/dist_factor, np.array(flux_err)/dist_factor, linestyle='',marker = 'o',color = 'xkcd:red', label='Data')

    x = np.linspace(np.min(wl),np.max(wl),1000)
    y = []
    for item in x :
        y.append(c/item)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,bbody_absorbed(x,T,R)/dist_factor,
             label='BB, T= '+str(np.round(T,1))+"$^{+"+str(np.round(T_errs[0],0))+"}_{-"+str(np.round(T_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R,2))+"$^{+"+str(np.round(R_errs[0],2))+"}_{-"+str(np.round(R_errs[1],2))+"}$x10$^{15}$ cm")      
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    #ax1.axes.set_xlim(wl[0]-2E3,wl[-1]+2E3)    
    
    #Comment one out
    if log == True:
    	ax1.axes.set_yscale("log")
    	
    if save_loc != None :
        plt.savefig(save_loc,dpi=300,bbox_inches="tight",facecolor="white")
    plt.show()

def bestplot_power_law(q,q_errs,scale,scale_errs,wl,flux_Jy,flux_Jy_err,phase,
                       save_loc=None,log=None,dist_factor=1):      
    plt.clf()
    c  = const.c

    fig = plt.figure(figsize=(8,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 

    ax1.errorbar(wl,np.array(flux_Jy)/dist_factor, np.array(flux_Jy_err)/dist_factor,
                 linestyle='',marker = 'o',color = 'xkcd:red', label='Data')

    x = np.linspace(np.min(wl),np.max(wl),1000)
    y = []
    for item in x :
        y.append(c/item)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,power_law(x,q,scale),#/dist_factor,
             label='Power law, q= '+str(np.round(q,1))+"$^{+"+str(np.round(q_errs[0],0))+"}_{-"+str(np.round(q_errs[1],0))+"}$ scale = " + \
             str(np.round(scale,2))+"$^{+"+str(np.round(scale_errs[0],2))+"}_{-"+str(np.round(scale_errs[1],2))+"}$")      
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    #ax1.axes.set_xlim(wl[0]-2E3,wl[-1]+2E3)    
    
    #Comment one out
    if log == True:
    	ax1.axes.set_yscale("log")
    	
    if save_loc != None :
        plt.savefig(save_loc,dpi=300,bbox_inches="tight",facecolor="white")
    plt.show()

def bestplot_modified(T,T_errs,R,R_errs,wl,flux,flux_err,phase,save_loc=None,log=None,dist_factor=1):      
    plt.clf()
    c  = const.c

    fig = plt.figure(figsize=(8,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 

    ax1.errorbar(wl,np.array(flux)/dist_factor, np.array(flux_err)/dist_factor, linestyle='',marker = 'o',color = 'xkcd:red', label='Data')

    x = np.linspace(np.min(wl),np.max(wl),1000)
    y = []
    for item in x :
        y.append(c/item)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,modified_bbody_absorbed(x,T,R)/dist_factor,
             label='BB, T= '+str(np.round(T,1))+"$^{+"+str(np.round(T_errs[0],0))+"}_{-"+str(np.round(T_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R,2))+"$^{+"+str(np.round(R_errs[0],2))+"}_{-"+str(np.round(R_errs[1],2))+"}$x10$^{15}$ cm")      
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    #ax1.axes.set_xlim(wl[0]-2E3,wl[-1]+2E3)    
    
    #Comment one out
    if log == True:
    	ax1.axes.set_yscale("log")
    	
    if save_loc != None :
        plt.savefig(save_loc,dpi=300,bbox_inches="tight",facecolor="white")
    plt.show()
    

def bestplot_2(results,wl,flux,flux_err,phase,save_loc=None,dist_factor=1,log=None):      
    plt.clf()
    c  = const.c

    T_1 = results["BB_1"]["T"][0]; T_2 = results["BB_2"]["T"][0]
    T_1_errs = results["BB_1"]["T"][1]; T_2_errs = results["BB_2"]["T"][1]
    R_1 = results["BB_1"]["scale"][0]; R_2 = results["BB_2"]["scale"][0]
    R_1_errs = results["BB_1"]["scale"][1]; R_2_errs = results["BB_2"]["scale"][1]
    
    fig = plt.figure(figsize=(8,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 

    ax1.errorbar(wl,np.array(flux)/dist_factor, np.array(flux_err)/dist_factor, linestyle='',marker = 'o',color = 'xkcd:red', label='Data')

    x = np.linspace(np.min(wl),np.max(wl),1000)
    y = []
    for item in x :
        y.append(c/item)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,bbody_absorbed(x,T_1,R_1)/dist_factor,
             label='BB_1, T= '+str(np.round(T_1,1))+"$_{-"+str(np.round(T_1_errs[0],0))+"}^{+"+str(np.round(T_1_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R_1,2))+"$_{-"+str(np.round(R_1_errs[0],2))+"}^{+"+str(np.round(R_1_errs[1],2))+"}$x10$^{15}$ cm")
    ax1.plot(x,bbody_absorbed(x,T_2,R_2)/dist_factor,
             label='BB_2, T= '+str(np.round(T_2,1))+"$_{-"+str(np.round(T_2_errs[0],0))+"}^{+"+str(np.round(T_2_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R_2,2))+"$^{+"+str(np.round(R_2_errs[0],2))+"}_{-"+str(np.round(R_2_errs[1],2))+"}$x10$^{15}$ cm")
    ax1.plot(x,bbody_absorbed_2(x,T_1,T_2,R_1,R_2)/dist_factor, label='BB sum')
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    #ax1.axes.set_xlim(wl[0]-1E3,wl[-1]+1E3)    
    #Here I am hacking for a specific plot, redo later!
    #ax1.set_yscale('log')
    #ax1.set_ylim(1E37,1E40)
    #Comment one out
    ax1.set_title("BB fit at:"+str(np.round(phase,0))+"d",fontsize=18,y=0)
    
    if log == True:
        ax1.axes.set_yscale("log")
        print(np.min(flux)/10/dist_factor,np.max(flux)/10/dist_factor)
        ax1.set_ylim(np.min(flux)/10/dist_factor,np.max(flux)*10/dist_factor)

    
    if save_loc != None :
        plt.savefig(save_loc,dpi=300,bbox_inches="tight",facecolor="white")
    plt.show()

def bestplot_modified_2(results,wl,flux,flux_err,phase,save_loc=None,dist_factor=1,log=None,q=1.8):      
    plt.clf()
    c  = const.c

    T_1 = results["BB_1"]["T"][0]; T_2 = results["BB_2"]["T"][0]
    T_1_errs = results["BB_1"]["T"][1]; T_2_errs = results["BB_2"]["T"][1]
    R_1 = results["BB_1"]["scale"][0]; R_2 = results["BB_2"]["scale"][0]
    R_1_errs = results["BB_1"]["scale"][1]; R_2_errs = results["BB_2"]["scale"][1]
    
    fig = plt.figure(figsize=(8,8))  # create a figure object
    ax1 = fig.add_subplot(1, 1, 1)  # create an axes object in the figure 

    ax1.errorbar(wl,np.array(flux)/dist_factor, np.array(flux_err)/dist_factor,
                 linestyle='',marker = 'o',color = 'xkcd:red', label='Data')

    x = np.linspace(np.min(wl),np.max(wl),1000)
    y = []
    for item in x :
        y.append(c/item)
    
#    x = np.linspace(0.3,0.9,2000)
    ax1.plot(x,modified_bbody_absorbed(x,T_1,R_1,q=0)/dist_factor,
             label='BB_1, T= '+str(np.round(T_1,1))+"$_{-"+str(np.round(T_1_errs[0],0))+"}^{+"+str(np.round(T_1_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R_1,2))+"$_{-"+str(np.round(R_1_errs[0],2))+"}^{+"+str(np.round(R_1_errs[1],2))+"}$x10$^{15}$ cm")
    ax1.plot(x,modified_bbody_absorbed(x,T_2,R_2,q=q)/dist_factor,
             label='BB_2, T= '+str(np.round(T_2,1))+"$_{-"+str(np.round(T_2_errs[0],0))+"}^{+"+str(np.round(T_2_errs[1],0))+"}$ K Rad = " + \
             str(np.round(R_2,2))+"$^{+"+str(np.round(R_2_errs[0],2))+"}_{-"+str(np.round(R_2_errs[1],2))+"}$x10$^{15}$ cm")
    ax1.plot(x,modified_bbody_absorbed_2(x,T_1,T_2,R_1,R_2,q=q)/dist_factor, label='BB sum')
    #ax = plt.gca()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, \
           ncol=2, mode="expand", borderaxespad=0.)      
    #for key_filter in central_wav_dict.keys():
    #    ax1.axvline(central_wav_dict[key_filter]/1E10/1.067,lw=0.4,color='k',ls='--')
    #ax1.axes.set_xlim(wl[0]-1E3,wl[-1]+1E3)    
    #Here I am hacking for a specific plot, redo later!
    #ax1.set_yscale('log')
    #ax1.set_ylim(1E37,1E40)
    #Comment one out
    ax1.set_title("BB fit at:"+str(np.round(phase,0))+"d",fontsize=18,y=0)
    
    if log == True:
        ax1.axes.set_yscale("log")
        print(np.min(flux)/10/dist_factor,np.max(flux)/10/dist_factor)
        ax1.set_ylim(np.min(flux)/10/dist_factor,np.max(flux)*10/dist_factor)

    
    if save_loc != None :
        plt.savefig(save_loc,dpi=300,bbox_inches="tight",facecolor="white")
    plt.show()

def ln_likelihood(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower,scale_upper = args[3]
    T_lower,T_upper = args[4]
    distance = args[5]

    #print fl
    scale, T = theta
    if not (scale_upper >= scale >= scale_lower) \
    or not (T_upper >= T >= T_lower):
        return -np.inf
        print(scale, T)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = sc_func(scale,distance)*bb_function(wavelength_value, T)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def ln_likelihood_power_law(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower,scale_upper = args[3]
    q_lower,q_upper = args[4]
    distance = args[5]

    #print fl
    scale, q = theta
    if not (scale_upper >= scale >= scale_lower) \
    or not (q_upper >= q >= q_lower):
        return -np.inf
        print(scale, q)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = power_law(wavelength_value, q, scale)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def opacity_ln_likelihood(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower,scale_upper = args[3]
    T_lower,T_upper = args[4]
    distance = args[5]
    opacity_file = args[6]

    #print fl
    scale, T = theta
    if not (scale_upper >= scale >= scale_lower) \
    or not (T_upper >= T >= T_lower):
        return -np.inf
        print(scale, T)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = bbody_fit_with_opacity(wavelength_value, T, scale, opacity_file) / (4*np.pi*(distance)**2)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def opacity_ln_likelihood_2(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower_1,scale_upper_1 = args[3]
    scale_lower_2,scale_upper_2 = args[4]
    T_lower_1,T_upper_1 = args[5]
    T_lower_2,T_upper_2 = args[6]
    distance = args[7]
    opacity_file = args[8]

    #print fl
    scale_1, T_1, scale_2, T_2 = theta
    if not (scale_upper_1 >= scale_1 >= scale_lower_1) \
    or not (T_upper_1 >= T_1 >= T_lower_1) \
    or not (scale_upper_2 >= scale_2 >= scale_lower_1) \
    or not (T_upper_2 >= T_2 >= T_lower_2):
        return -np.inf
        print(scale_1, T_1, scale_2, T_2)
    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = bbody_fit_with_opacity_2(wavelength_value, T_1,T_2, scale_1,scale_2,opacity_file) / (4*np.pi*(distance)**2)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def power_law(wav,q,scale):
    """
    wav units are m
    """
    #T *= 1000
    #R *= 1e15

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10
    lam_cm = wav * 1E-8
    
    nu=c/lam_cm
    flux = scale * nu**q
    return flux

def bb_function(wav,temp):
    """
    wav units are m
    """
    h=const.h ; c  = const.c ; e = math.exp(1) ; k  = const.k 
    fr=c/wav
    flux = 2*h*fr**3 / (c**2*(e**(h*fr/(k*temp))-1))
    return flux

def sc_func(radius,distance):
    sca = 1E29*np.pi*radius**2/(distance*1E6)**2
    return sca

def ln_likelihood_wav(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower,scale_upper = args[3]
    T_lower,T_upper = args[4]
    distance = args[5]

    #print fl
    scale, T = theta
    if not (scale_upper >= scale >= scale_lower) \
    or not (T_upper >= T >= T_lower):
        return -np.inf
        print(scale, T)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = bbody_absorbed(wavelength_value,T,scale) / (4*np.pi*(distance)**2)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def modified_ln_likelihood_wav(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower,scale_upper = args[3]
    T_lower,T_upper = args[4]
    distance = args[5]

    #print fl
    scale, T= theta
    if not (scale_upper >= scale >= scale_lower) \
    or not (T_upper >= T >= T_lower):
        return -np.inf
        print(scale, T)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = modified_bbody_absorbed(wavelength_value,T,scale) / (4*np.pi*(distance)**2)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def ln_likelihood_wav_2(theta, *args):
    wl = args[0]
    fl = args[1]
    fl_err = args[2]
    scale_lower_1,scale_upper_1 = args[3]
    scale_lower_2,scale_upper_2 = args[4]
    T_lower_1,T_upper_1 = args[5]
    T_lower_2,T_upper_2 = args[6]
    distance = args[7]

    #print fl
    scale_1, T_1, scale_2, T_2 = theta
    if not (scale_upper_1 >= scale_1 >= scale_lower_1) \
    or not (T_upper_1 >= T_1 >= T_lower_1) \
    or not (scale_upper_2 >= scale_2 >= scale_lower_1) \
    or not (T_upper_2 >= T_2 >= T_lower_2):
        return -np.inf
        print(scale_1, T_1, scale_2, T_2)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = bbody_absorbed_2(wavelength_value,T_1,T_2,scale_1,scale_2) / (4*np.pi*(distance)**2)#3.086e24
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl)**2)/(np.array(fl_err)**2)
    return -(chi_sq.sum())

def modified_ln_likelihood_wav_2(theta, *args):
    wl = args[0]
    fl_Jy = args[1]
    fl_Jy_err = args[2]
    scale_lower_1,scale_upper_1 = args[3]
    scale_lower_2,scale_upper_2 = args[4]
    T_lower_1,T_upper_1 = args[5]
    T_lower_2,T_upper_2 = args[6]
    distance = args[7]

    #print fl
    scale_1, T_1, scale_2, T_2 = theta
    if not (scale_upper_1 >= scale_1 >= scale_lower_1) \
    or not (T_upper_1 >= T_1 >= T_lower_1) \
    or not (scale_upper_2 >= scale_2 >= scale_lower_1) \
    or not (T_upper_2 >= T_2 >= T_lower_2):
        return -np.inf
        print(scale_1, T_1, scale_2, T_2)

    model_flux = np.zeros(len(wl))

    for i, wavelength_value in enumerate(wl):
 
        model_flux_value = modified_bbody_absorbed_2(wavelength_value,T_1,T_2,scale_1,scale_2) / (4*np.pi*(distance)**2)
        model_flux[i] = model_flux_value

    chi_sq = ((model_flux - fl_Jy)**2)/(np.array(fl_Jy_err)**2)
    return -(chi_sq.sum())


def bbody_absorbed(x,T,R,lambda_cutoff=1,alpha=0):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in Kelvin
    R:   Radius in 10^15 cm

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    #T *= 1000
    R *= 1e15

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10

    # Convert wavelength to cm
    lam_cm = x * 1E-8

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
    exponential = (h * c) / (lam_cm * k_B * T)
    B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
   # B_lam[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha

    # Multiply by the surface area
    A = 4*np.pi*R**2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = B_lam * A / 1E8

    return Radiance #/ 1e40

def bbody_absorbed_2(x,T1,T2,R1,R2,lambda_cutoff=1,alpha=0):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in Kelvin
    R:   Radius in cm

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    #T1 *= 1000;T2 *= 1000
    R1 *= 1e15; R2 *= 1e15

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10

    # Convert wavelength to cm
    lam_cm = x * 1E-8

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
    exponential1 = (h * c) / (lam_cm * k_B * T1)
    exponential2 = (h * c) / (lam_cm * k_B * T2)

    B_lam1 = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential1) - 1)
    B_lam2 = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential2) - 1)

    #B_lam1[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha
    #B_lam2[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha

    # Multiply by the surface area
    A1 = 4*np.pi*R1**2;  A2 = 4*np.pi*R2**2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = B_lam1 * A1 / 1E8 + B_lam2 * A2 / 1E8

    return Radiance #/ 1e40

def modified_bbody_absorbed(wav,T,R,q=1.8,lambda_cutoff=1,alpha=0):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Here modified to be a grey body, with the default factor 1.8 (Draine and Lee 1984, van Velzen 2016)

    

    Parameters
    ---------------
    T:   Temperature in Kelvin
    R:   Radius in 10^15 cm

    Output
    ---------------
    Spectral radiance in units of Jy 

    (calculation and constants checked by Sebastian Gomez)
    '''


    #T *= 1000
    R *= 1e15 # radius -> cm

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27

    # Speed of light in cm/s
    c = 2.99792458E10
    # wav in Å, convert to cm
    freq = c  / (wav*1E-8) # Hz

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_nu, in units of (erg / s) / cm ^ 2 / Hz
    exponential = (h * freq) / (k_B * T)
    modification_factor = (freq) ** q
    #modification_factor = 1
    modification_factor = ((2*np.pi *0.1E-4 * freq)/c) ** q

    B_nu =modification_factor * ((2 * np.pi * h * freq ** 3) / (c ** 2))  / (np.exp(exponential) - 1) # why is there a pi?
    

    #B_lam = modification_factor * ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
   # B_lam[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha

    # Multiply by the surface area
    C = 4*np.pi*R**2

    # Output radiance in units of Jy
    Radiance = B_nu * C * 1E23 #/ 1E8

    return Radiance  #/ 1e40

def modified_bbody_absorbed_2(wav,T1,T2,R1,R2,q=1.8,lambda_cutoff=1,alpha=0):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Here BB 2 is modified to be a grey body, with the default factor 1.8 (Draine and Lee 1984, van Velzen 2016)

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in Kelvin
    R:   Radius in cm

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    #T1 *= 1000;T2 *= 1000
    R1 *= 1e15; R2 *= 1e15

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10
    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    lam_cm = wav * 1E-8 # wavelength in cm
    freq = c / lam_cm
    #print(freq)
    # Calculate Radiance B_nu, in units of (erg / s) / cm ^ 2 / Hz
    exponential1 = (h * freq) / (k_B * T1)
    B_nu1 =  ((2 * np.pi * h * freq ** 3) / (c ** 2))  / (np.exp(exponential1) - 1) # why is there a pi?
    
    # Calculate Radiance B_nu, in units of (erg / s) / cm ^ 2 / Hz
    exponential2 = (h * freq) / (k_B * T2)
    #modification_factor = (freq)**q
    modification_factor = ((2*np.pi *0.1E-4 * freq)/c) ** q
    B_nu2 = modification_factor * ((2 * np.pi * h * freq ** 3) / (c ** 2))  / (np.exp(exponential2) - 1) # why is there a pi?

    #B_lam1[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha
    #B_lam2[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha

    # Multiply by the surface area
    A1 = 4*np.pi*R1**2;  A2 = 4*np.pi*R2**2

    # Output radiance in units of (erg / s) / Angstrom
    #Radiance = B_nu1 * A1 / 1E8 + B_nu2 * A2 / 1E8
    Radiance = B_nu1 * A1 *1E23 + B_nu2 * A2 *1E23

    return Radiance #/ 1e40


def bbody_fit_with_opacity(wav,T,R,opacity_dict):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Parameters
    ---------------
    T:   Temperature in Kelvin
    R:   Radius in 10^15 cm

    Output
    ---------------
    Spectral radiance in units of Jy 

    (calculation and constants checked by Sebastian Gomez)
    '''


    #T *= 1000
    R *= 1e15 # radius -> cm

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27

    # Speed of light in cm/s
    c = 2.99792458E10
    # wav in Å, convert to cm
    freq = c  / (wav*1E-8) # Hz
    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_nu, in units of (erg / s) / cm ^ 2 / Hz
    exponential = (h * freq) / (k_B * T)
    B_nu = ((2 * h * np.pi * freq ** 3) / (c ** 2))  / (np.exp(exponential) - 1)

    if isinstance(wav, float) :
        opacities =opacity_dict[wav]
    else :
        opacities = []
        for wl in wav:
            opacity = opacity_dict[wl]
            opacities.append(opacity)
        opacities=np.array(opacities)  

    B_nu_with_opacity = B_nu * opacities
    
    #B_lam = modification_factor * ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
   # B_lam[x <= lambda_cutoff] *= (x[x <= lambda_cutoff]/lambda_cutoff)**alpha
    # Multiply by the surface area
    C = 4*np.pi*R**2

    # Output radiance in units of Jy
    Radiance = B_nu_with_opacity * C * 1E23 #/ 1E8

    return Radiance  #/ 1e40

def bbody_fit_with_opacity_2(wav,T1,T2,R1,R2,opacity_dict):
    '''
    Calculate the blackbody radiance for a set
    of wavelengths given a temperature and radiance.
    Modified in the UV

    Parameters
    ---------------
    T:   Temperature in Kelvin
    
    Output
    ---------------
    Spectral radiance in units of Jy 
    '''


    #T *= 1000
    R1 *= 1e15; R2 *= 1e15

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10
    # wav in Å, convert to cm
    freq = c  / (wav*1E-8) # Hz
    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_nu, in units of (erg / s) / cm ^ 2 / Hz
    exponential1 = (h * freq) / (k_B * T1)
    B_nu1 =  ((2 * np.pi * h * freq ** 3) / (c ** 2))  / (np.exp(exponential1) - 1) # why is there a pi?

    # Calculate Radiance B_nu, in units of (erg / s) / cm ^ 2 / Hz
    exponential2 = (h * freq) / (k_B * T2)
    B_nu2 = ((2 * np.pi * h * freq ** 3) / (c ** 2))  / (np.exp(exponential2) - 1)

    if isinstance(wav, float) :
        opacities =opacity_dict[wav]
    else :
        print("this should only show up once!")
        opacities = []
        for wl in wav:
            opacity = opacity_dict[wl]
            opacities.append(opacity)
        opacities=np.array(opacities)  
    
    B_nu_with_opacity2 = B_nu2 * opacities

    # Multiply by the surface area
    A1 = 4*np.pi*R1**2;  A2 = 4*np.pi*R2**2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = (B_nu1 * A1 + B_nu_with_opacity2 * A2 ) * 1E23

    return Radiance  #/ 1e40





        

