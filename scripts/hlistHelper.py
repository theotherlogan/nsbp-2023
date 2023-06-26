import os
import pickle
import numpy as np
from helpers.SimulationAnalysis import SimulationAnalysis, readHlist

class hlist():

    '''
A helper class intended to read and manipulate uncompressed halo lists (hlists) produced by the Rockstar halo finder. This expands on the helper scripts found here: https://bitbucket.org/yymao/helpers/src/master/.
'''

    def __init__(self, PATH: str = '/central/groups/carnegie_poc/enadler/ncdm_resims/', halo_id: str = 'Halo004', model: str = 'cdm', hmb: np.ndarray = []) -> None:

        #...... can always add other admin/internal parameters here later.
        
        #...... internal path has been updated for cluster!
        if model == 'cdm':
            self.PATH = os.path.join(PATH, halo_id, model, 'output/rockstar/hlists') # sets total path
        else:
            self.PATH = os.path.join(PATH, halo_id, model, f'output_{model}/rockstar/hlists') # sets total path
        
        self.halo_id = halo_id
        self.model = model
        self.dict = {}
        
        #...... this has now been updated to pull the correct compressed data archive from the cluster.
        self.hmb = hmb # can be set during initialization or extracted later on.
        
        #...... storing the corresponding cdm model within the data object (only if the model is not cdm) for comparison/future expansion.
        if model == 'cdm':
            self.cdm = None
        else:
            self.cdm = hlist(self.PATH, self.halo_id, 'cdm')
            

            
    def load_hlists(self) -> bool:
        '''
        Loads the halo lists from a given path, halo, and dark matter model and sorts them into a dictionary by scale factor. Note that the default model is CDM.

        Returns True if the dictionary is populated.
        '''

        self.dict = {
        float(hlist[6:-5]): hlist
        for hlist in np.sort(os.listdir(self.PATH))
        }
        
        return True if self.dict != {} else False
        

        
    def load_hmb(self, high_res: bool = False) -> None:
        '''
        Extracts the main branch of the host halo, or the most 'recent' snapshot of the host galaxy and the surrounding halo (z = 0; a = 1).
        '''
        # sets the path based on simulation resolution (defualt is 8K)
        if high_res:
            PATH = '/central/groups/carnegie_poc/enadler/ncdm_resims/analysis/sim_data_16K.bin'
        else:
            PATH = '/central/groups/carnegie_poc/enadler/ncdm_resims/analysis/sim_data.bin'
            
        with open(PATH, "rb") as f:
            sim_data = pickle.load(f, encoding='latin1')
            
        self.hmb = sim_data[self.halo_id][self.model][0] # sets hmb
           
            

    def extract_halos(self, a: float, get_host_ind: bool = False) -> np.ndarray:
        '''
        Extracts the halo population from a given main host branch and returns the isolated halo population and subhalo population.
        '''

        # TODO: check to see if we need to check against a compressed HMB 

        halos = readHlist(os.path.join(self.PATH, self.dict[a])) # reads in all halos
        isolated_halos = halos[halos['upid'] == -1] # gets isolated population

        host_ind = np.argmin(np.abs(self.hmb['scale'] - a)) # smallest difference between hmb and desired scale factor.
        subhalos = halos[halos['upid'] == self.hmb[host_ind]['id']]
        ###
        if get_host_ind:
            return isolated_halos, subhalos, host_ind
        else:
            return isolated_halos, subhalos

        
        
    def get_z(self, z: float, get_host_ind: bool = False) -> np.ndarray:
        '''
        Returns the isolated halo population and subhalo population for a given redshift (z) using the closest absolute value of z in the hlist dictionary.
        '''

        return self.get_a( 1.0 / (1.0 + z), get_host_ind )

    
    
    def get_a(self, a: float, get_host_ind: bool = False) -> np.ndarray:
        '''
        Returns the isolated halo population and subhalo population for a given scale (a) using the closest absolute value of a in the hlist dictionary.
        '''
        scale_factors = np.array(list(self.dict.keys())) # list of scale factors
        scale = scale_factors[np.argmin(np.abs(scale_factors - a))] # closest scale factor

        return self.extract_halos(scale, get_host_ind)


    
    def hmf(self, z: float, bins: np.ndarray = np.linspace(5,11,10), return_masscut_idx: bool = False):
        '''
        Returns the isolated halo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = halos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(halos['Mpeak'][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        if return_masscut_idx:
            return values, cumulative_values, base, dist_ind_cdm
        else:
            return values, cumulative_values, base
    
    
    
    def hmf_plottables(self, z: float, bins: np.ndarray = np.linspace(5,11,10)):
        '''
        Returns the x and y values for the isolated halo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = halos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(halos['Mpeak'][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        return base[1:], len(halos['Mpeak'][dist_ind_cdm])-cumulative_values

    
    
    def shmf(self, z: float, bins: np.ndarray = np.linspace(5,11,10), return_masscut_idx: bool = False):
        '''
        Returns the subhalo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = subhalos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(subhalos['Mpeak'][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        if return_masscut_idx:
            return values, cumulative_values, base, dist_ind_cdm
        else:
            return values, cumulative_values, base
    
    
    
    def shmf_plottables(self, z: float, bins: np.ndarray = np.linspace(5,11,10)):
        '''
        Returns the x and y values for the subhalo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = subhalos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(subhalos['Mpeak'][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        return base[1:], len(subhalos['Mpeak'][dist_ind_cdm])-cumulative_values