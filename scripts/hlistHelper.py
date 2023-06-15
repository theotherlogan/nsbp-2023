import os
import pickle
import numpy as np
from helpers.SimulationAnalysis import SimulationAnalysis, readHlist

class hlist():

    '''
A helper class intended to read and manipulate uncompressed halo lists (hlists) produced by the Rockstar halo finder. This expands on the helper scripts found here: https://bitbucket.org/yymao/helpers/src/master/.
'''

    def __init__(self, PATH: str = '/central/groups/carnegie_poc/enadler/ncdm_resims/', halo_num: str = '001', model: str = 'cdm', hmb: np.ndarray = []) -> None:

        # TODO: come back and check this path structure once we have server access
        #...... can always add other admin/internal parameters here later.
        self.PATH = os.path.join(PATH, halo_num, f'hlists_{model}') # sets total path
        self.halo_num = halo_num
        self.model = model
        self.dict = {} # questionable naming here?
        self.hmb = hmb # can be set during initialization or extracted later on.

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
        

    def extract_main_branch(self) -> None:
        '''
        Extracts the main branch of the host halo, or the most 'recent' snapshot of the host galaxy and the surrounding halo (z = 0; a = 1).
        '''
        # TODO: fix this; need host halo id at minimum (also need mass accretion history -> maybe a method for this?)
        pass

    def extract_halos(self, a: float) -> np.ndarray:
        '''
        Extracts the halo population from a given main host branch and returns the isolated halo population and subhalo population.
        '''

        # TODO: check to see if we need to check against a compressed HMB 

        halos = readHlist(os.path.join(self.PATH, self.dict[a])) # reads in all halos
        isolated_halos = halos[halos['upid'] == -1] # gets isolated population

        host_ind = np.argmin(np.abs(self.hmb['scale'] - a)) # smallest difference between hmb and desired scale factor.
        subhalos = halos[halos['upid'] == self.hmb[host_ind]['id']]
        ###
        return isolated_halos, subhalos

    def get_z(self, z: float) -> np.ndarray:
        '''
        Returns the isolated halo population and subhalo population for a given redshift (z) using the closest absolute value of z in the hlist dictionary.
        '''

        return self.get_a( 1.0 / (1.0 + z) )

    def get_a(self, a: float) -> np.ndarray:
        '''
        Returns the isolated halo population and subhalo population for a given scale (a) using the closest absolute value of a in the hlist dictionary.
        '''
        scale_factors = np.array(list(self.dict.keys())) # list of scale factors
        scale = scale_factors[np.argmin(np.abs(scale_factors - a))] # closest scale factor

        return self.extract_halos(scale)


# TODO: check that these are spitting out the right deliverables.
    def hmf(self, z: float, independent_var: str = 'Mpeak', bins: np.ndarray = np.linspace(5,11,10)):
        '''
        Returns the isolated halo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = halos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(halos[independent_var][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        return values, cumulative_values, base
    
    def hmf_plottables(self, z: float, independent_var: str = 'Mpeak', bins: np.ndarray = np.linspace(5,11,10)):
        '''
        Returns the x and y values for the isolated halo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = halos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(halos[independent_var][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        return base[1:], len(halos[independent_var][dist_ind_cdm])-cumulative_values

    def shmf(self, z: float, independent_var: str = 'Mpeak', bins: np.ndarray = np.linspace(5,11,10)):
        '''
        Returns the subhalo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = subhalos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(subhalos[independent_var][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        return values, cumulative_values, base
    
    def shmf_plottables(self, z: float, independent_var: str = 'Mpeak', bins: np.ndarray = np.linspace(5,11,10)):
        '''
        Returns the x and y values for the subhalo mass function for a given redshift.
        '''
        halos, subhalos = self.get_z(z)

        dist_ind_cdm = subhalos['Mvir']/0.7 > 1.2e8 # mass and particle cut, by index

        values, base = np.histogram(np.log10(subhalos[independent_var][dist_ind_cdm]/0.7), bins=bins)
        cumulative_values = np.cumsum(values)

        return base[1:], len(subhalos[independent_var][dist_ind_cdm])-cumulative_values