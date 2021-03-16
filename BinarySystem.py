import numpy as np
import pandas as pd


class BinarySystem:
    """ A class for simulating and analysing the various steps in 
        stellar evolution in a binary system containing a massive
        primary star in the mass range 8-30 solar masses"""
        
    def __init__(self, M1, M2, a):
        """
        Takes initial arguments:
        -   M1 = mass of primary
        -   M2 = mass of secondary
        -   a = Initial semi-major axis
        """
        self.M1 = int(M1) # Must be an integer value
        self.M2 = M2
        self.a = a
        
        # Assume the primary star is more massive
        if M2 > M1:
            raise ValueError('M2 cannot be larger than M1')
        if M1 < 8 or M1 > 30:
            raise ValueError('M1 must be between 8 and 30 solar masses')

    def __str__(self):
        return('Primary Mass: ' + str(self.M1) + '\nSecondary Mass: ' +
               str(self.M2) + '\nSeparation: ' + str(self.a))

    def get_data(self):
        """
        Read in files with Hertzsprung-Russel track of the primary and save to
        pandas dataframe. Columns: time (years), radius (logRsun),
        temperature (log K), luminosity (log Lsun), Mass (Msun),
        Core Mass (Msun)
        """
        # Get the HR track of the primary
        file = 'DataFiles/{}Msun.dat'.format(self.M1)
        
        # Read and convert the data to a pandas DataFrame
        names = ['num', 'time', 'Radius', 'Temp', 'Lum', 'M1', 'Mcore']
        File = pd.read_table(file, header=None, sep=' ',
                             names=names, index_col=False).astype(float)
        
        # Clean the data
        File.drop('num', axis=1, inplace=True) # Drop the index
        File['Radius'] = 10**File['Radius']  # Convert log values to float
        File = File[['M1', 'Mcore', 'Radius', 'Temp', 'Lum', 'time']]
        
        self.File = File
        return self.File
    
    def RLOF(self, massLoss=True, printValues=False):
        """
        -   Calculates if/when primary fills its Roche lobe.

        -   When Radius of primary > Roche lobe radius, dataframe appended
            with updated values up to onset of RLOF.

        -   Implements mass loss by default, where semi-major axis is
            amended at each step such that (a * M_total = constant).

        -   Roche lobe radius calculated at each step using Eggleton fitting
            formula.

        -   If mass loss is not implemented, the semi-major axis and
            Roche lobe radius are constant. Chiefly for demonstration purposes.

        -   Columns for mass of secondary M2, semi-major axis a, and
            Roche lobe radius RL inserted into dataframe.

        -   printValues arg set to false as default. If one wishes to print
            the Roche lobe radius, semi-major axis and number of iterations
            at the onset of RLOF, set printValues=True. For single instance, could
            be helpful, but if running multiple instances, use default.
            If the Roche lobe is not filled, printValues arg alerts user.
            """
        
        self.File = self.get_data() # Ensures time t=0. Allows skipping of get_file
        
        # Simple case where effects of mass loss in primary star are ignored
        if massLoss is not True:
            q = self.M1/self.M2  # mass ratio
            
            # Calculate RocheLobe radius with Eggleton's formula
            RocheLobe = ((0.49 * (q)**(2/3)) /
                         (0.6 * (q)**(2/3) + np.log(1 + (q)**(1/3)))
                         * self.a)
            
            self.File.insert(4, 'Roche', RocheLobe)
            self.File.insert(2, 'semi-major axis', self.a)
            self.File.insert(1, 'M2', self.M2)
            
            for i in range(len(self.File)):
                if self.File.Radius[i] > RocheLobe:
                    if printValues is True:
                        print("Roche Lobe radius {0:.2f}: ".format(RocheLobe),
                              " after ", i, " iterations")
                    self.File = self.File.iloc[:i+1]
                    break
                elif self.File.Radius.max() < RocheLobe:
                    if printValues is True:
                        print("Roche lobe not filled!")
                    break
            return self.File
        
        else:   # Accounting for effect of mass loss on orbital separation
            const = (self.M1 + self.M2) * self.a
            a = []
            for i in range(len(self.File)):
                a.append(const / (self.File.M1[i] + self.M2)) 

            a = pd.Series(a)
            self.File.insert(2, 'semi-major axis', a)

            q = []  # A list to hold the changing mass ratio values
            RL = [] # A list to hold the changing Roche Lobe Radius values
            for k in range(len(self.File)):
                q.append(self.File.M1[k]/self.M2)
                RL.append((0.49 * (q[k])**(2/3)) /
                          (0.6 * (q[k])**(2/3) + np.log(1 + (q[k])**(1/3)))
                          * self.File['semi-major axis'][k])

            RL = pd.Series(RL)
            self.File.insert(4, 'Roche', RL)
            self.File.insert(1, 'M2', self.M2)

            # Check to see if the Roche lobe is filled
            for i in range(len(self.File)):
                if self.File.Radius[i] > self.File.Roche[i]:
                    if printValues is True:
                        print('Roche lobe radius {0:.2f}'.format(RL[i]))
                        print('Run through', i, 'iterations: ')
                        print("Semi-major axis {0:.3f}"
                              .format(self.File['semi-major axis'][i]))
                    self.File = self.File.iloc[:i+1]
                    break
                elif self.File.Radius.max() < self.File.Roche.max():
                    if printValues is True:
                        print("Roche lobe not filled!")
                    break

            return self.File
        
    def plot_HRD(self, color='blue'):
        """ Simple plot of HR-Diagram. 
            Should be implemented after RLOF function
            For other plots, pandas plot or pyplot can be used
            with self.File['Column name'] as x, y arguments
            """
        self.File = self.RLOF()           
        
        ax = self.File.plot('Temp', 'Lum', 
               title='HR Diagram for {} M$_\odot$ Star upto RLOF onset'.format(self.M1), 
               legend=None,
               linestyle='dotted',
               color=color)
        
        ax.set(xlabel="log $T_{eff}$ (k)", ylabel="log L $(L_\odot)$")
        ax.invert_xaxis()
        
        textstr = "\n".join((r"a = {0:.2f}".format(self.a), r"q = {0:.2f}".format(self.M1 / self.M2)))
        props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
        ax.text(0.75, 0.15, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props) 
   
    def __get_vals__(self):
        """
        Returns values at last instance
        """
        
        return self.File.iloc[-1]

    def mass_transfer(self, printValues=False):
        """
        -   Implement mass transfer.

        -   Stability criteria: mass ratio q = M1 / M2.
            Case A: Mcore < 1.4 ----> Merger
            Case B: if primary in RSG phase
                        If q <= 5/6, implement stable mass transfer.
                    if primary crossing the Hertzsprung gap
                        If q <= 4, implement stable mass transfer.
                    Else implement unstable mass transfer.
            Case C: Unstable mass transfer.

        -   Assume all mass transferred in one step after RLOF onset.

        -   Returns the updated dataframe with values after mass transfer
        """
        File = self.RLOF()
        File['Case'] = File.Lum*File.Temp  # Estimate commencement of RSG phase
        envelope = File['Case'].idxmin() # get the envelope status
        
        # Test for non-interacting wide binary
        if len(self.get_data()) == len(self.RLOF()):    #  RL not filled
            if printValues == True:
                print("Wide binary: No mass transfer")
            self.File.Case.iloc[-1] = 'Wide Binary'
            return self.File                                
        
        else:
            M_env = self.File.M1.iloc[-1] - self.File.Mcore.iloc[-1]
            M1 = self.File.M1.iloc[-1]
            M2 = self.File.M2.iloc[-1]
            Mc = self.File.Mcore.iloc[-1]
            a_i = self.File['semi-major axis'].iloc[-1]
            
            # Test for merger on main sequence
            if Mc < 1.4:
            
                if printValues == True:
                    print('Main sequence')
                
                merger = {'time': self.File.time.iloc[-1]*1.05,
                           'M2': self.M2 + M1 - Mc,
                           'M1' : 0, 
                           'Case': 'Merger'}
                
                self.File = self.File.append(merger, ignore_index=True)
                
                return self.File
            
            elif Mc >= 1.4:
                
                # Check if primary on Hertzsprung gap
                if len(self.File) <= envelope:
                
                    # Test for stable mass transfer
                    if M1 / M2 <= 4:
                        if printValues == True:
                            print("Implement stable mass transfer")
                
                        a_final = a_i * ((M1 * M2) / (Mc * (M2 + M_env)))**2
                
                        mt_data = {'time': self.File.time.iloc[-1]*1.05,
                               'M1': Mc,
                               'semi-major axis': a_final,
                               'M2': M2 + M1 - Mc,
                               'Case': 'Stable'}
                
                        mt_data = pd.Series(mt_data)
                        self.File = self.File.append(mt_data, ignore_index=True)
                
                        return self.File
                    else:
                        a_final = (a_i / ((2 * M1 * M_env) / ((Mc * M2 * 0.5) +1)))
                        if printValues == True:
                            print("Implement unstable mass transfer")
            
                        mt_data = {'time': self.File.time.iloc[-1] * 1.05,
                                   'M1': Mc,
                                   'semi-major axis': a_final,
                                   'M2': M2,
                                   'Case': 'Unstable'}
                
                        mt_data = pd.Series(mt_data)
                        self.File = self.File.append(mt_data, ignore_index=True)
                        
                        return self.File
                
                # Check if mass loss has reversed the mass ratio by
                # the onset of RLOF
                elif M1 / M2 < 0.8:
                    if printValues == True:
                        print("Implement stable mass transfer")
                
                    a_final = a_i * ((M1 * M2) / (Mc * (M2 + M_env)))**2
                
                    mt_data = {'time': self.File.time.iloc[-1]*1.05,
                           'M1': Mc,
                          'semi-major axis': a_final,
                          'M2': M2 + M1 - Mc,
                          'Case': 'Stable'}
                
                    mt_data = pd.Series(mt_data)
                    self.File = self.File.append(mt_data, ignore_index=True)
                
                    return self.File
                
                # The primary is in the RSG phase and is not significantly
                # less massive than the secondary
                else:
                    a_final = (a_i / ((2 * M1 * M_env) / ((Mc * M2 * 0.5) +1)))
                    if printValues == True:
                        print("Implement unstable mass transfer")
            
                    mt_data = {'time': self.File.time.iloc[-1] * 1.05,
                               'M1': Mc,
                               'semi-major axis': a_final,
                               'M2': M2,
                               'Case': 'Unstable'}
                
                    mt_data = pd.Series(mt_data)
                    self.File = self.File.append(mt_data, ignore_index=True)

                    return self.File

    def supernova(self, V_k=0, size=100):
        """ The primary explodes as a supernova.
            Establish whether the system remains bound
            or if the secondary star is ejected from the system
            to become a runaway star.
            
            keyword arguments:
                V_k  -- the magnitude of the velocity kick (default = 0)
                size -- the number of random values generated for the x and y
                        directions of the velocity kick (default = 100)
                
            returns an array of size=size of the total energy in the system after 
            the supernova explosion for velocity kicks in various directions.           
            If the total energy is negative the system remains bound otherwise
            unbound
            """
        M1 = self.__get_vals__().M1
        M2 = self.__get_vals__().M2
        a = self.__get_vals__()['semi-major axis']
        
        Mc = 1.4 
        mu = (Mc * M2) / (Mc + M2)
        G = 1.9e5
        
        V_orb = np.sqrt((G * (M1 + M2)) / a)
        
        if V_k == 0:
            V = V_orb
        else:
            xi = np.random.uniform(0, 1, size)
            xj = np.random.uniform(0, 1, size)
            theta = 2 * np.pi * xi
            z = 2 * xj - 1
            R = np.sqrt(1 - z**2)
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            self.x, self.y, self.z, self.theta  = x, y, z, theta
        
            V = np.sqrt((V_orb + (V_k * self.x))**2 + 
                    (V_k * self.y)**2 + (V_k * self.z)**2)
        
        E_kin = 0.5 * mu * V**2
        E_pot = - ( G * Mc * M2 ) / ( a )
        E_tot = E_kin + E_pot
        
        self.E_pot = E_pot
        self.E_kin = E_kin
        self.Del_M = M1 - Mc
        self.V_orb = V_orb
        self.V = V
        
        self.E_tot = E_tot
        return self.E_tot
    
    def escape_velocity(self):
        """ 
            Determine the minimum escape velocity of the pre-SN system
        """
        M1 = self.File.M1.iloc[-1]
        M2 = self.File.M2.iloc[-1]
        a = self.File['semi-major axis'].iloc[-1]
        v_esc = (M1 / (M1 + M2)) * self.V_orb
        self.v_esc = v_esc
        return self.v_esc
    