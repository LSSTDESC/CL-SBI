''' 
To give the individual tests import context
'''

import os
import sys

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import weaklensclustersbi
from weaklensclustersbi.simulations import population, wlprofile
